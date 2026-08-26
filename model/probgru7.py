import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def haversine_km(coords):
    """coords: [N, 2] lat/lon in degrees -> [N, N] pairwise distance in km"""
    lat = torch.deg2rad(coords[:, 0])
    lon = torch.deg2rad(coords[:, 1])
    dlat = lat.unsqueeze(1) - lat.unsqueeze(0)
    dlon = lon.unsqueeze(1) - lon.unsqueeze(0)
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat).unsqueeze(1) * torch.cos(lat).unsqueeze(0) * torch.sin(dlon / 2) ** 2
    return 2 * 6371.0 * torch.asin(torch.sqrt(a.clamp(min=0)))


def bearing_deg(coords):
    """coords: [N, 2] lat/lon in degrees -> [N, N] bearing FROM i TO j, in radians
    (standard forward-azimuth formula: 0 = north, pi/2 = east, clockwise)."""
    lat = torch.deg2rad(coords[:, 0])
    lon = torch.deg2rad(coords[:, 1])
    dlon = lon.unsqueeze(0) - lon.unsqueeze(1)          # [N, N], j - i
    y = torch.sin(dlon) * torch.cos(lat).unsqueeze(0)
    x = torch.cos(lat).unsqueeze(1) * torch.sin(lat).unsqueeze(0) - \
        torch.sin(lat).unsqueeze(1) * torch.cos(lat).unsqueeze(0) * torch.cos(dlon)
    return torch.atan2(y, x)                             # [N, N], i -> j


def _inv_softplus(y):
    """Initialize a raw parameter so that softplus(raw) == y (y > 0)."""
    return math.log(math.expm1(y))


class SpatiotemporalDispersionKernel(nn.Module):
    """
    Implements, as literally as the design allows, the pipeline:

        H_j(t) = sum_{i,k} alpha_ij(t,k) * C_i(t-k)
        alpha_ij(t,k) = normalize_i,k[ A_ij_adp * K_phys_ij(t,k) ]
        K_phys_ij(t,k) = K_time_ij(t,k) * K_space_ij(t,k)

    i.e. for target station j, pull a physics- and graph-weighted average
    of every OTHER station's PM2.5 reading from every recent lag k, where
    the weight is high exactly when (a) a learned station-to-station
    relationship says i matters to j at all, AND (b) station i's own wind
    reading AT THAT LAGGED TIME implies its emitted air would plausibly
    have reached j's location by now.

    K_time - the genuinely new piece relative to probgru5/6 - answers "is
    lag k the RIGHT lag for this (i,j) pair, given how fast i's air was
    moving k steps ago": for source i's wind speed u_i(t-k), the implied
    travel time to cover the i-j distance is tau_ij(t-k) = d_ij / u_i(t-k)
    (this uses the straight-line distance, not its downwind projection -
    matching the proposal's tau=d/u literally; directional correctness is
    enforced separately by K_space's downwind gate). K_time scores how
    close k's actual elapsed time (k * dt_hours) is to that implied travel
    time:

        K_time_ij(t,k) = exp[-(k*dt_hours - tau_ij(t-k))^2 / (2*sigma_tau^2)]

    K_space reuses probgru5/6's anisotropic advection-diffusion kernel
    (downwind/crosswind decomposition of the i->j offset relative to i's
    wind bearing, Gaussian crosswind spread widening with tau, smooth
    downwind gate, multiplicative terrain attenuation) rather than
    re-deriving the proposal's separate "directional Gaussian on the angle
    alone" (its step 4) plus "crosswind Gaussian on distance alone" (its
    step 7): applying both would score angular deviation twice, once
    through the raw angle and again through the distance the angle itself
    produces. One physically-composed spatial kernel, reused verbatim from
    the version that was already checked for the y_ij==0 upwind-degeneracy
    bug, is used instead. It is evaluated with i's wind AT LAG k (not
    "now"), since propagation from k steps ago depends on the wind that
    was blowing THEN.

    A_ij_adp = softmax_j(relu(E_c @ E_r^T)) is a small, STATIC (independent
    of t/k) learned station-to-station embedding, exactly as specified -
    kept at a modest `adaptive_dim` given v5's lesson that a large version
    of this term overfit a small training set.

    NORMALIZATION - one deliberate deviation from the proposal's literal
    per-(j,k) normalization over i only: that leaves every lag's alpha
    summing to 1 over i regardless of how physically implausible that lag
    is in absolute terms (a bad lag still gets fully redistributed weight
    among sources, just not down-weighted relative to a good lag). This
    class instead normalizes jointly over the combined (source i, lag k)
    index for each target j, so an implausible lag is actually suppressed
    in the final sum rather than merely reshuffled - closer to what "the
    model learns from the physically plausible historical lag" is meant to
    achieve. Both are one softmax call apart if this turns out to matter
    empirically.
    """
    def __init__(self, station_coords, station_elevation, adaptive_dim=8,
                 dist_threshold_km=300.0, sigma_h=1200.0,
                 diffusivity_init_km2h=50.0, sigma_min_km=15.0,
                 sigma_tau_init_h=3.0, downwind_gate_steepness_init=4.0,
                 dt_hours=3.0, speed_floor_kmh=0.5, eps=1e-6):
        super().__init__()

        self.eps = eps
        self.dt_hours = dt_hours
        self.sigma_min_km = sigma_min_km
        self.speed_floor_kmh = speed_floor_kmh
        self._sqrt_2pi = math.sqrt(2.0 * math.pi)

        N = station_coords.shape[0]
        self.node_emb_c = nn.Parameter(torch.randn(N, adaptive_dim) * 0.1)
        self.node_emb_r = nn.Parameter(torch.randn(N, adaptive_dim) * 0.1)

        self.log_D_base = nn.Parameter(torch.tensor(_inv_softplus(diffusivity_init_km2h)))
        self.w_pbl = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_tau = nn.Parameter(torch.tensor(_inv_softplus(sigma_tau_init_h)))
        self.gate_steepness_raw = nn.Parameter(torch.tensor(_inv_softplus(downwind_gate_steepness_init)))

        dist = haversine_km(station_coords)                           # [N, N], km
        neighbor_mask = dist <= dist_threshold_km
        self.register_buffer('neighbor_mask', neighbor_mask)
        self.register_buffer('dist_km', dist)
        self.register_buffer('bearing', bearing_deg(station_coords))  # [N, N], i -> j

        elev_diff = (station_elevation.unsqueeze(1) - station_elevation.unsqueeze(0)).abs()
        terrain_bias = torch.exp(-elev_diff / sigma_h)
        self.register_buffer('terrain_bias', terrain_bias)

    def forward(self, travel_bearing_lag, wind_speed_kmh_lag, pm25_lag, k_hours, pbl_lag=None):
        """
        travel_bearing_lag  : [B, K, N] radians - compass bearing each
                               station's air was heading toward AT that lag
        wind_speed_kmh_lag  : [B, K, N] wind speed (km/h) AT that lag
        pm25_lag            : [B, K, N] PM2.5 reading AT that lag (the C_i(t-k)
                               being propagated - same normalized scale
                               used throughout training)
        k_hours             : [K] elapsed time k*dt_hours for each lag,
                               oldest-lag-agnostic (any order, just must
                               match the lag axis of the other tensors)
        pbl_lag             : [B, K, N] normalized boundary-layer height at
                               that lag, or None
        returns              : H [B, N] transported-pollution feature per
                               target station, alpha [B, K, N, N] weights
                               (for inspection)
        """
        B, K, N = travel_bearing_lag.shape

        theta = travel_bearing_lag.unsqueeze(3) - self.bearing.view(1, 1, N, N)  # [B, K, N, N]
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        dist_b = self.dist_km.view(1, 1, N, N)
        y_ij = dist_b * sin_t                                                     # crosswind offset

        D_base = F.softplus(self.log_D_base)                                     # scalar, km^2/h
        if pbl_lag is not None:
            pbl_mod = torch.clamp(self.w_pbl * pbl_lag, -5.0, 5.0)
            D = D_base * torch.exp(pbl_mod)                                      # [B, K, N]
        else:
            D = D_base.view(1, 1, 1).expand(B, K, N)

        speed = wind_speed_kmh_lag.clamp(min=0.0)
        tau = dist_b / (speed.unsqueeze(3) + self.speed_floor_kmh)               # [B, K, N, N], hours

        # --- K_time: does lag k's elapsed time match i's implied travel time? ---
        sigma_tau = F.softplus(self.log_sigma_tau) + self.eps
        k_hours_b = k_hours.view(1, K, 1, 1)
        k_time = torch.exp(-((k_hours_b - tau) ** 2) / (2.0 * sigma_tau ** 2))

        # --- K_space: anisotropic advection-diffusion, wind AT that lag ---
        sigma_sq = 2.0 * D.unsqueeze(3) * tau.clamp(min=0.0) + self.sigma_min_km ** 2
        sigma = torch.sqrt(sigma_sq)
        gate_k = F.softplus(self.gate_steepness_raw)
        downwind_gate = torch.sigmoid(gate_k * cos_t)
        k_space = downwind_gate / (sigma * self._sqrt_2pi) * torch.exp(-(y_ij ** 2) / (2.0 * sigma_sq))
        k_space = k_space * self.terrain_bias.view(1, 1, N, N)

        k_phys = (k_time * k_space).clamp_min(self.eps)                          # [B, K, N, N]

        # --- static learned adjacency, row i normalized over receivers j ---
        a_adp = F.softmax(F.relu(torch.mm(self.node_emb_c, self.node_emb_r.t())), dim=1)  # [N, N]

        log_score = torch.log(a_adp).view(1, 1, N, N) + torch.log(k_phys)        # [B, K, N, N]
        log_score = log_score.masked_fill(~self.neighbor_mask.view(1, 1, N, N), float('-inf'))

        # joint softmax over (k, i) per target j - see class docstring
        flat = log_score.permute(0, 3, 1, 2).reshape(B, N, K * N)                # [B, N_j, K*N_i]
        alpha_flat = torch.nan_to_num(torch.softmax(flat, dim=-1), nan=0.0)
        alpha = alpha_flat.reshape(B, N, K, N).permute(0, 2, 3, 1)               # [B, K, N_i, N_j]

        H = (alpha * pm25_lag.unsqueeze(3)).sum(dim=(1, 2))                      # [B, N_j]
        return H, alpha


class ProbGRUModel7(nn.Module):
    """
    A structurally different encoder from probgru4-6: rather than folding
    spatial information through a recurrent per-step hidden state, this
    builds an EXPLICIT spatiotemporal feature H_j - a physics- and learned-
    graph-weighted average of every neighboring station's PM2.5 reading at
    every recent historical lag (see SpatiotemporalDispersionKernel) -
    directly from raw history, matching the wind reading AT EACH LAG
    against how physically plausible that lag's contribution is. This is
    the multi-lag Sum_k formulation the project's spatial-attention
    versions (v2-v6) only approximated implicitly through GRU recurrence;
    here it's materialized directly, at the cost of an explicit
    [B, max_lag, N, N] kernel instead of a cheap per-step hidden state.

    Encoder = two parallel branches, combined:
      - "local": a plain GRU over each station's own [PM2.5 + weather]
        history (same encoder v4-v6 use) - covers the proposal's
        f_local(C_j history) and f_weather(weather history) in one pass.
      - "transported": H_j from SpatiotemporalDispersionKernel, a scalar
        per station, projected to hidden_dim and combined with the local
        branch via the same concat-then-linear "mix_gate" pattern v4-v6
        use for their (different) spatial mixing step.

    VAE BOTTLENECK REMOVED (unlike v4/v5/v6): the combined state feeds the
    decoder directly, deterministically - no mu/logvar heads, no
    reparameterized latent z, no KL term. `last_kl_loss` is simply absent
    (train.py's `getattr(model, 'last_kl_loss', None)` already handles
    that as "no KL regularization for this model", same code path v1-v3
    benchmark models without a KL term use - no trainer changes needed).
    Rationale: the VAE's reparameterization noise was one more source of
    training-time stochasticity on top of an already-heavier explicit
    multi-lag kernel and a small (~680-sample) dataset, and the trainer
    only ever evaluates the mean (z = mu_q at eval, per v4-v6's own
    forward()) - so its practical effect here was a regularizer on the
    encoder, not a source of predictive ensembles anyone was sampling from.
    Dropping it isolates whether the spatiotemporal kernel idea helps on
    its own, and removes a hyperparameter (kl_weight) and a term competing
    for gradient signal during training. `class ProbGRUModel7` keeps its
    name for continuity with train.py's dispatch and the file's place in
    this repo's numbered lineage, even though it is no longer probabilistic
    in the VAE sense - a naming mismatch worth knowing about, not fixing
    silently.

    OUT OF SCOPE for this version (kept simple/cheap deliberately - flag
    for a v8 if the core idea proves out): no decode-time spatial mixing.
    The decoder is a plain per-node-independent GRUCell rollout, as in v4.
    Re-running the multi-lag kernel every decode step would need a
    growing pool of "lagged" values built from the model's own predictions
    (nothing conceptually wrong with that, just materially more compute -
    this class already pays for an explicit [B, max_lag, N, N] kernel once
    per forward, and repeating that at every one of pred_len decode steps
    was judged not worth the added runtime until the core idea is
    validated).

    max_lag bounds the k = 1..max_lag lags actually used (clamped to
    hist_len - 1); it trades fidelity (longer transport windows considered)
    against the O(max_lag * N^2) cost of the kernel - default 8 covers a
    24-hour lookback at this dataset's 3-hour native step.
    """
    def __init__(self, hist_len, pred_len, in_dim, city_num, batch_size, device,
                 edge_index, edge_attr, wind_mean, wind_std,
                 station_coords, station_elevation,
                 feature_mean, feature_std, pbl_channel_idx=None,
                 hidden_dim=32, adaptive_dim=8, max_lag=8,
                 num_layers=1, dropout=0.1,
                 dist_threshold_km=300.0, sigma_h=1200.0,
                 diffusivity_init_km2h=50.0, sigma_min_km=15.0,
                 sigma_tau_init_h=3.0, dt_hours=3.0):
        super(ProbGRUModel7, self).__init__()

        self.hist_len = hist_len
        self.pred_len = pred_len
        self.in_dim = in_dim
        self.city_num = city_num
        self.device = device
        self.hidden_dim = hidden_dim
        self.dt_hours = dt_hours
        self.max_lag = min(max_lag, hist_len - 1)
        if self.max_lag < 1:
            raise ValueError(
                f"hist_len={hist_len} leaves no usable lag (need hist_len >= 2 "
                f"so at least one historical step precedes 'now')."
            )

        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.wind_mean = wind_mean
        self.wind_std = wind_std

        self.feature_dim = in_dim - 1
        self.hour_idx = self.feature_dim - 4
        self.speed_idx = self.feature_dim - 2
        self.direc_idx = self.feature_dim - 1
        self.pbl_channel_idx = pbl_channel_idx

        feature_mean_t = torch.as_tensor(feature_mean, dtype=torch.float32)
        feature_std_t = torch.as_tensor(feature_std, dtype=torch.float32)
        assert feature_mean_t.shape[0] == self.feature_dim, (
            f"feature_mean has {feature_mean_t.shape[0]} entries but feature_dim "
            f"(in_dim - 1) is {self.feature_dim} - pass HazeData.feature_mean/"
            f"feature_std, computed over the same metero_use as this run."
        )
        self.register_buffer('feature_mean', feature_mean_t)
        self.register_buffer('feature_std', feature_std_t.clamp(min=1e-6))

        # lag k=1..max_lag -> history index (hist_len-1-k) and elapsed hours
        lag_idxs = [hist_len - 1 - k for k in range(1, self.max_lag + 1)]
        self.register_buffer('lag_idxs', torch.tensor(lag_idxs, dtype=torch.long))
        k_hours = [k * dt_hours for k in range(1, self.max_lag + 1)]
        self.register_buffer('k_hours', torch.tensor(k_hours, dtype=torch.float32))

        self.local_encoder = nn.GRU(
            input_size=in_dim, hidden_size=hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dispersion = SpatiotemporalDispersionKernel(
            station_coords=station_coords,
            station_elevation=station_elevation,
            adaptive_dim=adaptive_dim,
            dist_threshold_km=dist_threshold_km,
            sigma_h=sigma_h,
            diffusivity_init_km2h=diffusivity_init_km2h,
            sigma_min_km=sigma_min_km,
            sigma_tau_init_h=sigma_tau_init_h,
            dt_hours=dt_hours,
        )
        self.transport_proj = nn.Linear(1, hidden_dim)
        self.mix_gate = nn.Linear(hidden_dim * 2, hidden_dim)

        # bridge layer between the encoder's mixed representation and the
        # decoder's initial state - no VAE latent to concatenate here now
        self.decoder_init = nn.Linear(hidden_dim, hidden_dim)
        self.decoder_cell = nn.GRUCell(
            input_size=self.feature_dim, hidden_size=hidden_dim,
        )
        self.output_head = nn.Linear(hidden_dim, 1)
        self.last_transport_alpha = None  # [B, max_lag, N, N] from the most recent forward, for inspection

    def _lagged_wind_and_pbl(self, feature_hist, B, N):
        """Gather (travel_bearing, speed_kmh, pbl) at every lag index at once.
        See probgru5/6 for the same de-normalization fix applied per-lag here."""
        speed_z = feature_hist[:, self.lag_idxs, :, self.speed_idx]     # [B, K, N]
        direc_z = feature_hist[:, self.lag_idxs, :, self.direc_idx]
        speed_kmh = speed_z * self.feature_std[self.speed_idx] + self.feature_mean[self.speed_idx]
        direc_from_deg = direc_z * self.feature_std[self.direc_idx] + self.feature_mean[self.direc_idx]
        travel_bearing = torch.deg2rad(direc_from_deg + 180.0)
        speed_kmh = speed_kmh.clamp(min=0.0)

        pbl = None
        if self.pbl_channel_idx is not None:
            pbl = feature_hist[:, self.lag_idxs, :, self.pbl_channel_idx]
        return travel_bearing, speed_kmh, pbl

    def forward(self, pm25_hist, feature):
        feature_hist = feature[:, :self.hist_len]
        feature_future = feature[:, self.hist_len:self.hist_len + self.pred_len]
        inputs = torch.cat([pm25_hist, feature_hist], dim=-1)

        B, T, N, C = inputs.shape
        if N != self.city_num:
            raise ValueError(
                f"ProbGRUModel7 was built with city_num={self.city_num}, but got "
                f"N={N} nodes in this batch's data."
            )
        if feature_future.shape[1] != self.pred_len:
            raise ValueError(
                f"expected {self.pred_len} future feature steps, got "
                f"{feature_future.shape[1]} - check feature's time dimension "
                f"covers hist_len + pred_len."
            )

        x = inputs.permute(0, 2, 1, 3).reshape(B * N, T, C)
        feat_fut = feature_future.permute(0, 2, 1, 3).reshape(
            B * N, self.pred_len, self.feature_dim)

        # --- local branch: per-station GRU over own PM2.5 + weather history ---
        _, h_n = self.local_encoder(x)
        h_local = h_n[-1].reshape(B, N, self.hidden_dim)

        # --- transported branch: explicit multi-lag spatiotemporal kernel ---
        travel_bearing_lag, speed_kmh_lag, pbl_lag = self._lagged_wind_and_pbl(feature_hist, B, N)
        pm25_lag = pm25_hist[:, self.lag_idxs, :, 0]                     # [B, K, N]
        H, alpha = self.dispersion(travel_bearing_lag, speed_kmh_lag, pm25_lag, self.k_hours, pbl_lag)
        self.last_transport_alpha = alpha
        h_transport = self.transport_proj(H.unsqueeze(-1))               # [B, N, hidden_dim]

        h_mixed = self.mix_gate(torch.cat([h_local, h_transport], dim=-1))
        h_mixed = h_mixed.reshape(B * N, self.hidden_dim)

        h_dec = self.decoder_init(h_mixed)
        preds = []
        for t in range(self.pred_len):
            h_dec = self.decoder_cell(feat_fut[:, t], h_dec)
            preds.append(self.output_head(h_dec))

        out = torch.stack(preds, dim=1)
        pm25_pred = out.reshape(B, N, self.pred_len, 1).permute(0, 2, 1, 3)
        return pm25_pred
