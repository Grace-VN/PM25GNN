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


class FickianMultiLagSpatialAttention(nn.Module):
    """
    probgru8's MultiLagPhysicsAwareSpatialAttention, with its two
    time/distance-related biases - the static isotropic `dist_bias =
    exp(-d/sigma_d)` inherited unchanged since v1, and the ad hoc
    `lag_bias` Gaussian match between k*dt_hours and the wind-implied
    travel time tau_ij - replaced by ONE actually-derived term: the
    point-source solution to Fick's second law of diffusion.

    Fick's second law, dC/dt = D*Laplacian(C), has a well-known closed-form
    Green's function for an instantaneous point source in 2D:

        C(d, tau) = 1 / (4*pi*D*tau) * exp(-d^2 / (4*D*tau))

    - mass-conserving (integrates to a constant "released mass" at every
    tau, it just spreads over a larger area as tau grows, diluting the
    peak) and, critically, ALREADY a function of tau - the same implied
    travel time tau_ij(t-k) = d_ij / wind_speed_i(t-k) v8 computed for its
    separate lag_bias term. That means this one kernel does what v8's TWO
    separate terms were doing between them: a source with a short implied
    travel time concentrates sharply near itself (subsuming what dist_bias
    tried to express with a single fixed sigma_d, now correctly SHRINKING
    for a fast/close pair and GROWING for a slow/far one); a source whose
    implied tau is large relative to how close it is gets diluted
    (subsuming what lag_bias's ad hoc "does k*dt match tau" penalty was
    trying to approximate, but derived rather than calibrated against an
    arbitrary sigma_tau).

    Numerically: effective_area = 4*D*tau + area_floor_km2 (the floor plays
    the same role as v5-v8's sigma_min_km^2 - it keeps the kernel finite as
    tau -> 0 for self/very-close pairs instead of dividing by ~0). D is one
    learnable diffusivity (km^2/h, softplus-positive, initialized at a
    physically plausible ~50 km^2/h as in v5-v7). The kernel spans many
    orders of magnitude (near-instant, near-self pairs vs. distant, slow
    ones), so - exactly like v5-v7's K_physics - it is added to the
    attention score in LOG space, i.e. multiplicatively in probability
    space once softmax is applied, not as a small additive perturbation
    the way v1-v4's dist_bias was.

    wind_align (directional gate) and terrain_bias (topographic blocking)
    are UNCHANGED from v8 - Fick's law as used here is isotropic (it only
    knows about distance and elapsed time, not direction), so wind_align
    still supplies the one piece of anisotropy this module has, exactly as
    it did on top of v4's isotropic dist_bias. Reproducing v5-v7's full
    downwind/crosswind anisotropic decomposition on top of THIS kernel is
    a natural next step, deliberately left out here to keep this an
    isolated, attributable change on top of the version that has actually
    performed best so far.
    """
    def __init__(self, hidden_dim, station_coords, station_elevation,
                 attn_dim=32, dist_threshold_km=300.0, sigma_h=1200.0,
                 diffusivity_init_km2h=50.0, sigma_min_km=15.0,
                 speed_floor_kmh=0.5):
        super().__init__()

        self.q_proj = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.scale = attn_dim ** -0.5
        self.speed_floor_kmh = speed_floor_kmh
        self.area_floor_km2 = sigma_min_km ** 2
        self.eps = 1e-12

        self.w_wind = nn.Parameter(torch.tensor(1.0))
        self.w_terrain = nn.Parameter(torch.tensor(1.0))
        self.w_fick = nn.Parameter(torch.tensor(1.0))
        self.log_D_base = nn.Parameter(torch.tensor(_inv_softplus(diffusivity_init_km2h)))

        dist = haversine_km(station_coords)                          # [N, N]
        neighbor_mask = dist <= dist_threshold_km
        self.register_buffer('neighbor_mask', neighbor_mask)
        self.register_buffer('dist_km', dist)

        elev_diff = (station_elevation.unsqueeze(1) - station_elevation.unsqueeze(0)).abs()
        terrain_bias = torch.exp(-elev_diff / sigma_h)
        self.register_buffer('terrain_bias', terrain_bias)            # [N, N], static

        # bearing FROM source i TO receiver j, laid out as [j, i] (receiver
        # row, source col) to match this module's [B, N_j, K*N_i] score
        # layout - see forward()
        self.register_buffer('bearing_j_i', bearing_deg(station_coords).t())

    def forward(self, h_last, h_lag, travel_bearing_lag, wind_speed_kmh_lag, k_hours):
        """
        h_last              : [B, N, hidden_dim] - current/last-step states (query)
        h_lag                : [B, K, N, hidden_dim] - states at each of the K
                                most recent lags, k=0 (now) first (key/value)
        travel_bearing_lag   : [B, K, N] radians - each SOURCE station's wind
                                bearing (blowing toward) at that lag
        wind_speed_kmh_lag   : [B, K, N] each source's wind speed (km/h) at that lag
        k_hours              : [K] elapsed time k*dt_hours for each lag position
                                (unused by the Fick kernel itself - tau is
                                derived independently per lag from that
                                lag's own wind - kept in the signature for
                                interface parity with v8's spatial_attn call)
        returns              : [B, N, hidden_dim]
        """
        B, N, _ = h_last.shape
        K = h_lag.shape[1]

        q = self.q_proj(h_last)                                          # [B, N, attn_dim]
        k = self.k_proj(h_lag).reshape(B, K * N, -1)                     # [B, K*N, attn_dim]
        v = self.v_proj(h_lag).reshape(B, K * N, -1)                     # [B, K*N, hidden_dim]
        content_score = torch.bmm(q, k.transpose(1, 2)) * self.scale     # [B, N, K*N]

        # theta: angle between SOURCE i's wind (at lag k) and bearing i->j
        angle_diff = travel_bearing_lag.unsqueeze(2) - self.bearing_j_i.view(1, 1, N, N)  # [B,K,N_j,N_i]
        wind_align = torch.clamp(torch.cos(angle_diff), min=0.0)

        speed = wind_speed_kmh_lag.clamp(min=0.0)
        dist_b = self.dist_km.view(1, 1, N, N)
        tau = dist_b / (speed.unsqueeze(2) + self.speed_floor_kmh)                          # [B,K,N_j,N_i]

        # --- Fick's second law: 2D point-source Green's function ---
        D = F.softplus(self.log_D_base)                                                     # km^2/h
        effective_area = 4.0 * D * tau.clamp(min=0.0) + self.area_floor_km2
        k_fick = (1.0 / (math.pi * effective_area)) * torch.exp(-(dist_b ** 2) / effective_area)  # [B,K,N_j,N_i]

        def _flatten(t):
            return t.permute(0, 2, 1, 3).reshape(B, N, K * N)

        wind_align_flat = _flatten(wind_align)
        k_fick_flat = _flatten(k_fick)

        terrain_tiled = self.terrain_bias.unsqueeze(1).expand(N, K, N).reshape(N, K * N)
        mask_tiled = self.neighbor_mask.unsqueeze(1).expand(N, K, N).reshape(N, K * N)

        score = (content_score
                 + self.w_wind * wind_align_flat
                 + self.w_terrain * terrain_tiled.unsqueeze(0)
                 + self.w_fick * torch.log(k_fick_flat.clamp_min(self.eps)))

        score = score.masked_fill(~mask_tiled.unsqueeze(0), float('-inf'))
        weights = torch.nan_to_num(torch.softmax(score, dim=-1), nan=0.0)
        return torch.bmm(weights, v)


class ProbGRUModel9(nn.Module):
    """
    ProbGRUModel8 with its spatial mixing step's distance/lag biases
    replaced by FickianMultiLagSpatialAttention's Fick's-law-derived
    kernel (see that class's docstring). Everything else - the encoder's
    bottleneck/per_step fork reusing `nn.GRU`'s full per-step output as
    multi-lag keys/values, the VAE bottleneck, the per-node-independent
    decoder - is identical to v8, so this stays directly comparable
    through the same trainer. Same wind de-normalization / compass-bearing
    fixes carried over from v5-v8 (needed for tau = d/u to mean anything).
    """
    def __init__(self, hist_len, pred_len, in_dim, city_num, batch_size, device,
                 edge_index, edge_attr, wind_mean, wind_std,
                 station_coords, station_elevation,
                 feature_mean, feature_std,
                 hidden_dim=64, latent_dim=16, attn_dim=32, num_layers=1,
                 dropout=0.1, logvar_clamp=10.0,
                 spatial_mix_mode='bottleneck', max_lag=6,
                 dist_threshold_km=300.0, sigma_h=1200.0,
                 diffusivity_init_km2h=50.0, sigma_min_km=15.0,
                 dt_hours=3.0):
        super(ProbGRUModel9, self).__init__()
        assert spatial_mix_mode in ('bottleneck', 'per_step'), \
            f"spatial_mix_mode must be 'bottleneck' or 'per_step', got {spatial_mix_mode}"
        if spatial_mix_mode == 'per_step' and num_layers != 1:
            raise ValueError(
                "spatial_mix_mode='per_step' unrolls a single-layer GRUCell "
                "manually, so num_layers must be 1 (got {}).".format(num_layers)
            )

        self.hist_len = hist_len
        self.pred_len = pred_len
        self.in_dim = in_dim
        self.city_num = city_num
        self.device = device
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.logvar_clamp = logvar_clamp
        self.spatial_mix_mode = spatial_mix_mode
        self.dt_hours = dt_hours
        self.max_lag = min(max_lag, hist_len)

        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.wind_mean = wind_mean
        self.wind_std = wind_std

        self.feature_dim = in_dim - 1
        self.speed_idx = self.feature_dim - 2
        self.direc_idx = self.feature_dim - 1

        feature_mean_t = torch.as_tensor(feature_mean, dtype=torch.float32)
        feature_std_t = torch.as_tensor(feature_std, dtype=torch.float32)
        assert feature_mean_t.shape[0] == self.feature_dim, (
            f"feature_mean has {feature_mean_t.shape[0]} entries but feature_dim "
            f"(in_dim - 1) is {self.feature_dim} - pass HazeData.feature_mean/"
            f"feature_std, computed over the same metero_use as this run."
        )
        self.register_buffer('feature_mean', feature_mean_t)
        self.register_buffer('feature_std', feature_std_t.clamp(min=1e-6))

        if spatial_mix_mode == 'bottleneck':
            self.encoder = nn.GRU(
                input_size=in_dim, hidden_size=hidden_dim, num_layers=num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
            )
        else:  # 'per_step'
            self.encoder_cell = nn.GRUCell(input_size=in_dim, hidden_size=hidden_dim)
            self.step_dropout = nn.Dropout(dropout)

        self.spatial_attn = FickianMultiLagSpatialAttention(
            hidden_dim=hidden_dim,
            station_coords=station_coords,
            station_elevation=station_elevation,
            attn_dim=attn_dim,
            dist_threshold_km=dist_threshold_km,
            sigma_h=sigma_h,
            diffusivity_init_km2h=diffusivity_init_km2h,
            sigma_min_km=sigma_min_km,
        )
        self.mix_gate = nn.Linear(hidden_dim * 2, hidden_dim)

        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

        self.decoder_init = nn.Linear(hidden_dim + latent_dim, hidden_dim)
        self.decoder_cell = nn.GRUCell(
            input_size=self.feature_dim + latent_dim, hidden_size=hidden_dim,
        )
        self.output_head = nn.Linear(hidden_dim, 1)
        self.last_kl_loss = None

    def _wind_at_idxs(self, feature_hist, idxs, B, N):
        """De-normalize the dataset's precomputed speed(km/h)/direction(deg,
        meteorological 'from' convention) tail channels at each of the given
        history indices and convert to compass-bearing 'travel toward'
        angles. idxs: 1-D LongTensor or list of history indices.
        Returns (travel_bearing [B,len(idxs),N] rad, wind_speed_kmh [B,len(idxs),N])."""
        speed_z = feature_hist[:, idxs, :, self.speed_idx]
        direc_z = feature_hist[:, idxs, :, self.direc_idx]
        speed_kmh = speed_z * self.feature_std[self.speed_idx] + self.feature_mean[self.speed_idx]
        direc_from_deg = direc_z * self.feature_std[self.direc_idx] + self.feature_mean[self.direc_idx]
        travel_bearing = torch.deg2rad(direc_from_deg + 180.0)
        speed_kmh = speed_kmh.clamp(min=0.0)
        return travel_bearing, speed_kmh

    def _encode_bottleneck(self, x, feature_hist, B, N):
        """Fused GRU; the last max_lag steps' hidden states become the
        spatial attention's keys/values, each scored with its own wind."""
        output, h_n = self.encoder(x)                          # output: [B*N, T, hidden_dim]
        h_T = h_n[-1]
        h_grid = h_T.reshape(B, N, self.hidden_dim)

        T = output.shape[1]
        K = self.max_lag
        lag_idxs = list(range(T - K, T))                       # oldest -> newest, newest = "now"
        h_lag = output[:, lag_idxs, :].reshape(B, N, K, self.hidden_dim).permute(0, 2, 1, 3)  # [B,K,N,hidden]

        travel_bearing_lag, speed_kmh_lag = self._wind_at_idxs(feature_hist, lag_idxs, B, N)
        k_hours = torch.tensor(
            [(T - 1 - idx) * self.dt_hours for idx in lag_idxs],
            dtype=x.dtype, device=x.device,
        )

        context = self.spatial_attn(h_grid, h_lag, travel_bearing_lag, speed_kmh_lag, k_hours)
        h_mixed = self.mix_gate(torch.cat([h_grid, context], dim=-1))
        return h_mixed.reshape(B * N, self.hidden_dim)

    def _encode_per_step(self, x, feature_hist, B, N):
        """Unroll the encoder manually; at every step, K=1 (current step
        only) - the per-step analogue of v4's original attention, plus the
        Fick's-law kernel evaluated with that step's own wind."""
        T = x.shape[1]
        h = torch.zeros(B * N, self.hidden_dim, device=x.device, dtype=x.dtype)

        for t in range(T):
            x_t = x[:, t, :]
            h = self.encoder_cell(x_t, h)
            h = self.step_dropout(h)

            h_grid_t = h.reshape(B, N, self.hidden_dim)
            h_lag_t = h_grid_t.unsqueeze(1)                     # [B, 1, N, hidden_dim]
            travel_bearing_t, speed_kmh_t = self._wind_at_idxs(feature_hist, [t], B, N)
            k_hours = torch.zeros(1, dtype=x.dtype, device=x.device)

            context_t = self.spatial_attn(h_grid_t, h_lag_t, travel_bearing_t, speed_kmh_t, k_hours)
            h_mixed_t = self.mix_gate(torch.cat([h_grid_t, context_t], dim=-1))
            h = h_mixed_t.reshape(B * N, self.hidden_dim)

        return h

    def forward(self, pm25_hist, feature):
        feature_hist = feature[:, :self.hist_len]
        feature_future = feature[:, self.hist_len:self.hist_len + self.pred_len]
        inputs = torch.cat([pm25_hist, feature_hist], dim=-1)

        B, T, N, C = inputs.shape
        if N != self.city_num:
            raise ValueError(
                f"ProbGRUModel9 was built with city_num={self.city_num}, but got "
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

        if self.spatial_mix_mode == 'bottleneck':
            h_mixed = self._encode_bottleneck(x, feature_hist, B, N)
        else:
            h_mixed = self._encode_per_step(x, feature_hist, B, N)

        mu_q = self.mu_head(h_mixed)
        logvar_q = torch.clamp(self.logvar_head(h_mixed), -self.logvar_clamp, self.logvar_clamp)

        if self.training:
            eps = torch.randn_like(mu_q)
            z = mu_q + eps * torch.exp(0.5 * logvar_q)
        else:
            z = mu_q

        kl = -0.5 * torch.mean(
            torch.sum(1 + logvar_q - mu_q.pow(2) - logvar_q.exp(), dim=-1)
        )
        self.last_kl_loss = kl

        h_dec = self.decoder_init(torch.cat([h_mixed, z], dim=-1))
        preds = []
        for t in range(self.pred_len):
            step_in = torch.cat([feat_fut[:, t], z], dim=-1)
            h_dec = self.decoder_cell(step_in, h_dec)
            preds.append(self.output_head(h_dec))

        out = torch.stack(preds, dim=1)
        pm25_pred = out.reshape(B, N, self.pred_len, 1).permute(0, 2, 1, 3)
        return pm25_pred
