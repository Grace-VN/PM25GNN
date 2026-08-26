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


class PhysicsPriorSpatialAttention(nn.Module):
    """
    Same anisotropic advection-diffusion prior as probgru5's module of the
    same name (see that file for the full derivation of K_physics: downwind/
    crosswind decomposition via bearing/distance from lat-lon, tau=x/u,
    sigma^2=2*D*tau, calm-wind isotropic blending, PBL-modulated diffusivity,
    multiplicative terrain attenuation) - PRUNED after v5 underperformed
    probgru4 on a ~680-sample training set:

      - REMOVED the Graph-WaveNet-style static adaptive adjacency
        (node_emb1/node_emb2/w_adp). It was ~5.9k free parameters (2*N*16)
        with no physical grounding, added purely so the model could absorb
        "persistent structure content attention doesn't find on its own" -
        exactly the kind of capacity a few hundred training samples can
        memorize instead of generalizing from. Content attention
        (Q.K^T, already dynamic and learned) is what's left to fill this
        role.

      - g_theta SHRUNK from a 2-layer MLP (ctx -> 16 -> tanh -> 1) to a
        single linear layer (ctx -> 1, zero-initialized). It is still a
        multiplicative log-correction on the physics prior, still starts
        as the identity, but can now only learn a LINEAR reweighting of
        [z_i, z_j, C_i, C_j, sin(hour), cos(hour), pbl_i] - it can no longer
        fit an arbitrary nonlinear function of that context, which is
        exactly the capacity most likely to overfit a few hundred samples
        while contributing little the physics kernel + content attention
        don't already cover.

    The physics kernel's own free parameters (log_D_base, w_pbl, calm_center,
    calm_scale_raw, gate_steepness_raw, w_phys, w_gate - 7 scalars) are left
    as in v5: each is individually cheap, interpretable, and initialized at
    a sensible physical default, so they were not the likely source of
    v5's parameter-count/optimization problem - the anisotropic kernel shape
    itself (the one genuinely new physical idea over v1-v4's isotropic
    exp(-d/sigma)) is kept intact.
    """
    def __init__(self, hidden_dim, station_coords, station_elevation,
                 attn_dim=16, dist_threshold_km=300.0, sigma_h=1200.0,
                 gate_clamp=2.0, diffusivity_init_km2h=50.0, sigma_min_km=15.0,
                 calm_speed_kmh_init=5.0, calm_scale_kmh_init=3.0,
                 downwind_gate_steepness_init=4.0, dt_hours=3.0,
                 speed_floor_kmh=0.5, eps=1e-6):
        super().__init__()

        self.eps = eps
        self.dt_hours = dt_hours
        self.sigma_min_km = sigma_min_km
        self.speed_floor_kmh = speed_floor_kmh
        self.gate_clamp = gate_clamp
        self._sqrt_2pi = math.sqrt(2.0 * math.pi)

        # --- content-based attention (dynamic learned spatial relation) ---
        self.q_proj = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.scale = attn_dim ** -0.5

        # --- overall interpretable strengths for the two log-bias terms ---
        self.w_phys = nn.Parameter(torch.tensor(1.0))
        self.w_gate = nn.Parameter(torch.tensor(1.0))

        # --- learnable physics parameters, sensible defaults, free to move ---
        self.log_D_base = nn.Parameter(torch.tensor(_inv_softplus(diffusivity_init_km2h)))
        self.w_pbl = nn.Parameter(torch.tensor(0.0))               # 0 -> no PBL effect until learned
        self.calm_center = nn.Parameter(torch.tensor(calm_speed_kmh_init))
        self.calm_scale_raw = nn.Parameter(torch.tensor(_inv_softplus(calm_scale_kmh_init)))
        self.gate_steepness_raw = nn.Parameter(torch.tensor(_inv_softplus(downwind_gate_steepness_init)))

        # --- g_theta: single linear log-correction, zero-init = identity ---
        ctx_dim = 7  # [z_i, z_j, C_i, C_j, sin(hour), cos(hour), pbl_i]
        self.gate_linear = nn.Linear(ctx_dim, 1)
        nn.init.zeros_(self.gate_linear.weight)
        nn.init.zeros_(self.gate_linear.bias)

        # --- static geometry / physics buffers ---
        dist = haversine_km(station_coords)                           # [N, N], km
        neighbor_mask = dist <= dist_threshold_km
        self.register_buffer('neighbor_mask', neighbor_mask)
        self.register_buffer('dist_km', dist)
        self.register_buffer('bearing', bearing_deg(station_coords))  # [N, N], i -> j

        elev_diff = (station_elevation.unsqueeze(1) - station_elevation.unsqueeze(0)).abs()
        terrain_bias = torch.exp(-elev_diff / sigma_h)
        self.register_buffer('terrain_bias', terrain_bias)            # [N, N], static barrier factor

        elev_mean = station_elevation.mean()
        elev_std = station_elevation.std().clamp(min=eps)
        self.register_buffer('elev_norm', (station_elevation - elev_mean) / elev_std)  # [N]

    def forward(self, h, travel_bearing, wind_speed_kmh, pm25_now, hour_sin, hour_cos, pbl_z=None):
        """
        h               : [B, N, hidden_dim] - per-node states
        travel_bearing  : [B, N] radians - compass bearing each node's air
                          is currently heading TOWARD
        wind_speed_kmh  : [B, N] wind speed in km/h, >= 0
        pm25_now        : [B, N] current/most-recent PM2.5 reading, same
                          normalized scale the model trains on
        hour_sin, hour_cos : [B] cyclical encoding of hour-of-day
        pbl_z           : [B, N] normalized boundary-layer height, or None
        returns         : [B, N, hidden_dim]
        """
        B, N, _ = h.shape

        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)
        content_score = torch.bmm(q, k.transpose(1, 2)) * self.scale     # [B, N, N]

        # ---- anisotropic advection-diffusion kernel ----
        theta = travel_bearing.unsqueeze(2) - self.bearing.unsqueeze(0)  # [B, N, N]
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        dist_b = self.dist_km.unsqueeze(0)                               # [1, N, N]
        x_ij = dist_b * cos_t                                            # downwind distance
        y_ij = dist_b * sin_t                                            # crosswind offset

        D_base = F.softplus(self.log_D_base)                             # scalar, km^2/h
        if pbl_z is not None:
            pbl_mod = torch.clamp(self.w_pbl * pbl_z, -5.0, 5.0)
            D_i = D_base * torch.exp(pbl_mod)                            # [B, N]
        else:
            D_i = D_base.view(1, 1).expand(B, N)

        speed = wind_speed_kmh.clamp(min=0.0)
        tau = x_ij.clamp(min=0.0) / (speed.unsqueeze(2) + self.speed_floor_kmh)   # [B, N, N]
        sigma_sq = 2.0 * D_i.unsqueeze(2) * tau + self.sigma_min_km ** 2
        sigma = torch.sqrt(sigma_sq)

        gate_k = F.softplus(self.gate_steepness_raw)
        downwind_gate = torch.sigmoid(gate_k * cos_t)                    # smooth, in (0, 1)
        k_plume = downwind_gate / (sigma * self._sqrt_2pi) * torch.exp(-(y_ij ** 2) / (2.0 * sigma_sq))

        sigma_iso_sq = 2.0 * D_i.unsqueeze(2) * self.dt_hours + self.sigma_min_km ** 2
        k_iso = torch.exp(-(dist_b ** 2) / (2.0 * sigma_iso_sq)) / (torch.sqrt(sigma_iso_sq) * self._sqrt_2pi)

        calm_scale = F.softplus(self.calm_scale_raw) + self.eps
        blend = torch.sigmoid((speed - self.calm_center) / calm_scale).unsqueeze(2)   # [B, N, 1]
        k_dispersion = blend * k_plume + (1.0 - blend) * k_iso

        k_phys = (k_dispersion * self.terrain_bias.unsqueeze(0)).clamp_min(self.eps)  # [B, N, N]

        # ---- neural correction g_theta(z_i, z_j, C_i, C_j, weather, time) ----
        # single linear map now, not an MLP - see class docstring
        elev_i = self.elev_norm.unsqueeze(1).expand(N, N).unsqueeze(0).expand(B, N, N)
        elev_j = self.elev_norm.unsqueeze(0).expand(N, N).unsqueeze(0).expand(B, N, N)
        pm_i = pm25_now.unsqueeze(2).expand(B, N, N)
        pm_j = pm25_now.unsqueeze(1).expand(B, N, N)
        hs = hour_sin.view(B, 1, 1).expand(B, N, N)
        hc = hour_cos.view(B, 1, 1).expand(B, N, N)
        if pbl_z is not None:
            pbl_i = pbl_z.unsqueeze(2).expand(B, N, N)
        else:
            pbl_i = torch.zeros(B, N, N, device=h.device, dtype=h.dtype)

        ctx = torch.stack([elev_i, elev_j, pm_i, pm_j, hs, hc, pbl_i], dim=-1)  # [B, N, N, 7]
        gate_raw = self.gate_linear(ctx).squeeze(-1)                            # [B, N, N]
        gate_log = torch.clamp(gate_raw, -self.gate_clamp, self.gate_clamp)

        score = (content_score
                 + self.w_phys * torch.log(k_phys)
                 + self.w_gate * gate_log)

        score = score.masked_fill(~self.neighbor_mask.unsqueeze(0), float('-inf'))
        weights = torch.nan_to_num(torch.softmax(score, dim=-1), nan=0.0)
        return torch.bmm(weights, v)


class ProbGRUModel6(nn.Module):
    """
    probgru5's VAE-GRU skeleton and PhysicsPriorSpatialAttention (pruned -
    see that class's docstring), with ONE structural change made mandatory
    rather than optional: spatial mixing now runs at EVERY decode step, not
    just once at the encoder bottleneck.

    Why: `model/PM25_GNN.py`'s GraphGNN - the benchmark v5 lost to - barely
    "encodes history" at all. Its forward() seeds from only the LAST
    observed PM2.5 reading and then, for every one of pred_len steps, reruns
    wind-weighted spatial message passing on the model's own just-updated
    state before advancing the GRUCell:

        xn = pm25_hist[:, -1]
        for i in range(pred_len):
            x = cat([xn, feature[hist_len+i]])
            xn_gnn = self.graph_gnn(x)          # spatial mixing, EVERY step
            hn = self.gru_cell(cat([xn_gnn, x]), hn)
            xn = self.fc_out(hn)

    v4 and v5 both mix spatially ONCE (at the end of encoding) and then
    decode pred_len steps with each station running independently - exactly
    the gap that should hurt most as pred_len grows (this repo currently
    trains with pred_len=24), since spatial coupling compounding over a
    long autoregressive horizon is dropped after step 0. This class closes
    that gap while keeping what v4/v5 have that PM25_GNN doesn't: a VAE
    latent over the full history window (for the ensemble/uncertainty
    behavior the ProbGRU line exists for) and the anisotropic physics-prior
    kernel (a strictly more realistic dispersion shape than PM25_GNN's own
    `relu(3*u*cos(theta)/d)` edge weight).

    Mechanically, `decode_spatial_mix` is gone as a flag - every decode step
    now: computes a provisional prediction from the just-updated GRUCell
    state (used only as g_theta's "current pollution" context, exactly as
    PM25_GNN uses its own last state - never the label), spatially remixes
    that state through PhysicsPriorSpatialAttention using that step's own
    known future wind, then re-applies output_head to the remixed state for
    the actual prediction. This roughly doubles the per-decode-step cost
    (spatial attention + 2x output_head instead of 1x) relative to v4/v5
    with decode mixing off - worth it only if it actually buys accuracy;
    that is exactly the ablation this class exists to run.

    Capacity is also reduced from v5's defaults (hidden_dim 64->32,
    latent_dim 16->8, attn_dim 32->16) on top of the adaptive-embedding and
    gate-MLP removal in PhysicsPriorSpatialAttention - the working
    hypothesis is that v5 underperformed v4 from too much free capacity for
    a ~680-sample training set, not from the anisotropic kernel or
    decode-mixing ideas being wrong, so this class cuts capacity almost
    everywhere it can while keeping both of those ideas intact, to test
    them cleanly.

    spatial_mix_mode (encoder side, unchanged in spirit from v2-v5):
      - 'bottleneck': fused nn.GRU encodes each station independently; ONE
        physics-prior-plus-correction mixing pass at the end of history.
      - 'per_step': encoder unrolled with a GRUCell; mixing recomputed at
        every history step too, so BOTH history and forecast get per-step
        spatial coupling - the closest analogue to PM25_GNN's decode loop,
        just with a VAE-conditioned history phase in front of it.
    """
    def __init__(self, hist_len, pred_len, in_dim, city_num, batch_size, device,
                 edge_index, edge_attr, wind_mean, wind_std,
                 station_coords, station_elevation,
                 feature_mean, feature_std, pbl_channel_idx=None,
                 hidden_dim=32, latent_dim=8, attn_dim=16,
                 num_layers=1, dropout=0.1, logvar_clamp=10.0,
                 spatial_mix_mode='bottleneck',
                 dist_threshold_km=300.0, sigma_h=1200.0,
                 diffusivity_init_km2h=50.0, sigma_min_km=15.0,
                 calm_speed_kmh_init=5.0, calm_scale_kmh_init=3.0,
                 gate_clamp=2.0, dt_hours=3.0):
        super(ProbGRUModel6, self).__init__()
        assert spatial_mix_mode in ('bottleneck', 'per_step'), \
            f"spatial_mix_mode must be 'bottleneck' or 'per_step', got {spatial_mix_mode}"
        if spatial_mix_mode == 'per_step' and num_layers != 1:
            raise ValueError(
                "spatial_mix_mode='per_step' unrolls a single-layer GRUCell "
                "manually, so num_layers must be 1 (got {}). Stacking layers "
                "for per-step mixing would need a per-layer cell list; not "
                "implemented here to keep the ablation clean.".format(num_layers)
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

        # kept for signature parity across benchmark models / the trainer's
        # get_model() dispatch; superseded by feature_mean/feature_std below
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.wind_mean = wind_mean
        self.wind_std = wind_std

        self.feature_dim = in_dim - 1
        # Fixed tail layout dataset.py._process_feature always produces:
        # [...metero_use channels..., hour, weekday, speed_kmh, direc_deg].
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

        if spatial_mix_mode == 'bottleneck':
            self.encoder = nn.GRU(
                input_size=in_dim, hidden_size=hidden_dim, num_layers=num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
            )
        else:  # 'per_step'
            self.encoder_cell = nn.GRUCell(input_size=in_dim, hidden_size=hidden_dim)
            self.step_dropout = nn.Dropout(dropout)

        self.spatial_attn = PhysicsPriorSpatialAttention(
            hidden_dim=hidden_dim,
            station_coords=station_coords,
            station_elevation=station_elevation,
            attn_dim=attn_dim,
            dist_threshold_km=dist_threshold_km,
            sigma_h=sigma_h,
            gate_clamp=gate_clamp,
            diffusivity_init_km2h=diffusivity_init_km2h,
            sigma_min_km=sigma_min_km,
            calm_speed_kmh_init=calm_speed_kmh_init,
            calm_scale_kmh_init=calm_scale_kmh_init,
            dt_hours=dt_hours,
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

    def _wind_and_pbl_at(self, feature_slice, t, B, N):
        """De-normalize the dataset's precomputed speed(km/h)/direction(deg,
        meteorological "from" convention) tail channels and convert to a
        compass-bearing "travel toward" angle - exact z-score inversion,
        same fix as probgru5. Returns (travel_bearing [B,N] rad,
        wind_speed_kmh [B,N], pbl_z [B,N] or None)."""
        speed_z = feature_slice[:, t, :, self.speed_idx]
        direc_z = feature_slice[:, t, :, self.direc_idx]
        speed_kmh = speed_z * self.feature_std[self.speed_idx] + self.feature_mean[self.speed_idx]
        direc_from_deg = direc_z * self.feature_std[self.direc_idx] + self.feature_mean[self.direc_idx]
        travel_bearing = torch.deg2rad(direc_from_deg + 180.0)
        speed_kmh = speed_kmh.clamp(min=0.0)

        pbl_z = None
        if self.pbl_channel_idx is not None:
            pbl_z = feature_slice[:, t, :, self.pbl_channel_idx]
        return travel_bearing, speed_kmh, pbl_z

    def _hour_encoding_at(self, feature_slice, t):
        """Cyclical hour-of-day encoding, de-normalized from the dataset's
        hour channel (identical across stations at a given timestep, so
        node 0's value represents the whole batch). Returns (sin, cos),
        each [B]."""
        hour_z = feature_slice[:, t, :, self.hour_idx]
        hour_raw = hour_z * self.feature_std[self.hour_idx] + self.feature_mean[self.hour_idx]
        hour0 = hour_raw[:, 0]
        frac = hour0 / 24.0 * (2.0 * math.pi)
        return torch.sin(frac), torch.cos(frac)

    def _spatial_mix(self, h_grid, travel_bearing, speed_kmh, pm_now, hour_sin, hour_cos, pbl_z, B, N):
        context = self.spatial_attn(h_grid, travel_bearing, speed_kmh, pm_now, hour_sin, hour_cos, pbl_z)
        h_mixed = self.mix_gate(torch.cat([h_grid, context], dim=-1))
        return h_mixed.reshape(B * N, self.hidden_dim)

    def _encode_bottleneck(self, x, feature_hist, pm25_hist, B, N):
        """Fused GRU, single physics-prior-plus-correction mixing pass at
        the end, using the last observed history step's context."""
        _, h_n = self.encoder(x)
        h_T = h_n[-1]                                       # [B*N, hidden_dim]
        h_grid = h_T.reshape(B, N, self.hidden_dim)

        travel_bearing, speed_kmh, pbl_z = self._wind_and_pbl_at(feature_hist, -1, B, N)
        hour_sin, hour_cos = self._hour_encoding_at(feature_hist, -1)
        pm_now = pm25_hist[:, -1, :, 0]

        return self._spatial_mix(h_grid, travel_bearing, speed_kmh, pm_now, hour_sin, hour_cos, pbl_z, B, N)

    def _encode_per_step(self, x, feature_hist, pm25_hist, B, N):
        """Unroll the encoder manually; at every timestep recompute the
        full physics-prior-plus-correction mixing from that step's own
        wind/time/pollution context, and feed the mixed state back into
        the recurrence."""
        T = x.shape[1]
        h = torch.zeros(B * N, self.hidden_dim, device=x.device, dtype=x.dtype)

        for t in range(T):
            x_t = x[:, t, :]
            h = self.encoder_cell(x_t, h)
            h = self.step_dropout(h)

            h_grid_t = h.reshape(B, N, self.hidden_dim)
            travel_bearing_t, speed_kmh_t, pbl_t = self._wind_and_pbl_at(feature_hist, t, B, N)
            hour_sin_t, hour_cos_t = self._hour_encoding_at(feature_hist, t)
            pm_now_t = pm25_hist[:, t, :, 0]

            h = self._spatial_mix(h_grid_t, travel_bearing_t, speed_kmh_t, pm_now_t, hour_sin_t, hour_cos_t, pbl_t, B, N)

        return h

    def forward(self, pm25_hist, feature):
        feature_hist = feature[:, :self.hist_len]
        feature_future = feature[:, self.hist_len:self.hist_len + self.pred_len]
        inputs = torch.cat([pm25_hist, feature_hist], dim=-1)

        B, T, N, C = inputs.shape
        if N != self.city_num:
            raise ValueError(
                f"ProbGRUModel6 was built with city_num={self.city_num}, but got "
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
            h_mixed = self._encode_bottleneck(x, feature_hist, pm25_hist, B, N)
        else:
            h_mixed = self._encode_per_step(x, feature_hist, pm25_hist, B, N)

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

            # PM25_GNN-parity: spatial mixing EVERY decode step, mandatory
            # (see class docstring). The gate's "current pollution" context
            # can't use the (future, unknown) label, so it uses this step's
            # own just-computed provisional prediction instead - then
            # output_head is re-applied to the spatially-remixed state for
            # the actual prediction.
            pred_t_provisional = self.output_head(h_dec)          # [B*N, 1], gate context only
            h_dec_grid = h_dec.reshape(B, N, self.hidden_dim)
            travel_bearing_t, speed_kmh_t, pbl_t = self._wind_and_pbl_at(feature_future, t, B, N)
            hour_sin_t, hour_cos_t = self._hour_encoding_at(feature_future, t)
            pm_now_t = pred_t_provisional.reshape(B, N)

            h_dec = self._spatial_mix(h_dec_grid, travel_bearing_t, speed_kmh_t, pm_now_t, hour_sin_t, hour_cos_t, pbl_t, B, N)
            pred_t = self.output_head(h_dec)

            preds.append(pred_t)

        out = torch.stack(preds, dim=1)
        pm25_pred = out.reshape(B, N, self.pred_len, 1).permute(0, 2, 1, 3)
        return pm25_pred
