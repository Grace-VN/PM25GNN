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


class MultiLagPhysicsAwareSpatialAttention(nn.Module):
    """
    airlapse.py's MultiLagPhysicsAwareSpatialAttention, extended with
    exactly ONE new idea: a trend-augmented key/value, added inside the
    existing attention pipeline rather than as a fifth ad-hoc scalar bias
    weight the way w_lag was added on top of w_dist/w_wind/w_terrain.

    Everything from airlapse.py is unchanged here except this one addition
    - dist_bias/terrain_bias/wind_align/lag_bias and their w_* weights are
    untouched, so any behavior difference vs. AirLapse traces to exactly
    this.

    TREND-AUGMENTED KEY/VALUE (not a new w_* bias - this one has no scalar
    weight of its own; it's absorbed into k_proj/v_proj's own learned
    weights instead, so the model's existing content attention can use the
    trend directly rather than a hand-tuned kernel scoring it separately):
        delta_h[k] = h_lag[k] - h_lag[k+1]   for k = 0..K-2
        delta_h[K-1] = 0                     (no k+1 to diff against)
    h_lag is concatenated with delta_h along the hidden dim before
    k_proj/v_proj (which now take 2*hidden_dim in - q_proj is untouched,
    since the query is still just h_last), so both the content-matching
    keys and the values the softmax mixes carry "what this station looked
    like, AND how it was changing" instead of an instantaneous snapshot
    alone.

    Degrades to an exact no-op at K=1 (per_step mode): delta_h is exactly
    zero there (there's no k+1 for the only lag), and since k_proj/v_proj
    have no bias term, zero concatenated onto h_lag contributes exactly
    zero to k_proj(h_lag_aug)/v_proj(h_lag_aug) regardless of what the
    delta-half of those weights learn - the output is identical to using
    h_lag alone. That's a property of bias=False, not something that needs
    the delta-half weights to happen to be zero.
    """
    def __init__(self, hidden_dim, station_coords, station_elevation,
                 attn_dim=32, dist_threshold_km=300.0, sigma_d=200.0,
                 sigma_h=1200.0, sigma_tau_init_h=3.0, speed_floor_kmh=0.5):
        super().__init__()

        self.q_proj = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.k_proj = nn.Linear(2 * hidden_dim, attn_dim, bias=False)
        self.v_proj = nn.Linear(2 * hidden_dim, hidden_dim, bias=False)
        self.scale = attn_dim ** -0.5
        self.speed_floor_kmh = speed_floor_kmh

        # same "learnable but not softmax-tied" philosophy as v4
        self.w_dist = nn.Parameter(torch.tensor(1.0))
        self.w_wind = nn.Parameter(torch.tensor(1.0))
        self.w_terrain = nn.Parameter(torch.tensor(1.0))
        self.w_lag = nn.Parameter(torch.tensor(1.0))
        self.log_sigma_tau = nn.Parameter(torch.tensor(_inv_softplus(sigma_tau_init_h)))

        dist = haversine_km(station_coords)                          # [N, N]
        neighbor_mask = dist <= dist_threshold_km
        self.register_buffer('neighbor_mask', neighbor_mask)
        self.register_buffer('dist_km', dist)

        dist_bias = torch.exp(-dist / sigma_d)
        self.register_buffer('dist_bias', dist_bias)                  # [N, N], static

        elev_diff = (station_elevation.unsqueeze(1) - station_elevation.unsqueeze(0)).abs()
        terrain_bias = torch.exp(-elev_diff / sigma_h)
        self.register_buffer('terrain_bias', terrain_bias)            # [N, N], static

        # bearing FROM source i TO receiver j, laid out as [j, i] (receiver
        # row, source col) to match this module's [B, N_j, K*N_i] score
        # layout - see forward()
        self.register_buffer('bearing_j_i', bearing_deg(station_coords).t())

    @staticmethod
    def _trend_augment(h_lag):
        """h_lag: [B, K, N, hidden_dim] -> [B, K, N, 2*hidden_dim], h_lag
        concatenated with its own consecutive-lag delta (zero at the last
        lag position, and everywhere when K=1) - see class docstring."""
        K = h_lag.shape[1]
        delta_h = torch.zeros_like(h_lag)
        if K > 1:
            delta_h[:, :K - 1] = h_lag[:, :K - 1] - h_lag[:, 1:K]
        return torch.cat([h_lag, delta_h], dim=-1)

    def forward(self, h_last, h_lag, travel_bearing_lag, wind_speed_kmh_lag, k_hours):
        """
        h_last              : [B, N, hidden_dim] - current/last-step states (query)
        h_lag                : [B, K, N, hidden_dim] - states at each of the K
                                most recent lags, k=0 (now) first (key/value)
        travel_bearing_lag   : [B, K, N] radians - each SOURCE station's wind
                                bearing (blowing toward) at that lag
        wind_speed_kmh_lag   : [B, K, N] each source's wind speed (km/h) at that lag
        k_hours              : [K] elapsed time k*dt_hours for each lag position
        returns              : [B, N, hidden_dim]
        """
        B, N, _ = h_last.shape
        K = h_lag.shape[1]

        h_lag_aug = self._trend_augment(h_lag)                           # [B, K, N, 2*hidden_dim]

        q = self.q_proj(h_last)                                          # [B, N, attn_dim]
        k = self.k_proj(h_lag_aug).reshape(B, K * N, -1)                 # [B, K*N, attn_dim]
        v = self.v_proj(h_lag_aug).reshape(B, K * N, -1)                 # [B, K*N, hidden_dim]
        content_score = torch.bmm(q, k.transpose(1, 2)) * self.scale     # [B, N, K*N]

        # theta: angle between SOURCE i's wind (at lag k) and bearing i->j
        angle_diff = travel_bearing_lag.unsqueeze(2) - self.bearing_j_i.view(1, 1, N, N)  # [B,K,N_j,N_i]
        wind_align = torch.clamp(torch.cos(angle_diff), min=0.0)

        speed = wind_speed_kmh_lag.clamp(min=0.0)
        tau = self.dist_km.view(1, 1, N, N) / (speed.unsqueeze(2) + self.speed_floor_kmh)  # [B,K,N_j,N_i]
        sigma_tau = F.softplus(self.log_sigma_tau) + 1e-6
        k_hours_b = k_hours.view(1, K, 1, 1)
        lag_bias = torch.exp(-((k_hours_b - tau) ** 2) / (2.0 * sigma_tau ** 2))            # [B,K,N_j,N_i]

        # [B,K,N_j,N_i] -> [B,N_j,K,N_i] -> [B,N_j,K*N_i], matching k/v's flattening order
        def _flatten(t):
            return t.permute(0, 2, 1, 3).reshape(B, N, K * N)

        wind_align_flat = _flatten(wind_align)
        lag_bias_flat = _flatten(lag_bias)

        dist_tiled = self.dist_bias.unsqueeze(1).expand(N, K, N).reshape(N, K * N)
        terrain_tiled = self.terrain_bias.unsqueeze(1).expand(N, K, N).reshape(N, K * N)
        mask_tiled = self.neighbor_mask.unsqueeze(1).expand(N, K, N).reshape(N, K * N)

        score = (content_score
                 + self.w_dist * dist_tiled.unsqueeze(0)
                 + self.w_wind * wind_align_flat
                 + self.w_terrain * terrain_tiled.unsqueeze(0)
                 + self.w_lag * lag_bias_flat)

        score = score.masked_fill(~mask_tiled.unsqueeze(0), float('-inf'))
        weights = torch.nan_to_num(torch.softmax(score, dim=-1), nan=0.0)
        return torch.bmm(weights, v)


class AirLapse2(nn.Module):
    """
    AirLapse (model/airlapse.py), unchanged everywhere except
    MultiLagPhysicsAwareSpatialAttention gaining the trend-augmented
    key/value described in that class's docstring above. Constructor,
    encoder (both spatial_mix_mode forks), VAE bottleneck, and decoder are
    all identical to AirLapse's, so this stays directly comparable through
    the same trainer - any behavior difference between AirLapse and
    AirLapse2 traces to exactly the one change in the attention class.

    spatial_mix_mode:
      - 'bottleneck': `nn.GRU`'s full per-step output (not just h_n) is
        kept; the last `max_lag` steps become the attention's keys/values
        in one mixing pass at the end of encoding.
      - 'per_step': the encoder is unrolled with a GRUCell; at every step,
        MultiLagPhysicsAwareSpatialAttention is called with K=1 (that
        step's own state and wind only) - the trend-augmentation is an
        exact no-op here (see that class's docstring), so per_step mode
        behaves identically to AirLapse's per_step mode.

    `max_lag` (bottleneck mode only) bounds how far back the lag-matching
    keys/values reach, clamped to hist_len.
    """
    def __init__(self, hist_len, pred_len, in_dim, city_num, batch_size, device,
                 edge_index, edge_attr, wind_mean, wind_std,
                 station_coords, station_elevation,
                 feature_mean, feature_std,
                 hidden_dim=64, latent_dim=16, attn_dim=32, num_layers=1,
                 dropout=0.1, logvar_clamp=10.0,
                 spatial_mix_mode='bottleneck', max_lag=6,
                 dist_threshold_km=300.0, sigma_d=200.0, sigma_h=1200.0,
                 sigma_tau_init_h=3.0, dt_hours=3.0):
        super(AirLapse2, self).__init__()
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

        # kept for signature parity across benchmark models
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.wind_mean = wind_mean
        self.wind_std = wind_std

        self.feature_dim = in_dim - 1
        # fixed tail layout dataset.py._process_feature always produces:
        # [...metero_use channels..., hour, weekday, speed_kmh, direc_deg]
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

        self.spatial_attn = MultiLagPhysicsAwareSpatialAttention(
            hidden_dim=hidden_dim,
            station_coords=station_coords,
            station_elevation=station_elevation,
            attn_dim=attn_dim,
            dist_threshold_km=dist_threshold_km,
            sigma_d=sigma_d,
            sigma_h=sigma_h,
            sigma_tau_init_h=sigma_tau_init_h,
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
        only) - the trend-augmentation is an exact no-op here (see
        MultiLagPhysicsAwareSpatialAttention's docstring)."""
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
                f"AirLapse2 was built with city_num={self.city_num}, but got "
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
