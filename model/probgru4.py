import torch
import torch.nn as nn


def haversine_km(coords):
    """coords: [N, 2] lat/lon in degrees -> [N, N] pairwise distance in km"""
    lat = torch.deg2rad(coords[:, 0])
    lon = torch.deg2rad(coords[:, 1])
    dlat = lat.unsqueeze(1) - lat.unsqueeze(0)
    dlon = lon.unsqueeze(1) - lon.unsqueeze(0)
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat).unsqueeze(1) * torch.cos(lat).unsqueeze(0) * torch.sin(dlon / 2) ** 2
    return 2 * 6371.0 * torch.asin(torch.sqrt(a.clamp(min=0)))


def bearing_deg(coords):
    """coords: [N, 2] lat/lon in degrees -> [N, N] bearing FROM i TO j, in radians"""
    lat = torch.deg2rad(coords[:, 0])
    lon = torch.deg2rad(coords[:, 1])
    dlon = lon.unsqueeze(0) - lon.unsqueeze(1)          # [N, N], j - i
    y = torch.sin(dlon) * torch.cos(lat).unsqueeze(0)
    x = torch.cos(lat).unsqueeze(1) * torch.sin(lat).unsqueeze(0) - \
        torch.sin(lat).unsqueeze(1) * torch.cos(lat).unsqueeze(0) * torch.cos(dlon)
    return torch.atan2(y, x)                             # [N, N], i -> j


class PhysicsAwareSpatialAttention(nn.Module):
    """
    Spatial attention with three interpretable, independently-weighted biases:
      - distance decay (static)
      - terrain barrier (static)
      - wind alignment (dynamic - recomputed from the CURRENT wind reading,
        not baked into a fixed edge feature, since wind changes every step)

    Each bias has its own learnable scalar weight (no softmax mixing), so
    after training you can inspect self.w_dist / self.w_wind / self.w_terrain
    directly to see which physical factor the model actually leaned on.

    Neighborhood is a fixed distance threshold (auditable), NOT a learned
    per-station radius - trades some flexibility for a graph you can always
    explain.
    """
    def __init__(self, hidden_dim, station_coords, station_elevation,
                 attn_dim=32, dist_threshold_km=300.0, sigma_d=200.0, sigma_h=1200.0):
        super().__init__()

        self.q_proj = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.scale = attn_dim ** -0.5

        # learnable but NOT softmax-tied - each can independently grow, shrink,
        # or go to ~0 if that factor turns out not to matter
        self.w_dist = nn.Parameter(torch.tensor(1.0))
        self.w_wind = nn.Parameter(torch.tensor(1.0))
        self.w_terrain = nn.Parameter(torch.tensor(1.0))

        dist = haversine_km(station_coords)                          # [N, N]
        neighbor_mask = dist <= dist_threshold_km
        self.register_buffer('neighbor_mask', neighbor_mask)

        dist_bias = torch.exp(-dist / sigma_d)
        self.register_buffer('dist_bias', dist_bias)                  # [N, N], static

        elev_diff = (station_elevation.unsqueeze(1) - station_elevation.unsqueeze(0)).abs()
        terrain_bias = torch.exp(-elev_diff / sigma_h)
        self.register_buffer('terrain_bias', terrain_bias)            # [N, N], static

        self.register_buffer('bearing', bearing_deg(station_coords))  # [N, N], static geometry

    def forward(self, h, wind_dir):
        """
        h        : [B, N, hidden_dim] - per-node states
        wind_dir : [B, N] - current wind direction (radians) per node, from
                   the live feature stream, not a precomputed edge attribute
        returns  : [B, N, hidden_dim]
        """
        B, N, _ = h.shape

        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)
        content_score = torch.bmm(q, k.transpose(1, 2)) * self.scale   # [B, N, N]

        # wind alignment: how well does j lie downwind of i, RIGHT NOW
        angle_diff = wind_dir.unsqueeze(2) - self.bearing.unsqueeze(0)  # [B, N, N]
        wind_align = torch.clamp(torch.cos(angle_diff), min=0.0)

        score = (content_score
                 + self.w_dist * self.dist_bias.unsqueeze(0)
                 + self.w_wind * wind_align
                 + self.w_terrain * self.terrain_bias.unsqueeze(0))

        score = score.masked_fill(~self.neighbor_mask.unsqueeze(0), float('-inf'))
        weights = torch.nan_to_num(torch.softmax(score, dim=-1), nan=0.0)
        return torch.bmm(weights, v)


class ProbGRUModel4(nn.Module):
    """
    ProbGRUModel + physics-aware spatial mixing.

    spatial_mix_mode controls WHEN the spatial mixing happens:

      - 'bottleneck' (original behaviour): plain fused nn.GRU encodes each
        node's sequence independently, then ONE spatial mixing pass happens
        at the end, using only the final wind reading. Cheap, but throws
        away how the wind field evolved over the history window and how
        far spatial information could propagate in time.

      - 'per_step' (Strategy 1): the encoder is unrolled with a GRUCell.
        At EVERY history timestep, wind direction is recomputed from that
        timestep's reading, spatial context is mixed in, and the MIXED
        hidden state is fed back into the recurrence for the next step.
        This lets spatially-mixed information from step t influence the
        encoding at step t+1, t+2, ... instead of being injected only once
        after the fact - closer in spirit to how AirFormer interleaves
        DS_MSA and CT_MSA every block instead of once at the end.

    Both modes share the same downstream VAE bottleneck / decoder, so they
    can be ablated head-to-head through the same trainer.
    """
    def __init__(self, hist_len, pred_len, in_dim, city_num, batch_size, device,
                 edge_index, edge_attr, wind_mean, wind_std,
                 station_coords, station_elevation, wind_channel_idx=(0, 1),
                 hidden_dim=64, latent_dim=16, attn_dim=32, num_layers=1,
                 dropout=0.1, logvar_clamp=10.0,
                 spatial_mix_mode='bottleneck'):
        super(ProbGRUModel4, self).__init__()
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
        self.wind_channel_idx = wind_channel_idx  # (u_idx, v_idx) within feature's per-node channels
        self.spatial_mix_mode = spatial_mix_mode

        # kept for signature parity across benchmark models; the spatial
        # mixing here uses station_coords/elevation/wind directly instead
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.wind_mean = wind_mean
        self.wind_std = wind_std

        self.feature_dim = in_dim - 1

        if spatial_mix_mode == 'bottleneck':
            self.encoder = nn.GRU(
                input_size=in_dim, hidden_size=hidden_dim, num_layers=num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
            )
        else:  # 'per_step'
            self.encoder_cell = nn.GRUCell(input_size=in_dim, hidden_size=hidden_dim)
            self.step_dropout = nn.Dropout(dropout)

        self.spatial_attn = PhysicsAwareSpatialAttention(
            hidden_dim=hidden_dim,
            station_coords=station_coords,
            station_elevation=station_elevation,
            attn_dim=attn_dim,
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

    def _wind_dir_at(self, feature_hist, t, B, N):
        """feature_hist: [B, T, N, feature_dim] -> wind_dir at step t, [B, N] radians"""
        u_idx, v_idx = self.wind_channel_idx
        wind_u = feature_hist[:, t, :, u_idx]
        wind_v = feature_hist[:, t, :, v_idx]
        return torch.atan2(wind_v, wind_u)

    def _encode_bottleneck(self, x, feature_hist, B, N):
        """Original behaviour: fused GRU, single mixing pass at the end."""
        _, h_n = self.encoder(x)
        h_T = h_n[-1]                                       # [B*N, hidden_dim]
        h_grid = h_T.reshape(B, N, self.hidden_dim)

        # current wind direction, read from the LAST observed history step
        wind_dir = self._wind_dir_at(feature_hist, -1, B, N)  # [B, N]

        context = self.spatial_attn(h_grid, wind_dir)        # [B, N, hidden_dim]
        h_mixed = self.mix_gate(torch.cat([h_grid, context], dim=-1))
        return h_mixed.reshape(B * N, self.hidden_dim)

    def _encode_per_step(self, x, feature_hist, B, N):
        """
        Strategy 1: unroll the encoder manually. At every timestep, recompute
        wind direction from that step's reading, mix in spatial context, and
        feed the MIXED hidden state back into the recurrence so spatially-
        aware information can compound over the history window.
        """
        T = x.shape[1]
        h = torch.zeros(B * N, self.hidden_dim, device=x.device, dtype=x.dtype)

        for t in range(T):
            x_t = x[:, t, :]                                  # [B*N, in_dim]
            h = self.encoder_cell(x_t, h)                     # [B*N, hidden_dim]
            h = self.step_dropout(h)

            h_grid_t = h.reshape(B, N, self.hidden_dim)
            wind_dir_t = self._wind_dir_at(feature_hist, t, B, N)  # [B, N]

            context_t = self.spatial_attn(h_grid_t, wind_dir_t)    # [B, N, hidden_dim]
            h_mixed_t = self.mix_gate(torch.cat([h_grid_t, context_t], dim=-1))
            h = h_mixed_t.reshape(B * N, self.hidden_dim)          # feed mixed state forward

        return h  # already spatially mixed at the final step

    def forward(self, pm25_hist, feature):
        feature_hist = feature[:, :self.hist_len]
        feature_future = feature[:, self.hist_len:self.hist_len + self.pred_len]
        inputs = torch.cat([pm25_hist, feature_hist], dim=-1)

        B, T, N, C = inputs.shape
        if N != self.city_num:
            raise ValueError(
                f"ProbGRUModel3 was built with city_num={self.city_num}, but got "
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

        # --- encode + mix, mode-dependent ---
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