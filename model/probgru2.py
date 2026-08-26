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


class AdvectivePhysicsAwareSpatialAttention(nn.Module):
    """
    PhysicsAwareSpatialAttention (probgru3/4/5) + a wind-advected distance
    bias, implementing the "moving puff" geometry:

        A  = station i's position at time t
        A' = where i's air parcel drifts to after dt_hours, carried by
             i's current wind (A -> A' has length AA' = wind_speed_i * dt_hours,
             direction = wind_dir_i)
        B  = station j's position (receiver)

    We want A'B, the distance from the DRIFTED parcel to j, not the resting
    distance AB = d_ij. Dropping a perpendicular from A' onto line AB at H
    and letting alpha = angle(AA', AB) = angle between i's wind and the
    bearing from i to j:

        A'H = AA' - AB*cos(alpha),   HB = AB*sin(alpha)
        A'B^2 = A'H^2 + HB^2
              = AA'^2 - 2*AA'*AB*cos(alpha) + AB^2*cos^2(alpha) + AB^2*sin^2(alpha)
              = AA'^2 + AB^2 - 2*AA'*AB*cos(alpha)          <- law of cosines

    i.e.  d_eff(i,j)^2 = d_ij^2 + disp_i^2 - 2*d_ij*disp_i*cos(theta)

    where theta is the SAME angle already used for the upwind/downwind gate
    in PhysicsAwareSpatialAttention (wind_align = clamp(cos(theta), min=0)),
    so this costs one extra sqrt, not a second trig pass.

    Sanity checks:
      - disp_i = 0                -> d_eff = d_ij            (static case)
      - theta  = 0  (wind -> j)   -> d_eff = |d_ij - disp_i|  (parcel closes the gap)
      - theta  = pi (wind -> away)-> d_eff = d_ij + disp_i    (but wind_align
        already ~zeroes this pair via the upwind/downwind gate, so it rarely
        matters in practice)

    SINGLE-HOP BY DESIGN: dt_hours should be the dataset's native step
    (3h for KnowAir - see dataset.py's Arrow.interval('hour', ..., 3) and
    GraphGNN's hardcoded `3 * src_wind_speed` in PM25_GNN.py), not the full
    forecast horizon. Longer-range and multi-hop (i -> k -> j) effects are
    meant to emerge from chaining this single-hop bias across steps (see
    spatial_mix_mode='per_step' on ProbGRUModel6 below) using each step's
    own wind reading, rather than from one closed-form multi-step
    displacement - a single cumulative displacement can't express a wind
    direction that shifts mid-window, and can't route through intermediate
    stations. Note ProbGRUModel6 (like its ProbGRUModel4 base) only mixes
    during ENCODING - the decoder loop stays per-node-independent, same
    as v4 (unlike v5's optional decode_spatial_mix).

    NOTE ON INDEXING CONVENTION (carried over unchanged from probgru3/4/5,
    flagged here rather than silently changed): softmax is over dim=-1 (j),
    and context = weights @ v pulls CONTRIBUTOR j's value into RECEIVER i's
    output. The existing wind_align term (and this new advection term, to
    stay consistent/comparable with v3-v5) index wind at the ROW station
    (wind_dir[i], wind_speed[i]) against bearing(i -> j). A stricter
    "does j's air actually reach i" check would instead use wind at the
    COLUMN station j against bearing(j -> i). Left as-is here for
    apples-to-apples comparison with prior versions; worth a v6b ablation
    if the direction convention turns out to matter empirically.

    Distance units: dist_km (haversine) is kilometers. Wind speed is
    computed upstream from the raw u/v wind components, which are in m/s,
    so displacement is converted m/s -> km/h via the same 3.6 factor
    dataset.py uses for its own precomputed speed feature, then scaled by
    dt_hours.
    """
    def __init__(self, hidden_dim, station_coords, station_elevation,
                 attn_dim=32, dist_threshold_km=300.0, sigma_d=200.0,
                 sigma_h=1200.0, dt_hours=3.0):
        super().__init__()

        self.q_proj = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.scale = attn_dim ** -0.5

        # learnable, NOT softmax-tied - each can independently grow, shrink,
        # or go to ~0. w_adv is new; if advection doesn't add signal beyond
        # the static terms, training can push it toward 0 on its own.
        self.w_dist = nn.Parameter(torch.tensor(1.0))
        self.w_wind = nn.Parameter(torch.tensor(1.0))
        self.w_terrain = nn.Parameter(torch.tensor(1.0))
        self.w_adv = nn.Parameter(torch.tensor(1.0))

        self.dt_hours = dt_hours
        self.sigma_d = sigma_d
        self.ms_to_kmh = 3.6  # 1 m/s = 3.6 km/h, matches dataset.py's speed feature convention

        dist = haversine_km(station_coords)                          # [N, N], km
        neighbor_mask = dist <= dist_threshold_km
        self.register_buffer('neighbor_mask', neighbor_mask)
        self.register_buffer('dist_km', dist)                         # raw km, needed for law of cosines

        dist_bias = torch.exp(-dist / sigma_d)
        self.register_buffer('dist_bias', dist_bias)                  # [N, N], static

        elev_diff = (station_elevation.unsqueeze(1) - station_elevation.unsqueeze(0)).abs()
        terrain_bias = torch.exp(-elev_diff / sigma_h)
        self.register_buffer('terrain_bias', terrain_bias)            # [N, N], static

        self.register_buffer('bearing', bearing_deg(station_coords))  # [N, N], static geometry, i -> j

    def forward(self, h, wind_dir, wind_speed_ms):
        """
        h             : [B, N, hidden_dim] - per-node states
        wind_dir      : [B, N] - current wind direction (radians) per node
        wind_speed_ms : [B, N] - current wind speed in m/s per node (i.e.
                        sqrt(u^2 + v^2) from the raw wind components, same
                        source atan2(v, u) uses for wind_dir - NOT the
                        dataset's precomputed km/h "speed" feature, so the
                        conversion below is still needed)
        returns       : [B, N, hidden_dim]
        """
        B, N, _ = h.shape

        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)
        content_score = torch.bmm(q, k.transpose(1, 2)) * self.scale   # [B, N, N]

        # theta: angle between wind_i and bearing i->j - shared by the
        # upwind/downwind gate AND the law-of-cosines advected distance
        angle_diff = wind_dir.unsqueeze(2) - self.bearing.unsqueeze(0)  # [B, N, N]
        cos_theta = torch.cos(angle_diff)
        wind_align = torch.clamp(cos_theta, min=0.0)

        # law of cosines: d_eff(i,j)^2 = d_ij^2 + disp_i^2 - 2*d_ij*disp_i*cos(theta)
        disp_km = (wind_speed_ms * self.ms_to_kmh * self.dt_hours).unsqueeze(2)  # [B, N, 1], per source i
        dist_b = self.dist_km.unsqueeze(0)                                       # [1, N, N]
        d_eff_sq = (dist_b ** 2 + disp_km ** 2 - 2 * dist_b * disp_km * cos_theta).clamp(min=0.0)
        d_eff = torch.sqrt(d_eff_sq + 1e-8)
        adv_bias = torch.exp(-d_eff / self.sigma_d)                    # [B, N, N], dynamic

        score = (content_score
                 + self.w_dist * self.dist_bias.unsqueeze(0)
                 + self.w_wind * wind_align
                 + self.w_terrain * self.terrain_bias.unsqueeze(0)
                 + self.w_adv * adv_bias)

        score = score.masked_fill(~self.neighbor_mask.unsqueeze(0), float('-inf'))
        weights = torch.nan_to_num(torch.softmax(score, dim=-1), nan=0.0)
        return torch.bmm(weights, v)


class ProbGRUModel2(nn.Module):
    """
    ProbGRUModel4 + wind-advected effective distance (see
    AdvectivePhysicsAwareSpatialAttention above). Built on ProbGRUModel4,
    NOT v5 - there is no decode_spatial_mix here; spatial mixing only
    happens during ENCODING, same scope as v4. Everything else -
    spatial_mix_mode fork, VAE bottleneck, decoder - is unchanged from
    ProbGRUModel4, so results are directly comparable to v3/v4 ablations
    through the same trainer.

    spatial_mix_mode controls WHEN the (advection-aware) spatial mixing
    happens:

      - 'bottleneck' (original behaviour): plain fused nn.GRU encodes each
        node's sequence independently, then ONE mixing pass happens at the
        end, using only the final observed wind reading (direction AND
        speed, since d_eff needs both).

      - 'per_step' (Strategy 1): the encoder is unrolled with a GRUCell.
        At EVERY history timestep, wind direction and speed are recomputed
        from that timestep's reading, d_eff is re-derived, spatial context
        is mixed in, and the MIXED hidden state is fed back into the
        recurrence for the next step - this is what lets single-hop
        advection compound into effectively multi-step, multi-hop
        propagation over the history window, rather than needing a
        closed-form multi-step displacement formula.

    Both modes share the same downstream VAE bottleneck / decoder, so they
    can be ablated head-to-head through the same trainer.
    """
    def __init__(self, hist_len, pred_len, in_dim, city_num, batch_size, device,
                 edge_index, edge_attr, wind_mean, wind_std,
                 station_coords, station_elevation, wind_channel_idx=(0, 1),
                 hidden_dim=64, latent_dim=16, attn_dim=32, num_layers=1,
                 dropout=0.1, logvar_clamp=10.0,
                 spatial_mix_mode='bottleneck', dt_hours=3.0):
        super(ProbGRUModel2, self).__init__()
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
        self.dt_hours = dt_hours

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

        self.spatial_attn = AdvectivePhysicsAwareSpatialAttention(
            hidden_dim=hidden_dim,
            station_coords=station_coords,
            station_elevation=station_elevation,
            attn_dim=attn_dim,
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

    def _wind_vector_at(self, feature_hist, t, B, N):
        """feature_hist: [B, T, N, feature_dim] -> (wind_dir, wind_speed_ms) at
        step t, each [B, N]. wind_speed_ms is derived from the same raw u/v
        components as wind_dir (m/s), not the dataset's precomputed km/h
        speed feature - AdvectivePhysicsAwareSpatialAttention does its own
        m/s -> km/h conversion."""
        u_idx, v_idx = self.wind_channel_idx
        wind_u = feature_hist[:, t, :, u_idx]
        wind_v = feature_hist[:, t, :, v_idx]
        wind_dir = torch.atan2(wind_v, wind_u)
        wind_speed = torch.sqrt(wind_u ** 2 + wind_v ** 2 + 1e-8)
        return wind_dir, wind_speed

    def _encode_bottleneck(self, x, feature_hist, B, N):
        """Original behaviour: fused GRU, single advection-aware mixing
        pass at the end, using the last observed history step's wind."""
        _, h_n = self.encoder(x)
        h_T = h_n[-1]                                       # [B*N, hidden_dim]
        h_grid = h_T.reshape(B, N, self.hidden_dim)

        # current wind direction + speed, read from the LAST observed
        # history step
        wind_dir, wind_speed = self._wind_vector_at(feature_hist, -1, B, N)

        context = self.spatial_attn(h_grid, wind_dir, wind_speed)    # [B, N, hidden_dim]
        h_mixed = self.mix_gate(torch.cat([h_grid, context], dim=-1))
        return h_mixed.reshape(B * N, self.hidden_dim)

    def _encode_per_step(self, x, feature_hist, B, N):
        """
        Strategy 1: unroll the encoder manually. At every timestep,
        recompute wind direction AND speed from that step's reading,
        re-derive d_eff, mix in advection-aware spatial context, and feed
        the MIXED hidden state back into the recurrence so spatially-aware
        information can compound over the history window.
        """
        T = x.shape[1]
        h = torch.zeros(B * N, self.hidden_dim, device=x.device, dtype=x.dtype)

        for t in range(T):
            x_t = x[:, t, :]                                  # [B*N, in_dim]
            h = self.encoder_cell(x_t, h)                     # [B*N, hidden_dim]
            h = self.step_dropout(h)

            h_grid_t = h.reshape(B, N, self.hidden_dim)
            wind_dir_t, wind_speed_t = self._wind_vector_at(feature_hist, t, B, N)

            context_t = self.spatial_attn(h_grid_t, wind_dir_t, wind_speed_t)  # [B, N, hidden_dim]
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

        # --- decode, identical to ProbGRUModel4: per-node-independent for
        # the whole horizon, no spatial mixing inside this loop ---
        h_dec = self.decoder_init(torch.cat([h_mixed, z], dim=-1))
        preds = []
        for t in range(self.pred_len):
            step_in = torch.cat([feat_fut[:, t], z], dim=-1)
            h_dec = self.decoder_cell(step_in, h_dec)
            preds.append(self.output_head(h_dec))

        out = torch.stack(preds, dim=1)
        pm25_pred = out.reshape(B, N, self.pred_len, 1).permute(0, 2, 1, 3)
        return pm25_pred