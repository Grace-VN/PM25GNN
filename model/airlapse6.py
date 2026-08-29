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
    airlapse4.py's MultiLagPhysicsAwareSpatialAttention, with the explicit
    transport estimate upgraded from a 1D advection-diffusion projection to
    a full 2D downwind/crosswind decomposition - the actual Gaussian-plume
    picture, rather than airlapse4's simplification of it.

    Everything else is unchanged from airlapse4.py: the learned
    content+physics attention (`context`, using dist_bias/wind_align
    (clamped)/lag_bias/w_dist/w_wind/w_terrain/w_lag exactly as before) is
    completely untouched. Only the SECOND, non-learned output -
    `transported` - is recomputed with the fuller 2D physics.

    WHY airlapse4's version was a simplification: it modeled transport as
    1D advection-diffusion ALONG THE FIXED SOURCE-RECEIVER LINE, folding
    wind misalignment into a reduced effective speed along that one axis
    (v_radial = wind_speed * cos(theta)). That conflates two physically
    different things - "the wind isn't blowing very fast toward the
    receiver" and "the receiver sits off to the side of where the wind is
    actually going" - into a single scalar. A real plume's material
    travels along the ACTUAL wind direction and only reaches an off-axis
    receiver by additionally diffusing sideways to it; airlapse4's version
    never represents that sideways distance at all, so at 90 degrees
    misalignment it quietly falls back to isotropic pure diffusion instead
    of the sharp lateral-offset penalty a real plume would apply.

    EXPLICIT PHYSICAL TRANSPORT ESTIMATE, now via the 2D advection-
    diffusion Green's function decomposed into downwind/crosswind axes
    (still returned as `transported`, still no learned score weight of
    its own beyond the two diffusivities below - not a w_* attention
    bias, a direct physical computation in raw PM2.5 units):
        For source j, receiver i, lag k (elapsed time t_k, floored at
        t_eps_hours as in airlapse4), let theta_ijk be the angle between
        source j's wind bearing and the straight-line bearing from j to i
        (the same angle wind_align/v_radial used above). Decompose the
        source-receiver distance d_ij into coordinates ALONG and ACROSS
        the wind's actual direction:
            x_ijk = d_ij * cos(theta_ijk)     (downwind coordinate of the
                                                receiver, along the wind)
            y_ijk = d_ij * sin(theta_ijk)     (crosswind coordinate - how
                                                far off the wind's direct
                                                path the receiver sits)
        and evaluate the anisotropic 2D point-source Green's function
        (product of two independent 1D Gaussians along each axis, a
        standard construction for diffusion aligned with a wind axis -
        the real-world analogue is Pasquill-Gifford dispersion, where
        crosswind spread is tracked separately from downwind spread):
            G_ijk = 1/(4*pi*sqrt(D_x*D_y)*t_k)
                    * exp[-(x_ijk - U_ijk*t_k)^2 / (4*D_x*t_k)
                          - y_ijk^2 / (4*D_y*t_k)]
                    (U_ijk = source j's wind speed, unprojected - the full
                    magnitude now travels along the wind's OWN direction,
                    not a reduced radial component.)
        This directly answers "how much, and when" the way the user asked
        for it: alignment now enters through BOTH x and y (better
        alignment -> larger x, smaller y -> less crosswind penalty AND a
        sharper, sooner downwind peak), distance enters through both
        (d_ij feeds both x and y), and elapsed time enters through t_k in
        both exponential terms and the 1/t_k prefactor (2D spreading
        dilutes faster than airlapse4's 1D 1/sqrt(t_k) - a point source's
        mass spreads over an AREA in 2D, not just a length). Backward wind
        (x_ijk - U*t_k growing without bound) and calm wind (U=0,
        collapsing to pure 2D diffusion) degrade the same way airlapse4's
        1D version did, for the same reason - no special-casing needed.
        D_x, D_y (downwind/crosswind eddy diffusivity, km^2/hour) are the
        TWO learnable parameters this adds (airlapse4 had one, isotropic
        D) - softplus-parameterized, independently initialized to the
        same middling value (see __init__), free to diverge during
        training if the data supports asymmetric dispersion.
            w_ijk    = neighbor_mask_ij * (i != j) * G_ijk
            pi_ijk   = w_ijk / (sum_k w_ijk + eps)
            transport_j->i = sum_k pi_ijk * pm25_j(t-k)
            reach_ij = max_k w_ijk
            transported_i = sum_j reach_ij * transport_j->i
        (identical aggregation to airlapse4 - only G_ijk's derivation
        changed.)

        Known limitation, not fixed here: at exactly calm wind (U=0), the
        wind "direction" used to split d_ij into x/y is physically
        meaningless (there's no real axis to be aligned or misaligned
        with), yet the model still applies it, so a noisy recorded
        direction at zero speed can arbitrarily favor D_x over D_y or vice
        versa. This is a genuine edge case in truly calm conditions, left as a
        documented simplification rather than added complexity (e.g.
        blending toward isotropic diffusion as speed -> 0) for a case that
        real station wind logs rarely hit exactly.
    """
    def __init__(self, hidden_dim, station_coords, station_elevation,
                 attn_dim=32, dist_threshold_km=300.0, sigma_d=200.0,
                 sigma_h=1200.0, sigma_tau_init_h=3.0, speed_floor_kmh=0.5,
                 diffusivity_downwind_km2_per_hour_init=50.0,
                 diffusivity_crosswind_km2_per_hour_init=50.0,
                 t_eps_hours=0.25):
        super().__init__()

        self.q_proj = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.scale = attn_dim ** -0.5
        self.speed_floor_kmh = speed_floor_kmh
        self.t_eps_hours = t_eps_hours

        # same "learnable but not softmax-tied" philosophy as v4
        self.w_dist = nn.Parameter(torch.tensor(1.0))
        self.w_wind = nn.Parameter(torch.tensor(1.0))
        self.w_terrain = nn.Parameter(torch.tensor(1.0))
        self.w_lag = nn.Parameter(torch.tensor(1.0))
        self.log_sigma_tau = nn.Parameter(torch.tensor(_inv_softplus(sigma_tau_init_h)))

        # the two learnable parameters the transport estimate adds: eddy
        # diffusivity (km^2/hour) along the wind's downwind and crosswind
        # axes, for its 2D Green's function, below. airlapse4 had one
        # isotropic D; here they're free to diverge.
        self.log_D_downwind = nn.Parameter(
            torch.tensor(_inv_softplus(diffusivity_downwind_km2_per_hour_init)))
        self.log_D_crosswind = nn.Parameter(
            torch.tensor(_inv_softplus(diffusivity_crosswind_km2_per_hour_init)))

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

        # neighbor_mask with the diagonal zeroed, precomputed once - used
        # only by the explicit transport estimate below (not by the
        # content+physics attention path above, where a station attending
        # to its own recent history via neighbor_mask alone is normal/
        # intended, so that mask is kept separate and untouched).
        not_self = ~torch.eye(dist.shape[0], dtype=torch.bool)
        self.register_buffer('transport_mask', neighbor_mask & not_self)

    def forward(self, h_last, h_lag, travel_bearing_lag, wind_speed_kmh_lag, k_hours, pm25_lag):
        """
        h_last              : [B, N, hidden_dim] - current/last-step states (query)
        h_lag                : [B, K, N, hidden_dim] - states at each of the K
                                most recent lags, k=0 (now) first (key/value)
        travel_bearing_lag   : [B, K, N] radians - each SOURCE station's wind
                                bearing (blowing toward) at that lag
        wind_speed_kmh_lag   : [B, K, N] each source's wind speed (km/h) at that lag
        k_hours              : [K] elapsed time k*dt_hours for each lag position
        pm25_lag             : [B, K, N] each source's raw/normalized PM2.5 at
                                that lag, SAME (b, k, source) indexing as h_lag
        returns              : (context [B, N, hidden_dim], transported [B, N])
        """
        B, N, _ = h_last.shape
        K = h_lag.shape[1]

        q = self.q_proj(h_last)                                          # [B, N, attn_dim]
        k = self.k_proj(h_lag).reshape(B, K * N, -1)                     # [B, K*N, attn_dim]
        v = self.v_proj(h_lag).reshape(B, K * N, -1)                     # [B, K*N, hidden_dim]
        content_score = torch.bmm(q, k.transpose(1, 2)) * self.scale     # [B, N, K*N]

        # theta: angle between SOURCE i's wind (at lag k) and bearing i->j
        angle_diff = travel_bearing_lag.unsqueeze(2) - self.bearing_j_i.view(1, 1, N, N)  # [B,K,N_j,N_i]
        cos_theta = torch.cos(angle_diff)                                 # SIGNED - shared by wind_align below
        sin_theta = torch.sin(angle_diff)                                 # SIGNED, but only ever squared below -
        # sign doesn't matter for the crosswind term, only |sin_theta|'s
        # magnitude does, via y^2.
        wind_align = torch.clamp(cos_theta, min=0.0)                      # clamp is only for the score bonus below

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
        context = torch.bmm(weights, v)

        # --- explicit physical transport estimate: 2D advection-diffusion
        # Green's function, downwind/crosswind decomposed (see class
        # docstring) - independent of, and not feeding back into, the
        # learned attention above. ---
        d = self.dist_km.view(1, 1, N, N)
        x = d * cos_theta                                                  # downwind coordinate - [B,K,N_j,N_i]
        y = d * sin_theta                                                  # crosswind coordinate
        U = speed.unsqueeze(2)                                             # full (unprojected) wind speed
        t_eff = k_hours_b + self.t_eps_hours                               # [1,K,1,1], keeps t=0 well-defined
        D_x = F.softplus(self.log_D_downwind) + 1e-6
        D_y = F.softplus(self.log_D_crosswind) + 1e-6
        green = (
            1.0 / (4.0 * math.pi * torch.sqrt(D_x * D_y) * t_eff)
            * torch.exp(
                -((x - U * t_eff) ** 2) / (4.0 * D_x * t_eff)
                - (y ** 2) / (4.0 * D_y * t_eff)
            )
        )                                                                  # [B, K, N_receiver, N_source]

        w_transport = self.transport_mask.to(green.dtype).view(1, 1, N, N) * green

        pi = w_transport / (w_transport.sum(dim=1, keepdim=True) + 1e-8)  # sums to 1 over k
        transport_from_source = torch.einsum('bkij,bkj->bij', pi, pm25_lag)  # [B, N_receiver, N_source]
        reach = w_transport.max(dim=1).values                             # [B, N_receiver, N_source]
        transported = (reach * transport_from_source).sum(dim=-1)         # [B, N_receiver]

        return context, transported


class AirLapse6(nn.Module):
    """
    AirLapse4 (model/airlapse4.py), with its explicit transport estimate
    upgraded to a full 2D downwind/crosswind advection-diffusion
    decomposition - see MultiLagPhysicsAwareSpatialAttention's docstring
    above for the physics and for exactly what stayed the same (the
    learned content+physics attention, `context`, including lag_bias, is
    untouched - this variant builds on AirLapse4, not AirLapse5).
    Constructor, both spatial_mix_mode forks' control flow, VAE
    bottleneck, and decoder are otherwise identical to AirLapse4's - the
    wrapper class itself needed no changes beyond its name, this
    docstring, and forwarding two diffusivities instead of one.

    spatial_mix_mode:
      - 'bottleneck': `nn.GRU`'s full per-step output (not just h_n) is
        kept; the last `max_lag` steps become the attention's keys/values
        (and the transport estimate's source lags) in one pass at the end
        of encoding.
      - 'per_step': the encoder is unrolled with a GRUCell; at every step,
        MultiLagPhysicsAwareSpatialAttention is called with K=1 (that
        step's own state, wind, and PM2.5 only) - the Green's function
        transport estimate degrades to a single-lag evaluation there, same
        as any other K=1 case.

    `max_lag` (bottleneck mode only) bounds how far back the lag-matching
    keys/values (and transport source lags) reach, clamped to hist_len.
    """
    def __init__(self, hist_len, pred_len, in_dim, city_num, batch_size, device,
                 edge_index, edge_attr, wind_mean, wind_std,
                 station_coords, station_elevation,
                 feature_mean, feature_std,
                 hidden_dim=64, latent_dim=16, attn_dim=32, num_layers=1,
                 dropout=0.1, logvar_clamp=10.0,
                 spatial_mix_mode='bottleneck', max_lag=6,
                 dist_threshold_km=300.0, sigma_d=200.0, sigma_h=1200.0,
                 sigma_tau_init_h=3.0, dt_hours=3.0,
                 diffusivity_downwind_km2_per_hour_init=50.0,
                 diffusivity_crosswind_km2_per_hour_init=50.0,
                 t_eps_hours=0.25):
        super(AirLapse6, self).__init__()
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
            diffusivity_downwind_km2_per_hour_init=diffusivity_downwind_km2_per_hour_init,
            diffusivity_crosswind_km2_per_hour_init=diffusivity_crosswind_km2_per_hour_init,
            t_eps_hours=t_eps_hours,
        )
        # +1: the explicit scalar transport estimate, concatenated alongside
        # h_grid and the learned attention context.
        self.mix_gate = nn.Linear(hidden_dim * 2 + 1, hidden_dim)

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
        spatial attention's keys/values (and pm25_lag its transport
        estimate's source values), each scored with its own wind."""
        output, h_n = self.encoder(x)                          # output: [B*N, T, hidden_dim]
        h_T = h_n[-1]
        h_grid = h_T.reshape(B, N, self.hidden_dim)

        T = output.shape[1]
        K = self.max_lag
        lag_idxs = list(range(T - K, T))                       # oldest -> newest, newest = "now"
        h_lag = output[:, lag_idxs, :].reshape(B, N, K, self.hidden_dim).permute(0, 2, 1, 3)  # [B,K,N,hidden]

        # x's channel 0 is raw/normalized PM2.5 (see forward(): x = cat([
        # pm25_hist, feature_hist], -1)) - pull the same lag_idxs used for
        # h_lag, same (B, K, N) layout.
        pm25_lag = x[:, lag_idxs, 0].reshape(B, N, K).permute(0, 2, 1)  # [B, K, N]

        travel_bearing_lag, speed_kmh_lag = self._wind_at_idxs(feature_hist, lag_idxs, B, N)
        k_hours = torch.tensor(
            [(T - 1 - idx) * self.dt_hours for idx in lag_idxs],
            dtype=x.dtype, device=x.device,
        )

        context, transported = self.spatial_attn(
            h_grid, h_lag, travel_bearing_lag, speed_kmh_lag, k_hours, pm25_lag)
        h_mixed = self.mix_gate(torch.cat([h_grid, context, transported.unsqueeze(-1)], dim=-1))
        return h_mixed.reshape(B * N, self.hidden_dim)

    def _encode_per_step(self, x, feature_hist, B, N):
        """Unroll the encoder manually; at every step, K=1 (current step
        only) - the transport estimate degrades to a single-lag evaluation
        of the Green's function, see MultiLagPhysicsAwareSpatialAttention's
        docstring."""
        T = x.shape[1]
        h = torch.zeros(B * N, self.hidden_dim, device=x.device, dtype=x.dtype)

        for t in range(T):
            x_t = x[:, t, :]
            h = self.encoder_cell(x_t, h)
            h = self.step_dropout(h)

            h_grid_t = h.reshape(B, N, self.hidden_dim)
            h_lag_t = h_grid_t.unsqueeze(1)                     # [B, 1, N, hidden_dim]
            pm25_lag_t = x[:, t, 0].reshape(B, N).unsqueeze(1)  # [B, 1, N]
            travel_bearing_t, speed_kmh_t = self._wind_at_idxs(feature_hist, [t], B, N)
            k_hours = torch.zeros(1, dtype=x.dtype, device=x.device)

            context_t, transported_t = self.spatial_attn(
                h_grid_t, h_lag_t, travel_bearing_t, speed_kmh_t, k_hours, pm25_lag_t)
            h_mixed_t = self.mix_gate(
                torch.cat([h_grid_t, context_t, transported_t.unsqueeze(-1)], dim=-1))
            h = h_mixed_t.reshape(B * N, self.hidden_dim)

        return h

    def forward(self, pm25_hist, feature):
        feature_hist = feature[:, :self.hist_len]
        feature_future = feature[:, self.hist_len:self.hist_len + self.pred_len]
        inputs = torch.cat([pm25_hist, feature_hist], dim=-1)

        B, T, N, C = inputs.shape
        if N != self.city_num:
            raise ValueError(
                f"AirLapse6 was built with city_num={self.city_num}, but got "
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
