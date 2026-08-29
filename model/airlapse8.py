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
    airlapse7.py's MultiLagPhysicsAwareSpatialAttention, with the two paths
    that were previously fully independent - the learned content+physics
    attention (`context`) and the explicit physical transport estimate
    (`transported`) - now allowed to INTERACT at exactly one point: stage
    2's (intersource) softmax score is a learnable blend of the physical
    plausibility airlapse7 already used AND the learned attention's own
    per-source relevance. Everything else - stage 1 (over lags, per
    source), the Green's function itself, and the learned attention's own
    `context` output - is untouched.

    WHY: airlapse3 through airlapse7 kept `transported` fully decoupled
    from the learned attention on purpose, so it stays a physically
    interpretable quantity you can sanity-check in isolation (peak arrival
    time, misalignment penalty, etc. - all the checks this project has
    run on it). That's a real benefit, but it also means `transported`
    has no way to benefit from whatever the content attention has learned
    that pure physics can't see - e.g. two sources at the same distance
    and wind alignment whose recent hidden-state trajectories look very
    different might not be equally relevant, and only `context`'s learned
    Q/K matching would notice that. This variant asks the model to decide,
    per source, how much that content signal should also influence how
    much of `transported`'s weight that source gets - via ONE new
    learnable scalar, not a redesign of either path.

    COUPLED STAGE 2 (stage 1 is airlapse7's, unmodified):
        content_logit_ij = logsumexp_k(score_ijk)
            (the learned attention's OWN pre-softmax score - content
            match plus its four w_dist/w_wind/w_terrain/w_lag physics
            bonuses - marginalized over k the same way log_S below
            marginalizes the transport Green's function: total supporting
            evidence for source j across the lags considered, in log
            space. `score` already carries -inf for out-of-range sources
            via neighbor_mask, same mask family transport_mask is built
            from, so this stays consistent with transport's own masking.)
        combined_logit_ij = log_S_ij + w_context_couple * content_logit_ij
        alpha_ij = softmax_j(combined_logit_ij)      [re-masked to
                   transport_mask AFTER combining - see forward() for why
                   this can't just rely on -inf propagating on its own]
        transported_i = sum_j alpha_ij * transport_j->i
    `w_context_couple` starts at exactly 0.0, so AirLapse8 is IDENTICAL to
    AirLapse7 at initialization - training is what decides whether, and
    how much (the parameter is unconstrained, so it can also go negative:
    "sources the content attention likes LESS get more transport weight"
    is a hypothesis this lets the data reject, not one this rules out by
    construction), content relevance should influence transport weight.
    Inspecting w_context_couple after training tells you directly whether
    coupling the two paths helped: near 0 means no, the independent
    design was fine; a large positive value means physical plausibility
    alone was missing something content similarity captures.

    Numerical note - two separate -inf hazards, both real, both hit during
    development (not hypothetical):
    (1) content_logit_ij is -inf wherever neighbor_mask excludes a source
        (out of range). w_context_couple * content_logit_ij is therefore
        w * (-inf) there - if w is ever negative this becomes +inf, and
        log_S(-inf) + (+inf) = NaN once added. Fixed by masked_fill'ing
        combined_logit back to -inf at every transport_mask-excluded
        position AFTER the addition, overriding whatever the raw
        arithmetic produced, rather than trusting -inf to propagate
        correctly on its own.
    (2) Independent of (1), and easy to miss because it looks harmless:
        w_context_couple * content_logit_ij is 0 * (-inf) = NaN at those
        same masked positions whenever w_context_couple is exactly (or
        passes through) 0.0 - its own default initial value. The forward
        value there gets overwritten by masked_fill regardless, but the
        NaN still poisons w_context_couple's gradient: autograd's product
        rule contributes content_logit_ij (-inf) times the incoming
        gradient at that position, and even though masked_fill's backward
        correctly zeroes that incoming gradient, 0 * (-inf) = NaN, which
        then sums into w_context_couple's total gradient across every
        position and corrupts the parameter after the very first
        optimizer step - visible as train_loss going to NaN in epoch 0 of
        a real training run, not in any of this file's synthetic unit
        tests (whose station layouts never happened to combine the
        default w=0.0 with a genuinely out-of-range pair in the same
        test). Fixed with `torch.nan_to_num(content_logit, neginf=0.0)`
        before the multiply - those positions are getting overwritten by
        the masked_fill immediately after anyway, so replacing their
        value doesn't change the result, only removes the 0*inf hazard;
        nan_to_num's own backward stops gradient flow at replaced
        positions, so no spurious gradient reappears there either.
    """
    def __init__(self, hidden_dim, station_coords, station_elevation,
                 attn_dim=32, dist_threshold_km=300.0, sigma_d=200.0,
                 sigma_h=1200.0, sigma_tau_init_h=3.0, speed_floor_kmh=0.5,
                 diffusivity_km2_per_hour_init=50.0, t_eps_hours=0.25):
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

        # the one new learnable parameter airlapse7 added: eddy diffusivity
        # (km^2/hour) for the explicit transport estimate's Green's
        # function, below.
        self.log_D = nn.Parameter(torch.tensor(_inv_softplus(diffusivity_km2_per_hour_init)))

        # THE new parameter this variant adds: how much the learned
        # attention's per-source relevance should influence stage 2's
        # intersource softmax. Starts at 0 - see class docstring.
        self.w_context_couple = nn.Parameter(torch.tensor(0.0))

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
        # and the transport estimate's v_radial further down (that one
        # needs the sign kept; this clamp is only for the score bonus).
        wind_align = torch.clamp(cos_theta, min=0.0)

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

        # --- explicit physical transport estimate: 1D advection-diffusion
        # Green's function, in LOG SPACE throughout (see class docstring) -
        # log_green is computed analytically (never round-tripped through
        # exp then log), so it stays a normal finite number even for pairs
        # whose actual Green's-function value would underflow to exact 0.0
        # in float32. ---
        v_radial = speed.unsqueeze(2) * cos_theta                         # [B,K,N_j,N_i], toward-receiver speed (signed)
        t_eff = k_hours_b + self.t_eps_hours                              # [1,K,1,1], keeps t=0 well-defined
        D = F.softplus(self.log_D) + 1e-6
        d = self.dist_km.view(1, 1, N, N)
        log_green = (
            -0.5 * torch.log(4.0 * math.pi * D * t_eff)
            - ((d - v_radial * t_eff) ** 2) / (4.0 * D * t_eff)
        )                                                                  # [B, K, N_receiver, N_source]
        log_green = log_green.masked_fill(~self.transport_mask.view(1, 1, N, N), float('-inf'))

        # stage 1 (inner, over k, per source) - unchanged from airlapse7.
        pi = torch.nan_to_num(torch.softmax(log_green, dim=1), nan=0.0)   # [B, K, N_receiver, N_source]
        transport_from_source = torch.einsum('bkij,bkj->bij', pi, pm25_lag)  # [B, N_receiver, N_source]

        # stage 2 (intersource, over j) - THE COUPLING: blend physical
        # plausibility (log_S, airlapse7's score) with the learned
        # attention's own per-source relevance (content_logit), weighted
        # by the one new learnable w_context_couple (starts at 0 - see
        # class docstring for why the re-mask after combining is needed).
        log_S = torch.logsumexp(log_green, dim=1)                          # [B, N_receiver, N_source]
        content_logit = torch.logsumexp(score.reshape(B, N, K, N), dim=2)  # [B, N_receiver, N_source]
        # content_logit is -inf wherever neighbor_mask excluded a source (out
        # of range). Multiplying that by w_context_couple - even though the
        # result gets overwritten by the masked_fill below regardless of its
        # value - produces NaN, not just at those positions but poisoning
        # w_context_couple's ENTIRE gradient: d/dw(w * content_logit) at a
        # masked position is content_logit itself (-inf), and even though
        # the masked_fill's backward correctly zeroes the incoming gradient
        # there, 0 * (-inf) = NaN in IEEE-754 - and that NaN then sums into
        # w_context_couple's total gradient across ALL positions, corrupting
        # the parameter (and, from the next forward pass on, the whole
        # network) even when w_context_couple is sitting at its harmless-
        # looking default of 0.0. (Caught by an actual training run going
        # to NaN in epoch 0 - the synthetic unit tests happened to use
        # station layouts where this exact combination, default w=0.0 AND a
        # genuinely out-of-range pair, was never simultaneously exercised.)
        # nan_to_num replaces -inf with a finite placeholder before the
        # multiply; nan_to_num's own backward stops gradient flow at
        # replaced positions, so this is exactly the "doesn't matter, gets
        # overwritten anyway" value the masked_fill below expects, with none
        # of the 0*inf risk.
        content_logit_safe = torch.nan_to_num(content_logit, neginf=0.0)
        combined_logit = log_S + self.w_context_couple * content_logit_safe
        combined_logit = combined_logit.masked_fill(~self.transport_mask, float('-inf'))
        alpha = torch.nan_to_num(torch.softmax(combined_logit, dim=-1), nan=0.0)  # sums to 1 over valid j
        transported = (alpha * transport_from_source).sum(dim=-1)          # [B, N_receiver]

        return context, transported


class AirLapse8(nn.Module):
    """
    AirLapse7 (model/airlapse7.py), with stage 2's intersource softmax
    score changed from pure physical plausibility to a learnable blend of
    physical plausibility AND the learned attention's own per-source
    relevance - see MultiLagPhysicsAwareSpatialAttention's docstring above
    for the coupling formula and why it's safe to add on top of two
    previously-independent paths. Constructor, both spatial_mix_mode
    forks' control flow, VAE bottleneck, and decoder are otherwise
    identical to AirLapse7's - the wrapper class itself needed no changes
    beyond its name and this docstring, since the attention module's
    forward()/return signature is unchanged, and w_context_couple starting
    at 0 means this model is identical to AirLapse7 before any training
    happens.

    spatial_mix_mode:
      - 'bottleneck': `nn.GRU`'s full per-step output (not just h_n) is
        kept; the last `max_lag` steps become the attention's keys/values
        (and the transport estimate's source lags) in one pass at the end
        of encoding.
      - 'per_step': the encoder is unrolled with a GRUCell; at every step,
        MultiLagPhysicsAwareSpatialAttention is called with K=1 (that
        step's own state, wind, and PM2.5 only) - stage 1 degenerates the
        same way it does in airlapse7, and stage 2's coupled softmax over
        j still applies exactly as with any other K.

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
                 diffusivity_km2_per_hour_init=50.0, t_eps_hours=0.25):
        super(AirLapse8, self).__init__()
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
            diffusivity_km2_per_hour_init=diffusivity_km2_per_hour_init,
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
        only) - the transport estimate's stage 1 (over-k) softmax degrades
        to a single-lag evaluation, see MultiLagPhysicsAwareSpatialAttention's
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
                f"AirLapse8 was built with city_num={self.city_num}, but got "
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
