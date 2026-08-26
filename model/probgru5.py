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
    Spatial mixing built around one idea: the physics is a PRIOR, not the
    truth. probgru2/3/4's PhysicsAwareSpatialAttention family scores each
    station pair as

        score_ij = content(h_i,h_j) + w_dist*dist_bias_ij + w_wind*wind_align_ij
                   + w_terrain*terrain_bias_ij

    i.e. three independent, additively-blended exponential-decay bumps.
    That's a caricature of dispersion, not a model of it: real advection-
    diffusion says a station's influence travels preferentially DOWNWIND
    (elongated along the wind vector) and spreads slowly CROSSWIND, with
    the spread growing with travel time - none of which an isotropic
    exp(-d/sigma) term plus a separate 0/1-ish wind gate can express.

    This module instead builds a genuine (if still simplified) anisotropic
    dispersion kernel from advection-diffusion first principles, then
    treats it as a PRIOR that a small neural network is allowed to
    multiplicatively correct - never to override outright:

        K_final_ij = K_physics_ij * g_theta(z_i, z_j, C_i, C_j, weather, time)
        score_ij   = content(h_i,h_j) + w_adp*A_adp_ij
                     + w_phys*log(K_physics_ij) + w_gate*log(g_theta_ij)
        weights    = softmax_j(score_ij | neighbor_mask)

    Adding log(K_physics) and log(g_theta) to the attention LOGIT is the
    same thing as MULTIPLYING K_physics by g_theta in probability space
    (softmax(a + log b) = softmax(a) reweighted by b, up to renormalization)
    - so this is literally "physics prior x learned correction", expressed
    in a form that still plugs into an ordinary softmax attention block.
    g_theta's last layer is zero-initialized, so at the start of training
    g_theta == 1 everywhere and the model IS the physics kernel; whatever
    g_theta learns to deviate from 1 is, by construction, exactly what pure
    advection-diffusion could not explain.

    --- K_physics: anisotropic advection-diffusion, not isotropic decay ---

    For source station i with wind blowing toward compass bearing
    `travel_bearing_i` at speed `wind_speed_kmh_i`, and receiver j at
    distance d_ij and bearing_ij = bearing(i -> j), decompose the
    i->j offset into components along and across the wind vector
    (this is exactly the role lon/lat play here - they are not just
    input features, they parameterize the physical geometry: d_ij and
    bearing_ij come straight from haversine/great-circle formulas, and
    x_ij/y_ij below are that geometry projected onto the wind axis):

        theta_ij  = travel_bearing_i - bearing_ij
        x_ij      = d_ij * cos(theta_ij)     downwind distance (>0 ahead of i)
        y_ij      = d_ij * sin(theta_ij)     crosswind offset from the plume axis

    Advection time and diffusive spread follow the textbook relations the
    proposal names directly:

        tau_ij      = max(x_ij, 0) / wind_speed_i          (time to reach j)
        sigma_ij^2  = 2 * D_i * tau_ij + sigma_min^2        (Fickian spread)

    and the along-axis kernel is a 1-D Gaussian plume slice, normalized so
    a fast, narrow plume and a slow, wide one integrate comparably:

        K_plume_ij = downwind_gate(theta_ij) / (sigma_ij*sqrt(2*pi))
                     * exp(-y_ij^2 / (2*sigma_ij^2))

    `downwind_gate` is a SMOOTH sigmoid(k*cos(theta_ij)) rather than the
    previous versions' hard clamp(cos,min=0) - differentiable everywhere,
    including exactly crosswind, and it is what stops K_plume from firing
    on pairs directly behind the source (see note on y_ij==0 degeneracy
    below).

    D_i (diffusivity, km^2/h) is not a fixed constant: it is a learnable
    base value, OPTIONALLY modulated by boundary-layer height (a proxy for
    atmospheric stability/vertical mixing that's already in this repo's
    feature set) when `pbl_channel_idx` is supplied - deeper daytime
    boundary layers mix and dilute faster, shallow nocturnal ones trap
    pollution, and this lets the model learn that association instead of
    assuming one fixed spread rate at every hour.

    A pure advection-diffusion plume also breaks down exactly when it
    matters most for PM2.5: near-calm, stagnant conditions, where haze
    tends to BUILD UP precisely because there is no dominant wind to
    define "downwind". So K_physics blends the directional plume with an
    isotropic diffusion kernel (same D_i, a fixed nominal time step
    instead of a wind-derived tau) as wind speed drops, via a learnable
    sigmoid transition:

        K_iso_ij   = exp(-d_ij^2 / (2*sigma_iso_ij^2)) / (sigma_iso_ij*sqrt(2*pi))
        blend_i    = sigmoid((wind_speed_i - calm_center) / calm_scale)
        K_disp_ij  = blend_i * K_plume_ij + (1 - blend_i) * K_iso_ij

    Finally, topographic blocking is a genuinely separate physical
    mechanism from horizontal dispersion (a ridge attenuates transport
    independently of how far or how downwind two stations are), so it is
    combined MULTIPLICATIVELY as an independent attenuation factor - the
    same way optical depth / plume-depletion factors compose - rather than
    added as an unrelated bias term the way v1-v4 did:

        K_physics_ij = K_disp_ij * exp(-|z_i - z_j| / sigma_h)

    NOTE on the y_ij==0 degeneracy: at theta_ij = 180 deg (j directly
    behind i, upwind), x_ij < 0 so tau clamps to 0 and sigma shrinks to its
    floor - but y_ij = d_ij*sin(180 deg) = 0 too, i.e. j sits exactly on
    the now-collapsed plume axis. Without `downwind_gate` this would
    spuriously spike K_plume for stations directly upwind. The smooth gate
    (not the shrinking sigma) is what suppresses that case, and does so
    everywhere, not just at the singular angle.

    --- A_adp: a persistent learned relation, on top of content attention ---

    `content_score` (Q.K^T on the current hidden states) is already a
    dynamic, learned notion of "which stations relate to which" - that's
    the "adaptive spatial relation" half of the proposal's central
    equation. This module adds one more, complementary piece: a STATIC
    Graph-WaveNet-style adaptive adjacency built from two small learned
    per-station embeddings,

        A_adp_ij = relu(E1_i . E2_j)

    which can absorb persistent structure content attention has no
    incentive to find on its own (e.g. a fixed pair of stations that
    co-move for reasons outside this feature set - shared industrial
    sources, a shared valley not fully captured by the elevation/terrain
    prior, etc.), independent of D. It carries its own weight `w_adp` and
    can shrink to ~0 if it doesn't help.

    --- g_theta: the neural correction, and what it is allowed to see ---

    Per the proposal's g_theta(C, weather, time, location), the context fed
    to the correction MLP for pair (i, j) is:

        [z_i, z_j]                    location  (station elevation, z-scored)
        [C_i, C_j]                    current pollution state at i and j
                                       (the same normalized PM2.5 scale the
                                       rest of the model trains on)
        [sin(hour), cos(hour)]        time      (cyclical hour-of-day)
        [pbl_i]                       weather   (boundary-layer height, if
                                       available; zero-filled otherwise)

    g_theta's raw output is clamped to +-gate_clamp before being used as a
    log-correction (so the multiplicative correction is bounded to roughly
    [exp(-gate_clamp), exp(gate_clamp)] - it can meaningfully rescale the
    physics prior but cannot blow it up or erase it in one step), and its
    last linear layer starts at zero so training begins from "trust the
    physics" and only deviates where the data justifies it.

    NOTE ON INDEXING CONVENTION (carried over from v1-v4): softmax is over
    dim=-1 (j), and context = weights @ v pulls CONTRIBUTOR j's value into
    RECEIVER i's output; the wind/advection terms use wind measured at the
    ROW station i against bearing(i -> j) - i.e. "how much of i's plume
    reaches j", which is the physically correct direction for this
    convention (i emits, j receives), unlike v1-v4 where this alignment
    was noted but not re-derived from scratch here.
    """
    def __init__(self, hidden_dim, station_coords, station_elevation,
                 attn_dim=32, dist_threshold_km=300.0, sigma_h=1200.0,
                 adaptive_dim=16, gate_hidden_dim=16, gate_clamp=2.0,
                 diffusivity_init_km2h=50.0, sigma_min_km=15.0,
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

        # --- content-based attention (dynamic "learned spatial relation") ---
        self.q_proj = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.scale = attn_dim ** -0.5

        # --- static Graph-WaveNet-style adaptive adjacency (persistent
        # "learned spatial relation" A_adp, independent of the current
        # hidden state) ---
        N = station_coords.shape[0]
        self.node_emb1 = nn.Parameter(torch.randn(N, adaptive_dim) * 0.1)
        self.node_emb2 = nn.Parameter(torch.randn(N, adaptive_dim) * 0.1)
        self.w_adp = nn.Parameter(torch.tensor(1.0))

        # --- overall interpretable strengths for the two log-bias terms,
        # same "inspect after training" philosophy as v1-v4's w_dist etc. ---
        self.w_phys = nn.Parameter(torch.tensor(1.0))
        self.w_gate = nn.Parameter(torch.tensor(1.0))

        # --- learnable physics parameters (all initialized at sensible
        # physical defaults, then free to move) ---
        self.log_D_base = nn.Parameter(torch.tensor(_inv_softplus(diffusivity_init_km2h)))
        self.w_pbl = nn.Parameter(torch.tensor(0.0))               # 0 -> no PBL effect until learned
        self.calm_center = nn.Parameter(torch.tensor(calm_speed_kmh_init))
        self.calm_scale_raw = nn.Parameter(torch.tensor(_inv_softplus(calm_scale_kmh_init)))
        self.gate_steepness_raw = nn.Parameter(torch.tensor(_inv_softplus(downwind_gate_steepness_init)))

        # --- neural correction g_theta: context -> bounded log-multiplier,
        # zero-initialized so g_theta starts as the identity (pure physics) ---
        ctx_dim = 7  # [z_i, z_j, C_i, C_j, sin(hour), cos(hour), pbl_i]
        self.gate_mlp = nn.Sequential(
            nn.Linear(ctx_dim, gate_hidden_dim),
            nn.Tanh(),
            nn.Linear(gate_hidden_dim, 1),
        )
        nn.init.zeros_(self.gate_mlp[-1].weight)
        nn.init.zeros_(self.gate_mlp[-1].bias)

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
        travel_bearing  : [B, N] radians - compass bearing (i.e. SAME
                          clockwise-from-north convention as `bearing`)
                          each node's air is currently heading TOWARD
        wind_speed_kmh  : [B, N] wind speed in km/h, >= 0
        pm25_now        : [B, N] current/most-recent PM2.5 reading, same
                          normalized scale the model trains on
        hour_sin, hour_cos : [B] cyclical encoding of hour-of-day (shared
                          across nodes - time is global, not per-station)
        pbl_z           : [B, N] normalized boundary-layer height, or None
                          if that feature isn't in this run's metero_use
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

        # ---- persistent learned adjacency ----
        adp_logits = F.relu(torch.mm(self.node_emb1, self.node_emb2.t()))             # [N, N]

        # ---- neural correction g_theta(z_i, z_j, C_i, C_j, weather, time) ----
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
        gate_raw = self.gate_mlp(ctx).squeeze(-1)                               # [B, N, N]
        gate_log = torch.clamp(gate_raw, -self.gate_clamp, self.gate_clamp)

        score = (content_score
                 + self.w_adp * adp_logits.unsqueeze(0)
                 + self.w_phys * torch.log(k_phys)
                 + self.w_gate * gate_log)

        score = score.masked_fill(~self.neighbor_mask.unsqueeze(0), float('-inf'))
        weights = torch.nan_to_num(torch.softmax(score, dim=-1), nan=0.0)
        return torch.bmm(weights, v)


class ProbGRUModel5(nn.Module):
    """
    ProbGRUModel4's VAE-GRU skeleton (encoder -> latent bottleneck ->
    autoregressive decoder) with its spatial mixing block replaced by
    PhysicsPriorSpatialAttention: an anisotropic advection-diffusion prior
    that a small neural network is allowed to multiplicatively correct,
    instead of several unrelated additive bias terms standing in as "the
    physics". See PhysicsPriorSpatialAttention's docstring for the full
    derivation; this class docstring covers what changed at the model
    level relative to v2/v3/v4.

    TWO CORRECTNESS FIXES relative to v2/v3/v4, both load-bearing for a
    kernel that actually depends on real wind speed/direction:

      1. v2/v3/v4 read wind_channel_idx's u/v columns straight out of
         `feature` and fed them to atan2/sqrt - but dataset.py z-scores
         EVERY feature channel (see HazeData._norm), so those u/v values
         are standardized, not raw m/s. atan2 of two independently
         z-scored components does not, in general, recover the true wind
         angle (u and v don't share a mean/std), and sqrt(u_z^2+v_z^2) is
         not a wind speed in any physical unit - so v2/v3's advection
         term and v1-v4's wind-alignment gate were built from a
         quantity that only resembles wind direction/speed by
         coincidence. This class instead de-normalizes the dataset's own
         precomputed speed (km/h) and direction channels - which
         dataset.py appends as the LAST TWO feature columns, see
         `_process_feature` - via `feature_mean`/`feature_std`, the exact
         same z-score-inversion PM25_GNN.py's GraphGNN already relies on
         for the same two columns. This is an exact roundtrip (z-scoring
         is a lossless affine map), not an approximation.

      2. v1-v4's wind_dir = atan2(v, u) uses the mathematical convention
         (counter-clockwise from east); `bearing_deg` uses the standard
         compass/forward-azimuth convention (clockwise from north). Their
         difference was noted in v2/v3's docstrings as a known but
         unfixed simplification. Recovering wind FROM the dataset's own
         meteorological-convention direction column instead (also
         clockwise from north) and flipping it 180 degrees to get the
         direction air is heading TOWARD makes travel_bearing directly
         comparable to `bearing_deg` with no residual convention
         mismatch.

    Because of fix #1, `wind_channel_idx` is no longer needed and this
    class does not take it. `wind_mean`/`wind_std` are still accepted for
    call-signature parity with the trainer's `get_model()` dispatch, but
    are otherwise redundant with (a slice of) `feature_mean`/`feature_std`
    and are not read directly.

    spatial_mix_mode (unchanged in spirit from v2-v4):
      - 'bottleneck': fused nn.GRU encodes each station independently;
        ONE physics-prior-plus-correction mixing pass at the end, using
        the last observed history step's wind/time/pollution context.
      - 'per_step': encoder unrolled with a GRUCell; the full K_physics *
        g_theta mixing is recomputed at EVERY history step and the mixed
        state feeds back into the recurrence. This is the mode that
        actually realizes the proposal's Sum_{i,k} A_adp * K_physics^{t,k}
        * C_i(t-k) formulation: rather than materializing an explicit
        second sum over lag k (an O(T) stack of N x N kernels), the GRU
        recurrence integrates it implicitly - each step folds that step's
        own physics-weighted spatial mixing into a state that then
        persists (subject to the GRU's own gating) into every later step,
        which is the recurrent analogue of summing dispersion-weighted
        history without the memory cost of ever materializing the k axis.

    decode_spatial_mix (default False, parity with v3): also re-run the
    same physics-prior-plus-correction mixing at every decode step, using
    that step's own known future wind/time. Because g_theta needs a
    "current pollution" context and the true future PM2.5 is exactly what
    is being predicted, decode-time mixing uses the model's OWN
    just-computed prediction for that step as the C_i/C_j context (never
    the label) - i.e. output_head is evaluated once to get that proxy,
    the state is then spatially remixed, and output_head is evaluated
    again on the remixed state for the actual prediction. No leakage,
    same normalized PM2.5 scale used throughout training.
    """
    def __init__(self, hist_len, pred_len, in_dim, city_num, batch_size, device,
                 edge_index, edge_attr, wind_mean, wind_std,
                 station_coords, station_elevation,
                 feature_mean, feature_std, pbl_channel_idx=None,
                 hidden_dim=64, latent_dim=16, attn_dim=32, adaptive_dim=16,
                 gate_hidden_dim=16, num_layers=1, dropout=0.1, logvar_clamp=10.0,
                 spatial_mix_mode='bottleneck', decode_spatial_mix=False,
                 dist_threshold_km=300.0, sigma_h=1200.0,
                 diffusivity_init_km2h=50.0, sigma_min_km=15.0,
                 calm_speed_kmh_init=5.0, calm_scale_kmh_init=3.0,
                 gate_clamp=2.0, dt_hours=3.0):
        super(ProbGRUModel5, self).__init__()
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
        self.decode_spatial_mix = decode_spatial_mix
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
        # This holds regardless of which variables metero_use lists, since
        # these four are appended AFTER the metero_use subset is selected.
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
            adaptive_dim=adaptive_dim,
            gate_hidden_dim=gate_hidden_dim,
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
        compass-bearing "travel toward" angle - see class docstring, fixes
        #1/#2. Returns (travel_bearing [B,N] rad, wind_speed_kmh [B,N],
        pbl_z [B,N] or None)."""
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
        hour channel. Hour is identical across all N stations at a given
        timestep (dataset.py repeats the same global clock per node), so
        node 0's value is the representative per-batch scalar. Returns
        (sin, cos), each [B]."""
        hour_z = feature_slice[:, t, :, self.hour_idx]
        hour_raw = hour_z * self.feature_std[self.hour_idx] + self.feature_mean[self.hour_idx]
        hour0 = hour_raw[:, 0]
        frac = hour0 / 24.0 * (2.0 * math.pi)
        return torch.sin(frac), torch.cos(frac)

    def _encode_bottleneck(self, x, feature_hist, pm25_hist, B, N):
        """Fused GRU, single physics-prior-plus-correction mixing pass at
        the end, using the last observed history step's context."""
        _, h_n = self.encoder(x)
        h_T = h_n[-1]                                       # [B*N, hidden_dim]
        h_grid = h_T.reshape(B, N, self.hidden_dim)

        travel_bearing, speed_kmh, pbl_z = self._wind_and_pbl_at(feature_hist, -1, B, N)
        hour_sin, hour_cos = self._hour_encoding_at(feature_hist, -1)
        pm_now = pm25_hist[:, -1, :, 0]

        context = self.spatial_attn(h_grid, travel_bearing, speed_kmh, pm_now, hour_sin, hour_cos, pbl_z)
        h_mixed = self.mix_gate(torch.cat([h_grid, context], dim=-1))
        return h_mixed.reshape(B * N, self.hidden_dim)

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

            context_t = self.spatial_attn(h_grid_t, travel_bearing_t, speed_kmh_t, pm_now_t, hour_sin_t, hour_cos_t, pbl_t)
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
                f"ProbGRUModel5 was built with city_num={self.city_num}, but got "
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
            pred_t = self.output_head(h_dec)                 # [B*N, 1]

            if self.decode_spatial_mix:
                # PM25_GNN parity (as in v3): re-mix spatially at every
                # decode step using that step's own known future wind. The
                # gate's "current pollution" context can't use the (future,
                # unknown) label, so it uses this step's own just-computed
                # prediction instead - then output_head is re-applied to
                # the spatially-remixed state for the actual prediction.
                h_dec_grid = h_dec.reshape(B, N, self.hidden_dim)
                travel_bearing_t, speed_kmh_t, pbl_t = self._wind_and_pbl_at(feature_future, t, B, N)
                hour_sin_t, hour_cos_t = self._hour_encoding_at(feature_future, t)
                pm_now_t = pred_t.reshape(B, N)

                context_t = self.spatial_attn(h_dec_grid, travel_bearing_t, speed_kmh_t, pm_now_t, hour_sin_t, hour_cos_t, pbl_t)
                h_dec_grid = self.mix_gate(torch.cat([h_dec_grid, context_t], dim=-1))
                h_dec = h_dec_grid.reshape(B * N, self.hidden_dim)
                pred_t = self.output_head(h_dec)

            preds.append(pred_t)

        out = torch.stack(preds, dim=1)
        pm25_pred = out.reshape(B, N, self.pred_len, 1).permute(0, 2, 1, 3)
        return pm25_pred
