"""AirLapse V2 - a 2-D, context-adaptive-diffusivity upgrade of AirLapse's
explicit physical transport estimate.

Motivation (from real results/4-new numbers on dataset 4): AirLapse
finishes near the bottom of the 16-model leaderboard there (RMSE ~3.14-
3.18, vs. AirPhyNet 2.83 and WPMixer/TimeXer ~2.90 - see the comparison
this file's commit message/PR description quotes). The learned content+
physics attention (MultiLagPhysicsAwareSpatialAttention in model/
airlapse.py) is left untouched here; this file only replaces its
EXPLICIT physical transport term - the one derived directly from the
advection-diffusion equation's Green's function - with two upgrades:

1. 2-D anisotropic transport, not 1-D radial. V1's Green's function
   collapses source-receiver geometry to a single radial distance d_ij
   and a signed radial wind speed v_ij = speed*cos(theta) - it cannot
   distinguish "receiver is 50km downwind" from "receiver is 50km
   directly crosswind" of the same source; both get scored by the same
   1-D formula along the source-receiver line. Real plume transport
   isn't radial: pollution advects strongly along the wind direction and
   spreads much more slowly perpendicular to it. V2 decomposes the
   source->receiver displacement into a downwind component x (along the
   source's wind bearing) and a crosswind component y (perpendicular to
   it), and uses the standard 2-D anisotropic point-source advection-
   diffusion Green's function (see AdaptivePhysicsTransport2D's
   docstring for the exact formula) - a receiver directly crosswind of a
   fast, close source now correctly gets little estimated inflow, which
   V1's radial form could not express.

2. Context-adaptive diffusivity, not one global learned scalar. V1 has
   exactly one learnable diffusivity D for the entire graph, every
   station, every hour - a genuinely poor model of a real atmosphere,
   where turbulent mixing varies hugely with wind speed (more mechanical
   turbulence in stronger wind), time of day (a stable nocturnal boundary
   layer traps pollution near the surface; daytime convective mixing
   disperses it), and terrain (elevation is a workable proxy for terrain
   roughness/local orography where no direct roughness data exists).
   V2 replaces the single scalar with a small MLP that maps each
   SOURCE's own local conditions at each lag - wind speed, hour-of-day
   (as sin/cos), elevation, and its own recent PM2.5 level (a stagnation
   proxy: elevated PM2.5 often co-occurs with exactly the calm/inversion
   conditions that suppress mixing) - to a per-(source, lag) along-wind
   and cross-wind diffusivity pair. Diffusivity is modeled as a property
   of the source's local atmosphere at that moment (standard K-theory
   framing in atmospheric science - eddy diffusivity as a function of
   local stability/shear/roughness), not a joint property of each
   source-receiver pair, which keeps this cheap (per (B, K, N), not per
   (B, K, N, N)) and physically well-motivated.

Both upgrades are initialized to reduce, at the start of training, to
something close to V1's own defaults (isotropic ~50 km^2/hour) rather
than an arbitrary random starting point - see AdaptivePhysicsTransport2D.
__init__'s bias initialization.

3. Joint spatio-temporal encoding ("Option B", added after the two
   upgrades above), via TemporalGraphEncoder. Independent of everything
   above - it changes how each station's hidden states are FORMED before
   AdaptivePhysicsTransport2D ever sees them, not the transport Green's
   function itself. The base per-node GRU encoder has zero cross-station
   interaction while encoding (x folds N into the batch dimension), so
   "two neighbors rising together" or "a front moving through several
   stations" can only ever be inferred after the fact, by comparing two
   independently-finished summaries in the attention step. This module
   lets stations exchange information at every timestep WHILE their
   temporal trajectories are still forming instead - see its own
   docstring for the exact mechanism and why it's a single shared graph-
   attention layer rather than a full STAEformer/TCN-style stack.

4. Physics-fused attention logits ("A_final = A_learned + w_phys *
   log(K+eps)", added after the joint spatio-temporal encoder above).
   Until this change, the Green's function computed for `transported`
   was invisible to the softmax that produces `context` - the two were
   fully independent branches, combined only afterward by mix_gate. Now
   the same Green's-function value is also injected as one more additive
   term in the attention score, in log-space, alongside the four
   hand-crafted bonuses - see AdaptivePhysicsTransport2D's docstring
   ("PHYSICS-FUSED ATTENTION LOGITS") for the exact formula and
   reasoning. `transported` itself is unchanged; only how `context` is
   computed differs.

The GRU encoder/decoder, the mix_gate that combines context/transported/
h_grid, and (aside from the addition above) the learned content+physics
attention's four independent bonus terms are otherwise unchanged from
AirLapse (model/airlapse.py). One further difference from V1, unrelated
to the transport Green's function above: the latent bottleneck here is a
single deterministic Linear projection, not a VAE (no mu/logvar heads,
reparameterization sampling, or KL loss) - an Optuna search over V2's
own hyperparameters (tune_airlapse_v2.py) showed negligible improvement
over untuned defaults, not enough signal to justify the added training
noise and loss term the stochastic version costs for no measured
benefit here (see AirLapseV2's class docstring for the fuller reasoning).
This file is copied from AirLapse rather than subclassing it, so
V1 and V2 can be tuned/compared independently without one's changes
silently affecting the other.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint as _odeint
from torchdiffeq import odeint_adjoint as _odeint_adjoint

from model.airlapse import haversine_km, bearing_deg, _inv_softplus


class AdaptivePhysicsTransport2D(nn.Module):
    """
    Spatial mixing step for AirLapse V2. Structurally identical to
    MultiLagPhysicsAwareSpatialAttention (model/airlapse.py) - same
    learned content+physics attention (`context`), same four independent
    bonus terms (distance/wind/terrain/lag), same additive-across-sources
    aggregation for the explicit transport estimate - EXCEPT for how the
    Green's function itself is computed (2-D anisotropic, not 1-D radial;
    see below) AND one further addition not present in V1 at all:
    PHYSICS-FUSED ATTENTION LOGITS.

    Physics-fused attention logits. V1 (and V2 before this change) treats
    the learned attention (`context`) and the explicit physical estimate
    (`transported`) as fully independent branches, combined only
    afterward by the caller's mix_gate - the Green's function is computed
    but never actually seen by the softmax that produces `context`. Here,
    the same Green's-function value `green` (see formula below) is also
    injected directly into the attention logits before the softmax:
        A_final_ijk = A_learned_ijk + w_dist*dist_ij + w_wind*align_ijk
                      + w_terrain*terrain_ij + w_lag*lagbias_ijk
                      + w_phys * log(G_ijk + eps)
    w_phys is a free learnable scalar (init 1.0, same as the other four
    bonus weights) - it can shrink toward 0 during training if this
    doesn't earn its keep, exactly like w_dist/w_wind/w_terrain/w_lag.
    Unlike those four (each a hand-crafted proxy for one physical factor
    in isolation - static distance decay, instantaneous wind alignment,
    static terrain difference, a Gaussian arrival-time match), G_ijk is
    the actual advection-diffusion Green's function already being
    computed for `transported`: it jointly encodes distance, signed
    downwind advection, crosswind spread, AND per-source/per-lag
    context-adaptive diffusivity in one physically-grounded number. Log-
    space (not raw G_ijk added multiplicatively via softmax) matches the
    log-linear pooling used elsewhere in this repo's attention variants -
    a source/lag combination the physics says is implausible (wrong
    arrival time, wrong side of the plume) gets an extra strongly
    negative logit contribution, additively stacking with (or overriding)
    whatever the four hand-crafted bonuses alone would have said, while a
    plausible one adds close to 0. `transported` itself is UNCHANGED by
    this - it's still a separate output computed from the same `green`
    tensor, reaching mix_gate side by side with `context` exactly as
    before; only the attention logits that produce `context` are new.

    2-D anisotropic Green's function. For source j, receiver i, lag k
    (elapsed time t_k, floored at t_eps_hours):
        theta_ijk = angle between source j's wind bearing and the
                    straight-line bearing from j to i (same angle V1
                    uses for its radial term - nothing new to compute)
        x_ijk = d_ij * cos(theta_ijk)   (downwind offset, signed)
        y_ijk = d_ij * sin(theta_ijk)   (crosswind offset, signed - only
                                          ever squared below, so the sign
                                          convention doesn't matter)
        D_along_jk, D_cross_jk = softplus(MLP(context_jk))   (see below)
        G_ijk = 1 / (4*pi*sqrt(D_along_jk * D_cross_jk) * t_k)
                * exp[-(x_ijk - v_jk*t_k)^2 / (4*D_along_jk*t_k)
                      -(y_ijk)^2          / (4*D_cross_jk*t_k)]
    This is the direct 2-D generalization of V1's 1-D form (which is
    recovered exactly if y_ijk is dropped and D_along=D_cross): two
    independent Gaussians along orthogonal axes, one advecting with the
    wind, one purely diffusing sideways. Whatever aggregation V1 did with
    its `green` values (per-lag normalization to pi_ijk, "reach" as the
    per-source peak weight, additive summation across sources - see V1's
    docstring for why additive, not softmax-normalized) is unchanged
    here; only the Green's function itself differs.

    Context-adaptive diffusivity. context_jk = [wind_speed_kmh_jk / 10,
    sin(hour_jk), cos(hour_jk), elevation_j / 1000 (km), pm25_jk (already
    z-scored by HazeData._norm(), so already ~O(1))] - a small 2-layer
    MLP maps this to (D_along_jk, D_cross_jk) via softplus. The final
    layer's bias is initialized to diffusivity_along_init/
    diffusivity_cross_init (in the softplus-inverse sense - see
    _inv_softplus) so training starts from a sensible physical scale
    (V1's own tuned default was ~50 km^2/hour; cross-wind spread
    defaults lower, 20 km^2/hour, since lateral turbulent mixing is
    generally weaker than the combination of longitudinal turbulence and
    wind shear along the flow) rather than an arbitrary random value,
    while the MLP's weights (small, near-zero init from nn.Linear's
    default) let training adjust away from that starting point per
    actual local conditions once gradients make it worthwhile.
    """
    def __init__(self, hidden_dim, station_coords, station_elevation,
                 attn_dim=32, dist_threshold_km=300.0, sigma_d=200.0,
                 sigma_h=1200.0, sigma_tau_init_h=3.0, speed_floor_kmh=0.5,
                 diff_hidden_dim=16, diffusivity_along_init=50.0,
                 diffusivity_cross_init=20.0, t_eps_hours=0.25):
        super().__init__()

        self.q_proj = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.scale = attn_dim ** -0.5
        self.speed_floor_kmh = speed_floor_kmh
        self.t_eps_hours = t_eps_hours

        # learnable but not softmax-tied: each can shrink to ~0 if it
        # isn't earning its keep, independent of the others. Identical to
        # V1 - this is the learned content+physics attention path, not
        # the explicit transport estimate this file changes.
        self.w_dist = nn.Parameter(torch.tensor(1.0))
        self.w_wind = nn.Parameter(torch.tensor(1.0))
        self.w_terrain = nn.Parameter(torch.tensor(1.0))
        self.w_lag = nn.Parameter(torch.tensor(1.0))
        # weight on log(green + eps) fused directly into the attention
        # logits - see class docstring "PHYSICS-FUSED ATTENTION LOGITS".
        # Same free-scalar, can-shrink-to-0 treatment as the four bonuses
        # above; kept as a separate parameter (not folded into one of
        # them) so it can be inspected/ablated independently.
        self.w_phys = nn.Parameter(torch.tensor(1.0))
        self.log_sigma_tau = nn.Parameter(torch.tensor(_inv_softplus(sigma_tau_init_h)))

        # context-adaptive anisotropic diffusivity MLP - see class
        # docstring. 5 inputs: wind_speed/10, sin(hour), cos(hour),
        # elevation/1000, local pm25 (already z-scored).
        self.diff_mlp = nn.Sequential(
            nn.Linear(5, diff_hidden_dim),
            nn.Tanh(),
            nn.Linear(diff_hidden_dim, 2),
        )
        with torch.no_grad():
            self.diff_mlp[-1].weight.mul_(0.01)  # start near-constant, not near-random
            self.diff_mlp[-1].bias[0] = _inv_softplus(diffusivity_along_init)
            self.diff_mlp[-1].bias[1] = _inv_softplus(diffusivity_cross_init)

        dist = haversine_km(station_coords)                          # [N, N]
        neighbor_mask = dist <= dist_threshold_km
        self.register_buffer('neighbor_mask', neighbor_mask)
        self.register_buffer('dist_km', dist)

        dist_bias = torch.exp(-dist / sigma_d)
        self.register_buffer('dist_bias', dist_bias)                  # [N, N], static

        elev_diff = (station_elevation.unsqueeze(1) - station_elevation.unsqueeze(0)).abs()
        terrain_bias = torch.exp(-elev_diff / sigma_h)
        self.register_buffer('terrain_bias', terrain_bias)            # [N, N], static
        # station_elevation itself (not just the pairwise diff above) is
        # part of the diffusivity MLP's per-source context - see forward().
        self.register_buffer('station_elevation', station_elevation)  # [N]

        # bearing FROM source i TO receiver j, laid out as [j, i] (receiver
        # row, source col) to match this module's [B, N_j, K*N_i] score
        # layout - see forward()
        self.register_buffer('bearing_j_i', bearing_deg(station_coords).t())

        # neighbor_mask with the diagonal zeroed, precomputed once - used
        # only by the explicit transport estimate below.
        not_self = ~torch.eye(dist.shape[0], dtype=torch.bool)
        self.register_buffer('transport_mask', neighbor_mask & not_self)

    def forward(self, h_last, h_lag, travel_bearing_lag, wind_speed_kmh_lag, k_hours, pm25_lag, hour_lag):
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
        hour_lag             : [B, K, N] hour-of-day (0-23, real units) at
                                each source/lag - diffusivity context only
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
        cos_theta = torch.cos(angle_diff)
        sin_theta = torch.sin(angle_diff)                                 # NEW vs V1: crosswind component
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

        # --- 2-D anisotropic advection-diffusion Green's function with
        # context-adaptive diffusivity (see class docstring) - computed
        # here, BEFORE the learned-attention score, because it now feeds
        # BOTH the learned attention logits (log-fusion, right below) AND
        # the explicit `transported` estimate further down. ---
        d = self.dist_km.view(1, 1, N, N)
        x_downwind = d * cos_theta                                        # [B,K,N_j,N_i]
        y_crosswind = d * sin_theta                                       # [B,K,N_j,N_i]
        t_eff = k_hours_b + self.t_eps_hours                              # [1,K,1,1]

        elev_km = (self.station_elevation / 1000.0).view(1, 1, N).expand(B, K, N)
        hour_rad = hour_lag * (2.0 * math.pi / 24.0)
        context_jk = torch.stack([
            speed / 10.0, torch.sin(hour_rad), torch.cos(hour_rad), elev_km, pm25_lag,
        ], dim=-1)                                                        # [B,K,N_j,5]
        # Clamped to a physically-plausible range, not just floored above
        # zero: unlike V1's single global scalar D (a slowly-moving
        # learned parameter, stable by construction), D here is the
        # output of a freshly-initialized MLP re-evaluated every forward
        # pass - a few early, still-noisy gradient steps can otherwise
        # push it toward ~0, and both the prefactor 1/sqrt(D_along*D_cross)
        # and the exponent's 1/D terms blow up as D -> 0 (observed in
        # practice: an un-clamped smoke run's train_loss spiked to ~670
        # while val/test loss stayed normal - one exploding batch, not a
        # systemic break, but bad for training stability). The floor
        # (2 km^2/hour) is well below the physical default (50/20); the
        # ceiling (2000) just prevents the opposite failure mode
        # (transport washing out to ~0 everywhere) from wandering
        # unboundedly - both are generous, not tight, physical bounds.
        diff_raw = self.diff_mlp(context_jk)                              # [B,K,N_j,2]
        D_along = F.softplus(diff_raw[..., 0]).clamp(min=2.0, max=2000.0).unsqueeze(-1)  # [B,K,N_j,1]
        D_cross = F.softplus(diff_raw[..., 1]).clamp(min=2.0, max=2000.0).unsqueeze(-1)  # [B,K,N_j,1]

        v_signed = speed.unsqueeze(-1)                                    # [B,K,N_j,1]
        along_term = (x_downwind - v_signed * t_eff) ** 2 / (4.0 * D_along * t_eff)
        cross_term = y_crosswind ** 2 / (4.0 * D_cross * t_eff)
        green = (
            1.0 / (4.0 * math.pi * torch.sqrt(D_along * D_cross) * t_eff)
            * torch.exp(-(along_term + cross_term))
        )                                                                  # [B, K, N_receiver, N_source]

        # --- PHYSICS-FUSED ATTENTION LOGITS: A_final = A_learned +
        # w_phys * log(K_ij(tau) + eps) - see class docstring. green is
        # bounded in [0, ~0.16] given the D/t_eff clamps/floors above, so
        # log(green + 1e-6) is itself bounded below at ln(1e-6) ~= -13.8
        # with no extra clamp needed: masked-out (non-neighbor/self)
        # entries are still hard-excluded by mask_tiled below regardless
        # of this value, and for real neighbors a strongly negative value
        # here is the intended effect (physically-implausible source/lag
        # combinations - wrong arrival time, wrong side of the plume -
        # should be suppressed in the softmax, not just the four separate
        # hand-crafted bonuses that predate this kernel). w_phys is free
        # to shrink toward 0 during training if this term doesn't earn
        # its keep, exactly like w_dist/w_wind/w_terrain/w_lag.
        log_green_flat = _flatten(torch.log(green + 1e-6))

        score = (content_score
                 + self.w_dist * dist_tiled.unsqueeze(0)
                 + self.w_wind * wind_align_flat
                 + self.w_terrain * terrain_tiled.unsqueeze(0)
                 + self.w_lag * lag_bias_flat
                 + self.w_phys * log_green_flat)

        score = score.masked_fill(~mask_tiled.unsqueeze(0), float('-inf'))
        weights = torch.nan_to_num(torch.softmax(score, dim=-1), nan=0.0)
        context = torch.bmm(weights, v)

        # --- explicit physical transport estimate (unchanged): the same
        # `green` kernel above, aggregated additively across sources per
        # the superposition argument in the file/class docstring. This
        # stays a SEPARATE output from `context` - the log-fusion above
        # lets `green` also shape the learned attention, but the two
        # remain independent quantities reaching mix_gate side by side. ---
        w_transport = self.transport_mask.to(green.dtype).view(1, 1, N, N) * green

        # pm25_lag is clamped here, not upstream: it's z-scored PM2.5, and
        # this dataset's tail is extreme (one reading in the training
        # window sits at 82 std devs above the mean - real wildfire-smoke
        # spikes, not a data bug - see prepare_sensor_dataset.py). pi sums
        # to 1 over k, so transport_from_source alone can't exceed this
        # clamp regardless of how peaked pi gets - but transported then
        # sums ADDITIVELY across every neighbor (deliberately, per this
        # class's superposition argument - see file/class docstring), so
        # a smoke event hitting several neighboring sensors at once could
        # still add their clamped contributions into a large-but-bounded
        # value, rather than the unbounded one this fixes: an un-clamped
        # 50-epoch/5-repeat run on dataset 4 diverged on several repeats
        # (train_loss mean 44, std 72 - a handful of exploding runs, not
        # steady learning) traced to exactly this path. V1
        # (MultiLagPhysicsAwareSpatialAttention in model/airlapse.py) has
        # the same unclamped einsum and is likely exposed to a milder
        # version of this on this dataset too - not fixed here since V1's
        # own results are comparatively stable and this file's scope is
        # V2, but worth the same fix if V1 is revisited.
        pm25_lag_clamped = pm25_lag.clamp(-10.0, 10.0)
        pi = w_transport / (w_transport.sum(dim=1, keepdim=True) + 1e-8)  # sums to 1 over k
        transport_from_source = torch.einsum('bkij,bkj->bij', pi, pm25_lag_clamped)  # [B, N_receiver, N_source]
        reach = w_transport.max(dim=1).values                             # [B, N_receiver, N_source]
        transported = (reach * transport_from_source).sum(dim=-1)         # [B, N_receiver]
        # Hard backstop on the aggregated, additive-across-neighbors
        # result too - belt-and-suspenders given how much is riding on
        # this one number reaching mix_gate sanely.
        transported = transported.clamp(-20.0, 20.0)

        return context, transported


class ContinuousTransportODEFunc(nn.Module):
    """
    Continuous-time counterpart to AdaptivePhysicsTransport2D's Green's-
    function transport estimate. AdaptivePhysicsTransport2D (and
    everything upstream of it in this file) computes physics exactly
    ONCE, from historical lags, and hands the result to mix_gate as a
    fixed feature - the 24-step decoder that follows is then completely
    physics-blind (see AirLapseV2's class docstring "CONTINUOUS ODE-
    DRIVEN TRANSPORT" for the full motivation: both AirPhyNet and
    AirDualODE, model/airphynet.py and model/airdualode.py, outperform
    AirLapseV2 on dataset 4, and the one thing they share structurally is
    that diffusion/advection ARE the dz/dt of a neural ODE, integrated
    continuously across the WHOLE forecast horizon via torchdiffeq, not
    a one-shot estimate). This module is that same structural pattern,
    reusing AdaptivePhysicsTransport2D's own wind-aligned anisotropic
    kernel (downwind/crosswind decomposition, context-adaptive
    diffusivity) as the exchange conductance, rather than AirPhyNet/
    AirDualODE's simpler isotropic-ChebConv-diffusion + signed-flow-
    difference-advection - so this ODE branch is finally giving AirLapse's
    OWN, already-validated physics a chance to act continuously, not a
    weaker reimplementation of what those two papers do.

    Deliberately self-contained (own diff_mlp, not shared with
    AdaptivePhysicsTransport2D): refactoring that module's Green's-
    function computation into a function shared by both would risk
    regressing the already-tuned, currently-best-performing encoder-side
    attention for the sake of this new, unproven branch - the same
    reasoning AirDualODE's own file gives for reimplementing
    _GatedFusion locally instead of importing it from model/airphynet.py.

    AUTONOMOUS ODE - same simplification AirPhyNet/AirDualODE both make
    (see their set_wind()/set_context() methods): torchdiffeq's
    adaptive solver calls this function at arbitrary, solver-chosen t
    (not necessarily aligned to real hourly steps), so there's no clean
    way to look up "the actual future wind at exactly this t" the way a
    fixed-step decoder can. The exchange conductance g_ij is instead
    frozen once, from the LAST OBSERVED (historical) step, via
    set_context() before integration starts, and reused for the entire
    horizon - exactly what both reference architectures already do.

    DYNAMICS. For station i, at any solver-chosen t:
        dz_i/dt = sum_{j in neighbors(i)} g_ij * (z_j - z_i)   [exchange]
                  + beta_i * z_i                                [reaction, optional]
    g_ij (frozen - see set_context() for the exact Green's-function
    formula, t_eps_hours used as the "instantaneous" reference elapsed
    time rather than a lag-dependent one) plays the role a weighted graph
    Laplacian's off-diagonal entry would: mass flows from j to i in
    proportion to their CONCENTRATION DIFFERENCE (z_j - z_i) and how
    strongly the frozen wind connects j to i. This (z_j - z_i) form is
    what makes it a genuine RATE OF CHANGE, unlike AdaptivePhysicsTransport2D's
    `transported` (an estimate of absolute incoming mass at one instant
    from historical values, not a derivative) - and it is (ignoring the
    optional reaction term) approximately mass-conserving, unlike
    `transported`'s deliberate, superposition-based non-conservation (see
    that class's own docstring for why THAT design choice is correct for
    THAT quantity - a different quantity, a different choice here).
    beta_i (off by default - see reaction_term) is a learnable per-
    station scalar for local net production/removal that pure transport
    can't represent - AirDualODE's "open system" reaction term, borrowed
    directly (see model/airdualode.py's own docstring point 1 for the
    physical motivation): a station's own emissions or precipitation
    washout aren't transport from anywhere.

    Numerical stability: |dz/dt| is soft-bounded via tanh, same trick
    AirPhyNet's own _ODEFunc uses (see model/airphynet.py) and for the
    same reason - an unbounded neural-ODE vector field lets the state (or
    the adjoint method's required step size) blow up over a long-enough
    integration horizon, which torchdiffeq surfaces as an opaque
    "underflow in dt" solver failure rather than a normal training
    signal. This is a freshly-initialized, never-before-trained dynamical
    system (unlike AdaptivePhysicsTransport2D's diff_mlp, which starts
    biased toward V1's tuned defaults), so this safeguard matters more
    here, not less.
    """
    def __init__(self, latent_dim, station_coords, station_elevation,
                 dist_threshold_km=300.0, diff_hidden_dim=16,
                 diffusivity_along_init=50.0, diffusivity_cross_init=20.0,
                 t_eps_hours=0.25, max_deriv=10.0, reaction_term=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.t_eps_hours = t_eps_hours
        self.max_deriv = max_deriv
        self.reaction_term = reaction_term

        dist = haversine_km(station_coords)
        neighbor_mask = dist <= dist_threshold_km
        not_self = ~torch.eye(dist.shape[0], dtype=torch.bool)
        self.register_buffer('transport_mask', neighbor_mask & not_self)
        self.register_buffer('dist_km', dist)
        # bearing FROM source TO receiver, laid out [receiver, source] -
        # same convention as AdaptivePhysicsTransport2D.bearing_j_i.
        self.register_buffer('bearing_j_i', bearing_deg(station_coords).t())
        self.register_buffer('station_elevation', station_elevation)

        # Own diff_mlp (5 inputs: wind_speed/10, sin(hour), cos(hour),
        # elevation/1000, pm25 - identical context shape to
        # AdaptivePhysicsTransport2D's, but a SEPARATE set of weights -
        # see class docstring for why this isn't shared) - same bias-
        # initialization convention (start near V1's tuned isotropic
        # scale, let training adjust away from it).
        self.diff_mlp = nn.Sequential(
            nn.Linear(5, diff_hidden_dim), nn.Tanh(), nn.Linear(diff_hidden_dim, 2),
        )
        with torch.no_grad():
            self.diff_mlp[-1].weight.mul_(0.01)
            self.diff_mlp[-1].bias[0] = _inv_softplus(diffusivity_along_init)
            self.diff_mlp[-1].bias[1] = _inv_softplus(diffusivity_cross_init)

        if reaction_term:
            self.beta = nn.Parameter(torch.zeros(station_coords.shape[0], latent_dim))

        self.g_ij = None  # [B, N_receiver, N_source] - set per-forward via set_context()

    def set_context(self, travel_bearing, wind_speed_kmh, hour, pm25):
        """
        Freezes the exchange conductance g_ij for the upcoming
        integration - see class docstring "AUTONOMOUS ODE". Each arg is
        [B, N] - the LAST OBSERVED step's per-SOURCE-station context
        (same de-normalized quantities AdaptivePhysicsTransport2D reads
        per-lag, here evaluated once).
        """
        B, N = travel_bearing.shape
        angle_diff = travel_bearing.unsqueeze(1) - self.bearing_j_i.view(1, N, N)  # [B, N_recv, N_src]
        cos_theta = torch.cos(angle_diff)
        sin_theta = torch.sin(angle_diff)
        d = self.dist_km.view(1, N, N)
        x_downwind = d * cos_theta                                        # [B, N_recv, N_src]
        y_crosswind = d * sin_theta

        speed = wind_speed_kmh.clamp(min=0.0)                              # [B, N_src]
        elev_km = (self.station_elevation / 1000.0).view(1, N).expand(B, N)
        hour_rad = hour * (2.0 * math.pi / 24.0)
        context_j = torch.stack([
            speed / 10.0, torch.sin(hour_rad), torch.cos(hour_rad), elev_km, pm25,
        ], dim=-1)                                                        # [B, N_src, 5]
        diff_raw = self.diff_mlp(context_j)                                # [B, N_src, 2]
        # Same generous clamp range/reasoning as AdaptivePhysicsTransport2D's
        # own diff_mlp output - see that class's forward() comment.
        D_along = F.softplus(diff_raw[..., 0]).clamp(min=2.0, max=2000.0).view(B, 1, N)
        D_cross = F.softplus(diff_raw[..., 1]).clamp(min=2.0, max=2000.0).view(B, 1, N)

        t_eff = self.t_eps_hours  # frozen "instantaneous" reference elapsed time - see class docstring
        v_signed = speed.view(B, 1, N)                                     # [B, 1, N_src]
        along_term = (x_downwind - v_signed * t_eff) ** 2 / (4.0 * D_along * t_eff)
        cross_term = y_crosswind ** 2 / (4.0 * D_cross * t_eff)
        green = (
            1.0 / (4.0 * math.pi * torch.sqrt(D_along * D_cross) * t_eff)
            * torch.exp(-(along_term + cross_term))
        )                                                                  # [B, N_recv, N_src]

        self.g_ij = self.transport_mask.to(green.dtype).view(1, N, N) * green

    def forward(self, t, z):
        """z: [B, N*latent_dim] (flattened, as torchdiffeq requires a
        plain tensor state) -> dz/dt, same shape."""
        if self.g_ij is None:
            raise RuntimeError(
                "ContinuousTransportODEFunc.forward() called before set_context() - "
                "the exchange conductance must be frozen once before integration."
            )
        B = z.shape[0]
        N = self.g_ij.shape[-1]
        x = z.reshape(B, N, self.latent_dim)

        incoming = torch.bmm(self.g_ij, x)                                 # [B,N,latent_dim] = sum_j g_ij * x_j
        row_sum_g = self.g_ij.sum(dim=-1, keepdim=True)                    # [B,N,1] = sum_j g_ij, per receiver
        grad = incoming - row_sum_g * x                                    # sum_j g_ij * (x_j - x_i)

        if self.reaction_term:
            grad = grad + self.beta.unsqueeze(0) * x

        # Soft-bound - see class docstring "Numerical stability".
        grad = self.max_deriv * torch.tanh(grad / self.max_deriv)
        return grad.reshape(B, N * self.latent_dim)


class TemporalGraphEncoder(nn.Module):
    """
    "Option B" joint spatio-temporal encoder (in the terminology of the
    conversation that requested this): AirLapseV2's base 'bottleneck'
    encoder is a single per-node nn.GRU - x is folded to (B*N, T, C)
    before the GRU ever sees it, so no station's hidden state is
    influenced by any other station's history WHILE it's being formed.
    Cross-station information only enters once, in the single attention
    pass AdaptivePhysicsTransport2D does afterward, over each station's
    already-independently-compressed trajectory. That means the model
    structurally cannot represent "two neighbors rising together over
    the past few hours" or "a front visibly moving through several
    upstream stations" as a feature of the encoding itself - only as
    something inferred post-hoc from comparing two finished summaries.

    This module closes that gap with the lightest change that actually
    lets stations interact WHILE their temporal trajectories are still
    forming, not after: run the per-node GRU as before, then apply ONE
    graph-attention layer PER TIMESTEP (weight-shared across all T - the
    physical neighbor structure doesn't change from hour to hour, so
    there's no reason to learn a separate mixing function per timestep)
    with a residual + LayerNorm, before any lag is sliced out for the
    downstream physics-guided attention. Every h_lag entry
    AdaptivePhysicsTransport2D reads is now itself already
    spatially-aware, so "two neighbors rising together" is representable
    directly in the embeddings it attends over, not just inferable from
    comparing two independently-formed ones afterward.

    Deliberately NOT a full STAEformer/TCN-style stack (the other options
    raised in that conversation): with ~240 training windows on dataset
    4, minimizing added parameters matters more than expressiveness -
    one shared graph-attention layer adds roughly as many parameters as
    AdaptivePhysicsTransport2D's own q/k/v projections, not a second
    full model. num_layers lets you stack more if a dataset has enough
    data to support it; num_layers=1 is deliberately the default.

    Reuses the SAME distance-decay neighbor definition (dist_threshold_km,
    sigma_d, station_coords) as AdaptivePhysicsTransport2D, computed
    independently here rather than shared, so "neighbor" means the same
    physical thing everywhere in this model without coupling the two
    modules' internals together.

    This only ever touches the 'bottleneck' encoder path. 'per_step'
    already interleaves spatial mixing into every encoder step (calling
    AdaptivePhysicsTransport2D itself once per timestep, feeding the
    result back into the next GRUCell step) - stacking this module on
    top of that would double up on the same idea via two different
    mechanisms at once, and 'per_step' has empirically been the less
    stable, worse-scoring mode on dataset 4 throughout this repo's
    history, so it isn't a promising place to add more spatial mixing
    right now regardless.
    """
    def __init__(self, hidden_dim, station_coords, attn_dim=32,
                 dist_threshold_km=300.0, sigma_d=200.0, num_layers=1, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        dist = haversine_km(station_coords)
        neighbor_mask = dist <= dist_threshold_km
        self.register_buffer('neighbor_mask', neighbor_mask)
        dist_bias = torch.exp(-dist / sigma_d)
        self.register_buffer('dist_bias', dist_bias)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'q_proj': nn.Linear(hidden_dim, attn_dim, bias=False),
                'k_proj': nn.Linear(hidden_dim, attn_dim, bias=False),
                'v_proj': nn.Linear(hidden_dim, hidden_dim, bias=False),
                'norm': nn.LayerNorm(hidden_dim),
            })
            for _ in range(num_layers)
        ])
        self.scale = attn_dim ** -0.5
        self.w_dist = nn.Parameter(torch.ones(num_layers))
        self.dropout = nn.Dropout(dropout)

    def forward(self, output, B, N):
        """output: [B*N, T, hidden_dim] (a 'bottleneck'-mode GRU's raw
        output) -> same shape, now spatially mixed at every timestep."""
        T = output.shape[1]
        h = output.reshape(B, N, T, self.hidden_dim)
        for i, layer in enumerate(self.layers):
            # fold (B, T) into one batch axis - the same neighbor
            # structure/weights apply at every timestep, so this is one
            # batched attention call over all timesteps at once, not a
            # loop over T.
            h_bt = h.permute(0, 2, 1, 3).reshape(B * T, N, self.hidden_dim)
            q, k, v = layer['q_proj'](h_bt), layer['k_proj'](h_bt), layer['v_proj'](h_bt)
            score = torch.bmm(q, k.transpose(1, 2)) * self.scale + self.w_dist[i] * self.dist_bias.unsqueeze(0)
            score = score.masked_fill(~self.neighbor_mask.unsqueeze(0), float('-inf'))
            weights = torch.nan_to_num(torch.softmax(score, dim=-1), nan=0.0)
            mixed = self.dropout(torch.bmm(weights, v))
            mixed = mixed.reshape(B, T, N, self.hidden_dim).permute(0, 2, 1, 3)
            h = layer['norm'](h + mixed)
        return h.reshape(B * N, T, self.hidden_dim)


class AirLapseV2(nn.Module):
    """
    AirLapse V2 - identical outer architecture to AirLapse (model/
    airlapse.py: per-station-independent GRU forecaster) with its spatial
    mixing step swapped for AdaptivePhysicsTransport2D above (2-D
    anisotropic + context-adaptive diffusivity transport estimate, same
    learned content+physics attention otherwise). See this file's module
    docstring for the motivation and exact formulation.

    Unlike AirLapse V1, this is NOT a VAE: V1's stochastic latent (mu/
    logvar heads, reparameterization sampling, KL-divergence loss) was
    dropped after an Optuna search over V2's own hyperparameters (see
    tune_airlapse_v2.py) showed negligible improvement over the untuned
    defaults - not enough signal to justify keeping VAE-style uncertainty
    machinery that adds training noise and an extra loss term for no
    measured benefit on this dataset. `z` here is a single deterministic
    Linear projection of h_mixed (latent_head) - still a real bottleneck
    (latent_dim still meaningfully controls its width, still tunable),
    just without the sampling/regularization on top of it. If evidence
    later shows the stochasticity would help on a different dataset,
    re-adding it here is a small, self-contained change (see git history
    for the removed mu_head/logvar_head/reparameterization/KL code).

    spatial_mix_mode / max_lag: same meaning as AirLapse V1 - see its
    class docstring.

    st_encoder_layers: if > 0 (default 1) and spatial_mix_mode ==
    'bottleneck', a TemporalGraphEncoder (see above - "Option B", joint
    spatio-temporal encoding) is inserted right after the GRU, before any
    lag is sliced out for the physics-guided attention. 0 disables it,
    recovering the original per-node-independent encoding exactly (for
    ablating whether it actually helps). No effect in 'per_step' mode -
    see TemporalGraphEncoder's docstring for why.

    CONTINUOUS ODE-DRIVEN TRANSPORT (ode_transport, default False).
    Motivation: on dataset 4's own leaderboard (results/4-new), the two
    OTHER physics-guided benchmarks here - AirPhyNet (model/airphynet.py,
    RMSE ~2.83, best overall) and AirDualODE (model/airdualode.py, RMSE
    ~2.94, essentially tied with AirLapseV2's own best result) - both
    outperform or match AirLapseV2, and share one structural trait
    neither AirLapseV2 nor AirLapse V1 has: their diffusion/advection
    physics IS the dz/dt of a neural ODE, integrated continuously across
    the ENTIRE forecast horizon via torchdiffeq - not a value computed
    once from history and handed to a decoder as a static feature (which
    is exactly what AdaptivePhysicsTransport2D's `transported` is: the
    24-step decoder that follows it is completely physics-blind for the
    whole forecast). This is a different, more principled route to the
    same insight behind an earlier, reverted "decoder-side recurring
    transport" attempt (discrete, detached, autoregressive re-evaluation
    every step) that did not help in practice - continuous ODE
    integration gives smooth, adjoint-based gradients across the whole
    trajectory with no per-step discretization/teacher-forcing artifact,
    which the discrete version could not.

    When True: ContinuousTransportODEFunc (above) - reusing
    AdaptivePhysicsTransport2D's own wind-aligned anisotropic kernel as
    its exchange conductance, not AirPhyNet/AirDualODE's simpler
    isotropic-diffusion-plus-signed-advection - integrates a SEPARATE,
    physics-only latent trajectory across the full pred_len horizon,
    initialized from the last observed [pm25, wind] (ode_init_proj) -
    deliberately NOT from h_mixed/z, so this branch carries a genuinely
    distinct physical signal rather than reusing the same learned
    representation the GRUCell decoder already has. At every decode
    step, this trajectory is projected to hidden_dim (ode_phy_to_hidden)
    and blended with decoder_cell's own output via a learned sigmoid gate
    (ode_fusion_gate) - AFTER decoder_cell, not fed back into its
    recurrence - so the GRUCell's own hidden-state path is completely
    undisturbed and cannot reintroduce the instability the earlier
    discrete attempt risked; only the final representation reaching
    output_head changes. Off by default - this is a new, unproven branch
    (fresh ODE dynamics, no prior tuning) that should be A/B'd against
    the current best before trusting it.
    """
    def __init__(self, hist_len, pred_len, in_dim, city_num, batch_size, device,
                 edge_index, edge_attr, wind_mean, wind_std,
                 station_coords, station_elevation,
                 feature_mean, feature_std,
                 hidden_dim=64, latent_dim=16, attn_dim=32, num_layers=1,
                 dropout=0.1,
                 spatial_mix_mode='bottleneck', max_lag=6,
                 dist_threshold_km=300.0, sigma_d=200.0, sigma_h=1200.0,
                 sigma_tau_init_h=3.0, dt_hours=3.0,
                 diff_hidden_dim=16, diffusivity_along_init=50.0,
                 diffusivity_cross_init=20.0, t_eps_hours=0.25,
                 st_encoder_layers=1,
                 ode_transport=False, ode_latent_dim=8, ode_diff_hidden_dim=16,
                 ode_diffusivity_along_init=50.0, ode_diffusivity_cross_init=20.0,
                 ode_t_eps_hours=0.25, ode_max_deriv=10.0, ode_reaction_term=False,
                 ode_method='dopri5', ode_rtol=1e-3, ode_atol=1e-4, ode_adjoint=True):
        super(AirLapseV2, self).__init__()
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
        self.spatial_mix_mode = spatial_mix_mode
        self.dt_hours = dt_hours
        self.max_lag = min(max_lag, hist_len)
        # Opt-in (default False - see class docstring "CONTINUOUS
        # ODE-DRIVEN TRANSPORT"): a separate, continuously-integrated
        # physics branch fused into the decoder's output at every step.
        self.ode_transport = ode_transport
        self.ode_latent_dim = ode_latent_dim
        self.ode_method = ode_method
        self.ode_rtol = ode_rtol
        self.ode_atol = ode_atol
        self.ode_adjoint = ode_adjoint

        # kept for signature parity across benchmark models
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.wind_mean = wind_mean
        self.wind_std = wind_std

        self.feature_dim = in_dim - 1
        # fixed tail layout dataset.py._process_feature always produces:
        # [...metero_use channels..., hour, weekday, speed_kmh, direc_deg]
        self.hour_idx = self.feature_dim - 4
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
            # "Option B" - see TemporalGraphEncoder's docstring. None (not
            # just an empty ModuleList) when disabled, so _encode_bottleneck
            # can check `is not None` and skip it entirely - not just run
            # a zero-layer pass that reduces to a no-op anyway, but this
            # is clearer about "disabled" being a real, intentional state.
            self.st_encoder = TemporalGraphEncoder(
                hidden_dim=hidden_dim, station_coords=station_coords,
                attn_dim=attn_dim, dist_threshold_km=dist_threshold_km,
                sigma_d=sigma_d, num_layers=st_encoder_layers, dropout=dropout,
            ) if st_encoder_layers > 0 else None
        else:  # 'per_step'
            self.encoder_cell = nn.GRUCell(input_size=in_dim, hidden_size=hidden_dim)
            self.step_dropout = nn.Dropout(dropout)

        self.spatial_attn = AdaptivePhysicsTransport2D(
            hidden_dim=hidden_dim,
            station_coords=station_coords,
            station_elevation=station_elevation,
            attn_dim=attn_dim,
            dist_threshold_km=dist_threshold_km,
            sigma_d=sigma_d,
            sigma_h=sigma_h,
            sigma_tau_init_h=sigma_tau_init_h,
            diff_hidden_dim=diff_hidden_dim,
            diffusivity_along_init=diffusivity_along_init,
            diffusivity_cross_init=diffusivity_cross_init,
            t_eps_hours=t_eps_hours,
        )
        # +1: the explicit scalar transport estimate, concatenated alongside
        # h_grid and the learned attention context.
        self.mix_gate = nn.Linear(hidden_dim * 2 + 1, hidden_dim)

        # Deterministic bottleneck - see class docstring for why this
        # isn't a VAE (mu/logvar heads + reparameterization) like V1.
        self.latent_head = nn.Linear(hidden_dim, latent_dim)

        self.decoder_init = nn.Linear(hidden_dim + latent_dim, hidden_dim)
        self.decoder_cell = nn.GRUCell(
            input_size=self.feature_dim + latent_dim, hidden_size=hidden_dim,
        )
        self.output_head = nn.Linear(hidden_dim, 1)

        # --- CONTINUOUS ODE-DRIVEN TRANSPORT (see class docstring) ---
        if ode_transport:
            self.ode_func = ContinuousTransportODEFunc(
                latent_dim=ode_latent_dim,
                station_coords=station_coords,
                station_elevation=station_elevation,
                dist_threshold_km=dist_threshold_km,
                diff_hidden_dim=ode_diff_hidden_dim,
                diffusivity_along_init=ode_diffusivity_along_init,
                diffusivity_cross_init=ode_diffusivity_cross_init,
                t_eps_hours=ode_t_eps_hours,
                max_deriv=ode_max_deriv,
                reaction_term=ode_reaction_term,
            )
            # z0 built from [last_pm25, last_speed_kmh/10, sin(bearing),
            # cos(bearing)] - deliberately NOT h_mixed, so this branch
            # carries a genuinely distinct physics-only signal (see class
            # docstring). 4 inputs, matching AirPhyNet/AirDualODE's own
            # convention of a small [pm25, wind] initial-state projection.
            self.ode_init_proj = nn.Linear(4, ode_latent_dim)
            self.ode_phy_to_hidden = nn.Linear(ode_latent_dim, hidden_dim)
            self.ode_fusion_gate = nn.Linear(hidden_dim * 2, hidden_dim)

    def _wind_at_idxs(self, feature_hist, idxs, B, N):
        """De-normalize the dataset's precomputed speed(km/h)/direction(deg,
        meteorological 'from' convention) and hour-of-day tail/context
        channels at each of the given history indices, and convert wind to
        compass-bearing 'travel toward' angles. idxs: 1-D LongTensor or
        list of history indices.
        Returns (travel_bearing [B,len(idxs),N] rad, wind_speed_kmh
        [B,len(idxs),N], hour_of_day [B,len(idxs),N] in 0-23 real units)."""
        speed_z = feature_hist[:, idxs, :, self.speed_idx]
        direc_z = feature_hist[:, idxs, :, self.direc_idx]
        hour_z = feature_hist[:, idxs, :, self.hour_idx]
        speed_kmh = speed_z * self.feature_std[self.speed_idx] + self.feature_mean[self.speed_idx]
        direc_from_deg = direc_z * self.feature_std[self.direc_idx] + self.feature_mean[self.direc_idx]
        hour = hour_z * self.feature_std[self.hour_idx] + self.feature_mean[self.hour_idx]
        travel_bearing = torch.deg2rad(direc_from_deg + 180.0)
        speed_kmh = speed_kmh.clamp(min=0.0)
        return travel_bearing, speed_kmh, hour

    def _encode_bottleneck(self, x, feature_hist, B, N):
        """Fused GRU (optionally followed by TemporalGraphEncoder's
        "Option B" spatial mixing - see its docstring); the last max_lag
        steps' hidden states become the spatial attention's keys/values
        (and pm25_lag its transport estimate's source values), each
        scored with its own wind."""
        output, h_n = self.encoder(x)                          # output: [B*N, T, hidden_dim]
        if self.st_encoder is not None:
            output = self.st_encoder(output, B, N)              # same shape, now spatially-aware
            h_T = output[:, -1, :]                              # h_n[-1] would be pre-mixing - use the
                                                                 # mixed "now" state instead, consistent
                                                                 # with h_lag below.
        else:
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

        travel_bearing_lag, speed_kmh_lag, hour_lag = self._wind_at_idxs(feature_hist, lag_idxs, B, N)
        k_hours = torch.tensor(
            [(T - 1 - idx) * self.dt_hours for idx in lag_idxs],
            dtype=x.dtype, device=x.device,
        )

        context, transported = self.spatial_attn(
            h_grid, h_lag, travel_bearing_lag, speed_kmh_lag, k_hours, pm25_lag, hour_lag)
        h_mixed = self.mix_gate(torch.cat([h_grid, context, transported.unsqueeze(-1)], dim=-1))
        return h_mixed.reshape(B * N, self.hidden_dim)

    def _encode_per_step(self, x, feature_hist, B, N):
        """Unroll the encoder manually; at every step, K=1 (current step
        only) - the transport estimate degrades to a single-lag evaluation
        of the Green's function, see AdaptivePhysicsTransport2D's docstring."""
        T = x.shape[1]
        h = torch.zeros(B * N, self.hidden_dim, device=x.device, dtype=x.dtype)

        for t in range(T):
            x_t = x[:, t, :]
            h = self.encoder_cell(x_t, h)
            h = self.step_dropout(h)

            h_grid_t = h.reshape(B, N, self.hidden_dim)
            h_lag_t = h_grid_t.unsqueeze(1)                     # [B, 1, N, hidden_dim]
            pm25_lag_t = x[:, t, 0].reshape(B, N).unsqueeze(1)  # [B, 1, N]
            travel_bearing_t, speed_kmh_t, hour_t = self._wind_at_idxs(feature_hist, [t], B, N)
            k_hours = torch.zeros(1, dtype=x.dtype, device=x.device)

            context_t, transported_t = self.spatial_attn(
                h_grid_t, h_lag_t, travel_bearing_t, speed_kmh_t, k_hours, pm25_lag_t, hour_t)
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
                f"AirLapseV2 was built with city_num={self.city_num}, but got "
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

        z = self.latent_head(h_mixed)

        if self.ode_transport:
            # --- CONTINUOUS ODE-DRIVEN TRANSPORT (see class docstring) ---
            # Frozen context = the last OBSERVED (historical) step - same
            # autonomous-ODE simplification AirPhyNet/AirDualODE both make
            # (see ContinuousTransportODEFunc's docstring).
            last_bearing, last_speed, last_hour = self._wind_at_idxs(
                feature_hist, [self.hist_len - 1], B, N)               # each [B,1,N]
            last_bearing = last_bearing.squeeze(1)                      # [B,N]
            last_speed = last_speed.squeeze(1)
            last_hour = last_hour.squeeze(1)
            last_pm25 = pm25_hist[:, -1, :, 0]                          # [B,N]
            self.ode_func.set_context(last_bearing, last_speed, last_hour, last_pm25)

            # z0: physics-only initial state, deliberately NOT derived
            # from h_mixed - see class docstring for why.
            ode_init_input = torch.stack(
                [last_pm25, last_speed / 10.0, torch.sin(last_bearing), torch.cos(last_bearing)],
                dim=-1)                                                 # [B,N,4]
            z0_phy = self.ode_init_proj(ode_init_input).reshape(B, N * self.ode_latent_dim)

            time_steps = torch.linspace(
                0.0, self.pred_len * self.dt_hours, self.pred_len + 1,
                dtype=x.dtype, device=x.device)
            odeint_fn = _odeint_adjoint if self.ode_adjoint else _odeint
            phy_traj = odeint_fn(
                self.ode_func, z0_phy, time_steps,
                rtol=self.ode_rtol, atol=self.ode_atol, method=self.ode_method)
            phy_traj = phy_traj[1:]  # drop t=0 initial condition -> [pred_len, B, N*ode_latent_dim]
            phy_traj = phy_traj.reshape(self.pred_len, B, N, self.ode_latent_dim)
            phy_traj = phy_traj.permute(1, 2, 0, 3).reshape(
                B * N, self.pred_len, self.ode_latent_dim)              # [B*N, pred_len, ode_latent_dim]

        h_dec = self.decoder_init(torch.cat([h_mixed, z], dim=-1))
        preds = []
        for t in range(self.pred_len):
            step_in = torch.cat([feat_fut[:, t], z], dim=-1)
            h_dec = self.decoder_cell(step_in, h_dec)

            if self.ode_transport:
                # Fused AFTER decoder_cell, not fed back into its
                # recurrence - see class docstring for why this avoids
                # the earlier discrete-recurring-transport attempt's risk.
                phy_h = self.ode_phy_to_hidden(phy_traj[:, t])                        # [B*N, hidden_dim]
                gate = torch.sigmoid(self.ode_fusion_gate(torch.cat([h_dec, phy_h], dim=-1)))
                h_out = gate * h_dec + (1.0 - gate) * phy_h
            else:
                h_out = h_dec

            preds.append(self.output_head(h_out))

        out = torch.stack(preds, dim=1)
        pm25_pred = out.reshape(B, N, self.pred_len, 1).permute(0, 2, 1, 3)
        return pm25_pred
