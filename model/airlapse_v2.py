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

The GRU encoder/decoder, the learned content+physics attention and its
four independent bonus terms, and the mix_gate that combines context/
transported/h_grid are otherwise unchanged from AirLapse (model/
airlapse.py). One further difference from V1, unrelated to the transport
Green's function above: the latent bottleneck here is a single
deterministic Linear projection, not a VAE (no mu/logvar heads,
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

from model.airlapse import haversine_km, bearing_deg, _inv_softplus


class AdaptivePhysicsTransport2D(nn.Module):
    """
    Spatial mixing step for AirLapse V2. Structurally identical to
    MultiLagPhysicsAwareSpatialAttention (model/airlapse.py) - same
    learned content+physics attention (`context`), same four independent
    bonus terms (distance/wind/terrain/lag), same additive-across-sources
    aggregation - EXCEPT for how the explicit physical transport estimate
    (`transported`) is computed. See this module's file docstring for why.

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

        score = (content_score
                 + self.w_dist * dist_tiled.unsqueeze(0)
                 + self.w_wind * wind_align_flat
                 + self.w_terrain * terrain_tiled.unsqueeze(0)
                 + self.w_lag * lag_bias_flat)

        score = score.masked_fill(~mask_tiled.unsqueeze(0), float('-inf'))
        weights = torch.nan_to_num(torch.softmax(score, dim=-1), nan=0.0)
        context = torch.bmm(weights, v)

        # --- explicit physical transport estimate: 2-D anisotropic
        # advection-diffusion Green's function with context-adaptive
        # diffusivity (see class docstring) - independent of, and not
        # feeding back into, the learned attention above. ---
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
                 st_encoder_layers=1):
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

        h_dec = self.decoder_init(torch.cat([h_mixed, z], dim=-1))
        preds = []
        for t in range(self.pred_len):
            step_in = torch.cat([feat_fut[:, t], z], dim=-1)
            h_dec = self.decoder_cell(step_in, h_dec)
            preds.append(self.output_head(h_dec))

        out = torch.stack(preds, dim=1)
        pm25_pred = out.reshape(B, N, self.pred_len, 1).permute(0, 2, 1, 3)
        return pm25_pred
