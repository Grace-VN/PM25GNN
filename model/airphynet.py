"""
AirPhyNet (Hettige et al., "AirPhyNet: Physics-Guided Neural Networks for
Air Quality Prediction", arXiv:2402.03784, https://github.com/kethmih/AirPhyNet)
adapted as a benchmark model for this repo's harness.

A second physics-guided neural-ODE benchmark alongside model/airdde.py,
but a genuinely different design, not a re-skin of it:
  - Stochastic (VAE-style) initial state: a GRU encodes each station's
    PM2.5 history alone (not the full weather vector) into a Gaussian
    over a small `latent_dim` state, sampled once per forward pass -
    AirDDE's encoder is deterministic (AGCRN + memory banks) instead.
  - Two-term ODE dynamics (diffusion + advection only, combined by a
    learned gate) - AirDDE's ODE has a third, learned source/sink term
    plus a memory-based correction; AirPhyNet has neither.
  - Diffusion runs over the PLAIN (unweighted) station graph; AirDDE's
    diffusion is inverse-distance-weighted. Advection weights come from
    a small learned network over wind, not a closed-form trig formula.
  - Direct multi-horizon decoding straight from the ODE trajectory (one
    shot, no per-step decoder loop) - AirDDE instead re-decodes at every
    future step with an ST-decoder that also sees real future weather.

ARCHITECTURE (from the paper's own description)
---------------------------------------------------
  1. Encoder: a GRU reads each station's own PM2.5 history (channel 0
     only) and its final hidden state is projected (linear + tanh) to a
     Gaussian (mean, std) over a `latent_dim`-sized initial state z0 -
     reusing this repo's own model/cells.py LatentLayer/reparameterize
     for that projection, the same VAE building block model/probgru5.py+
     already uses elsewhere in this repo.
  2. ODE dynamics (`_ODEFunc`): at every solver step, the flattened state
     is graph-convolved (Chebyshev filtering, `gcn_step` hops) twice -
     once over the plain station graph (diffusion) and once over a
     wind-driven, per-batch edge weighting built from the last observed
     wind vector via a small linear network (advection) - and the two
     resulting gradients are combined through a learned sigmoid gate
     rather than a fixed mixing coefficient.
  3. Integration: `torchdiffeq.odeint_adjoint` rolls the state forward to
     `pred_len` evenly spaced points over the forecast horizon.
  4. Decoder: each trajectory point's latent state is linearly projected
     to the PM2.5 output, averaged over `n_traj_samples` (defaults to 1 -
     a single sampled trajectory - for a deterministic, comparable
     forecast; raise it for a cheap ensemble-style prediction).

WHAT HAD TO CHANGE, AND WHY
-----------------------------
  1. Contract. `Model(hist_len, pred_len, in_dim, city_num, batch_size,
     device, edge_index, edge_attr, wind_mean, wind_std, ...)` called as
     `model(pm25_hist, feature) -> [B, pred_len, N, 1]`, same as every
     other model here, instead of upstream's own
     `(inputs[T,B,N*D], ...)` trainer-specific signature.
  2. Graph convolution implementation. Upstream hand-rolls Chebyshev
     filtering over a manually scaled Laplacian
     (`calculate_scaled_laplacian`). This repo already vendored
     `torch_geometric`'s `ChebConv` for exactly this purpose in
     model/airdde.py (verified there to support both a shared dense-
     batched graph for the diffusion term and a per-batch, block-
     diagonal graph for the wind-driven advection term) - reused here
     rather than re-deriving the Laplacian scaling by hand.
  3. Advection edge weights. The paper computes a per-node scalar "flow"
     from wind via a small linear network, then an edge weight from the
     difference between its two endpoints' flow values; I implement
     exactly that (`_wind_flow_net` + endpoint difference) since it's
     described unambiguously, but note (unlike AirDDE's explicit
     `3*speed*cos(theta)/dist` formula, which is stated precisely in
     that repo) I could not verify the exact functional form beyond
     "linear network on wind, then node differences" - this is a
     faithful, principled implementation of that description, not a
     line-for-line port.
  4. KL regularization. AirPhyNet's stochastic initial state gives it a
     natural KL(q(z0|history) || N(0,1)) term, in the same spirit as
     this repo's other VAE-style benchmarks (AirFormer, ProbGRUModel*).
     Exposed via `self.last_kl_loss` so train.py's existing
     `kl_weight`-weighted loss hook picks it up automatically - upstream
     instead trains this as part of a full variational ELBO inside its
     own supervisor script, which this harness has no hook for.
  5. Wind input. `wind_mean`/`wind_std` (already computed by train.py
     for every graph-based model here) de-normalize `feature`'s last two
     channels (speed, direction) at the last history step, matching
     PM25_GNN/AirDDE's own convention for "which channels are wind."

Contract (matches every other model in model/, see train.py get_model()):
    AirPhyNetPM25(hist_len, pred_len, in_dim, city_num, batch_size, device,
                  edge_index, edge_attr, wind_mean, wind_std, ...)
    pm25_pred = model(pm25_hist, feature)   # -> [B, pred_len, N, 1]
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv
from torchdiffeq import odeint as _odeint
from torchdiffeq import odeint_adjoint as _odeint_adjoint

from model.cells import LatentLayer, reparameterize


class _WindFlowNet(nn.Module):
    """Small learned network mapping a node's (speed, direction) wind
    vector to a scalar 'flow potential' - see module docstring point 3."""

    def __init__(self, hidden_dim=16):
        super(_WindFlowNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, wind):
        # wind: [..., 2] -> [..., 1]
        return self.net(wind)


class _GatedFusion(nn.Module):
    """Learned sigmoid gate combining diffusion and advection gradients,
    per (node, latent channel)."""

    def __init__(self, latent_dim):
        super(_GatedFusion, self).__init__()
        self.gate = nn.Linear(2 * latent_dim, latent_dim)

    def forward(self, diff_grad, adv_grad):
        g = torch.sigmoid(self.gate(torch.cat([diff_grad, adv_grad], dim=-1)))
        return g * diff_grad + (1 - g) * adv_grad


class _ODEFunc(nn.Module):
    """Diffusion + advection dynamics over the latent state, combined by
    a learned gate. filter_type selects which term(s) are active:
    'diff', 'adv', 'diff_adv' (both, gated), or 'unkP' (a fully learned
    MLP over the flattened state, ignoring the graph - upstream's
    ablation for "unknown physics")."""

    def __init__(self, latent_dim, edge_index, num_nodes, gcn_step=2,
                 filter_type='diff_adv', gen_dim=64, gen_layers=1,
                 diff_coeff=0.1, wind_hidden_dim=16, max_deriv=10.0):
        super(_ODEFunc, self).__init__()
        self.latent_dim = latent_dim
        self.num_nodes = num_nodes
        self.filter_type = filter_type
        self.diff_coeff = diff_coeff
        self.max_deriv = max_deriv

        self.register_buffer('edge_index', torch.as_tensor(edge_index, dtype=torch.long))

        if filter_type in ('diff', 'diff_adv'):
            self.diff_conv = ChebConv(latent_dim, latent_dim, K=gcn_step, normalization='sym', bias=True)
        if filter_type in ('adv', 'diff_adv'):
            # normalization=None (raw/combinatorial Laplacian), not 'sym': the
            # advection edge weight below is a SIGNED flow difference
            # (directional, can be negative), and ChebConv's 'sym'/'rw'
            # normalization takes sqrt(degree) - a negative weighted degree
            # there produces NaN. An unnormalized Laplacian is also the more
            # physically sensible choice for a directional term anyway
            # (symmetric normalization is meant for undirected diffusion).
            self.adv_conv = ChebConv(latent_dim, latent_dim, K=gcn_step, normalization=None, bias=True)
            self.wind_flow_net = _WindFlowNet(wind_hidden_dim)
        if filter_type == 'diff_adv':
            self.gate = _GatedFusion(latent_dim)
        if filter_type == 'unkP':
            layers = [nn.Linear(latent_dim, gen_dim), nn.Tanh()]
            for _ in range(gen_layers - 1):
                layers += [nn.Linear(gen_dim, gen_dim), nn.Tanh()]
            layers.append(nn.Linear(gen_dim, latent_dim))
            self.unk_net = nn.Sequential(*layers)

        self.last_wind = None  # [B, N, 2] - set per-forward via set_wind()

    def set_wind(self, wind_bn2):
        self.last_wind = wind_bn2

    def _diff_grad(self, x):
        # x: [B, N, latent_dim] - shared graph, dense-batched (no edge weight: unweighted diffusion)
        return -self.diff_coeff * self.diff_conv(x, self.edge_index, lambda_max=2)

    def _adv_grad(self, x):
        # x: [B, N, latent_dim]; per-batch wind-driven edge weights, block-diagonal graph
        B, N, D = x.shape
        flow = self.wind_flow_net(self.last_wind).squeeze(-1)  # [B, N]
        edge_src, edge_dst = self.edge_index
        edge_weight = flow[:, edge_src] - flow[:, edge_dst]     # [B, M]

        batch = torch.arange(B, device=x.device).repeat_interleave(N)
        x_flat = x.reshape(B * N, D)
        edge_indices = [self.edge_index + i * N for i in range(B)]
        edge_index = torch.cat(edge_indices, dim=1)
        edge_weight = edge_weight.reshape(-1)

        # lambda_max=None (not a hardcoded 2): unlike diff_conv above,
        # this ChebConv uses normalization=None (a raw/combinatorial
        # Laplacian built from a SIGNED edge weight, since flow can be
        # negative) - its eigenvalues are not guaranteed to lie in [0, 2]
        # the way a symmetric-normalized Laplacian's are, so a hardcoded
        # lambda_max=2 silently mis-scales the Chebyshev recursion
        # whenever the learned wind_flow_net's actual weight magnitudes
        # push the true spectral radius past that. Passing None instead
        # lets ChebConv fall back to its own 2*edge_weight.max() estimate,
        # scaled to the weights actually present each forward call, which
        # is what fixed a real "underflow in dt" crash from the resulting
        # exponential blow-up compounding over a longer pred_len/ODE
        # integration horizon (reported on dataset 2, pred_len=24).
        out = self.adv_conv(x_flat, edge_index, edge_weight, batch=batch, lambda_max=None)
        return out.reshape(B, N, D)

    def forward(self, t, z):
        # z: [B, N*latent_dim] (flattened, as torchdiffeq requires a plain tensor state)
        B = z.shape[0]
        x = z.reshape(B, self.num_nodes, self.latent_dim)

        if self.filter_type == 'diff':
            grad = self._diff_grad(x)
        elif self.filter_type == 'adv':
            grad = self._adv_grad(x)
        elif self.filter_type == 'diff_adv':
            grad = self.gate(self._diff_grad(x), self._adv_grad(x))
        else:  # 'unkP'
            grad = self.unk_net(x)

        # Soft-bound the vector field's magnitude. An unbounded RHS lets a
        # neural ODE's state grow (or its adjoint's backward step size
        # shrink) without limit over a long-enough integration horizon -
        # torchdiffeq surfaces that as "underflow in dt" once its adaptive
        # solver can no longer take a step small enough to satisfy its
        # error tolerance. This saturates |dz/dt| well below that failure
        # point without materially changing the learned dynamics in the
        # normal (small-derivative) operating regime.
        grad = self.max_deriv * torch.tanh(grad / self.max_deriv)
        return grad.reshape(B, self.num_nodes * self.latent_dim)


class AirPhyNetPM25(nn.Module):
    def __init__(self, hist_len, pred_len, in_dim, city_num, batch_size, device,
                 edge_index, edge_attr, wind_mean, wind_std,
                 rnn_units=64, latent_dim=4, gcn_step=2, diff_coeff=0.1,
                 n_traj_samples=1, ode_method='dopri5', ode_rtol=1e-3, ode_atol=1e-4,
                 ode_adjoint=True, filter_type='diff_adv', gen_dim=64, gen_layers=1,
                 wind_hidden_dim=16, max_deriv=10.0):
        super(AirPhyNetPM25, self).__init__()
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.in_dim = in_dim
        self.city_num = city_num
        self.device = device
        self.batch_size = batch_size
        self.rnn_units = rnn_units
        self.latent_dim = latent_dim
        self.n_traj_samples = max(1, n_traj_samples)
        self.output_dim = 1
        self.last_kl_loss = None

        self.register_buffer('wind_mean', torch.as_tensor(np.float32(wind_mean))[-2:])
        self.register_buffer('wind_std', torch.as_tensor(np.float32(wind_std))[-2:])

        # --- Stage 1: recognition/encoder ---
        self.encoder_gru = nn.GRU(input_size=1, hidden_size=rnn_units, batch_first=True)
        self.latent_proj = nn.Sequential(nn.Linear(rnn_units, rnn_units), nn.Tanh())
        self.latent_layer = LatentLayer(in_dim=rnn_units, latent_dim=latent_dim, hidden_dim=rnn_units)

        # --- Stage 2/3: ODE dynamics + solver ---
        self.odefunc = _ODEFunc(
            latent_dim, edge_index, city_num, gcn_step=gcn_step, filter_type=filter_type,
            gen_dim=gen_dim, gen_layers=gen_layers, diff_coeff=diff_coeff, wind_hidden_dim=wind_hidden_dim,
            max_deriv=max_deriv,
        )
        self.ode_method = ode_method
        self.ode_rtol = ode_rtol
        self.ode_atol = ode_atol
        self.ode_adjoint = ode_adjoint

        # --- Stage 4: decoder ---
        self.decoder = nn.Linear(latent_dim, self.output_dim)

    def forward(self, pm25_hist, feature):
        """
        pm25_hist : [B, hist_len, N, 1]
        feature   : [B, hist_len + pred_len, N, F]   (F = in_dim - 1)
        returns   : [B, pred_len, N, 1]
        """
        B, T, N, _ = pm25_hist.shape
        if N != self.city_num:
            raise ValueError(
                f"AirPhyNetPM25 was built with city_num={self.city_num}, but got "
                f"N={N} nodes in this batch's data."
            )
        feature_hist = feature[:, :self.hist_len]

        # --- encoder: PM2.5-only GRU, per-station, shared weights ---
        pm25_flat = pm25_hist.permute(0, 2, 1, 3).reshape(B * N, self.hist_len, 1)
        _, h_n = self.encoder_gru(pm25_flat)          # h_n: [1, B*N, rnn_units]
        h_n = h_n[-1].reshape(B, N, self.rnn_units)     # [B, N, rnn_units]
        h_n = self.latent_proj(h_n)
        mu, sigma = self.latent_layer(h_n)               # [B, N, latent_dim] each

        eps_shape = (self.n_traj_samples, B, N, self.latent_dim)
        mu_rep = mu.unsqueeze(0).expand(eps_shape)
        sigma_rep = sigma.unsqueeze(0).expand(eps_shape)
        z0 = reparameterize(mu_rep, sigma_rep)           # [S, B, N, latent_dim]
        z0 = z0.reshape(self.n_traj_samples * B, N * self.latent_dim)

        # KL(q(z0|history) || N(0,1)), same VAE-style regularizer used
        # elsewhere in this repo (see module docstring point 4).
        self.last_kl_loss = 0.5 * torch.mean(mu.pow(2) + sigma.pow(2) - 2 * torch.log(sigma) - 1)

        # --- wind at the last observed history step, for the advection term ---
        last_wind_z = feature_hist[:, -1, :, -2:]                                  # [B, N, 2]
        last_wind = last_wind_z * self.wind_std.view(1, 1, 2) + self.wind_mean.view(1, 1, 2)
        wind_rep = last_wind.unsqueeze(0).expand(self.n_traj_samples, B, N, 2)
        self.odefunc.set_wind(wind_rep.reshape(self.n_traj_samples * B, N, 2))

        # --- integrate over the forecast horizon ---
        time_steps = torch.linspace(0.0, 1.0, self.pred_len + 1, device=pm25_hist.device)
        odeint = _odeint_adjoint if self.ode_adjoint else _odeint
        pred_z = odeint(self.odefunc, z0, time_steps,
                         rtol=self.ode_rtol, atol=self.ode_atol, method=self.ode_method)
        pred_z = pred_z[1:]  # drop t=0 initial condition -> [pred_len, S*B, N*latent_dim]

        pred_z = pred_z.reshape(self.pred_len, self.n_traj_samples, B, N, self.latent_dim)
        out = self.decoder(pred_z)                       # [pred_len, S, B, N, output_dim]
        out = out.mean(dim=1)                              # average over trajectory samples -> [pred_len, B, N, output_dim]
        pm25_pred = out.permute(1, 0, 2, 3)               # [B, pred_len, N, output_dim=1]
        return pm25_pred
