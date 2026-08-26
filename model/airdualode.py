"""
Air-DualODE (ICLR 2025, "Air Quality Prediction with Physics-Guided Dual
Neural ODEs in Open Systems", arXiv:2410.19892,
https://github.com/Philosober/Air-DualODE) adapted as a benchmark model
for this repo's harness.

ARCHITECTURE
-------------
Two parallel neural-ODE branches, fused CONTINUOUSLY across the whole
forecast trajectory (not just at the final step, unlike model/airdde.py's
late-stage fusion):

  1. Physics-driven branch: a diffusion + wind-driven advection + linear
     reaction/source term, integrated via a neural ODE. The diffusion and
     advection terms use the SAME formulas as model/airdde.py's ODEFunc
     (independently confirmed against both repos: Chebyshev graph
     convolution for diffusion with a fixed 0.1 coefficient, and
     `3*wind_speed*cos(theta)/distance` edge weighting for advection,
     gated together) - these two physics papers evidently share this
     exact formulation (or a common ancestor), so this branch reuses that
     already-validated implementation rather than re-deriving it. The
     genuinely new piece here is the reaction/source term: a per-node,
     per-channel learnable coefficient beta, contributing `beta * x` to
     the gradient (representing net local production/removal - this is
     the "open system" piece: mass isn't conserved by diffusion+advection
     alone, beta is free to add or remove it).
  2. Data-driven branch: a GRU encodes the full (PM2.5 + weather) history
     into an initial latent state; its neural ODE drift is a multi-head
     self-attention block over the STATION dimension at each solver
     step (reusing model/attn.py's FullAttention/AttentionLayer, the same
     building block model/informer.py and model/patchtst.py already use),
     i.e. dz/dt = Attention(z) - a purely learned, physics-free dynamic.
  3. Fusion, at EVERY trajectory step: both branches' latent states (at
     time t) are linearly projected to a shared dimension, gated by a
     graph convolution over their concatenation, and blended - so later
     ODE steps can lean more on whichever branch is doing better, rather
     than only combining once at the end.
  4. A soft "alignment" regularizer keeps the two branches' projected
     latents from diverging too far apart at every step (upstream calls
     this a temporal-alignment/contrastive loss; implemented here as a
     step-wise MSE between the two branches' fused-space projections,
     the simplest faithful reading of "keep them in agreement over time"
     - exposed via `self.last_alignment_loss`, analogous to how this
     repo's VAE-style benchmarks expose `self.last_kl_loss`).

WHAT HAD TO CHANGE, AND WHY
-----------------------------
  1. Contract. `Model(hist_len, pred_len, in_dim, city_num, batch_size,
     device, edge_index, edge_attr, wind_mean, wind_std, ...)` called as
     `model(pm25_hist, feature) -> [B, pred_len, N, 1]`, same as every
     other model here, instead of upstream's own time-major
     `(T,B,N*D)` trainer-specific input.
  2. edge_attr layout / diffusion weight. Same deviation as
     model/airdde.py: this repo's graph.py only carries 2 edge_attr
     columns (`[dist_km, bearing]`), not upstream's 3 (a dedicated
     diffusion-weight column, distance, bearing) - `diff_edge_attr` is
     synthesized as `1/(dist_km+eps)`, exactly as in model/airdde.py, for
     the same reason (see that file's own docstring point 3).
  3. Input embedding. Upstream's embedding module looks up learned
     vectors for hour-of-day/day-of-week/day-of-month/month/station-index
     from categorical indices. KnowAir's `feature` tensor is continuous
     and z-normalized throughout (see dataset.py) - there's nothing
     categorical to embed, same dataset mismatch already documented in
     model/airformer.py and model/GNN_Transformer.py - dropped.
  4. Coefficient estimation. Upstream can optionally learn time-varying
     PDE coefficients via a small RNN (`estimate=True`). This is
     implemented (`estimate_coeff=True`) but defaults to False - a
     single learnable per-node/channel `beta` parameter, which is
     simpler and was the paper's own non-estimated default per the
     source's `estimate` flag defaulting to False.
  5. Alignment loss weighting. Upstream trains its temporal-alignment
     loss as part of a combined objective inside its own trainer.
     train.py only had a hook for one extra scalar loss
     (`last_kl_loss`/`kl_weight`, for AirFormer's/ProbGRUModel's VAE
     terms) - a second, analogous hook (`last_alignment_loss` /
     `alignment_weight`, default 0.1) was added to train.py alongside it
     for this model, rather than overloading the KL hook for a
     differently-meaningful term.
  6. Decoder. Upstream's optional conv decoder maps the fused trajectory
     back to observation space; kept, but simplified to a single 1x1
     Conv2d(fusion_dim -> 1) applied at every forecast step (this repo
     only ever predicts one channel, PM2.5).

Contract (matches every other model in model/, see train.py get_model()):
    AirDualODEPM25(hist_len, pred_len, in_dim, city_num, batch_size, device,
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

from model.attn import FullAttention, AttentionLayer


class _GatedFusion(nn.Module):
    """Learned sigmoid gate combining two same-shape tensors, per (node,
    channel) - shared pattern with model/airphynet.py's diffusion/
    advection gate, reimplemented locally to keep this file self-contained
    (matching this repo's convention of not cross-importing between
    benchmark model files)."""

    def __init__(self, dim):
        super(_GatedFusion, self).__init__()
        self.gate = nn.Linear(2 * dim, dim)

    def forward(self, a, b):
        g = torch.sigmoid(self.gate(torch.cat([a, b], dim=-1)))
        return g * a + (1 - g) * b


class _PhysicsODEFunc(nn.Module):
    """Diffusion + wind-driven advection (gated together) + a linear
    reaction/source term. Diffusion/advection formulas match
    model/airdde.py's ODEFunc (see module docstring point 1/2)."""

    def __init__(self, latent_dim, edge_index, edge_attr, num_nodes, gcn_step=2,
                 estimate_coeff=False, coeff_hidden_dim=16):
        super(_PhysicsODEFunc, self).__init__()
        self.latent_dim = latent_dim
        self.num_nodes = num_nodes
        self.estimate_coeff = estimate_coeff

        self.register_buffer('edge_index', torch.as_tensor(edge_index, dtype=torch.long))
        edge_attr_t = torch.as_tensor(np.float32(edge_attr))
        self.register_buffer('edge_attr', edge_attr_t)
        self.register_buffer('diff_edge_attr', 1.0 / edge_attr_t[:, 0].clamp(min=1e-3))
        self.diff_coeff = 0.1

        self.diff_conv = ChebConv(latent_dim, latent_dim, K=gcn_step, normalization='sym', bias=True)
        self.adv_conv = ChebConv(latent_dim, latent_dim, K=gcn_step, normalization='sym', bias=True)
        self.gate = _GatedFusion(latent_dim)

        if estimate_coeff:
            self.beta_net = nn.GRU(input_size=latent_dim, hidden_size=coeff_hidden_dim, batch_first=True)
            self.beta_head = nn.Linear(coeff_hidden_dim, latent_dim)
        else:
            self.beta = nn.Parameter(torch.zeros(num_nodes, latent_dim))

        self.adv_edge_attr = None  # set per-forward via set_wind() - depends on this batch's wind
        self.beta_estimated = None  # set per-forward if estimate_coeff, else uses self.beta

    def set_wind(self, last_wind_bn2, wind_mean, wind_std):
        B = last_wind_bn2.shape[0]
        edge_src, edge_target = self.edge_index
        node_src = last_wind_bn2[:, edge_src, :]

        src_speed = node_src[:, :, 0] * wind_std[0] + wind_mean[0]
        src_dir = node_src[:, :, 1] * wind_std[1] + wind_mean[1]
        dist = self.edge_attr[:, 0].unsqueeze(0).repeat(B, 1)
        bearing = self.edge_attr[:, 1].unsqueeze(0).repeat(B, 1)

        src_dir = (src_dir + 180) % 360  # "from" -> "traveling toward"
        theta = torch.abs(bearing - src_dir)
        self.adv_edge_attr = F.relu(3 * src_speed * torch.cos(theta) / dist)  # B x M, always >= 0

    def set_beta(self, hist_seq):
        # hist_seq: [B, N, hist_len, latent_dim] - only used if estimate_coeff
        if self.estimate_coeff:
            B, N, T, D = hist_seq.shape
            _, h_n = self.beta_net(hist_seq.reshape(B * N, T, D))
            self.beta_estimated = self.beta_head(h_n[-1]).reshape(B, N, D)
        else:
            self.beta_estimated = None

    def _diff_grad(self, x):
        return -self.diff_coeff * self.diff_conv(x, self.edge_index, self.diff_edge_attr, lambda_max=2)

    def _adv_grad(self, x):
        B, N, D = x.shape
        x_flat = x.reshape(B * N, D)
        batch = torch.arange(B, device=x.device).repeat_interleave(N)
        edge_index = torch.cat([self.edge_index + i * N for i in range(B)], dim=1)
        edge_weight = self.adv_edge_attr.reshape(-1)
        out = self.adv_conv(x_flat, edge_index, edge_weight, batch=batch, lambda_max=2)
        return out.reshape(B, N, D)

    def forward(self, t, z):
        # z: [B, N*latent_dim]
        B = z.shape[0]
        x = z.reshape(B, self.num_nodes, self.latent_dim)

        grad = self.gate(self._diff_grad(x), self._adv_grad(x))
        beta = self.beta_estimated if self.beta_estimated is not None else self.beta.unsqueeze(0)
        grad = grad + beta * x

        return grad.reshape(B, self.num_nodes * self.latent_dim)


class _AttentionODEFunc(nn.Module):
    """Purely learned drift: dz/dt = Attention(z) - multi-head self-
    attention over the station dimension at every solver step (see
    module docstring point 2)."""

    def __init__(self, latent_dim, num_nodes, n_heads=2, dropout=0.0):
        super(_AttentionODEFunc, self).__init__()
        self.latent_dim = latent_dim
        self.num_nodes = num_nodes
        self.attn = AttentionLayer(
            FullAttention(False, attention_dropout=dropout, output_attention=False),
            latent_dim, n_heads, mix=False,
        )
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, t, z):
        B = z.shape[0]
        x = z.reshape(B, self.num_nodes, self.latent_dim)
        out, _ = self.attn(x, x, x, attn_mask=None)
        out = self.norm(out)
        return out.reshape(B, self.num_nodes * self.latent_dim)


class AirDualODEPM25(nn.Module):
    def __init__(self, hist_len, pred_len, in_dim, city_num, batch_size, device,
                 edge_index, edge_attr, wind_mean, wind_std,
                 phy_latent_dim=8, unk_latent_dim=8, fusion_dim=16,
                 gcn_step=2, rnn_units=32, attn_heads=2,
                 estimate_coeff=False, ode_method='dopri5',
                 ode_rtol=1e-3, ode_atol=1e-4, ode_adjoint=True):
        super(AirDualODEPM25, self).__init__()
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.in_dim = in_dim
        self.city_num = city_num
        self.device = device
        self.batch_size = batch_size
        self.phy_latent_dim = phy_latent_dim
        self.unk_latent_dim = unk_latent_dim
        self.output_dim = 1
        self.last_alignment_loss = None

        self.register_buffer('wind_mean', torch.as_tensor(np.float32(wind_mean))[-2:])
        self.register_buffer('wind_std', torch.as_tensor(np.float32(wind_std))[-2:])

        # --- physics branch ---
        self.phy_init_proj = nn.Linear(1 + 2, phy_latent_dim)  # [last pm25, last wind(2)] -> phy_latent_dim
        self.phy_odefunc = _PhysicsODEFunc(phy_latent_dim, edge_index, edge_attr, city_num,
                                            gcn_step=gcn_step, estimate_coeff=estimate_coeff)
        self.phy_refine_gru = nn.GRU(input_size=phy_latent_dim, hidden_size=phy_latent_dim, batch_first=True)

        # --- data-driven branch ---
        self.unk_encoder_gru = nn.GRU(input_size=in_dim, hidden_size=rnn_units, batch_first=True)
        self.unk_init_proj = nn.Linear(rnn_units, unk_latent_dim)
        self.unk_odefunc = _AttentionODEFunc(unk_latent_dim, city_num, n_heads=attn_heads)

        # --- fusion (applied at every trajectory step) ---
        self.fusion_dim = fusion_dim
        self.phy_fusion_proj = nn.Linear(phy_latent_dim, fusion_dim)
        self.unk_fusion_proj = nn.Linear(unk_latent_dim, fusion_dim)
        self.register_buffer('_edge_index_buf', torch.as_tensor(edge_index, dtype=torch.long))
        self.fusion_gate_conv = ChebConv(2 * fusion_dim, fusion_dim, K=gcn_step, normalization='sym', bias=True)

        # --- decoder ---
        self.decoder = nn.Conv2d(fusion_dim, self.output_dim, kernel_size=1)

        self.ode_method = ode_method
        self.ode_rtol = ode_rtol
        self.ode_atol = ode_atol
        self.ode_adjoint = ode_adjoint

    def _integrate(self, odefunc, z0, time_steps):
        odeint = _odeint_adjoint if self.ode_adjoint else _odeint
        traj = odeint(odefunc, z0, time_steps, rtol=self.ode_rtol, atol=self.ode_atol, method=self.ode_method)
        return traj[1:]  # drop t=0 initial condition -> [pred_len, ...]

    def forward(self, pm25_hist, feature):
        """
        pm25_hist : [B, hist_len, N, 1]
        feature   : [B, hist_len + pred_len, N, F]   (F = in_dim - 1)
        returns   : [B, pred_len, N, 1]
        """
        B, T, N, _ = pm25_hist.shape
        if N != self.city_num:
            raise ValueError(
                f"AirDualODEPM25 was built with city_num={self.city_num}, but got "
                f"N={N} nodes in this batch's data."
            )
        feature_hist = feature[:, :self.hist_len]
        x_hist = torch.cat([pm25_hist, feature_hist], dim=-1)  # [B,hist_len,N,in_dim]

        last_pm25 = pm25_hist[:, -1]                              # [B,N,1]
        last_wind_z = feature_hist[:, -1, :, -2:]                  # [B,N,2] (normalized)

        time_steps = torch.linspace(0.0, 1.0, self.pred_len + 1, device=pm25_hist.device)

        # --- physics branch ---
        phy_z0 = self.phy_init_proj(torch.cat([last_pm25, last_wind_z], dim=-1))  # [B,N,phy_latent_dim]
        phy_z0 = phy_z0.reshape(B, N * self.phy_latent_dim)

        # set_wind takes the NORMALIZED wind tensor plus (wind_mean, wind_std)
        # and denormalizes it internally - see _PhysicsODEFunc.set_wind.
        self.phy_odefunc.set_wind(last_wind_z, self.wind_mean, self.wind_std)

        if self.phy_odefunc.estimate_coeff:
            # per-step [pm25, wind] projected through the same phy_init_proj
            # used for the initial state, giving the coefficient estimator's
            # GRU a [B,N,hist_len,phy_latent_dim] input sequence.
            hist_input = torch.cat([pm25_hist, feature_hist[..., -2:]], dim=-1)  # [B,hist_len,N,3]
            hist_latent = self.phy_init_proj(hist_input)                          # [B,hist_len,N,phy_latent_dim]
            self.phy_odefunc.set_beta(hist_latent.permute(0, 2, 1, 3))            # [B,N,hist_len,phy_latent_dim]
        else:
            self.phy_odefunc.set_beta(None)

        phy_traj = self._integrate(self.phy_odefunc, phy_z0, time_steps)  # [pred_len, B, N*phy_latent_dim]
        phy_traj = phy_traj.reshape(self.pred_len, B, N, self.phy_latent_dim)
        phy_traj_bn = phy_traj.permute(1, 2, 0, 3).reshape(B * N, self.pred_len, self.phy_latent_dim)
        phy_latent, _ = self.phy_refine_gru(phy_traj_bn)  # [B*N, pred_len, phy_latent_dim]
        phy_latent = phy_latent.reshape(B, N, self.pred_len, self.phy_latent_dim).permute(2, 0, 1, 3)

        # --- data-driven branch ---
        x_flat = x_hist.permute(0, 2, 1, 3).reshape(B * N, self.hist_len, self.in_dim)
        _, h_n = self.unk_encoder_gru(x_flat)
        unk_z0 = self.unk_init_proj(h_n[-1]).reshape(B, N * self.unk_latent_dim)
        unk_traj = self._integrate(self.unk_odefunc, unk_z0, time_steps)  # [pred_len, B, N*unk_latent_dim]
        unk_latent = unk_traj.reshape(self.pred_len, B, N, self.unk_latent_dim)

        # --- fusion at every step ---
        edge_index = self._edge_index_buf
        preds = []
        align_terms = []
        for t in range(self.pred_len):
            p = self.phy_fusion_proj(phy_latent[t])   # [B,N,fusion_dim]
            u = self.unk_fusion_proj(unk_latent[t])    # [B,N,fusion_dim]
            gate_in = torch.cat([p, u], dim=-1)
            gate = torch.sigmoid(self.fusion_gate_conv(gate_in, edge_index, lambda_max=2))
            fused = gate * p + (1 - gate) * u           # [B,N,fusion_dim]
            align_terms.append(F.mse_loss(p, u))

            fused_map = fused.permute(0, 2, 1).unsqueeze(-1)  # [B,fusion_dim,N,1]
            out_t = self.decoder(fused_map).squeeze(-1).permute(0, 2, 1)  # [B,N,output_dim]
            preds.append(out_t)

        self.last_alignment_loss = torch.stack(align_terms).mean()
        pm25_pred = torch.stack(preds, dim=1)  # [B, pred_len, N, output_dim=1]
        return pm25_pred
