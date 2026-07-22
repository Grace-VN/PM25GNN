"""
AirFormer++ - compat wrapper for this repo's train.py

train.py instantiates every model with the same positional signature
(see PM25_GNN, GC_LSTM, etc.):

    Model(hist_len, pred_len, in_dim, city_num, batch_size, device,
          edge_index, edge_attr, wind_mean, wind_std)

and calls it as:

    pm25_pred = model(pm25_hist, feature)

where
    pm25_hist : [B, hist_len, N, 1]
    feature   : [B, hist_len + pred_len, N, F]     (F = in_dim - 1)

and expects a single tensor back:

    pm25_pred : [B, pred_len, N, 1]

The original AirFormer++ (see airformerplusplus.py) instead:
  - is built via `BaseModel(**args)` with named kwargs (name, dataset,
    device, num_nodes, seq_len, horizon, input_dim, output_dim)
  - forward(inputs, supports=None) takes ONE pre-combined [b, t, n, c]
    tensor and returns a 3-tuple (x_hat, x_rec, kl_loss) in stochastic mode

This file adapts ONLY that boundary. The core architecture (adaptive
sector spatial attention, adaptive-window temporal attention, hierarchical
stochastic VAE, multi-horizon fusion head) is unchanged.

This revision drops the AirFormer-specific time-categorical embedding
(confirmed dataset-mismatched). An earlier revision added GraphGNN-based
feature augmentation (reusing model/PM25_GNN.py's wind-weighted message
passing); that has since been REMOVED per request - `edge_index`,
`edge_attr`, `wind_mean`, `wind_std` are still accepted in the constructor
(to keep the positional signature identical to every other benchmark
model in train.py) but are no longer used anywhere in this file. If graph
awareness is wanted again, PM25_GNN.GraphGNN is still there to reuse.

TEMPORAL BACKBONE: AdaptiveTemporalAttention has been replaced with
SeqScaleTemporalMixer (layers/seqscale_temporal_mixer.py), a causal
trend/seasonal linear mixer derived from the user-supplied SeqScale model
(RevIN and the outer end-to-end SeqScale wrapper deliberately excluded -
only the temporal-mixing mechanism is reused, adapted to operate as a
same-shape [B,C,N,T]->[B,C,N,T] residual sub-layer per block instead of a
standalone seq_len->pred_len forecaster). This also fixes a real
correctness gap: the new mixer is explicitly causal, whereas the
generative/inference split just below (`d_shift`) shifts d back one
timestep assuming each d[...,t] only depends on information up to t - a
non-causal temporal module would undermine that assumption by smearing
near-future context backward within each window.

ASSUMPTIONS / THINGS TO VERIFY:
  1. Because output_dim is fixed at 1 (single PM2.5 value per node), the
     final conv output [B, pred_len*output_dim, N, 1] collapses to exactly
     [B, pred_len, N, 1] with no reshuffling needed - verified by shape,
     not by guessing channel ordering.
  2. SPATIAL BACKBONE: AdaptiveSectorMSA (learned-from-scratch sector
     clustering, no geographic locality prior) has been replaced with
     DS_MSA (layers/dartboard_spatial_attention.py) - real geographic
     dartboard sector attention using the assignment/mask that
     get_dartboard_info() already loaded but previously went unused.
     DS_MSA manages its own residual connections AND its own FeedForward
     internally, so it's called directly at the block-loop call site
     (`x = self.s_modules[i](x)`), NOT wrapped in an external `x + ...`
     like the plain-conv fallback is - see the inline comment there.
     The plain residual-conv fallback (used when spatial_flag=False, i.e.
     no assignment/mask available) still has its own Conv1d->Conv2d
     bugfix from before (upstream had an invalid Conv1d kernel size on a
     4D tensor).
  3. kl_loss / x_rec are computed (so the hierarchical VAE still trains its
     internal structure via the stochastic path) but are NOT added into
     the returned tensor directly - they're added into train.py's loss
     separately via model.last_kl_loss (see train.py's kl_weight).
  4. Added an MLP (feed-forward) sub-layer for AFTER the temporal mixer,
     which (unlike DS_MSA) has no FFN of its own. mlp_expansion was
     previously an accepted-but-unused constructor arg with nothing wired
     to it (true in the original airformerplusplus.py too). Each block is
     now: spatial sub-layer (DS_MSA, self-contained residual+FFN) ->
     temporal sub-layer (residual) -> MLP sub-layer (residual) ->
     BatchNorm.
"""

import os
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.base_model import BaseModel
from layers.dartboard_spatial_attention import DS_MSA
from layers.temporal_mixer import SeqScaleTemporalMixer
from layers.stochastics_layers import HierarchicalStochasticModel, HierarchicalKLLoss


class AirFormerPlusPlus(BaseModel):
    def __init__(self,
                 hist_len, pred_len, in_dim, city_num, batch_size, device,
                 edge_index, edge_attr, wind_mean, wind_std,
                 # --- AirFormer++ hyperparameters (unchanged knobs) ---
                 dropout=0.3,
                 spatial_flag=True,
                 stochastic_flag=True,
                 hidden_channels=128,
                 end_channels=64,
                 blocks=3,
                 mlp_expansion=2,
                 num_heads=2,
                 dartboard=0,
                 use_hierarchical_latent=True,
                 base_latent_dim=2,
                 latent_growth_factor=2.0,
                 # Moving-average kernel(s) for SeqScaleTemporalMixer's trend
                 # extraction. Single-element by default (classic Autoformer/
                 # DLinear-style single-scale decomposition rather than the
                 # module's own multi-kernel (3,5,9) default). NOTE: each
                 # kernel is clamped to that block's window_size internally,
                 # so 25 only has an effect distinct from "whole window" once
                 # hist_len >= 25 - see forward-pass docstring for detail.
                 temporal_kernel_sizes=(12,),
                 use_adaptive_temporal=True):
        super(AirFormerPlusPlus, self).__init__(
            name='AirFormerPlusPlus',
            dataset='haze',
            device=device,
            num_nodes=city_num,
            seq_len=hist_len,
            horizon=pred_len,
            input_dim=in_dim,
            output_dim=1,
        )

        # kept for parity with the other benchmark models / possible future use
        self.batch_size = batch_size
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.wind_mean = wind_mean
        self.wind_std = wind_std

        self.hist_len = hist_len
        self.pred_len = pred_len

        self.dropout = dropout
        self.blocks = blocks
        self.spatial_flag = spatial_flag
        self.stochastic_flag = stochastic_flag
        self.hidden_channels = hidden_channels
        self.mlp_expansion = mlp_expansion

        self.use_hierarchical_latent = use_hierarchical_latent
        self.base_latent_dim = base_latent_dim
        self.latent_growth_factor = latent_growth_factor
        self.use_adaptive_temporal = use_adaptive_temporal
        self.temporal_kernel_sizes = tuple(temporal_kernel_sizes)

        # edge_index/edge_attr/wind_mean/wind_std are still accepted (to
        # match train.py's calling convention shared with every other
        # benchmark model) but no longer used - graph augmentation via
        # GraphGNN was removed per request. If graph awareness is wanted
        # again later, PM25_GNN.GraphGNN is still available to reuse.
        conv_in_channels = in_dim
        self.conv_in_channels = conv_in_channels

        # Load real geographic dartboard sector assignment/mask BEFORE
        # building blocks, since DS_MSA needs them at construction time
        # (assignment.shape[-1] determines num_sectors). Previously this
        # only loaded when NOT using the (now-removed) learned adaptive
        # attention; now it's simply whenever spatial_flag=True.
        self.get_dartboard_info(dartboard)

        # --- spatial-temporal blocks ---
        self.residual_convs = nn.ModuleList()
        self.s_modules = nn.ModuleList()
        self.t_modules = nn.ModuleList()
        self.mlp_modules = nn.ModuleList()
        self.bn = nn.ModuleList()

        self.start_conv = nn.Conv2d(
            in_channels=conv_in_channels,
            out_channels=hidden_channels,
            kernel_size=(1, 1)
        )

        # DS_MSA is only usable if the real dartboard assignment/mask were
        # actually loaded (spatial_flag=True); otherwise fall back to a
        # plain residual conv.
        self._use_dartboard_path = self.spatial_flag and self.assignment is not None

        if self.use_adaptive_temporal and hist_len < 2 ** max(blocks - 1, 0):
            warnings.warn(
                f"hist_len={hist_len} is smaller than 2**(blocks-1)={2 ** max(blocks - 1, 0)} "
                f"(blocks={blocks}). Several SeqScaleTemporalMixer blocks will collapse to "
                f"window_size=1, losing the intended multi-scale hierarchy. Reduce `blocks` or "
                f"increase `hist_len` if that hierarchy matters for your experiment."
            )

        mlp_hidden = max(1, int(hidden_channels * mlp_expansion))

        for b in range(blocks):
            window_size = max(1, self.seq_len // 2 ** (blocks - b - 1))

            if self.use_adaptive_temporal:
                # Causal trend/seasonal linear mixer (from the user-supplied
                # SeqScale model), replacing AdaptiveTemporalAttention.
                # Same window_size role (bounds this block's receptive field
                # along T) and same [B,C,N,T]->[B,C,N,T] contract, so no
                # other changes are needed at the call site below. Unlike
                # AdaptiveTemporalAttention this is explicitly causal, which
                # the d_shift generative/inference split below relies on -
                # see SeqScaleTemporalMixer's docstring.
                self.t_modules.append(SeqScaleTemporalMixer(
                    dim=hidden_channels,
                    window_size=window_size,
                    kernel_sizes=self.temporal_kernel_sizes,
                ))

            if self._use_dartboard_path:
                # Real geographic dartboard sector attention, replacing the
                # learned-from-scratch AdaptiveSectorMSA. Uses the actual
                # assignment/mask loaded by get_dartboard_info() above -
                # these were previously loaded but never used anywhere.
                # depth=1: this file's own `blocks` loop already provides
                # the outer depth: DS_MSA's own `depth` param would stack
                # multiple spatial-attention+FFN layers inside a single
                # block, which isn't needed here.
                # NOTE: DS_MSA manages its own residual connections AND its
                # own FeedForward internally (x = attn(x)+x; x = ff(x)+x) -
                # do NOT wrap it in an external `x + ...` at the call site
                # below, that would double the residual.
                self.s_modules.append(DS_MSA(
                    dim=hidden_channels,
                    depth=1,
                    heads=num_heads,
                    mlp_dim=mlp_hidden,
                    assignment=self.assignment,
                    mask=self.mask,
                    dropout=dropout,
                ))
            else:
                # NOTE: upstream had nn.Conv1d(hidden_channels, hidden_channels, (1, 1))
                # here, which is invalid - Conv1d takes a 1-tuple/int kernel size, and
                # this path receives the same [b, c, n, t] 4D tensor as every other conv
                # in this model. Fixed to Conv2d to match. This path (unlike DS_MSA) has
                # no internal residual, so the call site below adds one externally.
                self.residual_convs.append(
                    nn.Conv2d(hidden_channels, hidden_channels, (1, 1))
                )

            # MLP (feed-forward) sub-layer for AFTER the temporal mixer,
            # which (unlike DS_MSA) has no FFN of its own. mlp_expansion was
            # previously an accepted-but-unused constructor arg with nothing
            # wired to it.
            self.mlp_modules.append(nn.Sequential(
                nn.Conv2d(hidden_channels, mlp_hidden, kernel_size=(1, 1)),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv2d(mlp_hidden, hidden_channels, kernel_size=(1, 1)),
                nn.Dropout(dropout),
            ))

            self.bn.append(nn.BatchNorm2d(hidden_channels))

        # --- stochastic (hierarchical VAE) modeling ---
        self.total_latent_dim = 0
        if stochastic_flag:
            # stochastics_layers.py only implements HierarchicalStochasticModel -
            # there is no separate "flat" VAE class, so use_hierarchical_latent
            # =False has nothing real to fall back to. The earlier version of
            # this file tried to fake a flat model by reusing
            # HierarchicalStochasticModel with growth_factor=1.0, but that
            # still returns lists of per-level tensors, which then broke the
            # forward-pass code that assumed plain tensors in that branch.
            # Rather than keep a path that silently produces wrong shapes,
            # always use the real hierarchical implementation.
            if not use_hierarchical_latent:
                warnings.warn(
                    "use_hierarchical_latent=False was requested, but "
                    "stochastics_layers.py only provides HierarchicalStochasticModel "
                    "- there is no flat VAE to fall back to. Proceeding with the "
                    "hierarchical model regardless."
                )
            self.use_hierarchical_latent = True

            # HierarchicalKLLoss hardcodes 4-element kl_weights/decorr_weights
            # tensors ([dim2, dim4, dim8, dim16]) - it silently breaks (wrong
            # weight per level, or IndexError) for any blocks != 4.
            if blocks != 4:
                raise ValueError(
                    f"blocks={blocks}, but HierarchicalKLLoss's per-level "
                    f"kl_weights/decorr_weights are hardcoded for exactly 4 "
                    f"hierarchy levels. Either set blocks=4, or extend "
                    f"HierarchicalKLLoss in stochastics_layers.py to size its "
                    f"weight tensors from num_blocks before using blocks={blocks}."
                )

            self.generative_model = HierarchicalStochasticModel(
                dm_dim=hidden_channels,
                base_latent_dim=base_latent_dim,
                num_blocks=blocks,
                growth_factor=latent_growth_factor
            )
            self.inference_model = HierarchicalStochasticModel(
                dm_dim=hidden_channels,
                base_latent_dim=base_latent_dim,
                num_blocks=blocks,
                growth_factor=latent_growth_factor
            )
            self.total_latent_dim = sum([
                int(base_latent_dim * (latent_growth_factor ** i))
                for i in range(blocks)
            ])
            self.kl_loss_fn = HierarchicalKLLoss(num_blocks=blocks)

            self.reconstruction_model = nn.Sequential(
                nn.Conv2d(hidden_channels * blocks, end_channels, (1, 1), bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(end_channels, in_dim, (1, 1), bias=True)  # reconstruct raw features, not GNN-augmented ones
            )

        # --- decoder / fusion head ---
        if self.stochastic_flag:
            input_to_decoder = hidden_channels * blocks + self.total_latent_dim + 4
        else:
            input_to_decoder = hidden_channels * blocks

        self.end_conv_1 = nn.Conv2d(
            in_channels=input_to_decoder,
            out_channels=end_channels,
            kernel_size=(1, 1),
            bias=True
        )
        self.end_conv_2 = nn.Conv2d(
            in_channels=end_channels,
            out_channels=self.pred_len * self.output_dim,
            kernel_size=(1, 1),
            bias=True
        )

        # populated on every forward() call for optional external use
        self.last_reconstruction = None
        self.last_kl_loss = None
        self.last_kl_dict = None

    def get_dartboard_info(self, dartboard):
        """Load the real geographic dartboard sector assignment/mask
        whenever spatial modeling is requested at all. Falls back to the
        plain residual-conv spatial path (see _use_dartboard_path in
        __init__) if those files aren't present on disk, rather than
        raising and breaking train.py's model construction - DS_MSA can't
        run without them, but the rest of the pipeline still can."""
        if self.spatial_flag:
            dartboard_map = {0: '50-200', 1: '50-200-500', 2: '50', 3: '25-100-250'}
            path_assignment = f'data/local_partition/{dartboard_map[dartboard]}/assignment.npy'
            path_mask = f'data/local_partition/{dartboard_map[dartboard]}/mask.npy'
            if os.path.exists(path_assignment) and os.path.exists(path_mask):
                self.assignment = torch.from_numpy(np.load(path_assignment)).float().to(self.device)
                self.mask = torch.from_numpy(np.load(path_mask)).bool().to(self.device)
            else:
                warnings.warn(
                    f"Dartboard assignment/mask files not found at '{path_assignment}' / "
                    f"'{path_mask}'. DS_MSA (real geographic spatial attention) needs these "
                    f"to run - falling back to the plain residual-conv spatial path instead. "
                    f"Generate the dartboard partition files to use DS_MSA, or pass "
                    f"spatial_flag=False to silence this warning."
                )
                self.assignment = None
                self.mask = None
        else:
            self.assignment = None
            self.mask = None

    def forward(self, pm25_hist, feature):  # type: ignore[override]
        """
        pm25_hist : [B, hist_len, N, 1]
        feature   : [B, hist_len + pred_len, N, F]
        returns   : [B, pred_len, N, 1]
        """
        feature_hist = feature[:, :self.hist_len]                # [B, hist_len, N, F]
        inputs = torch.cat([pm25_hist, feature_hist], dim=-1)    # [B, hist_len, N, in_dim]

        B, T, N, C = inputs.shape
        if N != self.num_nodes:
            raise ValueError(
                f"AirFormerPlusPlus was built with city_num={self.num_nodes}, but got "
                f"N={N} nodes in this batch's data. Check that the Graph()/HazeData node "
                f"ordering matches what was passed to the constructor."
            )

        x = inputs  # no graph augmentation (removed) - raw features go straight into start_conv

        # [b, t, n, c] -> [b, c, n, t]
        x = x.permute(0, 3, 2, 1)
        x = self.start_conv(x)

        d = []
        for i in range(self.blocks):
            # spatial sub-layer: DS_MSA manages its own residual (and its own
            # FFN) internally, so it's called directly, NOT wrapped in an
            # external `x + ...` (that would double the residual). The plain
            # conv fallback has no internal residual, so it still needs one.
            if self._use_dartboard_path:
                x = self.s_modules[i](x)
            else:
                x = x + self.residual_convs[i](x)

            # temporal sub-layer, residual-wrapped
            if self.use_adaptive_temporal:
                x = x + self.t_modules[i](x)

            # MLP (feed-forward) sub-layer, residual-wrapped - this is what was
            # missing: mlp_expansion previously had no layer attached to it
            x = x + self.mlp_modules[i](x)

            x = self.bn[i](x)
            d.append(x)
        d = torch.stack(d)  # [num_blocks, b, c, n, t]

        if self.stochastic_flag:
            d_shift = torch.stack([
                F.pad(d[i], pad=(1, 0))[..., :-1] for i in range(len(d))
            ])

            # d[-1] = block with the largest attention window (built with
            # window_size growing in b), i.e. the block that sees the most
            # history -> that's the "coarse / overall trend" level.
            # HierarchicalStochasticModel.forward expects exactly that: it
            # consumes d[-1] first as the unconditioned top level, then
            # walks down to d[0] (smallest window -> finest detail) as the
            # bottom level. This lines up correctly with block construction
            # order above without needing any re-indexing here.
            z_p, mu_p, sigma_p = self.generative_model(d_shift)
            z_q, mu_q, sigma_q = self.inference_model(d)

            kl_loss, kl_dict = self.kl_loss_fn(mu_q, sigma_q, mu_p, sigma_p, z_q)

            # Standard VAE convention: sample via reparameterization during
            # training (needed for the gradient signal), use the posterior
            # mean deterministically at eval time so point predictions are
            # stable and reproducible rather than re-randomized every call.
            z_for_pred = z_q if self.training else mu_q

            reconstruction_input = torch.cat([d[i] for i in range(d.shape[0])], dim=1)
            expected_in = self.reconstruction_model[0].in_channels
            if reconstruction_input.shape[1] != expected_in:
                if reconstruction_input.shape[1] < expected_in:
                    pad = torch.zeros(
                        reconstruction_input.shape[0],
                        expected_in - reconstruction_input.shape[1],
                        *reconstruction_input.shape[2:],
                        device=reconstruction_input.device,
                        dtype=reconstruction_input.dtype,
                    )
                    reconstruction_input = torch.cat([reconstruction_input, pad], dim=1)
                else:
                    reconstruction_input = reconstruction_input[:, :expected_in]
            x_rec = self.reconstruction_model(reconstruction_input)
            x_rec = x_rec.permute(0, 3, 2, 1)  # back to [b, t, n, c]

            num_blocks, B, C, N, T = d.shape
            d_flat = d.permute(1, 0, 2, 3, 4).reshape(B, -1, N, T)

            z_q_cat = torch.cat([z_for_pred[i] for i in range(num_blocks)], dim=1)

            # Multi-scale temporal fusion, scaled to hist_len rather than
            # fixed hour counts (the original 1h/6h/12h/24h assumed long
            # ~72h AirFormer windows; this dataset's hist_len is often much
            # shorter, e.g. 1/6/12/24 total, so fixed windows would collapse
            # into duplicates or silently cover the whole window).
            T_total = d_flat.shape[-1]
            w_short = max(1, T_total // 8)
            w_mid = max(1, T_total // 4)
            w_long = max(1, T_total // 2)
            d_1h = d_flat[..., -1:].mean(dim=1, keepdim=True)
            d_6h = d_flat[..., -w_short:].mean(dim=3, keepdim=True).mean(dim=1, keepdim=True)
            d_12h = d_flat[..., -w_mid:].mean(dim=3, keepdim=True).mean(dim=1, keepdim=True)
            d_24h = d_flat[..., -w_long:].mean(dim=3, keepdim=True).mean(dim=1, keepdim=True)

            x_hat = torch.cat([
                d_flat[..., -1:], z_q_cat[..., -1:], d_1h, d_6h, d_12h, d_24h
            ], dim=1)

            x_hat = F.relu(self.end_conv_1(x_hat))
            x_hat = self.end_conv_2(x_hat)  # [B, pred_len*output_dim, N, 1]

            self.last_reconstruction = x_rec
            self.last_kl_loss = kl_loss
            self.last_kl_dict = kl_dict
        else:
            num_blocks, B, C, N, T = d.shape
            d_flat = d.permute(1, 0, 2, 3, 4).reshape(B, -1, N, T)
            x_hat = F.relu(self.end_conv_1(d_flat[..., -1:]))
            x_hat = self.end_conv_2(x_hat)
            self.last_reconstruction = None
            self.last_kl_loss = None
            self.last_kl_dict = None

        # output_dim == 1, so [B, pred_len*output_dim, N, 1] IS [B, pred_len, N, 1] -
        # no reshuffling needed. If output_dim is ever changed, this needs revisiting.
        pm25_pred = x_hat
        return pm25_pred