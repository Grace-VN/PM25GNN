"""
AirFormer (Liang et al., AAAI 2023, "AirFormer: Predicting Nationwide Air
Quality in China with Transformers", https://github.com/yoshall/AirFormer)
adapted as a benchmark model for this repo's harness.

Rebuilt fresh (replaces the earlier model/airformerplusplus.py, which was
a heavily modified downstream variant - SeqScale-derived temporal mixer,
learned-clustering spatial attention experiments, etc. - not a faithful
port of the paper). This file targets the paper's own three mechanisms:

  1. DS-MSA (Dartboard Spatial Multi-head Self-Attention): each station
     attends to its geographic neighborhood, partitioned into "dartboard"
     sectors (concentric distance rings x compass-direction wedges, plus
     a self sector). Reused UNCHANGED from layers/dartboard_spatial_attention.py
     (`SpatialAttention`/`DS_MSA`) - that file already is a faithful,
     working port of this exact mechanism (confirmed against the paper:
     its default num_sectors=17 = 1 self + 2 rings x 8 compass wedges,
     matching the paper's "50-200" dartboard scheme).
  2. CT-MSA (Causal Temporal Multi-head Self-Attention): windowed, causal
     self-attention over the time axis, with window size shrinking at
     earlier blocks and growing to the full sequence at the deepest block
     (window_size = seq_len / 2**(blocks - block_idx - 1)) - finer
     resolution early, coarser/more global context late. Implemented here
     as `_CausalTemporalMSA` (a plain windowed+causal multi-head
     attention, reusing the same PreNorm/FeedForward wrapper as DS_MSA).
     NOT reusing this repo's own layers/temporal_attention.py
     (AdaptiveTemporalAttention) - that module adds repo-local
     experiments (a learnable window-size multiplier, a soft/temperature
     causal mask) on top of whatever it started from, and isn't a
     faithful stand-in for the paper's plain hard-causal windowed
     attention.
  3. Hierarchical stochastic latent variables: a coarse-to-fine ladder of
     latent variables (dims [2,4,8,16] by default) with paired
     generative (prior, conditioned on causally-shifted deterministic
     states) and inference (posterior, conditioned on the current
     states) models, trained via KL divergence - captures the
     uncertainty/variability the paper highlights around extreme
     pollution events. Reused UNCHANGED from layers/stochastics_layers.py
     (`HierarchicalStochasticModel`, `HierarchicalKLLoss`) - already a
     complete, self-contained implementation of this exact mechanism.

WHAT HAD TO CHANGE, AND WHY
-----------------------------
  1. Contract. `Model(hist_len, pred_len, in_dim, city_num, batch_size,
     device, edge_index, edge_attr, wind_mean, wind_std, ...)` called as
     `model(pm25_hist, feature) -> [B, pred_len, N, 1]`, same as every
     other model here. `edge_index`/`edge_attr`/`wind_mean`/`wind_std` are
     accepted for signature parity with the rest of train.py's
     get_model() but unused - AirFormer's spatial structure comes from
     the dartboard partition, not the wind-weighted graph PM25_GNN uses.
     A new `station_coords` argument (same [N,2] lat/lon tensor
     model/probgru2.py+ already build in train.py) is required instead,
     to build the dartboard partition.
  2. Dartboard partition, computed instead of loaded. Upstream ships
     precomputed assignment/mask .npy files (see the warning
     model/airformerplusplus.py used to print when they were missing).
     This repo doesn't have those files, so `_build_dartboard()` computes
     the same kind of partition directly from `station_coords` at model
     construction time - each station's neighbors within `ring_km_thresholds`
     (default (50, 200), matching the paper's own "50-200" scheme) are
     bucketed into `num_angle_bins` (default 8) compass wedges per ring,
     plus a self sector, so num_sectors defaults to 1 + 2*8 = 17 - the
     same default DS_MSA/AdaptiveSectorMSA already use elsewhere in this
     repo. A station with no neighbor in a given sector has that sector
     masked out of its attention (always safe: the self sector is never
     empty).
  3. Input embedding. Upstream's `AirEmbedding` embeds categorical
     channels (wind-direction bucket, weather-code, hour-of-day,
     day-of-week) specific to its own China AQ dataset's discretized
     features. KnowAir's `feature` tensor is continuous/z-normalized
     throughout (see dataset.py), so there's nothing categorical to embed
     - dropped, same conclusion model/airformerplusplus.py's docstring
     already reached ("confirmed dataset-mismatched"). A plain 1x1 conv
     projects [pm25, weather] to `hidden_channels` instead.
  4. KL weighting. Upstream fixes its KL loss coefficient at alpha=10
     inside the training script. This harness already has a generic hook
     for that - train.py adds `kl_weight * model.last_kl_loss` to the
     training loss whenever a model exposes `last_kl_loss` (see
     AirFormerPlusPlus's own use of the same hook, and train.py's
     `kl_weight` config default of 0.01) - so the raw (unweighted) KL is
     exposed via `self.last_kl_loss` and left to that existing,
     externally-configurable mechanism instead of a second hardcoded
     constant.
  5. No reconstruction/auxiliary decoder. Upstream's stochastic path also
     reconstructs the encoder's input from the prior samples, as an
     auxiliary training loss, via its own dedicated trainer script. This
     harness's train.py only knows how to add one MSE forecast loss plus
     one optional KL term (see point 4) - there's no hook for a second
     reconstruction loss, and adding a whole extra input-dimensional
     decoder head that nothing would ever train against isn't worth the
     complexity. Only the forecast head is implemented.
  6. Output head. Upstream's decoder takes the FINAL timestep's combined
     [all blocks' deterministic states ++ posterior latent samples]
     representation and projects it (1x1 conv -> ReLU -> 1x1 conv) to
     `pred_len * output_dim` channels, then reshapes into the multi-
     horizon forecast - kept as-is, just reshaped into this repo's
     [B, pred_len, N, 1] instead of upstream's own layout.

Contract (matches every other model in model/, see train.py get_model()):
    AirFormerPM25(hist_len, pred_len, in_dim, city_num, batch_size, device,
                  edge_index, edge_attr, wind_mean, wind_std, station_coords, ...)
    pm25_pred = model(pm25_hist, feature)   # -> [B, pred_len, N, 1]
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.dartboard_spatial_attention import DS_MSA, PreNorm, FeedForward
from layers.stochastics_layers import HierarchicalStochasticModel, HierarchicalKLLoss


def _build_dartboard(station_coords, ring_km_thresholds=(50.0, 200.0), num_angle_bins=8):
    """station_coords: [N,2] (lat, lon) in degrees.
    Returns (assignment [N,N,num_sectors] float32, mask [N,num_sectors] bool)
    - see module docstring point 2. Sector 0 is always "self"; sectors
    1..num_rings*num_angle_bins are (ring, compass-wedge) bins, in ring-major
    order. mask[i,s]=True means sector s is empty for station i (no station
    falls in it) and should be excluded from that station's attention."""
    coords = station_coords.detach().cpu().numpy() if torch.is_tensor(station_coords) else np.asarray(station_coords)
    lat, lon = coords[:, 0], coords[:, 1]
    N = len(lat)
    R = 6371.0
    lat_r, lon_r = np.deg2rad(lat), np.deg2rad(lon)
    dlat = lat_r[:, None] - lat_r[None, :]
    dlon = lon_r[None, :] - lon_r[:, None]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat_r[:, None]) * np.cos(lat_r[None, :]) * np.sin(dlon / 2) ** 2
    dist_km = 2 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))  # [N,N], i (row) -> j (col)

    y = np.sin(dlon) * np.cos(lat_r[None, :])
    x = np.cos(lat_r[:, None]) * np.sin(lat_r[None, :]) - np.sin(lat_r[:, None]) * np.cos(lat_r[None, :]) * np.cos(dlon)
    bearing_deg = (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0  # [N,N], compass bearing i -> j
    angle_bin = np.minimum((bearing_deg / (360.0 / num_angle_bins)).astype(np.int64), num_angle_bins - 1)

    num_rings = len(ring_km_thresholds)
    num_sectors = 1 + num_rings * num_angle_bins
    assignment = np.zeros((N, N, num_sectors), dtype=np.float32)
    np.fill_diagonal(assignment[:, :, 0], 1.0)  # self sector

    for i in range(N):
        for j in range(N):
            if j == i:
                continue
            ring = next((r for r, thresh in enumerate(ring_km_thresholds) if dist_km[i, j] <= thresh), None)
            if ring is None:
                continue  # beyond the outermost ring - no sector for this pair
            sector = 1 + ring * num_angle_bins + angle_bin[i, j]
            assignment[i, j, sector] = 1.0

    mask = assignment.sum(axis=1) == 0  # [N, num_sectors]
    return torch.from_numpy(assignment), torch.from_numpy(mask)


class _WindowedCausalAttention(nn.Module):
    """Plain multi-head self-attention, restricted to non-overlapping
    windows of the time axis, causally masked within each window (see
    module docstring point 2 for why this - not the repo's own
    AdaptiveTemporalAttention - is what's used here)."""

    def __init__(self, dim, heads, window_size, dropout=0.):
        super(_WindowedCausalAttention, self).__init__()
        assert dim % heads == 0, f"dim {dim} should be divisible by heads {heads}"
        self.heads = heads
        self.window_size = max(1, window_size)
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B', T, C]
        Bp, T, C = x.shape
        w = min(self.window_size, T)
        pad = (w - T % w) % w
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        Tp = x.shape[1]
        xw = x.reshape(-1, w, C)  # [B'*Tp/w, w, C]

        qkv = self.qkv(xw).reshape(xw.shape[0], w, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        causal = torch.tril(torch.ones(w, w, device=x.device, dtype=torch.bool))
        attn = attn.masked_fill(~causal, float('-inf'))
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(-1, w, C)
        out = out.reshape(Bp, Tp, C)[:, :T]
        out = self.proj(out)
        return self.drop(out)


class _CausalTemporalMSA(nn.Module):
    """CT-MSA: windowed causal self-attention + FFN, PreNorm-wrapped,
    operating per-station over the time axis. forward: [B,C,N,T]->[B,C,N,T]."""

    def __init__(self, dim, heads, mlp_dim, window_size, depth=1, dropout=0.):
        super(_CausalTemporalMSA, self).__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                PreNorm(dim, _WindowedCausalAttention(dim, heads=heads, window_size=window_size, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
            ]) for _ in range(depth)
        ])

    def forward(self, x):
        b, c, n, t = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b * n, t, c)  # [B*N, T, C]
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        x = x.reshape(b, n, t, c).permute(0, 3, 1, 2)  # [B, C, N, T]
        return x


class _AirFormerBlock(nn.Module):
    """One encoder block: spatial sub-layer (DS-MSA, or a residual conv
    fallback when spatial_flag=False) -> temporal sub-layer (CT-MSA,
    block-level residual) -> BatchNorm2d over the channel dim."""

    def __init__(self, dim, heads, mlp_dim, window_size, assignment, mask,
                 spatial_flag=True, depth=1, dropout=0.):
        super(_AirFormerBlock, self).__init__()
        self.spatial_flag = spatial_flag
        if spatial_flag:
            self.spatial = DS_MSA(dim, depth=depth, heads=heads, mlp_dim=mlp_dim,
                                   assignment=assignment, mask=mask, dropout=dropout)
        else:
            self.spatial = nn.Conv2d(dim, dim, kernel_size=1)
        self.temporal = _CausalTemporalMSA(dim, heads=heads, mlp_dim=mlp_dim,
                                            window_size=window_size, depth=depth, dropout=dropout)
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x):
        # x: [B, C, N, T]
        if self.spatial_flag:
            x = self.spatial(x)  # DS_MSA already applies its own residual internally
        else:
            x = x + self.spatial(x)
        x = x + self.temporal(x)
        x = self.norm(x)
        return x


class AirFormerPM25(nn.Module):
    def __init__(self, hist_len, pred_len, in_dim, city_num, batch_size, device,
                 edge_index, edge_attr, wind_mean, wind_std, station_coords,
                 hidden_channels=32, end_channels=512, blocks=4,
                 num_heads=2, mlp_expansion=2, depth=1, dropout=0.3,
                 spatial_flag=True, stochastic_flag=True,
                 ring_km_thresholds=(50.0, 200.0), num_angle_bins=8,
                 latent_base_dim=2, latent_growth=2.0, latent_num_layers=2):
        super(AirFormerPM25, self).__init__()
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.in_dim = in_dim
        self.city_num = city_num
        self.device = device
        self.batch_size = batch_size
        self.hidden_channels = hidden_channels
        self.num_blocks = blocks
        self.stochastic_flag = stochastic_flag
        self.output_dim = 1
        self.last_kl_loss = None

        mlp_dim = hidden_channels * mlp_expansion

        if spatial_flag:
            assignment, mask = _build_dartboard(station_coords, ring_km_thresholds, num_angle_bins)
        else:
            assignment, mask = None, None

        self.input_proj = nn.Conv2d(in_dim, hidden_channels, kernel_size=1)

        self.blocks = nn.ModuleList()
        for i in range(blocks):
            window_size = max(1, round(hist_len / (2 ** (blocks - i - 1))))
            self.blocks.append(_AirFormerBlock(
                hidden_channels, heads=num_heads, mlp_dim=mlp_dim, window_size=window_size,
                assignment=assignment, mask=mask, spatial_flag=spatial_flag,
                depth=depth, dropout=dropout,
            ))

        if stochastic_flag:
            self.generative = HierarchicalStochasticModel(
                hidden_channels, base_latent_dim=latent_base_dim, num_blocks=blocks,
                growth_factor=latent_growth, num_layers_per_block=latent_num_layers,
            )
            self.inference = HierarchicalStochasticModel(
                hidden_channels, base_latent_dim=latent_base_dim, num_blocks=blocks,
                growth_factor=latent_growth, num_layers_per_block=latent_num_layers,
            )
            self.kl_loss_fn = HierarchicalKLLoss(num_blocks=blocks)
            latent_total = sum(self.generative.latent_dims)
        else:
            self.generative = self.inference = self.kl_loss_fn = None
            latent_total = 0

        decoder_in = blocks * hidden_channels + latent_total
        self.end_conv1 = nn.Conv2d(decoder_in, end_channels, kernel_size=1)
        self.end_conv2 = nn.Conv2d(end_channels, pred_len * self.output_dim, kernel_size=1)

    def forward(self, pm25_hist, feature):
        """
        pm25_hist : [B, hist_len, N, 1]
        feature   : [B, hist_len + pred_len, N, F]   (F = in_dim - 1)
        returns   : [B, pred_len, N, 1]
        """
        B, T, N, _ = pm25_hist.shape
        if N != self.city_num:
            raise ValueError(
                f"AirFormerPM25 was built with city_num={self.city_num}, but got "
                f"N={N} nodes in this batch's data."
            )
        feature_hist = feature[:, :self.hist_len]
        x_hist = torch.cat([pm25_hist, feature_hist], dim=-1)   # [B,hist_len,N,in_dim]
        x = x_hist.permute(0, 3, 2, 1)                           # [B,in_dim,N,hist_len]

        x = self.input_proj(x)  # [B,hidden_channels,N,hist_len]

        d = []
        for block in self.blocks:
            x = block(x)
            d.append(x)
        d = torch.stack(d, dim=0)  # [blocks,B,C,N,T] - d[-1] is the deepest/coarsest block

        d_flat = d.permute(1, 0, 2, 3, 4).reshape(B, self.num_blocks * self.hidden_channels, N, self.hist_len)

        if self.stochastic_flag:
            d_shift = torch.cat([torch.zeros_like(d[..., :1]), d[..., :-1]], dim=-1)
            z_p, mu_p, sigma_p = self.generative(d_shift)
            z_q, mu_q, sigma_q = self.inference(d)
            kl, _ = self.kl_loss_fn(mu_q, sigma_q, mu_p, sigma_p, z_q)
            self.last_kl_loss = kl
            z_q_cat = torch.cat(z_q, dim=1)  # [B, sum(latent_dims), N, T]
            combined = torch.cat([d_flat, z_q_cat], dim=1)
        else:
            self.last_kl_loss = None
            combined = d_flat

        feat = combined[..., -1:]              # [B, decoder_in, N, 1] - final timestep only
        out = F.relu(self.end_conv1(feat))
        out = self.end_conv2(out)               # [B, pred_len*output_dim, N, 1]
        out = out.reshape(B, self.pred_len, self.output_dim, N)
        pm25_pred = out.permute(0, 1, 3, 2)     # [B, pred_len, N, output_dim=1]
        return pm25_pred
