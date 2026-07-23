import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import graph


class GRUCell(nn.Module):

    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        x = x.view(-1, x.size(-1))

        gate_x = self.x2h(x)
        gate_h = self.h2h(hidden)

        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()

        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)

        resetgate = F.sigmoid(i_r + h_r)
        inputgate = F.sigmoid(i_i + h_i)
        newgate = F.tanh(i_n + (resetgate * h_n))

        hy = newgate + inputgate * (hidden - newgate)

        return hy


class LSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        hx, cx = hidden

        x = x.view(-1, x.size(-1))

        gates = self.x2h(x) + self.h2h(hx)

        gates = gates.squeeze()

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)

        cy = torch.mul(cx, forgetgate) + torch.mul(ingate, cellgate)

        hy = torch.mul(outgate, F.tanh(cy))

        return (hy, cy)

import torch
import torch.nn.functional as F
from torch import nn


class LatentLayer(nn.Module):
    """
    Maps a conditioning vector to (mu, sigma) of a diagonal Gaussian latent.
    Used for both the prior (causal) and inference (sees target) nets —
    same role as LatentLayer in airformer.py, just linear instead of conv.
    """
    def __init__(self, in_dim, latent_dim, hidden_dim=32, num_layers=2):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True)]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        self.net = nn.Sequential(*layers)
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logsigma_head = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = self.net(x)
        mu = torch.clamp(self.mu_head(h), -10, 10)
        logsigma = torch.clamp(self.logsigma_head(h), -10, 10)
        sigma = F.softplus(logsigma) + 1e-3  # numerically safer than exp() here
        return mu, sigma


def reparameterize(mu, sigma):
    eps = torch.randn_like(sigma)
    return mu + eps * sigma

"""
Optimized MCASALayer -- same math as the original, two speed fixes:

1. Geometry caching: coords/altitude are static buffers that never change
   during training, but the original recomputed haversine distance,
   direction vectors, and the altitude barrier matrix from scratch on
   EVERY forward call (twice per call, in fact -- once in _soft_mask,
   once in _edge_features). These are now computed once and cached,
   keyed on the tensor's memory address, so repeated calls with the same
   coords/altitude buffer skip straight to the sigmoid gating (which does
   depend on trainable temperature params and must stay per-call).

2. Optional top-k neighbor sparsification (`neighbor_topk`): the real
   FLOPs cost is edge_mlp running over all N^2 pairs every call, most of
   which the soft mask gates to ~0 anyway (cities >dist_threshold apart).
   Set neighbor_topk=k to only compute Q.K, edge features, and edge_mlp
   over each node's k nearest neighbors (by raw distance) instead of all
   N-1 others. Import aggregation is fully sparse (O(N*k)). The export
   term still needs one O(N^2) scatter+matmul (cheap -- no MLP, just a
   zero-fill and a matmul) since "how much do others pull from me" isn't
   naturally expressible over a query-centric top-k gather. Set
   neighbor_topk=None (default) to keep exact dense behaviour.

Verified numerically equivalent to the dense version when neighbor_topk
covers all nodes (see cells_fast_test.py).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MCASALayer(nn.Module):
    """
    Single-layer, multi-head, meteorology-conditioned spatial attention.
    See cells.py docstring for the input/output contract -- unchanged here.

    New constructor args
    ---------------------
    cache_geometry : bool
        Cache static distance/direction/barrier tensors keyed on the
        coords/altitude buffer identity. Safe to leave True always --
        invalidates automatically if you pass a different coords tensor.
        Only gotcha: call model.to(device) BEFORE the first forward pass,
        since the cache captures whatever device coords/altitude were on
        at cache time.
    neighbor_topk : int or None
        If set, restrict attention (and the expensive edge_mlp) to each
        node's k nearest neighbors by raw distance instead of all N-1
        others. None = exact dense behaviour (default, unchanged).
        Rule of thumb: with dist_threshold_km=300 on a dataset like
        KnowAir (184 cities spanning ~2000km), the soft mask already
        gates most far pairs near zero, so a modest k (20-40) usually
        loses very little while cutting edge_mlp cost by ~5-10x.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        dist_threshold_km: float = 300.0,
        alt_threshold_m: float = 1200.0,
        extra_feat_dim: int = 0,
        export_lambda_init: float = 0.5,
        dropout: float = 0.1,
        cache_geometry: bool = True,
        neighbor_topk: int | None = None,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.cache_geometry = cache_geometry
        self.neighbor_topk = neighbor_topk
        self._geom_cache = None  # populated lazily on first forward

        self.dist_threshold = nn.Parameter(torch.tensor(float(dist_threshold_km)))
        self.alt_threshold = nn.Parameter(torch.tensor(float(alt_threshold_m)))
        self.tau_d = nn.Parameter(torch.tensor(30.0))
        self.tau_m = nn.Parameter(torch.tensor(100.0))
        self.head_scale = nn.Parameter(torch.linspace(0.5, 2.0, n_heads))

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        edge_feat_dim = 1 + 1 + 2 + 1 + extra_feat_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_feat_dim, d_model // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(d_model // 2, n_heads),
        )

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.export_lambda = nn.Parameter(torch.tensor(export_lambda_init))

    @staticmethod
    def _haversine_km(coords: torch.Tensor) -> torch.Tensor:
        R = 6371.0
        lat = torch.deg2rad(coords[:, 0])
        lon = torch.deg2rad(coords[:, 1])
        dlat = lat[:, None] - lat[None, :]
        dlon = lon[:, None] - lon[None, :]
        a = torch.sin(dlat / 2) ** 2 + torch.cos(lat[:, None]) * torch.cos(lat[None, :]) * torch.sin(dlon / 2) ** 2
        return 2 * R * torch.asin(torch.sqrt(a.clamp(min=1e-12)))

    def _compute_geometry(self, coords: torch.Tensor, altitude: torch.Tensor) -> dict:
        with torch.no_grad():
            dist = self._haversine_km(coords)  # (N, N)
            lat = torch.deg2rad(coords[:, 0])
            lon = torch.deg2rad(coords[:, 1])
            dx = lon[None, :] - lon[:, None]
            dy = lat[None, :] - lat[:, None]
            norm = torch.sqrt(dx ** 2 + dy ** 2).clamp(min=1e-8)
            ux, uy = dx / norm, dy / norm
            barrier = torch.maximum(altitude[:, None], altitude[None, :])
            geom = {"dist": dist, "ux": ux, "uy": uy, "barrier": barrier}
            if self.neighbor_topk is not None:
                k = min(self.neighbor_topk, dist.shape[0])
                _, idx = torch.topk(-dist, k=k, dim=-1)  # (N, k) nearest by raw distance
                geom["idx"] = idx
        return geom

    def _get_geometry(self, coords: torch.Tensor, altitude: torch.Tensor) -> dict:
        key = (coords.data_ptr(), tuple(coords.shape), altitude.data_ptr())
        if self.cache_geometry and self._geom_cache is not None and self._geom_cache.get("key") == key:
            return self._geom_cache
        geom = self._compute_geometry(coords, altitude)
        geom["key"] = key
        if self.cache_geometry:
            self._geom_cache = geom
        return geom

    def _dense_forward(self, h, geom, wind, pbl, extra_node_feats):
        B, N, D = h.shape
        Hh, Dh = self.n_heads, self.d_head
        dist, ux, uy, barrier = geom["dist"], geom["ux"], geom["uy"], geom["barrier"]

        q = self.q_proj(h).view(B, N, Hh, Dh).transpose(1, 2)
        k = self.k_proj(h).view(B, N, Hh, Dh).transpose(1, 2)
        v = self.v_proj(h).view(B, N, Hh, Dh).transpose(1, 2)

        content_logits = torch.einsum("bhid,bhjd->bhij", q, k) / (Dh ** 0.5)

        wu, wv = wind[..., 0], wind[..., 1]
        wind_speed = torch.sqrt(wu ** 2 + wv ** 2 + 1e-8)
        wind_proj = wu[:, None, :] * ux[None, :, :] + wv[:, None, :] * uy[None, :, :]
        wind_speed_src = wind_speed[:, None, :].expand(B, N, N)
        pbl_src = pbl[..., 0][:, None, :].expand(B, N, N)
        pbl_sink = pbl[..., 0][:, :, None].expand(B, N, N)
        dist_b = dist[None, :, :].expand(B, N, N)

        feats = [wind_proj, wind_speed_src, pbl_src, pbl_sink, dist_b]
        feats = [f.unsqueeze(-1) for f in feats]
        if extra_node_feats is not None:
            ef = extra_node_feats[:, None, :, :].expand(B, N, N, extra_node_feats.shape[-1])
            feats.append(ef)
        edge_feats = torch.cat(feats, dim=-1)
        edge_bias = self.edge_mlp(edge_feats).permute(0, 3, 1, 2)

        gate_m = torch.sigmoid((self.alt_threshold - barrier) / self.tau_m.clamp(min=1.0))
        masks = []
        for hscale in self.head_scale:
            gd = torch.sigmoid((self.dist_threshold - dist) / (self.tau_d.clamp(min=1.0) * hscale))
            masks.append(gd * gate_m)
        soft_mask = torch.stack(masks, dim=0)
        log_mask = torch.log(soft_mask.clamp(min=1e-6)).unsqueeze(0)

        logits = content_logits + edge_bias + log_mask
        attn = F.softmax(logits, dim=-1)
        attn = self.dropout(attn)

        imported = torch.einsum("bhij,bhjd->bhid", attn, v)
        exported = torch.einsum("bhji,bhjd->bhid", attn, v)
        out = imported - self.export_lambda * exported
        return out, attn

    def _sparse_forward(self, h, geom, wind, pbl, extra_node_feats):
        B, N, D = h.shape
        Hh, Dh = self.n_heads, self.d_head
        idx = geom["idx"]  # (N, k)
        k_top = idx.shape[1]
        dist, ux, uy, barrier = geom["dist"], geom["ux"], geom["uy"], geom["barrier"]

        q = self.q_proj(h).view(B, N, Hh, Dh).transpose(1, 2)      # (B, Hh, N, Dh)
        k_ = self.k_proj(h).view(B, N, Hh, Dh).transpose(1, 2)
        v_ = self.v_proj(h).view(B, N, Hh, Dh).transpose(1, 2)

        k_gather = k_[:, :, idx, :]  # (B, Hh, N, k, Dh)
        v_gather = v_[:, :, idx, :]  # (B, Hh, N, k, Dh)
        content_logits = torch.einsum("bhnd,bhnkd->bhnk", q, k_gather) / (Dh ** 0.5)

        dist_g = torch.gather(dist, 1, idx)          # (N, k)
        barrier_g = torch.gather(barrier, 1, idx)     # (N, k)
        ux_g = torch.gather(ux, 1, idx)
        uy_g = torch.gather(uy, 1, idx)

        wu, wv = wind[..., 0], wind[..., 1]                       # (B, N)
        wind_speed = torch.sqrt(wu ** 2 + wv ** 2 + 1e-8)
        wu_g = wu[:, idx]                 # (B, N, k) -- neighbor's wind, gathered
        wv_g = wv[:, idx]
        wind_speed_g = wind_speed[:, idx]                          # (B, N, k)
        wind_proj_g = wu_g * ux_g[None] + wv_g * uy_g[None]         # (B, N, k)

        pbl_flat = pbl[..., 0]                                     # (B, N)
        pbl_src_g = pbl_flat[:, idx]                                # (B, N, k)
        pbl_sink_g = pbl_flat.unsqueeze(-1).expand(B, N, k_top)     # (B, N, k)
        dist_b_g = dist_g.unsqueeze(0).expand(B, N, k_top)

        feats = [wind_proj_g, wind_speed_g, pbl_src_g, pbl_sink_g, dist_b_g]
        feats = [f.unsqueeze(-1) for f in feats]
        if extra_node_feats is not None:
            ef_g = extra_node_feats[:, idx, :]   # (B, N, k, F)
            feats.append(ef_g)
        edge_feats = torch.cat(feats, dim=-1)     # (B, N, k, edge_feat_dim)
        edge_bias = self.edge_mlp(edge_feats).permute(0, 3, 1, 2)  # (B, Hh, N, k)

        gate_m_g = torch.sigmoid((self.alt_threshold - barrier_g) / self.tau_m.clamp(min=1.0))  # (N, k)
        masks = []
        for hscale in self.head_scale:
            gd = torch.sigmoid((self.dist_threshold - dist_g) / (self.tau_d.clamp(min=1.0) * hscale))
            masks.append(gd * gate_m_g)
        soft_mask_g = torch.stack(masks, dim=0)   # (Hh, N, k)
        log_mask = torch.log(soft_mask_g.clamp(min=1e-6)).unsqueeze(0)  # (1, Hh, N, k)

        logits = content_logits + edge_bias + log_mask
        attn = F.softmax(logits, dim=-1)   # (B, Hh, N, k)
        attn = self.dropout(attn)

        imported = torch.einsum("bhnk,bhnkd->bhnd", attn, v_gather)  # (B, Hh, N, Dh)

        # export needs the reverse direction (how much each node is pulled from
        # by others); scatter the sparse attn back into a zero-filled dense
        # buffer and reuse the same matmul as the dense path. No MLP here --
        # just a zero-fill + scatter + one matmul, cheap relative to edge_mlp.
        dense_attn = torch.zeros(B, Hh, N, N, device=h.device, dtype=attn.dtype)
        idx_expand = idx.view(1, 1, N, k_top).expand(B, Hh, N, k_top)
        dense_attn.scatter_(-1, idx_expand, attn)
        exported = torch.einsum("bhji,bhjd->bhid", dense_attn, v_)

        out = imported - self.export_lambda * exported
        return out, attn

    def forward(self, h, coords, altitude, wind, pbl, extra_node_feats=None):
        B, N, D = h.shape
        geom = self._get_geometry(coords, altitude)

        if self.neighbor_topk is not None:
            out, attn = self._sparse_forward(h, geom, wind, pbl, extra_node_feats)
        else:
            out, attn = self._dense_forward(h, geom, wind, pbl, extra_node_feats)

        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)
        h_out = self.norm(h + self.dropout(out))
        return h_out, attn

import torch

def scatter_add(src, index, dim, dim_size):
    """Pure-PyTorch replacement for torch_scatter.scatter_add.
    index is 1D and indexes along `dim`; broadcasts over other dims."""
    out_shape = list(src.shape)
    out_shape[dim] = dim_size
    out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
    index_shape = [1] * src.dim()
    index_shape[dim] = index.size(0)
    idx = index.view(index_shape).expand_as(src)
    return out.scatter_add_(dim, idx, src)
