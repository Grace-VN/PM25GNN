"""
PatchTST (Nie et al., ICLR 2023, "A Time Series is Worth 64 Words: Long-Term
Forecasting with Transformers", https://github.com/yuqinie98/PatchTST)
adapted as a plain, non-graph benchmark model for this repo's harness.

Faithful to the paper's two headline ideas:
  1. Channel independence: every scalar channel (here: PM2.5 plus every
     weather variable, at every station) is forecast by ONE SHARED
     transformer backbone, run independently per channel - channels never
     attend to each other. Combined with folding the city dimension into
     batch (same convention as this repo's LSTM/GRU/MLP/InformerPM25), the
     effective batch for the backbone is B * city_num * in_dim.
  2. Patching: each channel's hist_len lookback window is split into
     overlapping patches (patch_len, stride) instead of feeding one token
     per timestep - shorter attention sequences, more context per token.
  Also includes RevIN (instance normalization per channel, denormalized
  back at the output) from the same lineage of work, which the official
  PatchTST implementation uses by default and which measurably helps.

Encoder blocks (EncoderLayer, bidirectional FullAttention) are reused from
model/encoder.py and model/attn.py, already used by model/informer.py.

IMPORTANT LIMITATION (inherent to PatchTST, not a bug): unlike
InformerPM25, PatchTST has no decoder and is purely autoregressive from
the lookback window - it does NOT consume the future known weather this
harness makes available (feature[:, hist_len:]). That future weather is
simply unused here, exactly as in the paper's own multivariate-forecast
setup (predict all channels from their own histories only). If you want
a transformer baseline that exploits future weather, use InformerPM25.

Contract (matches every other model in model/, see train.py get_model()):
    PatchTSTPM25(hist_len, pred_len, in_dim, city_num, batch_size, device, ...)
    pm25_pred = model(pm25_hist, feature)
    # pm25_hist: [B, hist_len, N, 1], feature: [B, hist_len+pred_len, N, F]
    # -> pm25_pred: [B, pred_len, N, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.encoder import Encoder, EncoderLayer
from model.attn import FullAttention, AttentionLayer


class _RevIN(nn.Module):
    """Reversible instance normalization (Kim et al. 2022), as used by the
    official PatchTST implementation: normalize each channel of each
    instance over the time dimension before the backbone, denormalize the
    forecast with the same per-instance statistics afterward."""

    def __init__(self, num_channels, eps=1e-5, affine=True):
        super(_RevIN, self).__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))
        self.mean = None
        self.stdev = None

    def normalize(self, x):
        # x: [B, L, C]
        self.mean = x.mean(dim=1, keepdim=True).detach()
        self.stdev = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
        x = (x - self.mean) / self.stdev
        if self.affine:
            x = x * self.weight + self.bias
        return x

    def denormalize(self, x):
        # x: [B, L, C]
        if self.affine:
            x = (x - self.bias) / (self.weight + self.eps * self.eps)
        x = x * self.stdev + self.mean
        return x


class _FlattenHead(nn.Module):
    def __init__(self, patch_num, d_model, pred_len, dropout=0.1):
        super(_FlattenHead, self).__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(patch_num * d_model, pred_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [.., patch_num, d_model] -> [.., pred_len]
        x = self.flatten(x)
        x = self.linear(x)
        return self.dropout(x)


class PatchTSTPM25(nn.Module):
    def __init__(self, hist_len, pred_len, in_dim, city_num, batch_size, device,
                 target_channel=0, patch_len=8, stride=4,
                 d_model=32, n_heads=4, e_layers=2, d_ff=128,
                 dropout=0.1, head_dropout=0.1, activation='gelu', revin=True):
        super(PatchTSTPM25, self).__init__()
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.in_dim = in_dim
        self.city_num = city_num
        self.device = device
        self.batch_size = batch_size
        self.target_channel = target_channel
        self.patch_len = patch_len
        self.stride = stride
        self.use_revin = revin

        # pad the end by `stride` (replication) so the last patch fully
        # covers the tail of the window - same trick the official
        # implementation uses (nn.ReplicationPad1d((0, stride))).
        padded_len = hist_len + stride
        self.patch_num = (padded_len - patch_len) // stride + 1

        self.revin = _RevIN(in_dim, affine=True) if revin else None
        self.patch_embedding = nn.Linear(patch_len, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.patch_num, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        self.embed_dropout = nn.Dropout(dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(FullAttention(False, attention_dropout=dropout,
                                                  output_attention=False),
                                   d_model, n_heads, mix=False),
                    d_model, d_ff, dropout=dropout, activation=activation,
                ) for _ in range(e_layers)
            ],
            conv_layers=None,
            norm_layer=nn.LayerNorm(d_model),
        )

        self.head = _FlattenHead(self.patch_num, d_model, pred_len, dropout=head_dropout)

    def forward(self, pm25_hist, feature):
        """
        pm25_hist : [B, hist_len, N, 1]
        feature   : [B, hist_len + pred_len, N, F]   (F = in_dim - 1)
        returns   : [B, pred_len, N, 1]
        """
        B, T, N, _ = pm25_hist.shape
        if N != self.city_num:
            raise ValueError(
                f"PatchTSTPM25 was built with city_num={self.city_num}, but got "
                f"N={N} nodes in this batch's data."
            )
        feature_hist = feature[:, :self.hist_len]
        x_hist = torch.cat([pm25_hist, feature_hist], dim=-1)  # [B,hist_len,N,in_dim]

        # fold city into batch (per-station, shared weights - same
        # convention as this repo's other baselines / InformerPM25)
        C = self.in_dim
        x = x_hist.permute(0, 2, 1, 3).reshape(B * N, self.hist_len, C)  # [B*N, L, C]

        if self.use_revin:
            x = self.revin.normalize(x)

        # -> [B*N, C, L] -> replication-pad the end -> unfold into patches
        x = x.permute(0, 2, 1)                                           # [B*N, C, L]
        x = F.pad(x, (0, self.stride), mode='replicate')                 # [B*N, C, L+stride]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # [B*N, C, patch_num, patch_len]

        BN, C_, P_num, P_len = x.shape
        x = x.reshape(BN * C_, P_num, P_len)                             # channel-independent: fold C into batch too

        x = self.patch_embedding(x) + self.pos_embedding                 # [B*N*C, patch_num, d_model]
        x = self.embed_dropout(x)
        x, _ = self.encoder(x, attn_mask=None)                           # [B*N*C, patch_num, d_model]

        out = self.head(x)                                               # [B*N*C, pred_len]
        out = out.view(BN, C_, self.pred_len).permute(0, 2, 1)           # [B*N, pred_len, C]

        if self.use_revin:
            out = self.revin.denormalize(out)

        out = out[..., self.target_channel:self.target_channel + 1]      # [B*N, pred_len, 1]
        pm25_pred = out.view(B, N, self.pred_len, 1).permute(0, 2, 1, 3)  # [B, pred_len, N, 1]
        return pm25_pred
