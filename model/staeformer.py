"""
STAEformer (Liu et al., AAAI 2023, "Spatio-Temporal Adaptive Embedding
Makes Vanilla Transformer SOTA for Traffic Forecasting",
https://github.com/XDZhelheim/STAEformer) adapted as a benchmark model
for this repo's harness.

CORRECTION vs. how this model was pitched: STAEformer does NOT actually
consume a station graph (edge_index/edge_attr/wind), unlike GMAN or this
repo's PM25_GNN. Its whole point is that a plain transformer, given a
learned spatio-temporal "adaptive embedding" plus attention factored
into a temporal pass (per station, across time) and a spatial pass (per
timestep, across stations), matches or beats graph-based models WITHOUT
a predefined adjacency at all. So this file takes no edge_index - spatial
structure is learned purely from data via the spatial attention pass and
the adaptive embedding table, not supplied.

Architecture (faithful to the paper):
  1. input_proj: per-(station, timestep) linear embedding of the raw
     [pm25, weather...] feature vector.
  2. tod/dow embeddings: learned lookup tables keyed by time-of-day slot
     and day-of-week, recovered by de-normalizing dataset.py's hour/
     weekday channels (same de-normalize-via-feature_mean/std pattern
     already used by model/probgru5.py+ for its own hour/wind channels).
  3. adaptive_embedding: a learned nn.Parameter of shape
     [hist_len, city_num, adaptive_dim], broadcast over the batch -
     exactly the paper's data-driven, graph-free stand-in for spatial
     priors.
  4. e_layers x (temporal self-attention over time, per station via
     EncoderLayer with city folded into batch) -> (spatial self-attention
     over stations, per timestep via EncoderLayer with time folded into
     batch). Both reuse model/encoder.py's EncoderLayer + model/attn.py's
     FullAttention/AttentionLayer (bidirectional, no causal mask, no
     distillation) - same building blocks model/patchtst.py and
     model/informer.py already use.
  5. output_proj: flatten (hist_len * model_dim) per station -> Linear ->
     pred_len, i.e. direct multi-step regression (no autoregressive
     decoding), matching the paper's own output head.

LIMITATION (inherent to the paper's design, not a bug): like PatchTST,
this is a pure lookback-window regressor - it does not consume the
future known weather this harness exposes via feature[:, hist_len:].

Contract (matches every other model in model/, see train.py get_model()):
    STAEformerPM25(hist_len, pred_len, in_dim, city_num, batch_size, device,
                    feature_mean, feature_std, ...)
    pm25_pred = model(pm25_hist, feature)
    # pm25_hist: [B, hist_len, N, 1], feature: [B, hist_len+pred_len, N, F]
    # -> pm25_pred: [B, pred_len, N, 1]
"""

import torch
import torch.nn as nn

from model.encoder import EncoderLayer
from model.attn import FullAttention, AttentionLayer


class _STAEformerLayer(nn.Module):
    """One temporal-attention pass (per station, across time) followed by
    one spatial-attention pass (per timestep, across stations). Both are
    plain bidirectional EncoderLayers; only the reshape around them
    differs."""

    def __init__(self, model_dim, n_heads, d_ff, dropout, activation):
        super(_STAEformerLayer, self).__init__()
        self.temporal_layer = EncoderLayer(
            AttentionLayer(FullAttention(False, attention_dropout=dropout, output_attention=False),
                           model_dim, n_heads, mix=False),
            model_dim, d_ff, dropout=dropout, activation=activation,
        )
        self.spatial_layer = EncoderLayer(
            AttentionLayer(FullAttention(False, attention_dropout=dropout, output_attention=False),
                           model_dim, n_heads, mix=False),
            model_dim, d_ff, dropout=dropout, activation=activation,
        )

    def forward(self, x):
        # x: [B, T, N, D]
        B, T, N, D = x.shape

        x_t = x.permute(0, 2, 1, 3).reshape(B * N, T, D)   # fold station into batch, attend across time
        x_t, _ = self.temporal_layer(x_t, attn_mask=None)
        x = x_t.view(B, N, T, D).permute(0, 2, 1, 3)        # back to [B, T, N, D]

        x_s = x.reshape(B * T, N, D)                        # fold time into batch, attend across stations
        x_s, _ = self.spatial_layer(x_s, attn_mask=None)
        x = x_s.view(B, T, N, D)

        return x


class STAEformerPM25(nn.Module):
    def __init__(self, hist_len, pred_len, in_dim, city_num, batch_size, device,
                 feature_mean, feature_std,
                 input_embedding_dim=24, tod_embedding_dim=24, dow_embedding_dim=24,
                 adaptive_embedding_dim=80, n_heads=4, e_layers=3, d_ff=256,
                 dropout=0.1, activation='gelu', dt_hours=3.0,
                 time_mark_channels=None):
        super(STAEformerPM25, self).__init__()
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.in_dim = in_dim
        self.city_num = city_num
        self.device = device
        self.batch_size = batch_size
        self.dt_hours = dt_hours
        self.tod_steps = max(1, round(24.0 / dt_hours))

        self.feature_dim = in_dim - 1
        # Fixed tail layout dataset.py._process_feature always produces:
        # [...metero_use channels..., hour, weekday, speed_kmh, direc_deg]
        # (see model/probgru5.py's identical convention). This holds
        # regardless of which variables metero_use lists, since these four
        # are appended AFTER the metero_use subset is selected.
        if time_mark_channels is None:
            self.hour_idx = self.feature_dim - 4
            self.weekday_idx = self.feature_dim - 3
        else:
            self.hour_idx, self.weekday_idx = time_mark_channels

        feature_mean_t = torch.as_tensor(feature_mean, dtype=torch.float32)
        feature_std_t = torch.as_tensor(feature_std, dtype=torch.float32)
        assert feature_mean_t.shape[0] == self.feature_dim, (
            f"feature_mean has {feature_mean_t.shape[0]} entries but feature_dim "
            f"(in_dim - 1) is {self.feature_dim} - pass HazeData.feature_mean/"
            f"feature_std, computed over the same metero_use as this run."
        )
        self.register_buffer('feature_mean', feature_mean_t)
        self.register_buffer('feature_std', feature_std_t.clamp(min=1e-6))

        model_dim = input_embedding_dim + tod_embedding_dim + dow_embedding_dim + adaptive_embedding_dim
        self.model_dim = model_dim

        self.input_proj = nn.Linear(in_dim, input_embedding_dim)
        self.tod_embedding = nn.Embedding(self.tod_steps, tod_embedding_dim) if tod_embedding_dim > 0 else None
        self.dow_embedding = nn.Embedding(7, dow_embedding_dim) if dow_embedding_dim > 0 else None
        if adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.Parameter(torch.empty(hist_len, city_num, adaptive_embedding_dim))
            nn.init.xavier_uniform_(self.adaptive_embedding)
        else:
            self.adaptive_embedding = None

        self.input_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            _STAEformerLayer(model_dim, n_heads, d_ff, dropout, activation)
            for _ in range(e_layers)
        ])
        self.output_proj = nn.Linear(hist_len * model_dim, pred_len)

    def _tod_dow_idx(self, feature_hist):
        """De-normalize the hour/weekday channels back to raw values and
        turn them into embedding-table indices. Identical across stations
        at a given timestep (dataset.py broadcasts the same global clock
        to every node), so this is safe to compute densely per node too."""
        hour_z = feature_hist[..., self.hour_idx]
        weekday_z = feature_hist[..., self.weekday_idx]
        hour_raw = hour_z * self.feature_std[self.hour_idx] + self.feature_mean[self.hour_idx]
        weekday_raw = weekday_z * self.feature_std[self.weekday_idx] + self.feature_mean[self.weekday_idx]

        tod_idx = torch.div(hour_raw, self.dt_hours, rounding_mode='floor').round().long() % self.tod_steps
        dow_idx = (weekday_raw.round().long() - 1).clamp(0, 6)  # dataset uses isoweekday() -> 1..7
        return tod_idx, dow_idx

    def forward(self, pm25_hist, feature):
        """
        pm25_hist : [B, hist_len, N, 1]
        feature   : [B, hist_len + pred_len, N, F]   (F = in_dim - 1)
        returns   : [B, pred_len, N, 1]
        """
        B, T, N, _ = pm25_hist.shape
        if N != self.city_num:
            raise ValueError(
                f"STAEformerPM25 was built with city_num={self.city_num}, but got "
                f"N={N} nodes in this batch's data."
            )
        feature_hist = feature[:, :self.hist_len]
        x_hist = torch.cat([pm25_hist, feature_hist], dim=-1)  # [B,hist_len,N,in_dim]

        parts = [self.input_proj(x_hist)]  # [B,hist_len,N,input_embedding_dim]

        if self.tod_embedding is not None or self.dow_embedding is not None:
            tod_idx, dow_idx = self._tod_dow_idx(feature_hist)
            if self.tod_embedding is not None:
                parts.append(self.tod_embedding(tod_idx))
            if self.dow_embedding is not None:
                parts.append(self.dow_embedding(dow_idx))

        if self.adaptive_embedding is not None:
            parts.append(self.adaptive_embedding.unsqueeze(0).expand(B, -1, -1, -1))

        x = torch.cat(parts, dim=-1)  # [B,hist_len,N,model_dim]
        x = self.input_dropout(x)

        for layer in self.layers:
            x = layer(x)

        x = x.permute(0, 2, 1, 3).reshape(B, N, self.hist_len * self.model_dim)  # [B,N,T*D]
        out = self.output_proj(x)                                                # [B,N,pred_len]
        pm25_pred = out.permute(0, 2, 1).unsqueeze(-1)                           # [B,pred_len,N,1]
        return pm25_pred
