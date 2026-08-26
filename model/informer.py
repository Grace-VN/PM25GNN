"""
Informer (https://github.com/zhouhaoyi/Informer2020) adapted as a plain,
non-graph sequence-to-sequence benchmark model for this repo's harness.

Unlike model/GNN_Transformer.py (an unfinished GNN+Informer hybrid that
flattens all city_num nodes into one wide feature vector, and imports
`models.encoder`/`models.attn`/`utils.masking` that don't exist in this
repo), this is "vanilla" Informer: one shared set of weights runs
independently per city/station, exactly like this repo's LSTM/GRU/MLP
baselines fold the city dimension into the batch dimension for their
per-node cell. That keeps enc_in/dec_in equal to in_dim (not
city_num * in_dim), which is what you want for a tractable, literal
Informer baseline on a ~184-node graph like KnowAir.

Building blocks (Encoder/EncoderLayer/ConvLayer, Decoder/DecoderLayer,
TokenEmbedding/PositionalEmbedding) come unmodified from model/encoder.py,
model/decoder.py and model/embed.py, which already carry Informer2020's
code verbatim. model/attn.py (FullAttention/ProbAttention/AttentionLayer)
was added alongside this file to complete the set.

Contract (matches every other model in model/, see train.py get_model()):
    InformerPM25(hist_len, pred_len, in_dim, city_num, batch_size, device, ...)
    pm25_pred = model(pm25_hist, feature)
    # pm25_hist: [B, hist_len, N, 1], feature: [B, hist_len+pred_len, N, F]
    # -> pm25_pred: [B, pred_len, N, 1]

Decoder input follows standard Informer usage (last `label_len`
ground-truth steps + a placeholder for the future), except the "future"
part isn't fully zero-filled: this harness hands every model known
FUTURE WEATHER (feature[:, hist_len:]), which PM25_GNN/ProbGRU*/
GNN_Transformer all already use during decoding - only the PM2.5 channel
itself (the actual forecast target) is zero-filled for the future part.

Time marks: KnowAir's `feature` tensor carries [..., hour, weekday,
wind_speed, wind_direction] as its last four channels (see dataset.py
_process_feature), so time_mark_channels defaults to (-4, -3). These
are z-normalized floats, not integer calendar indices, so marks go
through a plain linear layer rather than Informer's categorical/fixed
TemporalEmbedding (which expects raw month/day/weekday/hour integers).
"""

import torch
import torch.nn as nn

from model.embed import TokenEmbedding, PositionalEmbedding
from model.encoder import Encoder, EncoderLayer, ConvLayer
from model.decoder import Decoder, DecoderLayer
from model.attn import FullAttention, ProbAttention, AttentionLayer


class _Embedding(nn.Module):
    """value + positional + linear time-mark embedding, dropout at the end.
    Same idea as model/embed.py's DataEmbedding, but the mark projection
    is sized to however many mark channels this harness actually supplies
    (2: hour, weekday) instead of Informer's fixed freq -> dim lookup."""

    def __init__(self, c_in, n_mark, d_model, dropout=0.1):
        super(_Embedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.mark_embedding = nn.Linear(n_mark, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding(x) + self.mark_embedding(x_mark)
        return self.dropout(x)


class InformerPM25(nn.Module):
    def __init__(self, hist_len, pred_len, in_dim, city_num, batch_size, device,
                 label_len=None, time_mark_channels=(-4, -3),
                 d_model=64, n_heads=8, e_layers=2, d_layers=1, d_ff=256,
                 factor=5, dropout=0.1, attn='prob', activation='gelu',
                 output_attention=False, distil=True, mix=True):
        super(InformerPM25, self).__init__()
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.in_dim = in_dim
        self.city_num = city_num
        self.device = device
        self.batch_size = batch_size
        self.label_len = label_len if label_len is not None else max(1, hist_len // 2)
        self.time_mark_channels = list(time_mark_channels)
        self.output_attention = output_attention

        n_mark = len(time_mark_channels)
        c_out = 1

        self.enc_embedding = _Embedding(in_dim, n_mark, d_model, dropout)
        self.dec_embedding = _Embedding(in_dim, n_mark, d_model, dropout)

        Attn = ProbAttention if attn == 'prob' else FullAttention
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout,
                                         output_attention=output_attention),
                                   d_model, n_heads, mix=False),
                    d_model, d_ff, dropout=dropout, activation=activation,
                ) for _ in range(e_layers)
            ],
            [ConvLayer(d_model) for _ in range(e_layers - 1)] if distil else None,
            norm_layer=nn.LayerNorm(d_model),
        )
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout,
                                         output_attention=False), d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout,
                                                  output_attention=False), d_model, n_heads, mix=False),
                    d_model, d_ff, dropout=dropout, activation=activation,
                ) for _ in range(d_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )
        self.projection = nn.Linear(d_model, c_out, bias=True)

    @staticmethod
    def _fold(t):
        """[B, T, N, C] -> [B*N, T, C] - folds the city dim into batch so
        one shared Informer runs independently per node, same convention
        this repo's LSTM/GRU/MLP baselines use for their per-node cell."""
        B, T, N, C = t.shape
        return t.permute(0, 2, 1, 3).reshape(B * N, T, C)

    def forward(self, pm25_hist, feature):
        """
        pm25_hist : [B, hist_len, N, 1]
        feature   : [B, hist_len + pred_len, N, F]   (F = in_dim - 1)
        returns   : [B, pred_len, N, 1]
        """
        B, T, N, _ = pm25_hist.shape
        if N != self.city_num:
            raise ValueError(
                f"InformerPM25 was built with city_num={self.city_num}, but got "
                f"N={N} nodes in this batch's data."
            )
        feature_hist = feature[:, :self.hist_len]
        feature_future = feature[:, self.hist_len:self.hist_len + self.pred_len]
        if feature_future.shape[1] != self.pred_len:
            raise ValueError(
                f"expected {self.pred_len} future feature steps, got "
                f"{feature_future.shape[1]} - check feature's time dimension "
                f"covers hist_len + pred_len."
            )

        x_hist = torch.cat([pm25_hist, feature_hist], dim=-1)  # [B,hist_len,N,in_dim]

        pm25_future_placeholder = torch.zeros(
            B, self.pred_len, N, pm25_hist.shape[-1], device=pm25_hist.device, dtype=pm25_hist.dtype
        )
        x_dec_future = torch.cat([pm25_future_placeholder, feature_future], dim=-1)  # [B,pred_len,N,in_dim]
        x_dec_hist = x_hist[:, self.hist_len - self.label_len:]                       # [B,label_len,N,in_dim]
        x_dec = torch.cat([x_dec_hist, x_dec_future], dim=1)                          # [B,label_len+pred_len,N,in_dim]

        mark_hist = feature_hist[..., self.time_mark_channels]                        # [B,hist_len,N,n_mark]
        mark_dec_hist = feature_hist[:, self.hist_len - self.label_len:][..., self.time_mark_channels]
        mark_dec_future = feature_future[..., self.time_mark_channels]
        x_mark_dec = torch.cat([mark_dec_hist, mark_dec_future], dim=1)               # [B,label_len+pred_len,N,n_mark]

        enc_x = self._fold(x_hist)
        dec_x = self._fold(x_dec)
        enc_mark = self._fold(mark_hist)
        dec_mark = self._fold(x_mark_dec)

        enc_out = self.enc_embedding(enc_x, enc_mark)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.dec_embedding(dec_x, dec_mark)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
        dec_out = self.projection(dec_out)  # [B*N, label_len+pred_len, 1]

        pm25_pred = dec_out[:, -self.pred_len:, :]                                    # [B*N, pred_len, 1]
        pm25_pred = pm25_pred.view(B, N, self.pred_len, 1).permute(0, 2, 1, 3)         # [B, pred_len, N, 1]
        return pm25_pred
