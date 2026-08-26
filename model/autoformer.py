"""
Autoformer (Wu et al., NeurIPS 2021, "Autoformer: Decomposition
Transformers with Auto-Correlation for Long-Term Series Forecasting",
https://github.com/thuml/Autoformer) adapted as a plain, non-graph
benchmark model for this repo's harness.

Two headline ideas, both preserved from upstream:
  1. Series decomposition (`series_decomp`, a moving-average trend/
     seasonal split) is applied PROGRESSIVELY, inside every encoder and
     decoder layer, not just once at the input - each layer strips its
     own residual trend back out after every sub-block.
  2. Auto-Correlation (`AutoCorrelation`) replaces dot-product attention:
     it finds the dominant periods via FFT (query/key correlation in the
     frequency domain) and aggregates values by rolling/gathering them at
     the top-k lag offsets, instead of a full pairwise attention matrix.

Same per-station treatment as model/informer.py / model/patchtst.py:
one shared backbone runs independently per city, weights shared, city
folded into batch.

ADAPTATION NEEDED FOR c_out != dec_in (read before changing d_model/c_out)
---------------------------------------------------------------------------
Upstream Autoformer always forecasts every channel of its input (c_out ==
dec_in, e.g. all 7 ETT columns). Here we only want to output ONE channel
(PM2.5), while the decoder's VALUE embedding still needs the full
`in_dim` (PM2.5 + weather) to do its cross-correlation with the encoder.
Concretely, that means the decoder's running trend accumulator - which
upstream keeps at `dec_in` channels and only projects down to `c_out` at
the very end - has to be tracked at `c_out=1` channels from the start
here, since each DecoderLayer's own trend projection already outputs
`c_out` channels per layer (see `Decoder.forward`'s `trend = trend +
residual_trend`, which upstream only works because dec_in == c_out).
So `trend_init` below is built from ONLY the PM2.5 channel's decomposed
trend, while `seasonal_init` (fed into the embedding) still carries all
`in_dim` channels, exactly as upstream does.

FUTURE WEATHER (real, not the paper's own placeholder convention)
---------------------------------------------------------------------
Upstream builds the decoder's future span by zero-filling the seasonal
placeholder and mean-filling the trend placeholder, for every channel -
appropriate when every channel is itself an unknown target. Here only
the PM2.5 channel is unknown; weather is known (feature[:, hist_len:]),
same convention as InformerPM25 and model/GNN_Transformer.py. So the
future span is built channel-wise: PM2.5 gets the usual zero/mean
seasonal/trend placeholder; weather channels get the real future value
in `seasonal_init` and zero in the trend placeholder, so seasonal+trend
reconstructs the true known value (real + 0 = real) rather than a
guessed one.

Contract (matches every other model in model/, see train.py get_model()):
    AutoformerPM25(hist_len, pred_len, in_dim, city_num, batch_size, device, ...)
    pm25_pred = model(pm25_hist, feature)
    # pm25_hist: [B, hist_len, N, 1], feature: [B, hist_len+pred_len, N, F]
    # -> pm25_pred: [B, pred_len, N, 1]
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.embed import TokenEmbedding


# --------------------------------------------------------------------------
# series decomposition
# --------------------------------------------------------------------------

class _MovingAvg(nn.Module):
    """Moving average trend extractor, replicate-padded so the output keeps
    the input's sequence length (same trick as upstream's moving_avg)."""

    def __init__(self, kernel_size, stride=1):
        super(_MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x: [B, L, C]
        pad = (self.kernel_size - 1) // 2
        front = x[:, 0:1, :].repeat(1, pad, 1)
        end = x[:, -1:, :].repeat(1, self.kernel_size - 1 - pad, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


class _SeriesDecomp(nn.Module):
    def __init__(self, kernel_size):
        super(_SeriesDecomp, self).__init__()
        self.moving_avg = _MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        seasonal = x - moving_mean
        return seasonal, moving_mean


class _MyLayerNorm(nn.Module):
    """Special LayerNorm for the seasonal path: subtracts the post-norm
    per-instance mean over time so it stays (approximately) zero-mean,
    keeping the trend from leaking back into what's meant to be the pure
    seasonal component. Same as upstream's my_Layernorm."""

    def __init__(self, channels):
        super(_MyLayerNorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1, keepdim=True)
        return x_hat - bias


# --------------------------------------------------------------------------
# Auto-Correlation
# --------------------------------------------------------------------------

class AutoCorrelation(nn.Module):
    """FFT-based period discovery + time-delay aggregation, replacing
    dot-product attention. `mask_flag` is accepted for API parity with
    upstream but - as in the original implementation - isn't actually
    used to mask anything inside forward(); period-based aggregation
    doesn't have a natural causal-masking analogue the way QK^T does."""

    def __init__(self, mask_flag=True, factor=1, attention_dropout=0.1, output_attention=False):
        super(AutoCorrelation, self).__init__()
        self.factor = factor
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _time_delay_agg_training(self, values, corr):
        # values, corr: [B, H, C, L]
        head, channel, length = values.shape[1], values.shape[2], values.shape[3]
        top_k = max(1, int(self.factor * math.log(length)))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)  # [B, L]
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)  # [B, top_k]
        tmp_corr = torch.softmax(weights, dim=-1)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            pattern = torch.roll(values, -int(index[i]), dims=-1)
            delays_agg = delays_agg + pattern * tmp_corr[:, i].view(-1, 1, 1, 1)
        return delays_agg

    def _time_delay_agg_inference(self, values, corr):
        # values, corr: [B, H, C, L]
        batch, head, channel, length = values.shape
        top_k = max(1, int(self.factor * math.log(length)))
        init_index = torch.arange(length, device=values.device).view(1, 1, 1, -1).repeat(batch, head, channel, 1)
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)  # [B, L]
        weights, delay = torch.topk(mean_value, top_k, dim=-1)   # [B, top_k]
        tmp_corr = torch.softmax(weights, dim=-1)
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[:, i].view(-1, 1, 1, 1)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * tmp_corr[:, i].view(-1, 1, 1, 1)
        return delays_agg

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        if L > S:
            zeros = torch.zeros_like(queries[:, :(L - S), :, :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]

        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
        corr = torch.fft.irfft(q_fft * torch.conj(k_fft), n=L, dim=-1)  # [B,H,E,L]

        values_bhcl = values.permute(0, 2, 3, 1).contiguous()
        if self.training:
            V = self._time_delay_agg_training(values_bhcl, corr)
        else:
            V = self._time_delay_agg_inference(values_bhcl, corr)
        V = V.permute(0, 3, 1, 2)  # [B, L, H, D]
        V = self.dropout(V)

        if self.output_attention:
            return V.contiguous(), corr.permute(0, 3, 1, 2)
        return V.contiguous(), None


class AutoCorrelationLayer(nn.Module):
    def __init__(self, correlation, d_model, n_heads, d_keys=None, d_values=None):
        super(AutoCorrelationLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_correlation = correlation
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_correlation(queries, keys, values, attn_mask)
        out = out.view(B, L, -1)
        return self.out_projection(out), attn


# --------------------------------------------------------------------------
# Encoder / Decoder (decomposition-aware)
# --------------------------------------------------------------------------

class _EncoderLayer(nn.Module):
    def __init__(self, correlation, d_model, d_ff=None, moving_avg=25, dropout=0.1, activation='gelu'):
        super(_EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.correlation = correlation
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.decomp1 = _SeriesDecomp(moving_avg)
        self.decomp2 = _SeriesDecomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == 'relu' else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.correlation(x, x, x, attn_mask)
        x = x + self.dropout(new_x)
        x, _ = self.decomp1(x)

        y = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        res, _ = self.decomp2(x + y)
        return res, attn


class _Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(_Encoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        for layer in self.layers:
            x, attn = layer(x, attn_mask=attn_mask)
            attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)
        return x, attns


class _DecoderLayer(nn.Module):
    def __init__(self, self_correlation, cross_correlation, d_model, c_out, d_ff=None,
                 moving_avg=25, dropout=0.1, activation='gelu'):
        super(_DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_correlation = self_correlation
        self.cross_correlation = cross_correlation
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.decomp1 = _SeriesDecomp(moving_avg)
        self.decomp2 = _SeriesDecomp(moving_avg)
        self.decomp3 = _SeriesDecomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        # c_out here matches THIS repo's c_out=1 (PM2.5 only) - see module
        # docstring's "ADAPTATION NEEDED FOR c_out != dec_in" section.
        self.trend_projection = nn.Conv1d(d_model, c_out, kernel_size=3, stride=1,
                                           padding=1, padding_mode='circular', bias=False)
        self.activation = F.relu if activation == 'relu' else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_correlation(x, x, x, x_mask)[0])
        x, trend1 = self.decomp1(x)

        x = x + self.dropout(self.cross_correlation(x, cross, cross, cross_mask)[0])
        x, trend2 = self.decomp2(x)

        y = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.decomp3(x + y)

        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.trend_projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
        return x, residual_trend


class _Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(_Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, trend, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            trend = trend + residual_trend
        if self.norm is not None:
            x = self.norm(x)
        if self.projection is not None:
            x = self.projection(x)
        return x, trend


class _EmbeddingNoPos(nn.Module):
    """value + linear time-mark embedding, NO positional term - upstream
    Autoformer deliberately drops the sinusoidal position embedding
    (DataEmbedding_wo_pos), relying on the decomposition + auto-
    correlation to carry temporal structure instead. Mark projection is
    sized to this harness's 2 mark channels (hour, weekday), same as
    model/informer.py's _Embedding."""

    def __init__(self, c_in, n_mark, d_model, dropout=0.1):
        super(_EmbeddingNoPos, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.mark_embedding = nn.Linear(n_mark, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.mark_embedding(x_mark)
        return self.dropout(x)


class AutoformerPM25(nn.Module):
    def __init__(self, hist_len, pred_len, in_dim, city_num, batch_size, device,
                 label_len=None, time_mark_channels=(-4, -3),
                 d_model=64, n_heads=8, e_layers=2, d_layers=1, d_ff=256,
                 moving_avg=25, factor=1, dropout=0.1, activation='gelu',
                 output_attention=False):
        super(AutoformerPM25, self).__init__()
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
        c_out = 1  # only PM2.5 is actually forecast - see module docstring

        self.decomp = _SeriesDecomp(moving_avg)

        self.enc_embedding = _EmbeddingNoPos(in_dim, n_mark, d_model, dropout)
        self.dec_embedding = _EmbeddingNoPos(in_dim, n_mark, d_model, dropout)

        self.encoder = _Encoder(
            [
                _EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, factor, attention_dropout=dropout,
                                         output_attention=output_attention),
                        d_model, n_heads,
                    ),
                    d_model, d_ff, moving_avg=moving_avg, dropout=dropout, activation=activation,
                ) for _ in range(e_layers)
            ],
            norm_layer=_MyLayerNorm(d_model),
        )
        self.decoder = _Decoder(
            [
                _DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, factor, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads,
                    ),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, factor, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads,
                    ),
                    d_model, c_out, d_ff, moving_avg=moving_avg, dropout=dropout, activation=activation,
                ) for _ in range(d_layers)
            ],
            norm_layer=_MyLayerNorm(d_model),
            projection=nn.Linear(d_model, c_out, bias=True),
        )

    @staticmethod
    def _fold(t):
        """[B, T, N, C] -> [B*N, T, C] - fold the city dim into batch, one
        shared Autoformer runs independently per node (same convention as
        model/informer.py / model/patchtst.py)."""
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
                f"AutoformerPM25 was built with city_num={self.city_num}, but got "
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

        mark_hist = feature_hist[..., self.time_mark_channels]
        mark_dec_hist = feature_hist[:, self.hist_len - self.label_len:][..., self.time_mark_channels]
        mark_dec_future = feature_future[..., self.time_mark_channels]
        x_mark_dec = torch.cat([mark_dec_hist, mark_dec_future], dim=1)

        enc_x = self._fold(x_hist)                 # [B*N, hist_len, in_dim]
        enc_mark = self._fold(mark_hist)            # [B*N, hist_len, n_mark]
        dec_mark = self._fold(x_mark_dec)            # [B*N, label_len+pred_len, n_mark]
        feat_future_folded = self._fold(feature_future)  # [B*N, pred_len, F]

        BN = enc_x.shape[0]

        seasonal_hist, trend_hist = self.decomp(enc_x)  # both [B*N, hist_len, in_dim]

        # PM2.5-only trend accumulator (c_out=1) - see module docstring.
        pm25_hist_folded = enc_x[..., 0:1]                                   # [B*N, hist_len, 1]
        mean_pm25 = pm25_hist_folded.mean(dim=1, keepdim=True).expand(-1, self.pred_len, -1)
        trend_hist_pm25 = trend_hist[:, self.hist_len - self.label_len:, 0:1]  # [B*N, label_len, 1]
        trend_init = torch.cat([trend_hist_pm25, mean_pm25], dim=1)          # [B*N, label_len+pred_len, 1]

        # Full-width (in_dim) seasonal input fed to the embedding: PM2.5's
        # future gets the usual zero placeholder; weather's future is the
        # REAL known value (this harness exposes it, unlike upstream's
        # own fully-unknown multivariate setup - see module docstring).
        zeros_pm25_future = torch.zeros(BN, self.pred_len, 1, device=enc_x.device, dtype=enc_x.dtype)
        seasonal_future = torch.cat([zeros_pm25_future, feat_future_folded], dim=-1)  # [B*N,pred_len,in_dim]
        seasonal_hist_slice = seasonal_hist[:, self.hist_len - self.label_len:]        # [B*N,label_len,in_dim]
        seasonal_init = torch.cat([seasonal_hist_slice, seasonal_future], dim=1)       # [B*N,label_len+pred_len,in_dim]

        enc_out = self.enc_embedding(enc_x, enc_mark)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.dec_embedding(seasonal_init, dec_mark)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, trend_init, x_mask=None, cross_mask=None)

        out = seasonal_part + trend_part           # [B*N, label_len+pred_len, 1]
        out = out[:, -self.pred_len:, :]            # [B*N, pred_len, 1]
        pm25_pred = out.view(B, N, self.pred_len, 1).permute(0, 2, 1, 3)  # [B, pred_len, N, 1]
        return pm25_pred
