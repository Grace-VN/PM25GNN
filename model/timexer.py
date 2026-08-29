"""
TimeXer (Wang, Wu, Shi, Hu, Luo, Ma, Zhang, Long - "TimeXer: Empowering
Transformers for Time Series Forecasting with Exogenous Variables",
NeurIPS 2024) adapted as a benchmark model for this repo's harness.
Reference implementation: https://github.com/thuml/TimeXer (MIT-licensed;
same THUML lab as this repo's Autoformer baseline).

Everything below TimeXerPM25 (PositionalEmbedding, DataEmbedding_inverted,
FullAttention, AttentionLayer, FlattenHead, EnEmbedding, Encoder,
EncoderLayer, and the core TimeXer class) is the reference repo's
models/TimeXer.py + the two layers/ files it imports from, ported
verbatim in 'MS' mode (their term for single-target-variable-with-
exogenous-inputs forecasting, exactly this repo's per-city pm25-from-
pm25+weather setup) - the 'M' (predict-every-variable) code path in the
reference Model class is dropped since it doesn't apply here. FullAttention/
AttentionLayer are TimeXer's own copies (not this repo's model/attn.py) -
close but not identical (TimeXer's accept unused tau/delta kwargs for a
de-stationary-attention variant it doesn't otherwise use), so reusing
model/attn.py directly would raise a TypeError on the call signature.

WHY THIS ONE, ARCHITECTURALLY: TimeXer is the one benchmark in this repo
that treats "target history" and "exogenous covariate history" as two
distinct embedded streams from the start (a patched endogenous token
sequence cross-attending to inverted per-variable exogenous tokens),
rather than concatenating everything into one undifferentiated channel
stack the way Informer/Autoformer/PatchTST/STAEformer do here. That's a
closer conceptual match to this repo's own pm25_hist/feature split than
any of those, even though (see below) it still can't consume feature's
FUTURE portion.

STILL NOT using this harness's future-known weather: TimeXer's published
architecture is encoder-only - `forecast()` never references x_dec/
x_mark_dec at all (see the reference Model.forward's signature vs. what
forecast() actually reads). A FlattenHead linearly projects the encoded
history representation straight to pred_len steps; there's no decoder
input point for future covariates. So despite the endogenous/exogenous
framing above, this is - like PDFormer and MGSFformer - a history-only
model with respect to `feature`: it uses feature's HISTORICAL portion
(as exogenous variates) but not its future portion, unlike the GNN/RNN-
family models here. Documented at the get_model() call site too.

hist_len must be a multiple of patch_len (default 8, matching this repo's
patchtst_patch_len default) - EnEmbedding patches the target history via
`x.unfold(size=patch_len, step=patch_len)`; a non-exact division silently
drops the remainder instead of erroring, which would quietly shrink the
model's effective receptive field, so this raises instead.

Contract (matches every other model in model/, see train.py get_model()):
    TimeXerPM25(hist_len, pred_len, in_dim, city_num, batch_size, device, ...)
    pm25_pred = model(pm25_hist, feature)
    # pm25_hist: [B, hist_len, N, 1], feature: [B, hist_len+pred_len, N, F]
    #   (only feature[:, :hist_len] is used - see note above)
    # -> pm25_pred: [B, pred_len, N, 1]
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model]
        return self.dropout(x)


class FullAttention(nn.Module):
    def __init__(self, mask_flag=False, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            raise NotImplementedError(
                "mask_flag=True isn't supported in this port - TimeXer only ever "
                "calls FullAttention with mask_flag=False (see TimeXerPM25's docstring)."
            )

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(queries, keys, values, attn_mask, tau=tau, delta=delta)
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class FlattenHead(nn.Module):
    def __init__(self, nf, target_window, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class EnEmbedding(nn.Module):
    def __init__(self, n_vars, d_model, patch_len, dropout):
        super(EnEmbedding, self).__init__()
        self.patch_len = patch_len

        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
        self.position_embedding = PositionalEmbedding(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        glb = self.glb_token.repeat((x.shape[0], 1, 1, 1))

        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))
        x = torch.cat([x, glb], dim=2)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        return self.dropout(x), n_vars


class Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)
        if self.norm is not None:
            x = self.norm(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        B, L, D = cross.shape
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask, tau=tau, delta=None)[0])
        x = self.norm1(x)

        x_glb_ori = x[:, -1, :].unsqueeze(1)
        x_glb = torch.reshape(x_glb_ori, (B, -1, D))
        x_glb_attn = self.dropout(self.cross_attention(x_glb, cross, cross, attn_mask=cross_mask, tau=tau, delta=delta)[0])
        x_glb_attn = torch.reshape(x_glb_attn, (x_glb_attn.shape[0] * x_glb_attn.shape[1], x_glb_attn.shape[2])).unsqueeze(1)
        x_glb = x_glb_ori + x_glb_attn
        x_glb = self.norm2(x_glb)

        y = x = torch.cat([x[:, :-1, :], x_glb], dim=1)

        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class TimeXer(nn.Module):
    """Core model, 'MS' mode only (n_vars=1: forecast one target variable
    from itself + exogenous covariates) - the reference repo's Model class
    also supports an 'M' mode (forecast every variable) not needed here."""

    def __init__(self, seq_len, pred_len, enc_in, patch_len, d_model, n_heads,
                 e_layers, d_ff, dropout, factor, activation, use_norm):
        super(TimeXer, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.use_norm = use_norm
        self.patch_len = patch_len
        self.patch_num = int(seq_len // patch_len)
        n_vars = 1

        self.en_embedding = EnEmbedding(n_vars, d_model, patch_len, dropout)
        self.ex_embedding = DataEmbedding_inverted(seq_len, d_model, dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), d_model, n_heads),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), d_model, n_heads),
                    d_model, d_ff, dropout=dropout, activation=activation,
                ) for _ in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
        )
        self.head_nf = d_model * (self.patch_num + 1)
        self.head = FlattenHead(self.head_nf, pred_len, head_dropout=dropout)

    def forward(self, x_enc, x_mark_enc):
        """
        x_enc      : [B, seq_len, enc_in] - exogenous channels then target last
        x_mark_enc : [B, seq_len, n_mark] or None
        returns    : [B, pred_len, 1]
        """
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        en_embed, n_vars = self.en_embedding(x_enc[:, :, -1].unsqueeze(-1).permute(0, 2, 1))
        ex_embed = self.ex_embedding(x_enc[:, :, :-1], x_mark_enc)

        enc_out = self.encoder(en_embed, ex_embed)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out[:, -self.pred_len:, :]


class TimeXerPM25(nn.Module):
    def __init__(self, hist_len, pred_len, in_dim, city_num, batch_size, device,
                 patch_len=8, d_model=128, n_heads=8, e_layers=2, d_ff=256,
                 dropout=0.1, factor=5, activation='gelu', use_norm=True,
                 time_mark_channels=(-4, -3)):
        super(TimeXerPM25, self).__init__()
        if hist_len % patch_len != 0:
            raise ValueError(
                f"TimeXer's EnEmbedding patches the target history in non-overlapping "
                f"chunks of patch_len - hist_len must be a multiple of patch_len, got "
                f"hist_len={hist_len}, patch_len={patch_len}."
            )
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.city_num = city_num
        self.time_mark_channels = list(time_mark_channels)

        self.core = TimeXer(
            seq_len=hist_len, pred_len=pred_len, enc_in=in_dim, patch_len=patch_len,
            d_model=d_model, n_heads=n_heads, e_layers=e_layers, d_ff=d_ff,
            dropout=dropout, factor=factor, activation=activation, use_norm=use_norm,
        )

    @staticmethod
    def _fold(t):
        """[B, T, N, C] -> [B*N, T, C] - same per-node folding convention as
        informer.py/autoformer.py/patchtst.py in this repo."""
        B, T, N, C = t.shape
        return t.permute(0, 2, 1, 3).reshape(B * N, T, C)

    def forward(self, pm25_hist, feature):
        """
        pm25_hist : [B, hist_len, N, 1]
        feature   : [B, hist_len + pred_len, N, F] - only the historical
                    portion is used, see module docstring
        returns   : [B, pred_len, N, 1]
        """
        B, T, N, _ = pm25_hist.shape
        if N != self.city_num:
            raise ValueError(
                f"TimeXerPM25 was built with city_num={self.city_num}, but got "
                f"N={N} nodes in this batch's data."
            )
        feature_hist = feature[:, :self.hist_len]
        mark_hist = feature_hist[..., self.time_mark_channels]  # [B, hist_len, N, n_mark]

        # exogenous channels first, target last - matches x_enc[:,:,-1] convention
        x_enc = torch.cat([feature_hist, pm25_hist], dim=-1)  # [B, hist_len, N, in_dim]

        enc_x = self._fold(x_enc)          # [B*N, hist_len, in_dim]
        enc_mark = self._fold(mark_hist)   # [B*N, hist_len, n_mark]

        out = self.core(enc_x, enc_mark)   # [B*N, pred_len, 1]
        pm25_pred = out.view(B, N, self.pred_len, 1).permute(0, 2, 1, 3)  # [B, pred_len, N, 1]
        return pm25_pred
