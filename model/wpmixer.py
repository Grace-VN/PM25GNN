"""
WPMixer (Murad, Aktukmak & Yilmaz, AAAI 2025, "WPMixer: Efficient
Multi-Resolution Mixing for Long-Term Time Series Forecasting",
https://github.com/Secure-and-Intelligent-Systems-Lab/WPMixer, arXiv
2412.17176) adapted as a plain, non-graph benchmark model for this repo's
harness.

The paper's headline idea: instead of forecasting the raw time-domain
sequence directly, decompose the lookback window with a multi-level
discrete wavelet transform (DWT) into one approximation (low-frequency)
sub-band and `level` detail (high-frequency) sub-bands, forecast EACH
sub-band's own future coefficients independently with a shared patch+MLP-
mixer backbone, then recombine the predicted sub-bands with the inverse
DWT (IDWT) to get the time-domain forecast. Each resolution's backbone is:
patch the sub-band's coefficient sequence (replication-padded at the tail
so the last patch fully covers it, same trick PatchTST uses) -> embed each
patch (Linear) -> two residual Mixer blocks (an MLP mixing across patches,
alternating with an MLP mixing across the embedding dimension - the same
alternating-axis idea as the original MLP-Mixer vision paper) -> flatten +
Linear head to the target coefficient-sequence length for that sub-band.
RevIN (instance normalization, denormalized at the output) wraps both the
overall series and, independently, each sub-band's own coefficient series.

Ported directly from `models/wavelet_patch_mixer.py` + `models/
decomposition.py` + `utils/RevIN.py` (this repo's `RevIN` here is the
verbatim upstream one, not model/patchtst.py's simplified `_RevIN` - the
two differ slightly: this one supports `subtract_last` mode, unused here
but kept for fidelity) - class names, forward-pass structure, and the
patching/mixing math are unchanged. `pytorch_wavelets.DWT1DForward` /
`DWT1DInverse` (MIT-licensed for the DWT code path used here - see that
package's own LICENSE) do the actual wavelet transform; hand-rolling a
1D multi-level DWT/IDWT from scratch would risk subtly wrong filter-
coefficient or boundary-condition fidelity, so this is used as a real
(lightweight, pure-Python, no compiled-extension) dependency instead -
see requirements.txt for why PyWavelets is *also* listed explicitly
(pytorch_wavelets imports `pywt` internally but doesn't declare it as a
dependency of its own).

CHANNEL CONVENTION: like PatchTSTPM25 in this repo, every scalar channel
is forecast by ONE SHARED backbone, run independently per channel (no
cross-channel attention/mixing) - but UNLIKE PatchTSTPM25 (which also
folds the channel dimension into batch, discarding per-channel identity
entirely), WPMixer's own BatchNorm2d layers are indexed by channel, so
this port keeps `channel = in_dim` (PM2.5 plus every weather variable) as
a real dimension - each variable gets its own BatchNorm2d running
statistics, closer to the paper's actual multivariate treatment - and
only folds the city dimension into batch (same convention as this repo's
LSTM/GRU/MLP/InformerPM25/PatchTSTPM25). PM2.5 is concatenated LAST
(`cat([feature_hist, pm25_hist], -1)`) so the upstream model's own
`channel_out`-from-the-end slicing convention (`pred[:, :, -channel_out:]`)
can be reused verbatim instead of reordering channels after the fact.

IMPORTANT LIMITATION (inherent to WPMixer, not a bug): like PatchTSTPM25/
MGSFformer/TimeXer/AGCRN, WPMixer's published architecture is purely
autoregressive from the lookback window - its own short-term-forecast
wrapper (`WPMixerWrapperShortTermForecast.forward`, not ported here)
literally takes and discards three extra arguments a standard multi-input
dataloader would supply, confirming it never consumes future-known
covariates. `feature[:, hist_len:]` (the future weather this harness makes
available) is simply unused here, exactly as in the paper's own
multivariate setup. If you want a benchmark that exploits future weather,
use InformerPM25/AirLapse/MegaCRN.

Contract (matches every other model in model/, see train.py get_model()):
    WPMixerPM25(hist_len, pred_len, in_dim, city_num, batch_size, device, ...)
    pm25_pred = model(pm25_hist, feature)
    # pm25_hist: [B, hist_len, N, 1], feature: [B, hist_len+pred_len, N, F]
    # -> pm25_pred: [B, pred_len, N, 1]
"""

import torch
import torch.nn as nn

from pytorch_wavelets import DWT1DForward, DWT1DInverse


class RevIN(nn.Module):
    """Reversible instance normalization (code from
    https://github.com/ts-kim/RevIN, as vendored - unmodified - by the
    upstream WPMixer repo's utils/RevIN.py). Normalizes over every
    dimension except the first (batch) and last (channel); `subtract_last`
    centers on the window's final value instead of its mean (unused here,
    kept for fidelity to the upstream class)."""

    def __init__(self, num_features, eps=1e-5, affine=True, subtract_last=False):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError(mode)
        return x

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1:, :]
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - (self.last if self.subtract_last else self.mean)
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + (self.last if self.subtract_last else self.mean)
        return x


class TokenMixer(nn.Module):
    """MLP mixing across the PATCH axis (the "token-mixing" half of an
    MLP-Mixer block): x's last dim is patch_num on the way in, mapped
    through an expand-GELU-contract MLP to pred_seq (== patch_num here,
    since every ResolutionBranch keeps its own patch count fixed)."""

    def __init__(self, input_seq, pred_seq, dropout, factor, d_model):
        super(TokenMixer, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_seq, pred_seq * factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pred_seq * factor, pred_seq),
        )

    def forward(self, x):
        # x: [B, d_model, channel, patch_num] -> mix over the last axis
        x = x.transpose(1, 2)
        x = self.layers(x)
        x = x.transpose(1, 2)
        return x


class Mixer(nn.Module):
    """One residual MLP-Mixer block: BatchNorm2d -> token-mixing MLP
    (across patches) -> BatchNorm2d -> + residual embedding-mixing MLP
    (across the d_model axis). `channel` indexes BatchNorm2d's own
    per-channel running statistics - each input variable (PM2.5, each
    weather field) gets its own normalization here, not a global one."""

    def __init__(self, input_seq, out_seq, channel, d_model, dropout, tfactor, dfactor):
        super(Mixer, self).__init__()
        self.tMixer = TokenMixer(input_seq=input_seq, pred_seq=out_seq, dropout=dropout, factor=tfactor, d_model=d_model)
        self.dropoutLayer = nn.Dropout(dropout)
        self.norm1 = nn.BatchNorm2d(channel)
        self.norm2 = nn.BatchNorm2d(channel)
        self.embeddingMixer = nn.Sequential(
            nn.Linear(d_model, d_model * dfactor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * dfactor, d_model),
        )

    def forward(self, x):
        # x: [B, channel, patch_num, d_model]
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2)                  # [B, d_model, channel, patch_num]
        x = self.dropoutLayer(self.tMixer(x))
        x = x.permute(0, 2, 3, 1)                  # [B, channel, patch_num, d_model]
        x = self.norm2(x)
        x = x + self.dropoutLayer(self.embeddingMixer(x))
        return x


class ResolutionBranch(nn.Module):
    """Per-sub-band forecaster: patches the sub-band's coefficient series,
    embeds and mixes it (two Mixer blocks), then flattens + Linear-heads
    to the target coefficient-sequence length for this sub-band. One
    instance of this exists per wavelet sub-band (1 approximation + one
    per detail level), sharing no weights across sub-bands - the SAME
    weights ARE shared across every station/city, though (this port folds
    city into batch, per the module docstring)."""

    def __init__(self, input_seq, pred_seq, channel, d_model, dropout,
                 embedding_dropout, tfactor, dfactor, patch_len, patch_stride):
        super(ResolutionBranch, self).__init__()
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.patch_num = int((input_seq - patch_len) / patch_stride + 2)

        self.patch_norm = nn.BatchNorm2d(channel)
        self.patch_embedding_layer = nn.Linear(patch_len, d_model)
        self.mixer1 = Mixer(self.patch_num, self.patch_num, channel, d_model, dropout, tfactor, dfactor)
        self.mixer2 = Mixer(self.patch_num, self.patch_num, channel, d_model, dropout, tfactor, dfactor)
        self.norm = nn.BatchNorm2d(channel)
        self.dropoutLayer = nn.Dropout(embedding_dropout)
        self.head = nn.Sequential(
            nn.Flatten(start_dim=-2, end_dim=-1),
            nn.Linear(self.patch_num * d_model, pred_seq),
        )
        self.revin = RevIN(channel)

    def forward(self, x):
        # x: [B, channel, length_of_coefficient_series]
        x = x.transpose(1, 2)
        x = self.revin(x, 'norm')
        x = x.transpose(1, 2)

        x_patch = self._do_patching(x)                  # [B, channel, patch_num, patch_len]
        x_patch = self.patch_norm(x_patch)
        x_emb = self.dropoutLayer(self.patch_embedding_layer(x_patch))  # [B, channel, patch_num, d_model]

        out = self.mixer1(x_emb)
        out = out + self.mixer2(out)
        out = self.norm(out)

        out = self.head(out)                             # [B, channel, pred_seq]
        out = out.transpose(1, 2)
        out = self.revin(out, 'denorm')
        out = out.transpose(1, 2)
        return out

    def _do_patching(self, x):
        # replication-pad the tail by `patch_stride` so the last patch
        # fully covers the sequence end (same trick PatchTST uses).
        x_padding = x[:, :, -1:].repeat(1, 1, self.patch_stride)
        x_new = torch.cat((x, x_padding), dim=-1)
        return x_new.unfold(dimension=-1, size=self.patch_len, step=self.patch_stride)


class WPMixerCore(nn.Module):
    """Wavelet-decompose the (RevIN-normalized) lookback window into one
    approximation + `level` detail sub-bands, forecast each sub-band's
    future coefficients with its own ResolutionBranch, then inverse-
    wavelet-transform the predicted sub-bands back into a time-domain
    forecast (cropped to the last `pred_length` steps, then RevIN-
    denormalized). `no_decomposition=True` bypasses the wavelet transform
    entirely (a single ResolutionBranch operates directly on the raw
    series) - a built-in ablation the upstream repo also supports."""

    def __init__(self, input_length, pred_length, wavelet_name, level,
                 channel, d_model, dropout, embedding_dropout, tfactor,
                 dfactor, device, patch_len, patch_stride, no_decomposition):
        super(WPMixerCore, self).__init__()
        self.pred_length = pred_length
        self.no_decomposition = no_decomposition

        if not no_decomposition:
            self.dwt = DWT1DForward(wave=wavelet_name, J=level).to(device)
            self.idwt = DWT1DInverse(wave=wavelet_name).to(device)
            input_w_dim = self._probe_dims(self.dwt, input_length, channel, device)
            pred_w_dim = self._probe_dims(self.dwt, pred_length, channel, device)
        else:
            input_w_dim = [input_length]
            pred_w_dim = [pred_length]

        if patch_len > min(input_w_dim):
            raise ValueError(
                f"patch_len={patch_len} exceeds the shortest wavelet sub-band "
                f"length {min(input_w_dim)} (wavelet={wavelet_name}, level={level}, "
                f"input_length={input_length}) - use a smaller patch_len, a "
                f"shorter wavelet, or fewer decomposition levels."
            )

        self.resolutionBranch = nn.ModuleList([
            ResolutionBranch(input_seq=input_w_dim[i], pred_seq=pred_w_dim[i],
                              channel=channel, d_model=d_model, dropout=dropout,
                              embedding_dropout=embedding_dropout, tfactor=tfactor,
                              dfactor=dfactor, patch_len=patch_len, patch_stride=patch_stride)
            for i in range(len(input_w_dim))
        ])
        self.revin = RevIN(channel, eps=1e-5, affine=True, subtract_last=False)

    @staticmethod
    def _probe_dims(dwt, length, channel, device):
        """Wavelet sub-band lengths depend only on (sequence length,
        wavelet, level), not on the actual values - run the transform once
        on a dummy tensor to read them off, exactly as the upstream
        Decomposition._dummy_forward does."""
        with torch.no_grad():
            dummy = torch.ones((1, channel, length), device=device)
            yl, yh = dwt(dummy)
        return [yl.shape[-1]] + [h.shape[-1] for h in yh]

    def forward(self, xL):
        # xL: [Batch, look_back_length, channel]
        x = self.revin(xL, 'norm')
        x = x.transpose(1, 2)                            # [B, channel, look_back_length]

        if not self.no_decomposition:
            xA, xD = self.dwt(x)
        else:
            xA, xD = x, []

        yA = self.resolutionBranch[0](xA)
        yD = [self.resolutionBranch[i + 1](xD[i]) for i in range(len(xD))]

        y = self.idwt((yA, yD)) if not self.no_decomposition else yA
        y = y.transpose(1, 2)                             # [B, T, channel]
        y = y[:, -self.pred_length:, :]
        return self.revin(y, 'denorm')


class WPMixerPM25(nn.Module):
    def __init__(self, hist_len, pred_len, in_dim, city_num, batch_size, device,
                 d_model=16, dropout=0.1, embedding_dropout=0.1,
                 tfactor=3, dfactor=5, wavelet='db2', level=1,
                 patch_len=4, stride=2, no_decomposition=False):
        super(WPMixerPM25, self).__init__()
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.in_dim = in_dim
        self.city_num = city_num
        self.device = device

        self.core = WPMixerCore(
            input_length=hist_len, pred_length=pred_len, wavelet_name=wavelet,
            level=level, channel=in_dim, d_model=d_model, dropout=dropout,
            embedding_dropout=embedding_dropout, tfactor=tfactor, dfactor=dfactor,
            device=device, patch_len=patch_len, patch_stride=stride,
            no_decomposition=no_decomposition,
        )

    def forward(self, pm25_hist, feature):
        """
        pm25_hist : [B, hist_len, N, 1]
        feature   : [B, hist_len + pred_len, N, F]   (F = in_dim - 1)
        returns   : [B, pred_len, N, 1]
        """
        B, T, N, _ = pm25_hist.shape
        if N != self.city_num:
            raise ValueError(
                f"WPMixerPM25 was built with city_num={self.city_num}, but got "
                f"N={N} nodes in this batch's data."
            )
        feature_hist = feature[:, :self.hist_len]
        # PM2.5 last, matching the upstream model's own channel_out-from-
        # the-end slicing convention (see module docstring).
        x_hist = torch.cat([feature_hist, pm25_hist], dim=-1)  # [B,hist_len,N,in_dim]

        C = self.in_dim
        x = x_hist.permute(0, 2, 1, 3).reshape(B * N, self.hist_len, C)  # [B*N, L, C]

        out = self.core(x)                      # [B*N, pred_len, C]
        out = out[..., -1:]                      # PM2.5 channel only -> [B*N, pred_len, 1]
        pm25_pred = out.view(B, N, self.pred_len, 1).permute(0, 2, 1, 3)  # [B, pred_len, N, 1]
        return pm25_pred
