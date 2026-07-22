import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalMovingAvg(nn.Module):
    """Trend extraction via LEFT-PADDED-ONLY moving average.
    Replaces MovingAvgPool's symmetric padding, which leaked future info."""
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x):  # x: [*, T, C]
        x = x.transpose(-1, -2)                       # [*, C, T]
        x = F.pad(x, (self.kernel_size - 1, 0))        # left-pad only
        x = F.avg_pool1d(x, kernel_size=self.kernel_size, stride=1)
        return x.transpose(-1, -2)                     # [*, T, C]


class CausalLinear(nn.Module):
    """Linear time-mixing with a lower-triangular weight mask: output[t]
    can only be a function of input[<=t]. This is the linear-model analog
    of TemporalAttention's `torch.tril` causal mask."""
    def __init__(self, seq_len):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(seq_len) + 0.01 * torch.randn(seq_len, seq_len))
        self.bias = nn.Parameter(torch.zeros(seq_len))
        self.register_buffer('causal_mask', torch.tril(torch.ones(seq_len, seq_len)))

    def forward(self, x):  # x: [*, T, C]
        w = self.weight * self.causal_mask          # [T_out, T_in]
        x = x.transpose(-1, -2)                      # [*, C, T_in]
        x = x @ w.transpose(0, 1) + self.bias         # [*, C, T_out]
        return x.transpose(-1, -2)                    # [*, T_out, C]


class SeqScaleTemporalMixer(nn.Module):
    """
    Causal trend/seasonal linear mixer, drop-in for AdaptiveTemporalAttention.
    forward(x: [B, C, N, T]) -> [B, C, N, T]. No internal residual -
    AirFormerPlusPlus's block loop adds the residual itself. No internal
    normalization either - normalization is handled upstream (StandardScaler)
    and downstream (BatchNorm2d per block); see module docstring.

    `window_size` plays the same role as in AdaptiveTemporalAttention:
    it bounds this block's receptive field along T (matching the
    2**(blocks-b-1) hierarchy) rather than mixing over all of history at
    every block. Windows are non-overlapping and reset positions at each
    boundary, same scheme AdaptiveTemporalAttention uses (end-zero-padded
    if T isn't divisible by window_size).
    """
    def __init__(self, dim, window_size, kernel_sizes=(3, 5, 9),
                 causal=True, individual=False):
        super().__init__()
        assert causal, "AirFormerPlusPlus's stochastic stage requires causal temporal modules"
        self.window_size = max(1, window_size)
        self.individual = individual
        # kernel can't exceed the window it operates inside
        self.kernel_sizes = [min(k, self.window_size) for k in kernel_sizes]

        self.moving_avgs = nn.ModuleList([CausalMovingAvg(k) for k in self.kernel_sizes])

        # individual=True -> one CausalLinear per channel (matches DLinear's
        # "individual" mode); expensive at hidden_channels>=32, off by default.
        if individual:
            self.trend_mixers = nn.ModuleList([
                nn.ModuleList([CausalLinear(self.window_size) for _ in range(dim)])
                for _ in self.kernel_sizes
            ])
            self.seasonal_mixers = nn.ModuleList([
                nn.ModuleList([CausalLinear(self.window_size) for _ in range(dim)])
                for _ in self.kernel_sizes
            ])
        else:
            self.trend_mixers = nn.ModuleList([CausalLinear(self.window_size) for _ in self.kernel_sizes])
            self.seasonal_mixers = nn.ModuleList([CausalLinear(self.window_size) for _ in self.kernel_sizes])

        self.scale_weights = nn.Parameter(torch.ones(len(self.kernel_sizes)))

    def _apply_individual(self, mixers, x):  # x: [*, T, C]
        outs = [mixers[c](x[..., c:c+1]) for c in range(x.shape[-1])]
        return torch.cat(outs, dim=-1)

    def _mix(self, x):  # x: [*, w, C]
        scale_outs = []
        for i, avg in enumerate(self.moving_avgs):
            trend = avg(x)
            seasonal = x - trend
            if self.individual:
                trend_out = self._apply_individual(self.trend_mixers[i], trend)
                seasonal_out = self._apply_individual(self.seasonal_mixers[i], seasonal)
            else:
                trend_out = self.trend_mixers[i](trend)
                seasonal_out = self.seasonal_mixers[i](seasonal)
            scale_outs.append(trend_out + seasonal_out)

        weights = F.softmax(self.scale_weights, dim=0)
        return sum(wt * o for wt, o in zip(weights, scale_outs))

    def forward(self, x):  # x: [B, C, N, T]
        b, c, n, t = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b * n, t, c)  # [B', T, C]
        B_merged, T, C = x.shape
        w = self.window_size

        if T % w != 0:
            pad_len = w - (T % w)
            x_in = torch.cat([x, x.new_zeros(B_merged, pad_len, C)], dim=1)
            T_padded = T + pad_len
        else:
            x_in, T_padded = x, T

        x_windowed = x_in.reshape(-1, w, C)
        out = self._mix(x_windowed)
        out = out.reshape(B_merged, T_padded, C)
        if T_padded != T:
            out = out[:, :T, :]

        return out.reshape(b, n, t, C).permute(0, 3, 1, 2)  # [B, C, N, T]