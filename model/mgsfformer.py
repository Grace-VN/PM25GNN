"""
MGSFformer (Yu, Wang, Wang, Shao, Sun, Yao, Xu - "MGSFformer: A
Multi-Granularity Spatiotemporal Fusion Transformer for air quality
prediction", Information Fusion, 2025) adapted as a benchmark model for
this repo's harness, ported faithfully from the authors' reference
implementation: https://github.com/GestaltCogTeam/MGSFformer (no LICENSE
file in that repo at the time of porting - ported for academic benchmark
comparison with attribution, per the paper's own citation request).

Everything below `MGSFformerPM25` (RevIN, IE_block, STA_block_att and its
Time_att/space_att2/cross_att, DF_block/RF_att, and the core MGSFformer
class) is that repository's model1/*.py verbatim, merged into one file
since none of it is shared with any other model in this repo. RevIN
itself carries the original authors' own attribution comment forward
unchanged (it's borrowed by them from https://github.com/ts-kim/RevIN).

DELIBERATELY NOT using this harness's `feature` (meteorological) input:
unlike every other model here, MGSFformer's published architecture takes
only the target series' own history - its multi-granularity decomposition
operates on the PM2.5 signal itself, not on exogenous covariates. This
makes it a target-only baseline in this repo's comparison table (see
train.py's get_model() docstring note on this), not a like-for-like
ablation - extending it to consume weather features would mean real
architectural surgery beyond the published design, so that wasn't done.

Contract (matches every other model in model/, see train.py get_model()):
    MGSFformerPM25(hist_len, pred_len, in_dim, city_num, batch_size, device, ...)
    pm25_pred = model(pm25_hist, feature)
    # pm25_hist: [B, hist_len, N, 1], feature: [B, hist_len+pred_len, N, F] (unused)
    # -> pm25_pred: [B, pred_len, N, 1]

hist_len must be a multiple of 24 - the architecture's coarse-graining
scales are hardcoded to factors of 24 (see MGSFformer.__init__: `Input_len
// 24`), inherited unchanged from the reference implementation rather than
re-derived for this dataset's 3-hour native cadence (the original targets
hourly data, where 24 literally means "one day"; here it's just the
coarsest of four fixed granularity levels). This repo's config.yaml
default of hist_len=24 satisfies it exactly (the coarsest scale degenerates
to a single point, which still runs correctly - just not usefully coarse).
"""

import torch
from torch import nn


class RevIN(nn.Module):
    # code from https://github.com/ts-kim/RevIN, with minor modifications
    # (unchanged from MGSFformer's own copy of it)
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


class IE_block(nn.Module):
    def __init__(self, input_num, out_num, IE_Input_len):
        super(IE_block, self).__init__()
        self.IE_Input_len = IE_Input_len
        self.output = nn.Linear(input_num, out_num)

    def forward(self, x):
        x = x.reshape((x.shape[0], x.shape[1], x.shape[2], 1))
        # piecewise sampling
        x = IE_block.piecewise_sample(x, self.IE_Input_len)
        # dimension transformation
        x = self.output(x)
        return x

    @staticmethod
    def piecewise_sample(data, n):
        result = 0.0
        data_len = data.shape[2] // n
        for i in range(n):
            line = data[:, :, data_len * i:data_len * (i + 1), :]
            if i == 0:
                result = line
            else:
                result = torch.cat([result, line], dim=3)
        result = result.transpose(2, 3)
        return result


# temporal attention
class Time_att(nn.Module):
    def __init__(self, Input_len, dim_input, dropout, num_head):
        super(Time_att, self).__init__()
        self.query = nn.Conv2d(in_channels=dim_input, out_channels=dim_input, kernel_size=1)
        self.key = nn.Conv2d(in_channels=dim_input, out_channels=dim_input, kernel_size=1)
        self.value = nn.Conv2d(in_channels=dim_input, out_channels=dim_input, kernel_size=1)
        self.laynorm = nn.LayerNorm([Input_len])
        self.softmax = nn.Softmax(dim=-1)
        self.num_head = num_head
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(num_head, 1)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        result = 0.0
        for i in range(self.num_head):
            q = self.dropout(self.query(x)).transpose(-3, -2)
            k = self.dropout(self.key(x)).permute(0, 2, 3, 1)
            v = self.dropout(self.value(x)).transpose(-3, -2)
            kd = torch.sqrt(torch.tensor(k.shape[-1]).to(torch.float32) / self.num_head)
            line = self.dropout(self.softmax(q @ k / kd)) @ v
            if i < 1:
                result = line.unsqueeze(-1)
            else:
                result = torch.cat([result, line.unsqueeze(-1)], dim=-1)
        result = self.output(result)
        result = result.squeeze(-1)
        x = x + result.transpose(-3, -2)
        x = self.laynorm(x)
        return x


# space attention
class space_att2(nn.Module):
    def __init__(self, dim_input, dropout, num_head):
        super(space_att2, self).__init__()
        self.query = nn.Linear(dim_input, dim_input)
        self.key = nn.Linear(dim_input, dim_input)
        self.value = nn.Linear(dim_input, dim_input)
        self.softmax = nn.Softmax(dim=-1)
        self.num_head = num_head
        self.linear1 = nn.Linear(num_head, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        result = 0.0
        q = self.dropout(self.query(x))
        k = self.dropout(self.key(x))
        k = k.transpose(-2, -1)
        v = self.dropout(self.value(x))
        kd = torch.sqrt(torch.tensor(k.shape[-1]).to(torch.float32) / self.num_head)

        for i in range(self.num_head):
            line = self.dropout(self.softmax(q @ k / kd)) @ v
            if i < 1:
                result = line.unsqueeze(-1)
            else:
                result = torch.cat([result, line.unsqueeze(-1)], dim=-1)
        result = self.linear1(result)
        result = result.squeeze(-1)
        result = result.permute(0, 2, 3, 1)
        return result


# cross attention
class cross_att(nn.Module):
    def __init__(self, Input_len, dim_input, dropout, num_head):
        super(cross_att, self).__init__()
        self.query = nn.Conv2d(in_channels=dim_input, out_channels=dim_input, kernel_size=1)
        self.key = nn.Conv2d(in_channels=dim_input, out_channels=dim_input, kernel_size=1)
        self.value = nn.Conv2d(in_channels=dim_input, out_channels=dim_input, kernel_size=1)
        self.laynorm = nn.LayerNorm([Input_len])
        self.softmax = nn.Softmax(dim=-1)
        self.num_head = num_head
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(num_head, 1)

    def forward(self, x, x2):
        result = 0.0
        for i in range(self.num_head):
            q = self.dropout(self.query(x2)).transpose(-3, -2)
            k = self.dropout(self.key(x)).transpose(-3, -2)
            k = k.transpose(-2, -1)
            v = self.dropout(self.value(x)).transpose(-3, -2)

            kd = torch.sqrt(torch.tensor(k.shape[-1]).to(torch.float32) / self.num_head)
            line = self.dropout(self.softmax(q @ k / kd)) @ v
            if i < 1:
                result = line.unsqueeze(-1)
            else:
                result = torch.cat([result, line.unsqueeze(-1)], dim=-1)
        result = self.output(result)
        result = result.squeeze(-1)
        x = x.transpose(-3, -2) + result
        x = self.laynorm(x)
        return x


class STA_block_att(nn.Module):
    def __init__(self, Input_len, num_id, IE_dim, out_len, dropout, num_head):
        super(STA_block_att, self).__init__()
        self.Time_att = Time_att(Input_len, IE_dim, dropout, num_head)
        self.space_att = space_att2(num_id, dropout, num_head)
        self.cross_att = cross_att(Input_len, IE_dim, dropout, num_head)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Conv1d(in_channels=Input_len * IE_dim, out_channels=out_len, kernel_size=1)

    def forward(self, x):
        x = self.cross_att(self.Time_att(x), self.space_att(x))
        x = x.reshape((x.shape[0], x.shape[1], -1))
        x = self.linear(x.transpose(-2, -1))
        return x.transpose(-2, -1)


class RF_att(nn.Module):
    def __init__(self, dim_input):
        super(RF_att, self).__init__()
        self.QK = nn.Linear(dim_input, dim_input)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        Q_K = self.QK(x)
        Q_K = self.softmax(Q_K)
        x = x * Q_K
        return x


class DF_block(nn.Module):
    def __init__(self, num_ga, out_len):
        super(DF_block, self).__init__()
        self.att1 = RF_att(num_ga)
        self.att2 = RF_att(num_ga)
        self.att3 = RF_att(num_ga)
        self.out_len = out_len // 4

    def forward(self, x):
        line1 = x[:, :, 0:self.out_len, :]
        line1 = self.att1(line1)
        line1 = line1.sum(dim=-1)

        line2 = x[:, :, self.out_len:self.out_len * 2, :]
        line2 = self.att1(line2)
        line2 = line2.sum(dim=-1)

        line3 = x[:, :, self.out_len * 2:, :]
        line3 = self.att1(line3)
        line3 = line3.sum(dim=-1)
        x = torch.cat([line1, line2, line3], dim=2)

        return x


class MGSFformer(nn.Module):
    def __init__(self, Input_len, out_len, num_id, IE_dim, dropout, num_head):
        """
        Input_len: Historical length
        out_len: Future length
        num_id: Number of time series
        IE_dim: Embedding size
        dropout: Dropout
        num_head: Number of multi-head attention
        """
        super(MGSFformer, self).__init__()

        self.RevIN = RevIN(num_id)

        # RD-block
        self.IE_Input_len = Input_len // 24
        self.IE_block1 = IE_block(1, IE_dim, self.IE_Input_len)
        self.IE_block2 = IE_block(2, IE_dim, self.IE_Input_len)
        self.IE_block3 = IE_block(4, IE_dim, self.IE_Input_len)
        self.IE_block4 = IE_block(8, IE_dim, self.IE_Input_len)
        self.IE_block5 = IE_block(24, IE_dim, self.IE_Input_len)

        self.lay_norm1 = nn.LayerNorm([num_id, self.IE_Input_len, IE_dim])
        self.lay_norm2 = nn.LayerNorm([num_id, self.IE_Input_len, IE_dim])
        self.lay_norm3 = nn.LayerNorm([num_id, self.IE_Input_len, IE_dim])
        self.lay_norm4 = nn.LayerNorm([num_id, self.IE_Input_len, IE_dim])
        self.lay_norm5 = nn.LayerNorm([num_id, self.IE_Input_len, IE_dim])

        # STA-block
        self.ST_block1 = STA_block_att(self.IE_Input_len, num_id, IE_dim, out_len, dropout, num_head)
        self.ST_block2 = STA_block_att(self.IE_Input_len, num_id, IE_dim, out_len, dropout, num_head)
        self.ST_block3 = STA_block_att(self.IE_Input_len, num_id, IE_dim, out_len, dropout, num_head)
        self.ST_block4 = STA_block_att(self.IE_Input_len, num_id, IE_dim, out_len, dropout, num_head)
        self.ST_block5 = STA_block_att(self.IE_Input_len, num_id, IE_dim, out_len, dropout, num_head)

        # DF_block
        self.DF_block = DF_block(5, out_len)

    def forward(self, history_data):
        # Input [B,H,N,1]: B is batch size. N is the number of variables. H is the history length
        # Output [B,L,N,1]: B is batch size. N is the number of variables. L is the future length

        x = history_data[:, :, :, 0]
        x = self.RevIN(x, 'norm').transpose(-2, -1)

        x_day = MGSFformer.Get_Coarse_grain(x, 24)
        x_12h = MGSFformer.Get_Coarse_grain(x, 12)
        x_6h = MGSFformer.Get_Coarse_grain(x, 6)
        x_3h = MGSFformer.Get_Coarse_grain(x, 3)

        # RD-block
        x_day = self.IE_block1(x_day)
        x_12h = self.IE_block2(x_12h)
        x_6h = self.IE_block3(x_6h)
        x_3h = self.IE_block4(x_3h)
        x = self.IE_block5(x)

        x_day = self.lay_norm1(x_day)
        x_12h = self.lay_norm2(x_12h)
        x_6h = self.lay_norm1(x_6h)
        x_3h = self.lay_norm2(x_3h)
        x = self.lay_norm3(x)

        x_12h = x_12h - x_day
        x_6h = x_6h - x_12h
        x_3h = x_3h - x_6h
        x = x - x_3h

        # STA-block
        x_day = self.ST_block1(x_day)
        x_12h = self.ST_block2(x_12h)
        x_6h = self.ST_block3(x_6h)
        x_3h = self.ST_block4(x_3h)
        x = self.ST_block5(x)

        # DF_block
        x_day = x_day.unsqueeze(-1)
        x_12h = x_12h.unsqueeze(-1)
        x_6h = x_6h.unsqueeze(-1)
        x_3h = x_3h.unsqueeze(-1)
        x = x.unsqueeze(-1)
        x = torch.cat([x_day, x_12h, x_6h, x_3h, x], dim=-1)
        x = self.DF_block(x).transpose(-2, -1)
        x = self.RevIN(x, 'denorm').unsqueeze(-1)
        return x

    @staticmethod
    def Get_Coarse_grain(data, n):
        result = 0.0
        for i in range(n):
            line = data[:, :, i::n]
            if i == 0:
                result = line
            else:
                result = result + line
        result = result / n
        return result


class MGSFformerPM25(nn.Module):
    def __init__(self, hist_len, pred_len, in_dim, city_num, batch_size, device,
                 ie_dim=8, dropout=0.1, num_head=4):
        super(MGSFformerPM25, self).__init__()
        if hist_len % 24 != 0:
            raise ValueError(
                f"MGSFformer's coarse-graining scales are hardcoded to factors of 24 "
                f"(see model/mgsfformer.py's docstring) - hist_len must be a multiple "
                f"of 24, got hist_len={hist_len}. Try 24, 48, 72, ..."
            )
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.city_num = city_num
        self.core = MGSFformer(
            Input_len=hist_len, out_len=pred_len, num_id=city_num,
            IE_dim=ie_dim, dropout=dropout, num_head=num_head,
        )

    def forward(self, pm25_hist, feature):
        """
        pm25_hist : [B, hist_len, N, 1]
        feature   : [B, hist_len + pred_len, N, F] - NOT used, see module docstring
        returns   : [B, pred_len, N, 1]
        """
        B, T, N, _ = pm25_hist.shape
        if N != self.city_num:
            raise ValueError(
                f"MGSFformerPM25 was built with city_num={self.city_num}, but got "
                f"N={N} nodes in this batch's data."
            )
        return self.core(pm25_hist)
