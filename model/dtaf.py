"""
DTAF (Lu, Chen, Guo, Shu, Wang & Yang, AAAI 2026, "Towards Non-Stationary
Time Series Forecasting with Temporal Stabilization and Frequency
Differencing", arXiv:2511.08229) adapted as a plain, non-graph benchmark
model for this repo's harness.

NOT A PORT - AN INDEPENDENT REIMPLEMENTATION FROM THE PAPER, AND WHY:
DTAF's official repository has no LICENSE file (confirmed via the GitHub
API: license field is null) - only a pointer to its own experiment
scripts for reproducing the paper's results. Under default copyright,
that means the source code is all-rights-reserved: publishing a repo
without a license grants the right to view and run it as distributed, not
the separate right to copy its source into a different published project.
This file is therefore written from scratch, based only on the published
paper's own description of the method (its arXiv abstract and
methodology), never the official repo's code - the same approach used for
model/timefilter.py in this repo, for the same reason. Where the paper's
description leaves an implementation detail unspecified, that is called
out explicitly below as this file's own design choice - this is a good-
faith functional reimplementation of the architecture described, for
benchmarking purposes, not a claim of bit-for-bit fidelity to the
authors' own code.

THE METHOD, as described in the paper - a dual-branch (temporal +
frequency) architecture built on overlapping patches:
  1. Instance normalization (RevIN-style), then the lookback window is
     split into overlapping patches (patch_len, stride) and linearly
     embedded, the same patching convention as PatchTST/WPMixer.
  2. Temporal Stabilizing Fusion (TFS): a mixture-of-experts filter
     estimates each patch's own non-stationary component (a router scores
     several small expert networks, their outputs combined by the
     router's softmax weights) and SUBTRACTS that estimate from the patch
     embedding, leaving a more stationary residual - encouraged toward
     genuinely diverse (not redundant) experts via a KL-divergence
     diversity term among the experts' own output distributions. The
     stabilized patches are then fused with their own causally-masked
     history: a learned weighting scores each patch against every
     earlier (never later) patch, aggregates that weighted history
     through an MLP, and combines it with a learned gate on the current
     patch's own contribution.
  3. Frequency Wave Modeling (FWM): the temporal branch's output is
     Fourier-transformed and consecutive patches' spectra are DIFFERENCED
     (magnitude at patch t minus patch t-1) to score which frequency
     components are shifting the most from one patch to the next: only
     the top-k most-volatile components survive (the rest zeroed) before
     an inverse transform back to the embedding domain, highlighting
     exactly the non-stationary spectral content the temporal branch's
     causal fusion might smooth over.
  4. Both branches run their own self-attention, the two outputs are
     concatenated and projected to the forecast horizon.

THIS FILE'S OWN DESIGN CHOICES (unspecified, or ambiguous between the
paper's abstract-level description and what a from-scratch reader would
otherwise have to guess at, decided independently here):
  - Experts are small two-layer MLPs (Linear - GELU - Linear, reducing to
    d_model // expert_reduction and back) - the paper's own description
    calls these "multiple linear layers," which this follows directly;
    no more exotic per-expert architecture is implied by that phrasing.
  - The expert-diversity KL term is computed as the average pairwise
    KL-divergence between each pair of experts' own softmax-normalized
    output distributions (over the d_model axis), exposed as
    `last_moe_diversity_loss` - a NEW generic opt-in aux-loss hook added
    to train.py's training loop alongside the existing last_kl_loss/
    last_alignment_loss/last_memory_loss ones (kept separate from
    last_kl_loss's existing name since that already means something
    different - VAE-style latent regularization - for AirFormer/
    AirPhyNet/AirLapse; reusing it here for an unrelated purpose would
    tie two independent hyperparameters to one config key).
  - The causal historical-fusion weighting is applied directly to the
    stabilized patch embeddings (no explicit separate trend/seasonal
    moving-average decomposition beforehand) - the paper mentions trend-
    seasonal decomposition as part of this module but doesn't specify
    precisely how its output feeds the rest of the fusion, so this keeps
    the causal-attention idea (the part unambiguous from the paper's own
    description) and skips the decomposition step it left underspecified,
    rather than guessing at machinery neither confirmed by the paper's
    own text nor safe to reconstruct from the unlicensed repo.
  - The frequency-differencing FFT operates over the embedding (d_model)
    axis, treating each patch's own embedding vector as the signal to
    transform, and differencing is taken across consecutive PATCHES for
    each embedding-frequency bin - i.e. "which parts of the learned patch
    representation are changing fastest, patch to patch" - matching the
    paper's own "frequency differencing across patches" framing.

CHANNEL CONVENTION: like PatchTSTPM25, every scalar channel (PM2.5 plus
every weather variable, at every station) is forecast by ONE SHARED
backbone, run independently per channel - nothing in the paper's own
description implies cross-channel modeling the way TimeFilter's
explicitly cross-channel graph does, so this folds BOTH the city
dimension AND the channel dimension into batch (same convention as this
repo's PatchTSTPM25).

IMPORTANT LIMITATION (inherent to the architecture as described, not a
bug): like PatchTSTPM25/WPMixerPM25/MGSFformer/TimeXer/AGCRN, this is
purely autoregressive from the lookback window - nothing in the described
architecture has a point where a future-known covariate would enter, so
`feature[:, hist_len:]` (the future weather this harness makes available)
goes unused here, same as every other patch-based baseline in this repo.

Contract (matches every other model in model/, see train.py get_model()):
    DTAFPM25(hist_len, pred_len, in_dim, city_num, batch_size, device, ...)
    pm25_pred = model(pm25_hist, feature)
    # pm25_hist: [B, hist_len, N, 1], feature: [B, hist_len+pred_len, N, F]
    # -> pm25_pred: [B, pred_len, N, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _RevIN(nn.Module):
    """Reversible instance normalization (Kim et al. 2022), as used by
    this repo's own PatchTSTPM25: normalize each instance over the time
    dimension before the backbone, denormalize the forecast with the same
    per-instance statistics afterward."""

    def __init__(self, eps=1e-5):
        super(_RevIN, self).__init__()
        self.eps = eps
        self.mean = None
        self.stdev = None

    def normalize(self, x):
        self.mean = x.mean(dim=1, keepdim=True).detach()
        self.stdev = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
        return (x - self.mean) / self.stdev

    def denormalize(self, x):
        return x * self.stdev + self.mean


class _NonStationaryFilter(nn.Module):
    """Mixture-of-experts subtractive filter: several small experts each
    estimate a candidate non-stationary component of the patch embedding,
    a router combines them by softmax weight, and that combined estimate
    is SUBTRACTED from the input, leaving a (hopefully more stationary)
    residual. `last_diversity_score` (populated each forward call) feeds
    the KL-diversity auxiliary loss described in the module docstring."""

    def __init__(self, d_model, expert_num, expert_reduction):
        super(_NonStationaryFilter, self).__init__()
        hidden = max(1, d_model // expert_reduction)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, d_model))
            for _ in range(expert_num)
        ])
        self.router = nn.Linear(d_model, expert_num)
        self.last_diversity_score = None

    def forward(self, x):
        # x: [B, patch_num, d_model]
        expert_out = torch.stack([e(x) for e in self.experts], dim=-2)  # [B, patch_num, expert_num, d_model]
        route_w = torch.softmax(self.router(x), dim=-1)                  # [B, patch_num, expert_num]
        non_stationary = torch.einsum('bpe,bped->bpd', route_w, expert_out)

        # average pairwise KL divergence between experts' own (softmax-
        # normalized-over-d_model) output distributions - MAXIMIZED via a
        # negative sign, so experts are pushed toward genuinely different
        # non-stationary hypotheses rather than redundant ones.
        log_p = torch.log_softmax(expert_out, dim=-1)                    # [B, patch_num, expert_num, d_model]
        p = log_p.exp()
        E = len(self.experts)
        if E > 1:
            kl_sum = 0.0
            count = 0
            for a in range(E):
                for b in range(E):
                    if a == b:
                        continue
                    kl_sum = kl_sum + F.kl_div(log_p[:, :, b], p[:, :, a], reduction='batchmean')
                    count += 1
            self.last_diversity_score = -(kl_sum / count)
        else:
            self.last_diversity_score = torch.zeros((), device=x.device)

        return x - non_stationary


class _CausalHistoryFusion(nn.Module):
    """Fuses each (stabilized) patch with a causally-masked weighted
    aggregate of its own history (never its future - the mask is lower-
    triangular), plus a learned gate on the current patch's own
    contribution. See module docstring for what this simplifies relative
    to the paper's own (underspecified, from its abstract-level
    description) trend/seasonal step."""

    def __init__(self, d_model, patch_num, dropout):
        super(_CausalHistoryFusion, self).__init__()
        self.weight_proj = nn.Linear(d_model, patch_num)
        self.history_mlp = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        causal = torch.tril(torch.ones(patch_num, patch_num, dtype=torch.bool))
        self.register_buffer('causal_mask', causal)

    def forward(self, x):
        # x: [B, patch_num, d_model]
        raw_weight = self.weight_proj(x)                                  # [B, patch_num, patch_num]
        raw_weight = raw_weight.masked_fill(~self.causal_mask, float('-inf'))
        weight = torch.softmax(raw_weight, dim=-1)                        # each row: distribution over j <= i
        history = torch.bmm(weight, x)
        history = self.dropout(self.history_mlp(history))

        gate = torch.sigmoid(self.gate(x))
        return gate * x + history


class _FrequencyWaveFilter(nn.Module):
    """Highlights the embedding-frequency components changing fastest
    from one patch to the next (see module docstring): FFT each patch's
    own d_model embedding, difference consecutive patches' magnitudes to
    score every frequency bin, keep only the top-k most-volatile bins per
    patch, then inverse-transform back. The first patch (no predecessor
    to difference against) passes through unfiltered."""

    def __init__(self, d_model, topk):
        super(_FrequencyWaveFilter, self).__init__()
        self.topk = min(topk, d_model // 2 + 1)

    def forward(self, x):
        # x: [B, patch_num, d_model]
        freq = torch.fft.rfft(x, dim=-1)                                  # [B, patch_num, d_model//2+1] complex
        mag = freq.abs()
        wave = torch.zeros_like(mag)
        wave[:, 1:, :] = mag[:, 1:, :] - mag[:, :-1, :]                   # differencing across patches

        topk_idx = wave.topk(self.topk, dim=-1).indices
        keep = torch.zeros_like(wave, dtype=torch.bool)
        keep.scatter_(-1, topk_idx, True)
        filtered = torch.where(keep, freq, torch.zeros_like(freq))

        out = torch.fft.irfft(filtered, n=x.shape[-1], dim=-1)
        out = out.clone()
        out[:, 0, :] = x[:, 0, :]                                          # no predecessor to difference patch 0 against
        return out


class _SelfAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super(_SelfAttentionBlock, self).__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        return self.norm(x + self.dropout(attn_out))


class DTAFBlock(nn.Module):
    """One Temporal Stabilizing Fusion block (non-stationary filtering +
    causal history fusion) - see module docstring."""

    def __init__(self, d_model, patch_num, expert_num, expert_reduction, dropout):
        super(DTAFBlock, self).__init__()
        self.filter = _NonStationaryFilter(d_model, expert_num, expert_reduction)
        self.fusion = _CausalHistoryFusion(d_model, patch_num, dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        stabilized = self.filter(x)
        fused = self.fusion(stabilized)
        return self.norm(x + fused), self.filter.last_diversity_score


class DTAFPM25(nn.Module):
    def __init__(self, hist_len, pred_len, in_dim, city_num, batch_size, device,
                 patch_len=8, stride=4, d_model=32, n_heads=4, e_layers=2,
                 expert_num=4, expert_reduction=2, topk_freq=8,
                 dropout=0.1, target_channel=0):
        super(DTAFPM25, self).__init__()
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.in_dim = in_dim
        self.city_num = city_num
        self.device = device
        self.patch_len = patch_len
        self.stride = stride
        self.target_channel = target_channel

        padded_len = hist_len + stride
        self.patch_num = (padded_len - patch_len) // stride + 1

        self.revin = _RevIN()
        self.patch_embedding = nn.Linear(patch_len, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.patch_num, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        self.embed_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            DTAFBlock(d_model, self.patch_num, expert_num, expert_reduction, dropout)
            for _ in range(e_layers)
        ])
        self.freq_filter = _FrequencyWaveFilter(d_model, topk_freq)

        self.temporal_attn = _SelfAttentionBlock(d_model, n_heads, dropout)
        self.frequency_attn = _SelfAttentionBlock(d_model, n_heads, dropout)

        self.head = nn.Sequential(
            nn.Flatten(start_dim=-2, end_dim=-1),
            nn.Dropout(dropout),
            nn.Linear(2 * self.patch_num * d_model, pred_len),
        )
        self.last_moe_diversity_loss = None

    def forward(self, pm25_hist, feature):
        """
        pm25_hist : [B, hist_len, N, 1]
        feature   : [B, hist_len + pred_len, N, F]   (F = in_dim - 1)
        returns   : [B, pred_len, N, 1]
        """
        B, T, N, _ = pm25_hist.shape
        if N != self.city_num:
            raise ValueError(
                f"DTAFPM25 was built with city_num={self.city_num}, but got "
                f"N={N} nodes in this batch's data."
            )
        feature_hist = feature[:, :self.hist_len]
        x_hist = torch.cat([pm25_hist, feature_hist], dim=-1)  # [B,hist_len,N,in_dim]

        C = self.in_dim
        x = x_hist.permute(0, 2, 1, 3).reshape(B * N, self.hist_len, C)  # [B*N, L, C]
        x = self.revin.normalize(x)

        x = x.permute(0, 2, 1)                                            # [B*N, C, L]
        x = F.pad(x, (0, self.stride), mode='replicate')
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # [B*N, C, patch_num, patch_len]
        BN, C_, P, _ = x.shape
        x = x.reshape(BN * C_, P, self.patch_len)                         # channel-independent: fold C into batch too

        h = self.patch_embedding(x) + self.pos_embedding
        h = self.embed_dropout(h)

        diversity_terms = []
        for block in self.blocks:
            h, diversity = block(h)
            diversity_terms.append(diversity)
        self.last_moe_diversity_loss = torch.stack(diversity_terms).mean()

        h_freq = self.freq_filter(h)

        h_t = self.temporal_attn(h)
        h_f = self.frequency_attn(h_freq)
        fused = torch.cat([h_t, h_f], dim=-2)                              # [B*N*C, 2*patch_num, d_model]

        out = self.head(fused)                                             # [B*N*C, pred_len]
        out = out.view(BN, C_, self.pred_len).permute(0, 2, 1)             # [B*N, pred_len, C]
        out = self.revin.denormalize(out)

        out = out[..., self.target_channel:self.target_channel + 1]        # [B*N, pred_len, 1]
        pm25_pred = out.view(B, N, self.pred_len, 1).permute(0, 2, 1, 3)   # [B, pred_len, N, 1]
        return pm25_pred
