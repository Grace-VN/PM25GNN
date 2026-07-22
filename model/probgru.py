"""
Probabilistic GRU with a single global latent variable for PM2.5
forecasting - GRU encoder/decoder baseline with an explicit noise/
uncertainty channel, no graph structure (mirrors SeqScaleModel's role
as a non-spatial baseline).

Architecture (single-shot VAE, NOT per-timestep hierarchical like
AirFormerPlusPlus's StochasticModel):
  encoder GRU(pm25_hist, feature_hist) -> h_T
  q(z|h_T)  = N(mu_q, diag(sigma_q^2))          [inference network]
  p(z)      = N(0, I)                            [fixed prior, not learned]
  decoder: GRUCell unrolled over pred_len, input = [feature_future_t, z]
           at every step (z re-injected each step, not just at h0, so it
           doesn't get washed out over a long horizon - same concern
           flagged for AirFormer's decoder bottleneck, d[..., -1:]).

Uses KNOWN FUTURE weather (feature[:, hist_len:]) as decoder input -
this is exactly the signal AirFormerPlusPlus's docstring flags as
available but unused during decoding.

Loss wiring: train.py already computes
    loss = criterion(pred, label) + kl_weight * model.last_kl_loss
whenever last_kl_loss is set (see AirFormerPlusPlus). That is, up to
constants, a negative ELBO: MSE ~ Gaussian NLL with fixed variance,
kl_weight ~ the KL coefficient. No train.py loss changes needed here -
only get_model() needs a new elif branch.

Watch for posterior collapse: the decoder has a strong non-latent signal
(future weather every step), so q(z|h_T) can collapse toward the prior
(mu_q -> 0, logvar_q -> 0, KL -> 0), making z inert. Log raw KL during
training; if it flatlines near 0 early, that's the signature. Mitigation
(KL annealing / free bits) is not implemented here - diagnose first.
"""

import torch
import torch.nn as nn


class ProbGRUModel(nn.Module):
    """
    train.py-compatible probabilistic GRU forecaster. Constructor matches
    every other benchmark model's positional signature so it drops into
    get_model()'s dispatch with only a new elif branch needed.
    """
    def __init__(self, hist_len, pred_len, in_dim, city_num, batch_size, device,
                 edge_index, edge_attr, wind_mean, wind_std,
                 hidden_dim=64, latent_dim=16, num_layers=1, dropout=0.1,
                 logvar_clamp=10.0):
        super(ProbGRUModel, self).__init__()
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.in_dim = in_dim
        self.city_num = city_num
        self.device = device
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.logvar_clamp = logvar_clamp

        # accepted for signature parity with every other benchmark model;
        # unused by design - this model has no graph awareness
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.wind_mean = wind_mean
        self.wind_std = wind_std

        # feature_dim excludes the pm25 channel that's concatenated into
        # in_dim (in_dim = feature_dim + 1); decoder only ever sees future
        # WEATHER, never future pm25 (that's the prediction target)
        self.feature_dim = in_dim - 1

        self.encoder = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

        self.decoder_init = nn.Linear(hidden_dim + latent_dim, hidden_dim)
        self.decoder_cell = nn.GRUCell(
            input_size=self.feature_dim + latent_dim,
            hidden_size=hidden_dim,
        )
        self.output_head = nn.Linear(hidden_dim, 1)

        # populated on every forward() call; train.py's train() picks this
        # up via getattr(model, 'last_kl_loss', None) - no train.py loss
        # changes needed, same hook AirFormerPlusPlus uses
        self.last_kl_loss = None

    def forward(self, pm25_hist, feature):
        """
        pm25_hist : [B, hist_len, N, 1]
        feature   : [B, hist_len + pred_len, N, F]
        returns   : [B, pred_len, N, 1]
        """
        feature_hist = feature[:, :self.hist_len]                     # [B, hist_len, N, F]
        feature_future = feature[:, self.hist_len:self.hist_len + self.pred_len]  # [B, pred_len, N, F]
        inputs = torch.cat([pm25_hist, feature_hist], dim=-1)         # [B, hist_len, N, in_dim]

        B, T, N, C = inputs.shape
        if N != self.city_num:
            raise ValueError(
                f"ProbGRUModel was built with city_num={self.city_num}, but got "
                f"N={N} nodes in this batch's data."
            )
        if feature_future.shape[1] != self.pred_len:
            raise ValueError(
                f"expected {self.pred_len} future feature steps, got "
                f"{feature_future.shape[1]} - check feature's time dimension "
                f"covers hist_len + pred_len."
            )

        # fold node dim into batch - no cross-node mixing, same convention
        # as SeqScaleModel
        x = inputs.permute(0, 2, 1, 3).reshape(B * N, T, C)             # [B*N, hist_len, in_dim]
        feat_fut = feature_future.permute(0, 2, 1, 3).reshape(
            B * N, self.pred_len, self.feature_dim)                     # [B*N, pred_len, F]

        # --- encode + inference network q(z|h_T) ---
        _, h_n = self.encoder(x)
        h_T = h_n[-1]  # last layer's final hidden state: [B*N, hidden_dim]

        mu_q = self.mu_head(h_T)
        logvar_q = torch.clamp(self.logvar_head(h_T), -self.logvar_clamp, self.logvar_clamp)

        if self.training:
            eps = torch.randn_like(mu_q)
            z = mu_q + eps * torch.exp(0.5 * logvar_q)
        else:
            # deterministic posterior mean at eval, same convention as
            # AirFormerPlusPlus's z_for_pred = z_q if training else mu_q
            z = mu_q

        # KL(q(z|h_T) || N(0, I)), closed form, mean over the batch
        kl = -0.5 * torch.mean(
            torch.sum(1 + logvar_q - mu_q.pow(2) - logvar_q.exp(), dim=-1)
        )
        self.last_kl_loss = kl

        # --- decode, conditioned on known future weather + z every step ---
        h_dec = self.decoder_init(torch.cat([h_T, z], dim=-1))
        preds = []
        for t in range(self.pred_len):
            step_in = torch.cat([feat_fut[:, t], z], dim=-1)  # re-inject z each step
            h_dec = self.decoder_cell(step_in, h_dec)
            preds.append(self.output_head(h_dec))

        out = torch.stack(preds, dim=1)  # [B*N, pred_len, 1]
        pm25_pred = out.reshape(B, N, self.pred_len, 1).permute(0, 2, 1, 3)  # [B, pred_len, N, 1]
        return pm25_pred