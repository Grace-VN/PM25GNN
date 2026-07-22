import math
import torch
import torch.nn as nn

"""
Hierarchical latent variables with scaled dimensions.

Dimension progression (coarse -> fine):
- Level 4 (top):    2-dim  (overall trend)
- Level 3:          4-dim  (daily cycle)
- Level 2:          8-dim  (hourly dynamics)
- Level 1 (bottom): 16-dim (fine fluctuations)
"""


class LatentLayer(nn.Module):
    '''
    The latent layer to compute mean and std
    '''

    def __init__(self,
                 dm_dim,          # the dimension of deterministic states
                 latent_dim_in,   # the dimension of input latent variables
                 latent_dim_out,  # the dimension of output latent variables
                 hidden_dim,      # the intermediate dimension
                 num_layers=2,
                 learn_sigma=True,       # if False, sigma is fixed (TLAE-style ablation)
                 fixed_sigma_value=1.0): # sigma value used when learn_sigma=False
        super(LatentLayer, self).__init__()

        self.num_layers = num_layers
        self.latent_dim_in = latent_dim_in
        self.input_dim = dm_dim + latent_dim_in
        self.learn_sigma = learn_sigma
        self.enc_in = nn.Sequential(
            nn.Conv2d(self.input_dim, hidden_dim, 1))

        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv2d(hidden_dim, hidden_dim, 1))
            layers.append(nn.ReLU(inplace=True))
        self.enc_hidden = nn.Sequential(*layers)
        self.enc_out_1 = nn.Conv2d(hidden_dim, latent_dim_out, 1)

        if self.learn_sigma:
            self.enc_out_2 = nn.Conv2d(hidden_dim, latent_dim_out, 1)
        else:
            # No sigma head at all: nothing to learn, nothing to collapse.
            # We store a constant "logsigma" such that, after the caller's
            # sigma = exp(logsigma) + 1e-3 step, sigma == fixed_sigma_value.
            self.enc_out_2 = None
            const = math.log(max(fixed_sigma_value - 1e-3, 1e-6))
            self.register_buffer(
                'fixed_logsigma', torch.tensor(float(const))
            )

    def forward(self, x):
        # x: [b, c, n, t]
        # NOTE: this shape-repair fallback should never fire once the
        # layer ordering in HierarchicalStochasticModel is correct. It's
        # kept only as a loud safety net (see assertion below) rather
        # than a silent fixer, since silently padding/truncating channels
        # is what masked the original ordering bug.
        if x.shape[1] != self.input_dim:
            raise RuntimeError(
                f'LatentLayer received {x.shape[1]} channels but expected '
                f'{self.input_dim}. This means a caller is feeding the '
                f'wrong level into this layer — check the hierarchy '
                f'ordering rather than padding/truncating here.'
            )

        h = self.enc_in(x)
        for layer in self.enc_hidden:
            h = layer(h)

        mu_raw = self.enc_out_1(h)
        mu = torch.minimum(mu_raw, torch.ones_like(mu_raw) * 10)

        if self.learn_sigma:
            sigma_raw = self.enc_out_2(h)
            logsigma = torch.minimum(sigma_raw, torch.ones_like(sigma_raw) * 10)
        else:
            # Broadcast the fixed constant to mu's shape so downstream
            # code (exp(logsigma) + 1e-3, reparameterize, etc.) needs no
            # special-casing for the fixed-variance path.
            logsigma = self.fixed_logsigma.expand_as(mu)

        return mu, logsigma


class HierarchicalStochasticModel(nn.Module):
    """
    Hierarchical latent variables with scaled dimensions.

    Dimension progression (coarse -> fine):
    - Level 4 (top):    2-dim  (overall trend)
    - Level 3:          4-dim  (daily cycle)
    - Level 2:          8-dim  (hourly dynamics)
    - Level 1 (bottom): 16-dim (fine fluctuations)
    """

    def __init__(self, dm_dim, base_latent_dim=2, num_blocks=4,
                 growth_factor=2.0, num_layers_per_block=2,
                 learn_variance=True, fixed_sigma_value=1.0):
        """
        Args:
            learn_variance: Controls the TLAE-style fixed-variance ablation
                (Nguyen & Quanz, 2021, sec. "Probabilistic Prediction": they
                fix latent sigma^2=1 instead of learning it, to reduce
                overfitting risk). Accepts either:
                  - a single bool applied to every level, e.g.
                    learn_variance=False fixes ALL levels' sigma to
                    fixed_sigma_value (full ablation), or
                  - a list/tuple of `num_blocks` bools ordered coarse -> fine
                    ([dim2, dim4, dim8, dim16]), letting you fix only some
                    levels, e.g. [False, False, True, True] fixes the two
                    coarse levels and learns sigma only at the two finest
                    levels.
            fixed_sigma_value: sigma used wherever learn_variance is False.

            NOTE: for a fair posterior-vs-prior KL comparison, instantiate
            the generative (prior) and inference (posterior) models with
            the *same* learn_variance / fixed_sigma_value settings.
        """
        super().__init__()

        self.num_blocks = num_blocks
        self.base_latent_dim = base_latent_dim
        self.growth_factor = growth_factor

        # latent_dims[0] = 2 (top/coarsest) ... latent_dims[-1] = 16 (bottom/finest)
        self.latent_dims = [int(base_latent_dim * (growth_factor ** i))
                            for i in range(num_blocks)]
        print(f"Hierarchical latent dims (coarse->fine): {self.latent_dims}")

        if isinstance(learn_variance, bool):
            learn_variance_per_level = [learn_variance] * num_blocks
        else:
            learn_variance_per_level = list(learn_variance)
            assert len(learn_variance_per_level) == num_blocks, (
                f'learn_variance list must have {num_blocks} entries '
                f'(coarse->fine), got {len(learn_variance_per_level)}'
            )
        self.learn_variance_per_level = learn_variance_per_level
        print(f"Learn sigma per level (coarse->fine): {learn_variance_per_level}")

        self.layers = nn.ModuleList()
        # Build from FINEST to COARSEST so that self.layers[-1] ends up
        # being the top layer (latent_dim_in=0), matching forward()'s
        # assumption that self.layers[-1] is the unconditioned starting
        # point of the top-down pass.
        for level_idx in reversed(range(num_blocks)):
            latent_dim_out = self.latent_dims[level_idx]
            latent_dim_in = self.latent_dims[level_idx - 1] if level_idx > 0 else 0
            hidden_dim = latent_dim_out * 2

            self.layers.append(
                LatentLayer(
                    dm_dim=dm_dim,
                    latent_dim_in=latent_dim_in,
                    latent_dim_out=latent_dim_out,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers_per_block,
                    learn_sigma=learn_variance_per_level[level_idx],
                    fixed_sigma_value=fixed_sigma_value
                )
            )
        # self.layers[-1] -> dim 2,  latent_dim_in=0  (top, unconditioned)
        # self.layers[0]  -> dim 16, latent_dim_in=8  (bottom, finest)

    def reparameterize(self, mu, sigma):
        """Standard reparameterization trick"""
        eps = torch.randn_like(sigma, requires_grad=False)
        return mu + eps * sigma

    def forward(self, d):
        """
        Forward pass through hierarchical model.

        Args:
            d: Deterministic states [num_blocks, b, c, n, t]

        Returns:
            z, mus, sigmas: lists ordered coarse -> fine
                (index 0 = 2-dim top level, index -1 = 16-dim bottom level).
                Kept as lists (not stacked) because latent dims differ
                across levels.
        """
        # Top-down sampling: start at the top (coarsest, unconditioned) level
        _mu, _logsigma = self.layers[-1](d[-1])
        _sigma = torch.exp(_logsigma) + 1e-3

        mus = [_mu]
        sigmas = [_sigma]
        z = [self.reparameterize(_mu, _sigma)]

        # Walk down through progressively finer levels, each conditioned
        # on the latent sample from the level above.
        for i in reversed(range(len(self.layers) - 1)):
            combined = torch.cat((d[i], z[-1]), dim=1)
            _mu, _logsigma = self.layers[i](combined)
            _sigma = torch.exp(_logsigma) + 1e-3

            mus.append(_mu)
            sigmas.append(_sigma)
            z.append(self.reparameterize(_mu, _sigma))

        # mus/sigmas/z are now ordered [dim2, dim4, dim8, dim16]
        return z, mus, sigmas


class HierarchicalKLLoss(nn.Module):
    """
    Per-level KL divergence with decorrelation penalty.
    Based on Valpola (2015) Ladder Networks.

    Expects mu_q/sigma_q/mu_p/sigma_p/z_q ordered coarse -> fine
    (index 0 = dim 2 / top, index -1 = dim 16 / bottom), matching
    HierarchicalStochasticModel.forward's output order.
    """

    def __init__(self, num_blocks=4):
        super().__init__()

        # Per-level KL weights, coarse -> fine: [dim2, dim4, dim8, dim16].
        # Higher weight at fine scales to encourage learning details.
        self.kl_weights = nn.Parameter(
            torch.tensor([0.5, 1.0, 2.0, 10.0]),
            requires_grad=False
        )

        # Decorrelation weights (prevent posterior collapse), same order.
        self.decorr_weights = nn.Parameter(
            torch.tensor([0.005, 0.01, 0.05, 0.1]),
            requires_grad=False
        )

    def forward(self, mu_q, sigma_q, mu_p, sigma_p, z_q):
        """
        Compute hierarchical KL loss with decorrelation.

        Args:
            mu_q, sigma_q: Inference posterior params, list of [b, d_i, n, t]
            mu_p, sigma_p: Generative prior params, list of [b, d_i, n, t]
            z_q: Sampled latents from inference, list of [b, d_i, n, t]

        Returns:
            total_kl: Total weighted KL (scalar tensor)
            kl_dict: Per-level breakdown for logging
        """

        num_blocks = len(mu_q)
        total_kl = 0
        kl_dict = {}

        for i in range(num_blocks):
            p_dist = torch.distributions.Normal(mu_p[i], sigma_p[i])
            q_dist = torch.distributions.Normal(mu_q[i], sigma_q[i])

            kl_i = torch.distributions.kl_divergence(q_dist, p_dist).mean()
            weighted_kl = self.kl_weights[i] * kl_i

            total_kl = total_kl + weighted_kl
            kl_dict[f'kl_level_{i+1}'] = kl_i.item()
            kl_dict[f'kl_weighted_{i+1}'] = weighted_kl.item()

            # Decorrelation penalty (prevent collapse): lightweight
            # per-dimension variance penalty instead of a full
            # covariance/eigendecomposition.
            h = z_q[i]  # [b, d, n, t]
            h_flat = h.reshape(h.shape[0], h.shape[1], -1)  # [b, d, n*t]
            var = h_flat.var(dim=-1, unbiased=False).clamp_min(1e-6)
            decorr_penalty = torch.sum(var - torch.log(var + 1e-8) - 1.0)
            weighted_decorr = self.decorr_weights[i] * decorr_penalty

            total_kl = total_kl + weighted_decorr
            kl_dict[f'decorr_level_{i+1}'] = weighted_decorr.item()

        return total_kl, kl_dict


# ==================== CHANGE SUMMARY ====================
# FIXED:   layer construction order so self.layers[-1] is the top
#          (unconditioned) level, matching forward()'s assumption
# FIXED:   kl_weights / decorr_weights now increase coarse->fine
#          (0.5,1.0,2.0,10.0 / 0.005,0.01,0.05,0.1) matching design notes
# FIXED:   LatentLayer.forward now raises on shape mismatch instead of
#          silently padding/truncating, so future ordering bugs fail loudly
# ADDED:   learn_variance ablation (TLAE, Nguyen & Quanz 2021), fixing
#          sigma^2 instead of learning it, globally or per level, via
#          HierarchicalStochasticModel(..., learn_variance=..., fixed_sigma_value=...)
# =====================================================

# ==================== USAGE: fixed-variance ablation ====================
# All levels learn sigma (current default / original behavior):
#   model = HierarchicalStochasticModel(dm_dim)
#
# TLAE-style full ablation: fix sigma=1.0 everywhere, learn only mu:
#   model = HierarchicalStochasticModel(dm_dim, learn_variance=False)
#
# Partial ablation: fix sigma at the two coarse levels (dim2, dim4),
# still learn it at the two finest levels (dim8, dim16):
#   model = HierarchicalStochasticModel(
#       dm_dim, learn_variance=[False, False, True, True])
#
# IMPORTANT: use the SAME learn_variance / fixed_sigma_value for both
# the generative (prior) and inference (posterior) model instances so
# the KL divergence in HierarchicalKLLoss compares like with like.
# ==========================================================================