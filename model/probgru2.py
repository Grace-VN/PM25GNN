import torch
import torch.nn as nn


class GraphRelevanceAttention(nn.Module):
    """
    One-shot cross-node mixing: for each node i, pulls a relevance-weighted
    combination of its graph neighbors' summary vectors. Restricted to real
    edges (edge_index) - never computes relevance for non-neighbor pairs -
    and edge_attr (distance/wind/etc.) biases the score, so it isn't scoring
    from learned similarity alone.

    N is small (184 for KnowAir), so a dense [N, N] score matrix is cheap;
    non-edges are masked to -inf before softmax rather than skipped, which
    keeps the implementation simple without meaningful compute cost at this
    scale. Revisit with sparse ops only if N grows into the AirFormer
    station-count range.
    """
    def __init__(self, hidden_dim, edge_dim, attn_dim=32, num_nodes=None,
                 edge_index=None, edge_attr=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attn_dim = attn_dim

        self.q_proj = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.edge_bias_proj = nn.Linear(edge_dim, 1)

        self.scale = attn_dim ** -0.5

        # precompute the fixed mask + edge-bias matrix once, since the graph
        # structure (edge_index/edge_attr) is static, passed at construction
        assert num_nodes is not None and edge_index is not None and edge_attr is not None
        src, dst = edge_index[0], edge_index[1]  # src -> dst, i.e. dst attends to src

        mask = torch.full((num_nodes, num_nodes), False)
        mask[dst, src] = True  # mask[i, j] = True if j is a real neighbor of i
        self.register_buffer('edge_mask', mask)

        # edge_attr bias scattered into [N, N]; non-edges stay 0 but get
        # masked to -inf anyway via edge_mask before softmax
        bias = torch.zeros(num_nodes, num_nodes, edge_attr.shape[-1])
        bias[dst, src] = edge_attr.float()
        self.register_buffer('edge_attr_dense', bias)

    def forward(self, h):
        """
        h: [B, N, hidden_dim] - per-node summary vectors
        returns: [B, N, hidden_dim] - relevance-mixed summaries
        """
        B, N, _ = h.shape

        q = self.q_proj(h)                                  # [B, N, attn_dim]
        k = self.k_proj(h)                                  # [B, N, attn_dim]
        v = self.v_proj(h)                                  # [B, N, hidden_dim]

        content_score = torch.bmm(q, k.transpose(1, 2)) * self.scale   # [B, N, N]
        edge_bias = self.edge_bias_proj(self.edge_attr_dense).squeeze(-1)  # [N, N]

        score = content_score + edge_bias.unsqueeze(0)      # [B, N, N]
        score = score.masked_fill(~self.edge_mask.unsqueeze(0), float('-inf'))

        weights = torch.softmax(score, dim=-1)               # [B, N, N]
        # nodes with zero real neighbors would produce all -inf -> nan;
        # guard by zeroing weights for isolated rows
        weights = torch.nan_to_num(weights, nan=0.0)

        context = torch.bmm(weights, v)                      # [B, N, hidden_dim]
        return context


class ProbGRUModel2(nn.Module):
    """
    ProbGRUModel + one bottleneck-level graph relevance mixing step.
    Per-node GRU encoding is untouched (still one fused nn.GRU call);
    cross-node mixing happens once, on the pooled h_T summaries, before
    the latent heads. Isolated on purpose - do not also add encoder
    self-attention pooling here, that's ProbGRUAttnModel's ablation arm.
    """
    def __init__(self, hist_len, pred_len, in_dim, city_num, batch_size, device,
                 edge_index, edge_attr, wind_mean, wind_std,
                 hidden_dim=64, latent_dim=16, edge_attn_dim=32, num_layers=1,
                 dropout=0.1, logvar_clamp=10.0):
        super(ProbGRUModel2, self).__init__()
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.in_dim = in_dim
        self.city_num = city_num
        self.device = device
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.logvar_clamp = logvar_clamp

        # edge_index/edge_attr are load-bearing here, unlike ProbGRUModel
        # where they're accepted-but-unused for signature parity - this
        # class is no longer the "no graph awareness" baseline
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.wind_mean = wind_mean
        self.wind_std = wind_std

        self.feature_dim = in_dim - 1

        self.encoder = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.graph_attn = GraphRelevanceAttention(
            hidden_dim=hidden_dim,
            edge_dim=edge_attr.shape[-1],
            attn_dim=edge_attn_dim,
            num_nodes=city_num,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )

        # combine the pure per-node summary with the graph-mixed context via
        # a learned gate, rather than replacing h_T outright - lets the model
        # fall back toward the ungated baseline if mixing doesn't help
        self.mix_gate = nn.Linear(hidden_dim * 2, hidden_dim)

        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

        self.decoder_init = nn.Linear(hidden_dim + latent_dim, hidden_dim)
        self.decoder_cell = nn.GRUCell(
            input_size=self.feature_dim + latent_dim,
            hidden_size=hidden_dim,
        )
        self.output_head = nn.Linear(hidden_dim, 1)
        self.last_kl_loss = None

    def forward(self, pm25_hist, feature):
        feature_hist = feature[:, :self.hist_len]
        feature_future = feature[:, self.hist_len:self.hist_len + self.pred_len]
        inputs = torch.cat([pm25_hist, feature_hist], dim=-1)

        B, T, N, C = inputs.shape
        if N != self.city_num:
            raise ValueError(
                f"ProbGRUGraphAttnModel was built with city_num={self.city_num}, but got "
                f"N={N} nodes in this batch's data."
            )
        if feature_future.shape[1] != self.pred_len:
            raise ValueError(
                f"expected {self.pred_len} future feature steps, got "
                f"{feature_future.shape[1]} - check feature's time dimension "
                f"covers hist_len + pred_len."
            )

        x = inputs.permute(0, 2, 1, 3).reshape(B * N, T, C)
        feat_fut = feature_future.permute(0, 2, 1, 3).reshape(
            B * N, self.pred_len, self.feature_dim)

        # --- per-node encoding, unchanged from ProbGRUModel ---
        _, h_n = self.encoder(x)
        h_T = h_n[-1]                                     # [B*N, hidden_dim]

        # --- one-shot cross-node mixing at the bottleneck ---
        h_grid = h_T.reshape(B, N, self.hidden_dim)
        context = self.graph_attn(h_grid)                  # [B, N, hidden_dim]
        h_mixed = self.mix_gate(torch.cat([h_grid, context], dim=-1))
        h_mixed = h_mixed.reshape(B * N, self.hidden_dim)   # [B*N, hidden_dim]

        # --- inference network q(z|h_mixed), same as ProbGRUModel otherwise ---
        mu_q = self.mu_head(h_mixed)
        logvar_q = torch.clamp(self.logvar_head(h_mixed), -self.logvar_clamp, self.logvar_clamp)

        if self.training:
            eps = torch.randn_like(mu_q)
            z = mu_q + eps * torch.exp(0.5 * logvar_q)
        else:
            z = mu_q

        kl = -0.5 * torch.mean(
            torch.sum(1 + logvar_q - mu_q.pow(2) - logvar_q.exp(), dim=-1)
        )
        self.last_kl_loss = kl

        # --- decode, identical to ProbGRUModel ---
        h_dec = self.decoder_init(torch.cat([h_mixed, z], dim=-1))
        preds = []
        for t in range(self.pred_len):
            step_in = torch.cat([feat_fut[:, t], z], dim=-1)
            h_dec = self.decoder_cell(step_in, h_dec)
            preds.append(self.output_head(h_dec))

        out = torch.stack(preds, dim=1)
        pm25_pred = out.reshape(B, N, self.pred_len, 1).permute(0, 2, 1, 3)
        return pm25_pred