"""
AGCRN (Bai, Yao, Li, Wang, Wang - "Adaptive Graph Convolutional Recurrent
Network for Traffic Forecasting", NeurIPS 2020) adapted as a benchmark
model for this repo's harness. Reference implementation:
https://github.com/LeiBAI/AGCRN (MIT-licensed).

Ported essentially verbatim from the reference repo's model/AGCN.py,
model/AGCRNCell.py, model/AGCRN.py (the entire architecture is these
three small files - everything else in that repo is training/data-loading
plumbing). The only change from the reference is AGCRN.__init__ taking
plain keyword arguments instead of an `args` namespace object.

WHY THIS ONE IS A GOOD FIT HERE: every graph-based model in this repo -
PM25_GNN, GC_LSTM, AirLapse itself - uses a FIXED graph (physical
distance / wind / geography baked in as a prior). AGCRN's whole premise
is the opposite: `node_embeddings` (self.node_embeddings below) is a
freely-learned [N, embed_dim] parameter with no physical meaning
whatsoever, and the adjacency ("supports" in AVWGCN.forward) is
recomputed from it every forward pass as softmax(relu(E @ E^T)) - the
graph structure itself is learned end-to-end from data, not supplied.
Its graph convolution weights are also node-specific, generated from
each node's embedding (weights_pool projected through node_embeddings) -
"Node Adaptive Parameter Learning" in the paper - rather than a single
shared weight matrix. That makes it a natural point of comparison for
AirLapse's explicitly physics-informed design: does a fully learned
graph beat domain knowledge on this problem, or does domain knowledge
still win?

NOT using this harness's future-known weather, like MGSFformer/PDFormer/
TimeXer before it: AGCRN.forward takes only the historical sequence
(`source`); its `targets` parameter (for teacher forcing) exists in the
reference signature but is never actually read inside forward() - the
model is a direct history -> multi-step Conv2d projection, not an
autoregressive decoder, so there's no real teacher-forcing path and
nothing this port is discarding by dropping that parameter. As with the
other three, this is inherent to a generic (traffic-forecasting-derived)
architecture that has no concept of "future weather is known", not a
simplification made for this port.

Contract (matches every other model in model/, see train.py get_model()):
    AGCRNPM25(hist_len, pred_len, in_dim, city_num, batch_size, device, ...)
    pm25_pred = model(pm25_hist, feature)
    # pm25_hist: [B, hist_len, N, 1], feature: [B, hist_len+pred_len, N, F]
    #   (only feature[:, :hist_len] is used - see note above)
    # -> pm25_pred: [B, pred_len, N, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AVWGCN(nn.Module):
    """Adaptive-Vertex-Weighted Graph Convolution: both the adjacency
    ("supports") and the per-node conv weights/bias are generated from
    node_embeddings rather than supplied - see module docstring."""

    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super(AVWGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))

    def forward(self, x, node_embeddings):
        # x shaped [B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        # output shape [B, N, C]
        node_num = node_embeddings.shape[0]
        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        support_set = [torch.eye(node_num).to(supports.device), supports]
        # default cheb_k = 3
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)
        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)  # N, cheb_k, dim_in, dim_out
        bias = torch.matmul(node_embeddings, self.bias_pool)  # N, dim_out
        x_g = torch.einsum("knm,bmc->bknc", supports, x)  # B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias  # b, N, dim_out
        return x_gconv


class AGCRNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim):
        super(AGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AVWGCN(dim_in + self.hidden_dim, 2 * dim_out, cheb_k, embed_dim)
        self.update = AVWGCN(dim_in + self.hidden_dim, dim_out, cheb_k, embed_dim)

    def forward(self, x, state, node_embeddings):
        # x: B, num_nodes, input_dim
        # state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z * state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings))
        h = r * state + (1 - r) * hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)


class AVWDCRNN(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers=1):
        super(AVWDCRNN, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim))

    def forward(self, x, init_state, node_embeddings):
        # shape of x: (B, T, N, D)
        # shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        # current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)  # (num_layers, B, N, hidden_dim)


class AGCRN(nn.Module):
    """Core model - takes plain kwargs instead of the reference repo's
    `args` namespace object; architecture otherwise unchanged."""

    def __init__(self, num_node, input_dim, rnn_units, output_dim, horizon,
                 num_layers, cheb_k, embed_dim):
        super(AGCRN, self).__init__()
        self.num_node = num_node
        self.input_dim = input_dim
        self.hidden_dim = rnn_units
        self.output_dim = output_dim
        self.horizon = horizon
        self.num_layers = num_layers

        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, embed_dim), requires_grad=True)

        self.encoder = AVWDCRNN(num_node, input_dim, rnn_units, cheb_k, embed_dim, num_layers)

        # predictor
        self.end_conv = nn.Conv2d(1, horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)

    def forward(self, source):
        # source: B, T, N, D
        init_state = self.encoder.init_hidden(source.shape[0])
        output, _ = self.encoder(source, init_state, self.node_embeddings)  # B, T, N, hidden
        output = output[:, -1:, :, :]  # B, 1, N, hidden

        # CNN based predictor
        output = self.end_conv(output)  # B, T*C, N, 1
        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node)
        output = output.permute(0, 1, 3, 2)  # B, T, N, C

        return output


class AGCRNPM25(nn.Module):
    def __init__(self, hist_len, pred_len, in_dim, city_num, batch_size, device,
                 rnn_units=64, num_layers=2, cheb_k=2, embed_dim=10):
        super(AGCRNPM25, self).__init__()
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.city_num = city_num

        self.core = AGCRN(
            num_node=city_num, input_dim=in_dim, rnn_units=rnn_units, output_dim=1,
            horizon=pred_len, num_layers=num_layers, cheb_k=cheb_k, embed_dim=embed_dim,
        )
        # AVWGCN.weights_pool/bias_pool are built via plain torch.FloatTensor(shape)
        # (uninitialized memory - not zeros, not a real distribution) and rely on
        # this init being applied externally, same as the reference repo's Run.py
        # does right after construction (not part of the model class itself).
        # Skipping this reliably trains to NaN from step one - confirmed by testing
        # without it before finding this in the reference training script.
        for p in self.core.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

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
                f"AGCRNPM25 was built with city_num={self.city_num}, but got "
                f"N={N} nodes in this batch's data."
            )
        feature_hist = feature[:, :self.hist_len]
        x = torch.cat([pm25_hist, feature_hist], dim=-1)  # [B, hist_len, N, in_dim]
        return self.core(x)
