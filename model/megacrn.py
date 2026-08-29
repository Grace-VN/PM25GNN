"""
MegaCRN (Jiang, Han, Zhao, Wang, Wang - "Spatio-Temporal Meta-Graph
Learning for Traffic Forecasting", AAAI 2023) adapted as a benchmark
model for this repo's harness. Reference implementation:
https://github.com/deepkashiwa20/MegaCRN (MIT-licensed).

Ported verbatim from the reference repo's model/MegaCRN.py (fully
self-contained - no framework dependency beyond plain torch/numpy,
unlike PDFormer's LibCity coupling).

WHY THIS ONE IS A GOOD FIT HERE, on top of AGCRN (model/agcrn.py):
MegaCRN also learns its graph structure end-to-end rather than using a
fixed physical prior, but two ways it goes further than AGCRN are
directly relevant to this repo's own design:
  - It's ASYMMETRIC: `g1`/`g2` below are two separate directional graphs
    (node_embeddings1 @ node_embeddings2^T and its transpose-ish
    counterpart), not one symmetric adjacency - i.e. "does station A
    influence B" and "does B influence A" get independently learned
    weights. That's the same asymmetry AirLapse's wind-direction bias
    encodes from physics (upwind/downwind aren't symmetric); MegaCRN
    tries to discover it purely from data instead.
  - Its graph comes from a shared MEMORY BANK (`self.memory['Memory']`,
    `mem_num` learned prototype patterns) rather than a free [N,
    embed_dim] parameter per node the way AGCRN's node_embeddings is -
    the paper's actual contribution ("meta-graph learning") is that
    tying the graph to a compact, shared set of patterns generalizes
    better than a fully free-form per-node embedding.

UNLIKE MGSFformer/PDFormer/TimeXer/AGCRN, this one DOES use this
harness's future-known weather: MegaCRN.forward takes `y_cov`, a
per-future-step covariate fed to the decoder at every horizon step
(concatenated with the previous step's own prediction) - exactly this
repo's "future weather is known" convention that AirLapse/GC_LSTM/
PM25_GNN/AirDDE/AirPhyNet/AirDualODE already rely on. This is the first
of the newly-added literature benchmarks (MGSFformer, PDFormer [dropped],
TimeXer, AGCRN) that can be a genuine like-for-like comparison on that
front rather than a documented history-only exception.

ONE SIMPLIFICATION MADE HERE: the reference decoder does scheduled
sampling / curriculum-learning teacher forcing during training (`labels`
+ `batches_seen` args to forward(), consulted only when
use_curriculum_learning=True) - occasionally substituting the decoder's
own last prediction with the true label to stabilize early training. This
repo's train()/val()/test() call every model identically as
`model(pm25_hist, feature)` with no path to pass the ground-truth label
into forward() itself (the label is only used afterward, for the loss) -
changing that shared calling convention for one model would be a much
bigger change than skipping teacher forcing for this one. So this port
hardcodes use_curriculum_learning=False; decoding is always the model's
own autoregressive rollout, every epoch - which is exactly how the
reference model already behaves at inference/test time regardless, so
only the early-training dynamics differ from the paper, not the
architecture or the test-time behavior.

THE PAPER'S AUXILIARY MEMORY LOSSES ARE KEPT, via this repo's existing
generic auxiliary-loss hook (see train.py's train(), same mechanism
AirFormer's `last_kl_loss` and AirDualODE's `last_alignment_loss` already
use): forward() exposes `self.last_memory_loss`, a triplet-margin
"separate" loss plus an MSE "compact" loss over the memory attention's
top-2 matches (`query`/`pos`/`neg` below) - identical formula and default
weights (lamb=lamb1=0.01) to the reference repo's traintest_MegaCRN.py.
train.py/sweep_all_models.py's training loops add a third opt-in hook for
this alongside the existing KL/alignment ones.

Contract (matches every other model in model/, see train.py get_model()):
    MegaCRNPM25(hist_len, pred_len, in_dim, city_num, batch_size, device, ...)
    pm25_pred = model(pm25_hist, feature)
    # pm25_hist: [B, hist_len, N, 1], feature: [B, hist_len+pred_len, N, F]
    #   (feature's future portion IS used here, as y_cov - see above)
    # -> pm25_pred: [B, pred_len, N, 1]
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AGCN(nn.Module):
    """Standard (non node-adaptive) Chebyshev graph conv - a single shared
    weight matrix applied over two supports' Chebyshev expansions, unlike
    model/agcrn.py's AVWGCN (which generates per-node weights from node
    embeddings). MegaCRN's adaptivity lives in how the supports themselves
    are derived from the memory bank (see MegaCRN.forward), not in the
    conv weights."""

    def __init__(self, dim_in, dim_out, cheb_k):
        super(AGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights = nn.Parameter(torch.FloatTensor(2 * cheb_k * dim_in, dim_out))  # 2 is the length of support
        self.bias = nn.Parameter(torch.FloatTensor(dim_out))
        nn.init.xavier_normal_(self.weights)
        nn.init.constant_(self.bias, val=0)

    def forward(self, x, supports):
        x_g = []
        support_set = []
        for support in supports:
            support_ks = [torch.eye(support.shape[0]).to(support.device), support]
            for k in range(2, self.cheb_k):
                support_ks.append(torch.matmul(2 * support, support_ks[-1]) - support_ks[-2])
            support_set.extend(support_ks)
        for support in support_set:
            x_g.append(torch.einsum("nm,bmc->bnc", support, x))
        x_g = torch.cat(x_g, dim=-1)  # B, N, 2 * cheb_k * dim_in
        x_gconv = torch.einsum('bni,io->bno', x_g, self.weights) + self.bias  # b, N, dim_out
        return x_gconv


class AGCRNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k):
        super(AGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AGCN(dim_in + self.hidden_dim, 2 * dim_out, cheb_k)
        self.update = AGCN(dim_in + self.hidden_dim, dim_out, cheb_k)

    def forward(self, x, state, supports):
        # x: B, num_nodes, input_dim
        # state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, supports))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z * state), dim=-1)
        hc = torch.tanh(self.update(candidate, supports))
        h = r * state + (1 - r) * hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)


class ADCRNN_Encoder(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, num_layers):
        super(ADCRNN_Encoder, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k))

    def forward(self, x, init_state, supports):
        # shape of x: (B, T, N, D), shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, supports)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return init_states


class ADCRNN_Decoder(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, num_layers):
        super(ADCRNN_Decoder, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Decoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k))

    def forward(self, xt, init_state, supports):
        # xt: (B, N, D); init_state: (num_layers, B, N, hidden_dim)
        assert xt.shape[1] == self.node_num and xt.shape[2] == self.input_dim
        current_inputs = xt
        output_hidden = []
        for i in range(self.num_layers):
            state = self.dcrnn_cells[i](current_inputs, init_state[i], supports)
            output_hidden.append(state)
            current_inputs = state
        return current_inputs, output_hidden


class MegaCRN(nn.Module):
    def __init__(self, num_nodes, input_dim, output_dim, horizon, rnn_units, num_layers=1, cheb_k=3,
                 ycov_dim=1, mem_num=20, mem_dim=64, cl_decay_steps=2000, use_curriculum_learning=True):
        super(MegaCRN, self).__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.rnn_units = rnn_units
        self.output_dim = output_dim
        self.horizon = horizon
        self.num_layers = num_layers
        self.cheb_k = cheb_k
        self.ycov_dim = ycov_dim
        self.cl_decay_steps = cl_decay_steps
        self.use_curriculum_learning = use_curriculum_learning

        # memory
        self.mem_num = mem_num
        self.mem_dim = mem_dim
        self.memory = self.construct_memory()

        # encoder
        self.encoder = ADCRNN_Encoder(self.num_nodes, self.input_dim, self.rnn_units, self.cheb_k, self.num_layers)

        # decoder
        self.decoder_dim = self.rnn_units + self.mem_dim
        self.decoder = ADCRNN_Decoder(self.num_nodes, self.output_dim + self.ycov_dim, self.decoder_dim, self.cheb_k, self.num_layers)

        # output
        self.proj = nn.Sequential(nn.Linear(self.decoder_dim, self.output_dim, bias=True))

    def compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def construct_memory(self):
        memory_dict = nn.ParameterDict()
        memory_dict['Memory'] = nn.Parameter(torch.randn(self.mem_num, self.mem_dim), requires_grad=True)  # (M, d)
        memory_dict['Wq'] = nn.Parameter(torch.randn(self.rnn_units, self.mem_dim), requires_grad=True)  # project to query
        memory_dict['We1'] = nn.Parameter(torch.randn(self.num_nodes, self.mem_num), requires_grad=True)  # project memory to embedding
        memory_dict['We2'] = nn.Parameter(torch.randn(self.num_nodes, self.mem_num), requires_grad=True)  # project memory to embedding
        for param in memory_dict.values():
            nn.init.xavier_normal_(param)
        return memory_dict

    def query_memory(self, h_t: torch.Tensor):
        query = torch.matmul(h_t, self.memory['Wq'])  # (B, N, d)
        att_score = torch.softmax(torch.matmul(query, self.memory['Memory'].t()), dim=-1)  # alpha: (B, N, M)
        value = torch.matmul(att_score, self.memory['Memory'])  # (B, N, d)
        _, ind = torch.topk(att_score, k=2, dim=-1)
        pos = self.memory['Memory'][ind[:, :, 0]]  # B, N, d
        neg = self.memory['Memory'][ind[:, :, 1]]  # B, N, d
        return value, query, pos, neg

    def forward(self, x, y_cov, labels=None, batches_seen=None):
        node_embeddings1 = torch.matmul(self.memory['We1'], self.memory['Memory'])
        node_embeddings2 = torch.matmul(self.memory['We2'], self.memory['Memory'])
        g1 = F.softmax(F.relu(torch.mm(node_embeddings1, node_embeddings2.T)), dim=-1)
        g2 = F.softmax(F.relu(torch.mm(node_embeddings2, node_embeddings1.T)), dim=-1)
        supports = [g1, g2]
        init_state = self.encoder.init_hidden(x.shape[0])
        h_en, state_en = self.encoder(x, init_state, supports)  # B, T, N, hidden
        h_t = h_en[:, -1, :, :]  # B, N, hidden (last state)

        h_att, query, pos, neg = self.query_memory(h_t)
        h_t = torch.cat([h_t, h_att], dim=-1)

        ht_list = [h_t] * self.num_layers
        go = torch.zeros((x.shape[0], self.num_nodes, self.output_dim), device=x.device)
        out = []
        for t in range(self.horizon):
            h_de, ht_list = self.decoder(torch.cat([go, y_cov[:, t, ...]], dim=-1), ht_list, supports)
            go = self.proj(h_de)
            out.append(go)
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self.compute_sampling_threshold(batches_seen):
                    go = labels[:, t, ...]
        output = torch.stack(out, dim=1)

        return output, h_att, query, pos, neg


class MegaCRNPM25(nn.Module):
    def __init__(self, hist_len, pred_len, in_dim, city_num, batch_size, device,
                 rnn_units=64, num_layers=1, cheb_k=3, mem_num=20, mem_dim=64,
                 memory_lamb=0.01, memory_lamb1=0.01):
        super(MegaCRNPM25, self).__init__()
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.city_num = city_num
        self.memory_lamb = memory_lamb
        self.memory_lamb1 = memory_lamb1
        ycov_dim = in_dim - 1  # weather channels only - pm25 itself isn't "covariate"

        # use_curriculum_learning=False: see module docstring - this harness's
        # model(pm25_hist, feature) contract has no path for ground-truth
        # labels during training, which teacher forcing needs.
        self.core = MegaCRN(
            num_nodes=city_num, input_dim=in_dim, output_dim=1, horizon=pred_len,
            rnn_units=rnn_units, num_layers=num_layers, cheb_k=cheb_k,
            ycov_dim=ycov_dim, mem_num=mem_num, mem_dim=mem_dim,
            use_curriculum_learning=False,
        )
        self.last_memory_loss = None

    def forward(self, pm25_hist, feature):
        """
        pm25_hist : [B, hist_len, N, 1]
        feature   : [B, hist_len + pred_len, N, F] - BOTH portions used:
                    historical as encoder input, future as decoder y_cov
        returns   : [B, pred_len, N, 1]
        """
        B, T, N, _ = pm25_hist.shape
        if N != self.city_num:
            raise ValueError(
                f"MegaCRNPM25 was built with city_num={self.city_num}, but got "
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

        x = torch.cat([pm25_hist, feature_hist], dim=-1)  # [B, hist_len, N, in_dim]
        output, h_att, query, pos, neg = self.core(x, feature_future)

        # Auxiliary memory losses from the reference repo's traintest_MegaCRN.py
        # (loss2/loss3 there), surfaced via this repo's generic aux-loss hook -
        # see module docstring.
        separate_loss = F.triplet_margin_loss(query, pos.detach(), neg.detach(), margin=1.0)
        compact_loss = F.mse_loss(query, pos.detach())
        self.last_memory_loss = self.memory_lamb * separate_loss + self.memory_lamb1 * compact_loss

        return output
