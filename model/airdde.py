"""
AirDDE (w2obin, AAAI, https://github.com/w2obin/airdde-aaai) adapted to
this repo's benchmark harness.

WHAT THIS MODEL IS
-------------------
A hybrid of three mechanisms, all preserved from upstream:
  1. An adaptive-graph AGCRN encoder/decoder (AGCN/AGCRNCell/STEncoder/
     STDecoder): a GRU whose gate/update transforms are Chebyshev graph
     convolutions over a FULLY LEARNED node-embedding adjacency (no
     edge_index needed for this part - `glo_memory['We1']/['We2']`
     generate it).
  2. A global + local memory bank (`construct_global_memory` /
     `LocalMemoryModule`): the global memory attends station encodings
     against a shared learned memory bank; the local memory picks each
     station's top-k wind-similar neighbours over a short lookback window
     and attends over their encodings.
  3. A physics-flavoured neural ODE (`ODEFunc` + `DiffeqSolver`, via
     `torchdiffeq`): integrates a diffusion term (Chebyshev graph conv
     over the REAL geographic graph), an advection term (Chebyshev graph
     conv over a wind-driven edge weighting, same "3*speed*cos(theta)/
     dist" formula this repo's own PM25_GNN.GraphGNN uses), and a learned
     source/sink term, to roll the PM2.5 field forward before the
     ST-decoder refines it with future weather at each step.

The class/module structure (AGCN, AGCRNCell, STEncoder, STDecoder,
LocalMemoryModule, DiffeqSolver, ODEFunc) is upstream's, functionally
unchanged except where noted below. Only `Model` (renamed `AirDDEPM25`)
and a few ODEFunc details were rewritten to fit this repo.

WHAT HAD TO CHANGE, AND WHY
-----------------------------
  1. Contract. Every other model here is
     `Model(hist_len, pred_len, in_dim, city_num, batch_size, device,
     edge_index, edge_attr, wind_mean, wind_std, ...)` called as
     `model(pm25_hist, feature) -> [B, pred_len, N, 1]`. Upstream's
     `Model.forward(x, y_cov, labels=None, batches_seen=None)` expected a
     TIME-MAJOR `x: [T,B,N,D]` and `y_cov: [T_pred,B,N,ycov_dim]` built by
     its own dataloader, plus unused curriculum-learning args (`labels`,
     `batches_seen`, `cl_decay_steps`, `use_curriculum_learning` never
     actually branch on anything in the pasted forward()) - dropped.
     Rewritten to batch-major throughout, built from (pm25_hist, feature)
     the way every other model in this repo already does; `y_cov` is
     `feature[:, hist_len:]` - the REAL future weather this harness
     exposes (PM25_GNN/ProbGRU*/Informer* all already use it), so
     `ycov_dim = in_dim - 1` instead of upstream's hardcoded 5.
  2. Hardcoded dims/device. Upstream hardcodes `device="cuda:0"` in two
     places, `num_nodes=184` in ODEFunc's construction (coincidentally
     matches KnowAir's own city_num - this model was very likely
     developed against the same dataset as this repo, which is why the
     wind/graph formulas line up so closely), `nn.Linear(128+128, 64)`
     for `source_sink_pred` (assumes rnn_units=mem_dim=64 exactly) and
     `nn.Linear(24*5, decoder_dim)` for `y_cov_embed_layer` (assumes
     pred_len=24, ycov_dim=5 exactly). All now computed from the actual
     constructor args, so changing rnn_units/mem_dim/pred_len/ycov_dim
     doesn't silently break shapes.
  3. edge_attr layout. Upstream's ODEFunc expects a 3-column edge_attr:
     `[:,0]` a dedicated diffusion weight, `[:,1]` distance, `[:,2]`
     bearing. This repo's graph.py only carries 2 columns, `[dist_km,
     bearing]` (see PM25_GNN.GraphGNN / GNN_Transformer.py's identical
     convention) - there is no separate diffusion-weight column to
     reuse. `diff_edge_attr` is therefore SYNTHESIZED as
     `1 / (dist_km + eps)` (closer stations diffuse into each other more
     strongly) - a documented deviation, not a literal reproduction.
     The advection term's distance/bearing use this repo's 2 columns
     directly, and keeps upstream's `(wind_dir + 180) % 360` correction
     (meteorological "blowing from" -> "travelling toward" convention) -
     the same fix this repo's own model/probgru5.py independently applies
     to the same convention issue, so it's kept rather than reverted to
     PM25_GNN's older uncorrected formula.
  4. Device-safety. `edge_index`/`edge_attr`/`wind_mean`/`wind_std` are
     now `register_buffer`s (so `model.to(device)` in train.py moves them
     automatically) instead of upstream's manual `.to("cuda:0")` calls
     baked into `__init__`.
  5. `LocalMemoryModule` assumed a time-major `x_orig: [T,B,N,D]` and
     permuted it to batch-major internally. Since `x_orig` is now already
     batch-major `[B,T,N,D]` when it reaches this module, that internal
     permute would silently swap the batch and time axes - removed; the
     module now expects batch-major input directly, matching `h_e`'s own
     `[B,T,N,hidden]` layout (unchanged from upstream).
  6. Final output. Upstream ends with `output.permute(1,0,2)`, producing
     a TIME-MAJOR `[pred_len,B,N]` result for its own trainer. Removed -
     this repo wants `[B,pred_len,N,1]`, so the last step is just
     `output.unsqueeze(-1)`.
  7. `time_steps_to_predict` (the ODE solver's query times) is now moved
     onto the input tensor's device explicitly; upstream left it on CPU,
     which errors as soon as this runs on a GPU.

Requires `torch_geometric` (already used elsewhere via
torch_geometric.utils in graph.py) and `torchdiffeq` (new dependency for
this file only - `pip install torchdiffeq`).

Contract (matches every other model in model/, see train.py get_model()):
    AirDDEPM25(hist_len, pred_len, in_dim, city_num, batch_size, device,
               edge_index, edge_attr, wind_mean, wind_std, ...)
    pm25_pred = model(pm25_hist, feature)   # -> [B, pred_len, N, 1]
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import ChebConv
from torchdiffeq import odeint_adjoint as odeint


class AGCN(nn.Module):
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


class STEncoder(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, num_layers):
        super(STEncoder, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k))

    def forward(self, x, init_state, supports):
        # x: (B, T, N, D)
        # init_state: (num_layers, B, N, hidden_dim)
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


class STDecoder(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, num_layers):
        super(STDecoder, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Decoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k))

    def forward(self, xt, init_state, supports):
        # xt: (B, N, D)
        # init_state: (num_layers, B, N, hidden_dim)
        assert xt.shape[1] == self.node_num and xt.shape[2] == self.input_dim
        current_inputs = xt
        output_hidden = []
        for i in range(self.num_layers):
            state = self.dcrnn_cells[i](current_inputs, init_state[i], supports)
            output_hidden.append(state)
            current_inputs = state
        return current_inputs, output_hidden


class LocalMemoryModule(nn.Module):
    def __init__(self, num_nodes, d_model, tau=3, k_neighbors=8):
        super().__init__()
        self.num_nodes = num_nodes
        self.d_model = d_model
        self.tau = tau
        self.k_neighbors = k_neighbors
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, h_e, x_orig):
        """h_e: [B, T, N, d_model] (encoder hidden states).
        x_orig: [B, T, N, D] - already batch-major (see module docstring
        point 5: upstream expected time-major here and permuted it into
        batch-major; we build it batch-major up front instead)."""
        if x_orig.dim() != 4:
            raise ValueError(f"LocalMemoryModule expects a 4D [B,T,N,D] x_orig, got dim={x_orig.dim()}")
        batch_size, seq_len, num_nodes, d_model = h_e.shape
        x_reshape = x_orig

        # last two channels = wind speed, wind direction (see dataset.py's
        # _process_feature - fixed tail layout regardless of metero_use).
        wind_vars = x_reshape[:, :, :, -2:]
        last_wind = wind_vars[:, -1]
        b, n, _ = last_wind.shape
        wind_flat = last_wind.reshape(b, n, -1)

        # for each station i, pick the k stations whose wind vector is
        # most similar (closest in speed/direction space) - a proxy for
        # "likely to be transporting pollution toward/with i right now".
        dist = torch.cdist(wind_flat, wind_flat)
        sim = -dist
        k = min(self.k_neighbors, n)
        topk_idx = sim.topk(k=k, dim=-1).indices

        t0 = seq_len - 1
        t_start = max(0, t0 - self.tau + 1)
        hist = h_e[:, t_start:t0 + 1].permute(0, 2, 1, 3)
        tau_eff = hist.size(2)

        batch_idx = torch.arange(b, device=h_e.device).view(b, 1, 1).expand(b, n, k)
        neighbor_hist = hist[batch_idx, topk_idx]
        neighbor_hist = neighbor_hist.reshape(b, n, k * tau_eff, d_model)

        # local attention
        q = h_e[:, t0]
        q = self.q_proj(q).unsqueeze(2)
        k_feat = self.k_proj(neighbor_hist)
        v_feat = self.v_proj(neighbor_hist)

        scale = math.sqrt(d_model)
        attn_scores = (q * k_feat).sum(-1) / scale
        attn_weights = torch.softmax(attn_scores, dim=-1).unsqueeze(-1)
        context = (attn_weights * v_feat).sum(2)
        h_l = self.mlp(context)
        return h_l


class DiffeqSolver:
    def __init__(self, method, odeint_rtol=1e-5, odeint_atol=1e-5, adjoint=True):
        self.ode_method = method
        self.odeint = odeint
        self.rtol = odeint_rtol
        self.atol = odeint_atol

    def solve(self, odefunc, first_point, time_steps_to_pred):
        pred_y = self.odeint(odefunc, first_point, time_steps_to_pred,
                              rtol=self.rtol, atol=self.atol, method=self.ode_method)
        return pred_y


class ODEFunc(nn.Module):
    def __init__(self, gcn_hidden_dim, input_dim, edge_index, edge_attr,
                 K_neighbour, num_nodes, decoder_dim, num_layers=2, activation='tanh'):
        super(ODEFunc, self).__init__()
        self.num_nodes = num_nodes
        self.gcn_hidden_dim = gcn_hidden_dim
        self.input_dim = input_dim
        self.num_layers = num_layers
        self._activation = torch.tanh if activation == 'tanh' else torch.relu

        self.register_buffer('edge_index', torch.as_tensor(edge_index, dtype=torch.long))
        edge_attr_t = torch.as_tensor(np.float32(edge_attr))
        self.register_buffer('edge_attr', edge_attr_t)
        # dist_km, bearing_deg - this repo's 2-column edge_attr convention
        # (see PM25_GNN.GraphGNN / GNN_Transformer.py). Upstream expected a
        # 3rd, dedicated diffusion-weight column that doesn't exist here;
        # synthesize one as inverse distance (see module docstring #3).
        self.register_buffer('diff_edge_attr', 1.0 / edge_attr_t[:, 0].clamp(min=1e-3))
        self.K_neighbour = K_neighbour

        self.adv_edge_attr = None  # set per-forward via create_equation() - depends on this batch's wind
        self.source_sink = None    # set per-forward via create_source_matrix()

        self.source_sink_pred = nn.Linear(2 * decoder_dim, gcn_hidden_dim)
        self.source_embed = nn.Linear(self.gcn_hidden_dim + 1, 1)
        self.norm = nn.LayerNorm(self.num_nodes)

        self.residual = nn.Identity()

        self.diff_cheb_conv = self.laplacian_operator()
        self.adv_cheb_conv = self.laplacian_operator()

        # placeholder only - always overwritten by Model.forward() before
        # the ODE solver runs; shape doesn't matter beyond "won't crash if
        # ever read before being set".
        self.previous_x = torch.zeros(1, num_nodes, gcn_hidden_dim)

    def create_adv_matrix(self, last_wind_vars, wind_mean, wind_std):
        batch_size = last_wind_vars.shape[0]
        edge_src, edge_target = self.edge_index
        node_src = last_wind_vars[:, edge_src, :]

        src_wind_speed = node_src[:, :, 0] * wind_std[0] + wind_mean[0]  # km/h
        src_wind_dir = node_src[:, :, 1] * wind_std[1] + wind_mean[1]    # deg, meteorological "from" convention
        dist = self.edge_attr[:, 0].unsqueeze(dim=0).repeat(batch_size, 1)
        dist_dir = self.edge_attr[:, 1].unsqueeze(dim=0).repeat(batch_size, 1)

        src_wind_dir = (src_wind_dir + 180) % 360  # "from" -> "traveling toward"
        theta = torch.abs(dist_dir - src_wind_dir)
        adv_edge_attr = F.relu(3 * src_wind_speed * torch.cos(theta) / dist)  # B x M

        return adv_edge_attr

    def create_equation(self, last_wind_vars, wind_mean, wind_std):
        self.adv_edge_attr = self.create_adv_matrix(last_wind_vars, wind_mean, wind_std)

    def create_source_matrix(self, features):
        self.source_sink = self.source_sink_pred(features)  # B,N,gcn_hidden_dim

    def forward(self, t_local, Xt):
        grad_diff = self.ode_func_net_diff(Xt, self.diff_edge_attr)
        grad_adv = self.ode_func_net_adv(Xt, self.adv_edge_attr)
        grad_source = self.ode_func_net_source_sink(Xt, self.source_sink)
        grad = 0.1 * grad_diff + grad_adv + grad_source
        return grad

    def ode_func_net_source_sink(self, x, source):
        out = torch.cat([x.unsqueeze(-1), source], dim=-1)
        out = self.norm(self.source_embed(out).squeeze(-1))
        return out

    def ode_func_net_diff(self, x, edge_attr):
        # x: B x N*var_dim - shared graph, dense-batched (no `batch` kwarg
        # needed: ChebConv broadcasts a single edge_index/edge_weight over
        # a leading [B, N, F] input).
        batch_size = x.shape[0]
        x = torch.reshape(x, (batch_size, self.num_nodes, self.input_dim))

        x = self.diff_cheb_conv[0](x, self.edge_index, edge_attr, lambda_max=2)
        x = self._activation(x)

        for op in self.diff_cheb_conv[1:-1]:
            residual = self.residual(x)
            x = op(x, self.edge_index, edge_attr, lambda_max=2)
            x = self._activation(x) + residual

        x = self.diff_cheb_conv[-1](x, self.edge_index, edge_attr, lambda_max=2)

        return x.reshape((batch_size, self.num_nodes * self.input_dim))

    def ode_func_net_adv(self, x, edge_attr):
        # edge_weight varies per batch element (wind-driven), so each
        # graph in the batch is block-diagonally concatenated into one
        # big disconnected graph and passed with a `batch` vector.
        batch_size = x.shape[0]
        batch = torch.arange(0, batch_size, device=x.device)
        batch = torch.repeat_interleave(batch, self.num_nodes)
        x = x.reshape(batch_size * self.num_nodes, -1)  # B*N x input_dim
        x = x + 0.01 * self.previous_x.sum(dim=1).sum(dim=-1).reshape(batch_size * self.num_nodes, -1)

        edge_indices = [self.edge_index + i * self.num_nodes for i in range(batch_size)]
        edge_index = torch.cat(edge_indices, dim=1)  # 2 x B*M
        edge_attr = edge_attr.reshape(-1)  # B*M

        x = self.adv_cheb_conv[0](x, edge_index, edge_attr, batch=batch, lambda_max=2)
        x = self._activation(x)

        for op in self.adv_cheb_conv[1:-1]:
            residual = self.residual(x)
            x = op(x, edge_index, edge_attr, batch=batch, lambda_max=2)
            x = self._activation(x) + residual

        x = self.adv_cheb_conv[-1](x, edge_index, edge_attr, batch=batch, lambda_max=2)

        x = x.reshape(batch_size, self.num_nodes, self.input_dim)
        return x.reshape((batch_size, self.num_nodes * self.input_dim))

    def laplacian_operator(self):
        operator = nn.ModuleList()
        operator.append(
            ChebConv(in_channels=self.input_dim, out_channels=self.gcn_hidden_dim,
                     K=self.K_neighbour, normalization='sym', bias=True)
        )
        for _ in range(self.num_layers - 2):
            operator.append(
                ChebConv(in_channels=self.gcn_hidden_dim, out_channels=self.gcn_hidden_dim,
                         K=self.K_neighbour, normalization='sym', bias=True)
            )
        operator.append(
            ChebConv(in_channels=self.gcn_hidden_dim, out_channels=self.input_dim,
                     K=self.K_neighbour, normalization='sym', bias=True)
        )
        return operator


class AirDDEPM25(nn.Module):
    def __init__(self, hist_len, pred_len, in_dim, city_num, batch_size, device,
                 edge_index, edge_attr, wind_mean, wind_std,
                 rnn_units=64, rnn_num_layers=1, agcn_cheb_k=3,
                 mem_num=20, mem_dim=64, local_mem_tau=3, local_mem_k=8,
                 ode_gcn_hidden_dim=64, ode_cheb_k=3, ode_num_layers=3,
                 ode_method='dopri5', ode_rtol=1e-2, ode_atol=1e-2, ode_adjoint=True,
                 output_dim=1):
        super(AirDDEPM25, self).__init__()
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.horizon = pred_len
        self.in_dim = in_dim
        self.city_num = city_num
        self.num_nodes = city_num
        self.device = device
        self.batch_size = batch_size
        self.input_dim = in_dim
        self.output_dim = output_dim
        self.ycov_dim = in_dim - 1  # every future weather channel this harness exposes
        self.rnn_units = rnn_units
        self.num_layers = rnn_num_layers
        self.cheb_k = agcn_cheb_k
        self.mem_num = mem_num
        self.mem_dim = mem_dim

        self.register_buffer('wind_mean', torch.as_tensor(np.float32(wind_mean))[-2:])
        self.register_buffer('wind_std', torch.as_tensor(np.float32(wind_std))[-2:])

        # global memory
        self.glo_memory = self.construct_global_memory()
        self.loc_memory = LocalMemoryModule(num_nodes=self.num_nodes, d_model=self.rnn_units,
                                             tau=local_mem_tau, k_neighbors=local_mem_k)
        self.memory_embed = nn.Sequential(nn.Linear(self.mem_dim + self.mem_dim, self.mem_dim, bias=True))

        # encoder / decoder (adaptive-graph AGCRN, no edge_index needed)
        self.encoder = STEncoder(self.num_nodes, self.input_dim, self.rnn_units, self.cheb_k, self.num_layers)
        self.decoder_dim = self.rnn_units + self.mem_dim
        self.decoder = STDecoder(self.num_nodes, self.output_dim + self.ycov_dim, self.decoder_dim,
                                  self.cheb_k, self.num_layers)

        # physics ODE branch (real geographic + wind graph)
        self.phy_solver = DiffeqSolver(method=ode_method, odeint_atol=ode_atol,
                                        odeint_rtol=ode_rtol, adjoint=ode_adjoint)
        self.phy_odefunc = ODEFunc(gcn_hidden_dim=ode_gcn_hidden_dim, input_dim=1,
                                    edge_index=edge_index, edge_attr=edge_attr,
                                    K_neighbour=ode_cheb_k, num_nodes=self.num_nodes,
                                    decoder_dim=self.decoder_dim, num_layers=ode_num_layers,
                                    activation='tanh')

        self.y_cov_embed_layer = nn.Linear(self.horizon * self.ycov_dim, self.decoder_dim)
        self.proj = nn.Sequential(nn.Linear(self.decoder_dim, self.output_dim, bias=True))

    def construct_global_memory(self):
        memory_dict = nn.ParameterDict()
        memory_dict['Memory'] = nn.Parameter(torch.randn(self.mem_num, self.mem_dim), requires_grad=True)
        memory_dict['Wq'] = nn.Parameter(torch.randn(self.rnn_units, self.mem_dim), requires_grad=True)
        memory_dict['We1'] = nn.Parameter(torch.randn(self.num_nodes, self.mem_num), requires_grad=True)
        memory_dict['We2'] = nn.Parameter(torch.randn(self.num_nodes, self.mem_num), requires_grad=True)
        for param in memory_dict.values():
            nn.init.xavier_normal_(param)
        return memory_dict

    def global_memory_modeling(self, h_t: torch.Tensor):
        query = torch.matmul(h_t, self.glo_memory['Wq'])                                    # (B, N, d)
        att_score = torch.softmax(torch.matmul(query, self.glo_memory['Memory'].t()), dim=-1)  # (B, N, M)
        value = torch.matmul(att_score, self.glo_memory['Memory'])                          # (B, N, d)
        return value, query

    def forward(self, pm25_hist, feature):
        """
        pm25_hist : [B, hist_len, N, 1]
        feature   : [B, hist_len + pred_len, N, F]   (F = in_dim - 1)
        returns   : [B, pred_len, N, 1]
        """
        B, T, N, _ = pm25_hist.shape
        if N != self.city_num:
            raise ValueError(
                f"AirDDEPM25 was built with city_num={self.city_num}, but got "
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

        x = torch.cat([pm25_hist, feature_hist], dim=-1)  # [B,hist_len,N,in_dim] - batch-major throughout
        y_cov = feature_future                            # [B,pred_len,N,ycov_dim] - real future weather

        # adaptive adjacency for the AGCRN encoder/decoder (learned, no edge_index)
        node_embeddings1 = torch.matmul(self.glo_memory['We1'], self.glo_memory['Memory'])
        node_embeddings2 = torch.matmul(self.glo_memory['We2'], self.glo_memory['Memory'])
        g1 = F.softmax(F.relu(torch.mm(node_embeddings1, node_embeddings2.T)), dim=-1)
        g2 = F.softmax(F.relu(torch.mm(node_embeddings2, node_embeddings1.T)), dim=-1)
        supports = [g1, g2]

        init_state = self.encoder.init_hidden(B)
        init_state = [s.to(x.device) for s in init_state]
        h_en, _ = self.encoder(x, init_state, supports)  # B, T, N, rnn_units
        h_t = h_en[:, -1, :, :]                           # B, N, rnn_units (last state)

        h_global, _ = self.global_memory_modeling(h_t)
        h_local = self.loc_memory(h_en, x)
        h_memory = self.memory_embed(torch.cat([h_global, h_local], dim=-1))  # B, N, mem_dim
        h_embed = torch.cat([h_t, h_memory], dim=-1)                          # B, N, decoder_dim
        ht_list = [h_embed] * self.num_layers

        # --- physics ODE branch ---
        last_wind_vars = x[:, -1, :, -2:]  # B x N x 2 - last hist step's (speed, direction)
        self.phy_odefunc.create_equation(last_wind_vars, self.wind_mean, self.wind_std)

        tau_back = 3
        self.phy_odefunc.previous_x = h_en[:, -1 - tau_back:-1, :, :]

        phy_input = x[:, -1, :, 0]  # B x N - last hist step's PM2.5 concentration
        y_cov_embed = self.y_cov_embed_layer(
            y_cov.permute(0, 2, 1, 3).reshape(B, self.num_nodes, -1)
        )  # B, N, decoder_dim
        self.phy_odefunc.create_source_matrix(torch.cat([h_embed, y_cov_embed], dim=-1))

        time_steps_to_predict = torch.arange(0, self.horizon + 1, dtype=torch.float32, device=x.device)
        time_steps_to_predict = time_steps_to_predict / len(time_steps_to_predict)

        phy_y = self.phy_solver.solve(self.phy_odefunc, phy_input, time_steps_to_predict)  # (horizon+1) x B x N
        phy_y = phy_y[1:]  # drop the t=0 initial condition -> horizon x B x N

        # --- ST decoder, one step at a time, using the ODE's rolled-forward
        # PM2.5 field plus real future weather at each step ---
        out = []
        for t in range(self.horizon):
            dec_in = torch.cat([phy_y[t].unsqueeze(-1), y_cov[:, t]], dim=-1)  # B, N, in_dim
            h_de, ht_list = self.decoder(dec_in, ht_list, supports)
            out.append(self.proj(h_de))  # B, N, output_dim

        pm25_pred = torch.stack(out, dim=1)  # B, pred_len, N, output_dim
        return pm25_pred
