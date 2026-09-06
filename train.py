import os
import sys
proj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(proj_dir)
from util import config, file_dir
from graph import Graph
from dataset import HazeData

from model.MLP import MLP
from model.LSTM import LSTM
from model.GRU import GRU
from model.GC_LSTM import GC_LSTM
from model.nodesFC_GRU import nodesFC_GRU
from model.PM25_GNN import PM25_GNN
from model.PM25_GNN_nosub import PM25_GNN_nosub
from model.airformer import AirFormerPM25
from model.informer import InformerPM25
from model.autoformer import AutoformerPM25
from model.patchtst import PatchTSTPM25
from model.staeformer import STAEformerPM25
from model.airdde import AirDDEPM25
from model.airphynet import AirPhyNetPM25
from model.airdualode import AirDualODEPM25
from model.airlapse import AirLapse
from model.airlapse_v2 import AirLapseV2
from model.mgsfformer import MGSFformerPM25
from model.timexer import TimeXerPM25
from model.wpmixer import WPMixerPM25
from model.dtaf import DTAFPM25
from model.agcrn import AGCRNPM25
from model.megacrn import MegaCRNPM25

import arrow
import torch
from torch import nn
import numpy as np
import pickle
import glob
import shutil
import time

try:
    import psutil
    _psutil_process = psutil.Process()
except ImportError:
    # Efficiency reporting degrades gracefully without it: peak_memory_mb
    # reports NaN on CPU runs instead of a real number (GPU runs are
    # unaffected - torch.cuda's own memory stats don't need psutil).
    psutil = None
    _psutil_process = None

# Numbering + publication year for each benchmark, matching config.yaml's
# commented model catalog exactly (same grouping: generic neural-network/
# graph architectures, generic transformers, air-quality-forecasting-
# domain-specific models, AirLapse last) - keep the two in sync if either
# changes. Used only to label the saved report file below; has no effect
# on model selection/dispatch (that's still the plain exp_model string).
# PM25_GNN_nosub/GC_LSTM/nodesFC_GRU are deliberately unnumbered (order
# None - PM25_GNN ablation/baseline-suite siblings, not separately
# ranked); numbering runs 1-20 across the rest.
MODEL_CATALOG = {
    'MLP': (1, 1986), 'LSTM': (2, 1997), 'GRU': (3, 2014),
    'AGCRN': (4, 2020), 'MegaCRN': (5, 2023),
    'Informer': (6, 2021), 'Autoformer': (7, 2021), 'PatchTST': (8, 2023),
    'STAEformer': (9, 2023), 'MGSFformer': (10, 2025), 'TimeXer': (11, 2024),
    'WPMixer': (12, 2025), 'DTAF': (13, 2026),
    'PM25_GNN': (14, 2020),
    'PM25_GNN_nosub': (None, 2020), 'GC_LSTM': (None, 2020), 'nodesFC_GRU': (None, 2020),
    'AirDDE': (15, 2026), 'AirPhyNet': (16, 2024), 'AirDualODE': (17, 2025),
    'AirFormer': (18, 2023),
    'AirLapse': (19, 2026),
    'AirLapseV2': (20, 2026),
}


def _catalog_label(exp_model_name):
    """'AirLapse' -> '19. AirLapse 2026'; an unnumbered entry (order None,
    e.g. 'GC_LSTM') -> 'GC_LSTM 2020' (no leading 'N. '). Falls back to the
    plain name if exp_model_name isn't in MODEL_CATALOG at all, e.g. right
    after adding a new model here before its catalog entry is added."""
    entry = MODEL_CATALOG.get(exp_model_name)
    if entry is None:
        return exp_model_name
    order, year = entry
    if order is None:
        return '%s %d' % (exp_model_name, year)
    return '%d. %s %d' % (order, exp_model_name, year)


torch.set_num_threads(1)
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

batch_size = config['train']['batch_size']
epochs = config['train']['epochs']
hist_len = config['train']['hist_len']
pred_len = config['train']['pred_len']
weight_decay = config['train']['weight_decay']
early_stop = config['train']['early_stop']
lr = config['train']['lr']
results_dir = file_dir['results_dir']
dataset_num = config['experiments']['dataset_num']
exp_model = config['experiments']['model']

# dataset_num must be read before building Graph(): datasets 1-3 share the
# same 184-city China graph (city.txt + altitude.npy), but a dataset with
# its own city_fp (e.g. 4's US sensor network) needs a different Graph()
# call - see graph.py's Graph.__init__ and config.yaml's dataset: 4 block.
_ds_cfg = config['dataset'][dataset_num]
if 'city_fp' in _ds_cfg:
    graph = Graph(
        city_fp=os.path.join(proj_dir, _ds_cfg['city_fp']),
        use_altitude=_ds_cfg.get('use_altitude', False),
        k_neighbors=_ds_cfg.get('k_neighbors', 5),
    )
else:
    graph = Graph()
city_num = graph.node_num
exp_repeat = config['train']['exp_repeat']
save_npy = config['experiments']['save_npy']
criterion = nn.MSELoss()
# Weight for an optional KL-divergence regularization term. Only models
# that expose a `last_kl_loss` attribute after forward() (AirFormer's
# hierarchical stochastic latent module, AirPhyNet's VAE-style initial
# state) use this; every other model is unaffected. Uses .get() with a
# default so existing config.yaml files don't need a new key to keep
# working. Override via config['train']['kl_weight'].
kl_weight = config['train'].get('kl_weight', 0.01)
# Weight for an optional temporal-alignment regularization term (currently
# just AirDualODE's physics/data-branch agreement loss, exposed via
# `last_alignment_loss`) - same opt-in mechanism as kl_weight above, just
# for a differently-meaning auxiliary loss so the two aren't conflated.
alignment_weight = config['train'].get('alignment_weight', 0.1)
# Weight for an optional expert-diversity regularization term (currently
# just DTAF's mixture-of-experts KL-diversity loss, exposed via
# `last_moe_diversity_loss`) - same opt-in mechanism as kl_weight/
# alignment_weight above, kept as its own name since it isn't the same
# quantity as either (a diversity-among-experts term, not a VAE-style
# latent KL or a physics/data-branch alignment loss).
moe_diversity_weight = config['train'].get('moe_diversity_weight', 0.01)

train_data = HazeData(graph, hist_len, pred_len, dataset_num, flag='Train')
val_data = HazeData(graph, hist_len, pred_len, dataset_num, flag='Val')
test_data = HazeData(graph, hist_len, pred_len, dataset_num, flag='Test')
in_dim = train_data.feature.shape[-1] + train_data.pm25.shape[-1]
wind_mean, wind_std = train_data.wind_mean, train_data.wind_std
pm25_mean, pm25_std = test_data.pm25_mean, test_data.pm25_std


def get_metric(predict_epoch, label_epoch):
    haze_threshold = 75
    predict_haze = predict_epoch >= haze_threshold
    predict_clear = predict_epoch < haze_threshold
    label_haze = label_epoch >= haze_threshold
    label_clear = label_epoch < haze_threshold
    hit = np.sum(np.logical_and(predict_haze, label_haze))
    miss = np.sum(np.logical_and(label_haze, predict_clear))
    falsealarm = np.sum(np.logical_and(predict_haze, label_clear))
    csi = hit / (hit + falsealarm + miss)
    pod = hit / (hit + miss)
    far = falsealarm / (hit + falsealarm)
    predict = predict_epoch[:,:,:,0].transpose((0,2,1))
    label = label_epoch[:,:,:,0].transpose((0,2,1))
    predict = predict.reshape((-1, predict.shape[-1]))
    label = label.reshape((-1, label.shape[-1]))
    mae = np.mean(np.mean(np.abs(predict - label), axis=1))
    rmse = np.mean(np.sqrt(np.mean(np.square(predict - label), axis=1)))
    mape = get_mape(predict, label)
    return rmse, mae, mape, csi, pod, far


def get_mape(predict, label, eps_threshold=1.0):
    """Mean Absolute Percentage Error, masking out |label| < eps_threshold
    (ug/m3). PM2.5 readings near zero make the percentage denominator
    blow up / become meaningless - that's measurement-noise-floor
    territory, not a real relative-error signal, so those points are
    excluded rather than fudged with a denominator epsilon."""
    mask = np.abs(label) >= eps_threshold
    if not np.any(mask):
        return float('nan')
    return float(np.mean(np.abs((predict[mask] - label[mask]) / label[mask])) * 100)


def get_exp_info():
    exp_info =  '============== Train Info ==============\n' + \
                'Dataset number: %s\n' % dataset_num + \
                'Model: %s\n' % exp_model + \
                'Train: %s --> %s\n' % (train_data.start_time, train_data.end_time) + \
                'Val: %s --> %s\n' % (val_data.start_time, val_data.end_time) + \
                'Test: %s --> %s\n' % (test_data.start_time, test_data.end_time) + \
                'City number: %s\n' % city_num + \
                'Use metero: %s\n' % config['experiments']['metero_use'] + \
                'batch_size: %s\n' % batch_size + \
                'epochs: %s\n' % epochs + \
                'hist_len: %s\n' % hist_len + \
                'pred_len: %s\n' % pred_len + \
                'weight_decay: %s\n' % weight_decay + \
                'early_stop: %s\n' % early_stop + \
                'lr: %s\n' % lr + \
                '========================================\n'
    return exp_info

# graph is your existing Graph() instance
idx, citys, lons, lats = graph.traverse_graph()

coords = torch.tensor(
    np.stack([lats, lons], axis=1), dtype=torch.float32
)  # (N, 2) — lat, lon order, matches MCASALayer's expectation

altitude = torch.tensor(
    graph.node_attr[:, 0], dtype=torch.float32
)  # (N,) — per-node altitude, already extracted by _add_node_attr()

def get_model():
    # Ordered in three groups - generic neural-network/graph architectures,
    # generic transformers, then air-quality-forecasting-domain-specific
    # models - with AirLapse (this repo's proposed model) last.
    if exp_model == 'MLP':
        return MLP(hist_len, pred_len, in_dim)
    elif exp_model == 'LSTM':
        return LSTM(hist_len, pred_len, in_dim, city_num, batch_size, device)
    elif exp_model == 'GRU':
        return GRU(hist_len, pred_len, in_dim, city_num, batch_size, device)
    elif exp_model == 'AGCRN':
        # History-only: AGCRN's forward() only ever takes the historical
        # sequence - see model/agcrn.py's docstring. It also has no fixed
        # graph at all - node_embeddings and the adjacency they imply are
        # learned end-to-end from data, in contrast to every physics/
        # geography-based graph model in this repo (including AirLapse).
        return AGCRNPM25(
            hist_len, pred_len, in_dim, city_num, batch_size, device,
            rnn_units=config['experiments'].get('agcrn_rnn_units', 64),
            num_layers=config['experiments'].get('agcrn_num_layers', 2),
            cheb_k=config['experiments'].get('agcrn_cheb_k', 2),
            embed_dim=config['experiments'].get('agcrn_embed_dim', 10),
        )
    elif exp_model == 'MegaCRN':
        # Unlike AGCRN above, this one DOES use future-known weather (as
        # decoder y_cov) - see model/megacrn.py's docstring. Its graph is
        # also learned (like AGCRN's) but from a shared memory bank rather
        # than free per-node embeddings, and is asymmetric (two directional
        # graphs, not one symmetric adjacency).
        return MegaCRNPM25(
            hist_len, pred_len, in_dim, city_num, batch_size, device,
            rnn_units=config['experiments'].get('megacrn_rnn_units', 64),
            num_layers=config['experiments'].get('megacrn_num_layers', 1),
            cheb_k=config['experiments'].get('megacrn_cheb_k', 3),
            mem_num=config['experiments'].get('megacrn_mem_num', 20),
            mem_dim=config['experiments'].get('megacrn_mem_dim', 64),
            memory_lamb=config['experiments'].get('megacrn_memory_lamb', 0.01),
            memory_lamb1=config['experiments'].get('megacrn_memory_lamb1', 0.01),
        )
    elif exp_model == 'Informer':
        return InformerPM25(
            hist_len, pred_len, in_dim, city_num, batch_size, device,
            label_len=config['experiments'].get('informer_label_len', None),
            d_model=config['experiments'].get('informer_d_model', 64),
            n_heads=config['experiments'].get('informer_n_heads', 8),
            e_layers=config['experiments'].get('informer_e_layers', 2),
            d_layers=config['experiments'].get('informer_d_layers', 1),
            d_ff=config['experiments'].get('informer_d_ff', 256),
            factor=config['experiments'].get('informer_factor', 5),
            dropout=config['experiments'].get('informer_dropout', 0.1),
            attn=config['experiments'].get('informer_attn', 'prob'),
            activation=config['experiments'].get('informer_activation', 'gelu'),
            distil=config['experiments'].get('informer_distil', True),
            mix=config['experiments'].get('informer_mix', True),
        )
    elif exp_model == 'Autoformer':
        return AutoformerPM25(
            hist_len, pred_len, in_dim, city_num, batch_size, device,
            label_len=config['experiments'].get('autoformer_label_len', None),
            d_model=config['experiments'].get('autoformer_d_model', 64),
            n_heads=config['experiments'].get('autoformer_n_heads', 8),
            e_layers=config['experiments'].get('autoformer_e_layers', 2),
            d_layers=config['experiments'].get('autoformer_d_layers', 1),
            d_ff=config['experiments'].get('autoformer_d_ff', 256),
            moving_avg=config['experiments'].get('autoformer_moving_avg', 25),
            factor=config['experiments'].get('autoformer_factor', 1),
            dropout=config['experiments'].get('autoformer_dropout', 0.1),
            activation=config['experiments'].get('autoformer_activation', 'gelu'),
        )
    elif exp_model == 'PatchTST':
        return PatchTSTPM25(
            hist_len, pred_len, in_dim, city_num, batch_size, device,
            patch_len=config['experiments'].get('patchtst_patch_len', 8),
            stride=config['experiments'].get('patchtst_stride', 4),
            d_model=config['experiments'].get('patchtst_d_model', 32),
            n_heads=config['experiments'].get('patchtst_n_heads', 4),
            e_layers=config['experiments'].get('patchtst_e_layers', 2),
            d_ff=config['experiments'].get('patchtst_d_ff', 128),
            dropout=config['experiments'].get('patchtst_dropout', 0.1),
            head_dropout=config['experiments'].get('patchtst_head_dropout', 0.1),
            activation=config['experiments'].get('patchtst_activation', 'gelu'),
            revin=config['experiments'].get('patchtst_revin', True),
        )
    elif exp_model == 'STAEformer':
        return STAEformerPM25(
            hist_len, pred_len, in_dim, city_num, batch_size, device,
            feature_mean=train_data.feature_mean,
            feature_std=train_data.feature_std,
            input_embedding_dim=config['experiments'].get('staeformer_input_dim', 24),
            tod_embedding_dim=config['experiments'].get('staeformer_tod_dim', 24),
            dow_embedding_dim=config['experiments'].get('staeformer_dow_dim', 24),
            adaptive_embedding_dim=config['experiments'].get('staeformer_adaptive_dim', 80),
            n_heads=config['experiments'].get('staeformer_n_heads', 4),
            e_layers=config['experiments'].get('staeformer_e_layers', 3),
            d_ff=config['experiments'].get('staeformer_d_ff', 256),
            dropout=config['experiments'].get('staeformer_dropout', 0.1),
            activation=config['experiments'].get('staeformer_activation', 'gelu'),
            dt_hours=config['experiments'].get('staeformer_dt_hours', 3.0),
        )
    elif exp_model == 'MGSFformer':
        # Target-only baseline: unlike every other model here, MGSFformer's
        # published architecture doesn't take `feature` (meteorological
        # covariates) at all - see model/mgsfformer.py's docstring for why
        # that's a deliberate choice, not an oversight.
        return MGSFformerPM25(
            hist_len, pred_len, in_dim, city_num, batch_size, device,
            ie_dim=config['experiments'].get('mgsfformer_ie_dim', 8),
            dropout=config['experiments'].get('mgsfformer_dropout', 0.1),
            num_head=config['experiments'].get('mgsfformer_num_head', 4),
        )
    elif exp_model == 'TimeXer':
        # History-only baseline like MGSFformer above: TimeXer's published
        # architecture is encoder-only, with no decoder input point for
        # feature's FUTURE portion - see model/timexer.py's docstring.
        # It does use feature's historical portion as exogenous covariates
        # though, unlike MGSFformer's target-only design.
        return TimeXerPM25(
            hist_len, pred_len, in_dim, city_num, batch_size, device,
            patch_len=config['experiments'].get('timexer_patch_len', 8),
            d_model=config['experiments'].get('timexer_d_model', 128),
            n_heads=config['experiments'].get('timexer_n_heads', 8),
            e_layers=config['experiments'].get('timexer_e_layers', 2),
            d_ff=config['experiments'].get('timexer_d_ff', 256),
            dropout=config['experiments'].get('timexer_dropout', 0.1),
            factor=config['experiments'].get('timexer_factor', 5),
            activation=config['experiments'].get('timexer_activation', 'gelu'),
            use_norm=config['experiments'].get('timexer_use_norm', True),
        )
    elif exp_model == 'WPMixer':
        # History-only baseline like TimeXer/MGSFformer above: WPMixer's
        # published architecture is purely autoregressive from the lookback
        # window - see model/wpmixer.py's docstring for confirmation (its
        # own short-term-forecast wrapper discards three extra dataloader
        # args a future-covariate-aware model would use).
        return WPMixerPM25(
            hist_len, pred_len, in_dim, city_num, batch_size, device,
            d_model=config['experiments'].get('wpmixer_d_model', 16),
            dropout=config['experiments'].get('wpmixer_dropout', 0.1),
            embedding_dropout=config['experiments'].get('wpmixer_embedding_dropout', 0.1),
            tfactor=config['experiments'].get('wpmixer_tfactor', 3),
            dfactor=config['experiments'].get('wpmixer_dfactor', 5),
            wavelet=config['experiments'].get('wpmixer_wavelet', 'db2'),
            level=config['experiments'].get('wpmixer_level', 1),
            patch_len=config['experiments'].get('wpmixer_patch_len', 4),
            stride=config['experiments'].get('wpmixer_stride', 2),
            no_decomposition=config['experiments'].get('wpmixer_no_decomposition', False),
        )
    elif exp_model == 'DTAF':
        # History-only baseline like WPMixer/TimeXer/MGSFformer above: the
        # architecture described in DTAF's own paper has no point where a
        # future-known covariate would enter - see model/dtaf.py's
        # docstring. Also note that file's docstring explains this is an
        # INDEPENDENT REIMPLEMENTATION from the paper's own description,
        # not a port - the official repo has no license.
        return DTAFPM25(
            hist_len, pred_len, in_dim, city_num, batch_size, device,
            patch_len=config['experiments'].get('dtaf_patch_len', 8),
            stride=config['experiments'].get('dtaf_stride', 4),
            d_model=config['experiments'].get('dtaf_d_model', 32),
            n_heads=config['experiments'].get('dtaf_n_heads', 4),
            e_layers=config['experiments'].get('dtaf_e_layers', 2),
            expert_num=config['experiments'].get('dtaf_expert_num', 4),
            expert_reduction=config['experiments'].get('dtaf_expert_reduction', 2),
            topk_freq=config['experiments'].get('dtaf_topk_freq', 8),
            dropout=config['experiments'].get('dtaf_dropout', 0.1),
        )
    elif exp_model == 'PM25_GNN':
        return PM25_GNN(hist_len, pred_len, in_dim, city_num, batch_size, device, graph.edge_index, graph.edge_attr, wind_mean, wind_std)
    elif exp_model == 'PM25_GNN_nosub':
        return PM25_GNN_nosub(hist_len, pred_len, in_dim, city_num, batch_size, device, graph.edge_index, graph.edge_attr, wind_mean, wind_std)
    elif exp_model == 'GC_LSTM':
        return GC_LSTM(hist_len, pred_len, in_dim, city_num, batch_size, device, graph.edge_index)
    elif exp_model == 'nodesFC_GRU':
        return nodesFC_GRU(hist_len, pred_len, in_dim, city_num, batch_size, device)
    elif exp_model == 'AirDDE':
        return AirDDEPM25(
            hist_len, pred_len, in_dim, city_num, batch_size, device,
            graph.edge_index, graph.edge_attr, wind_mean, wind_std,
            rnn_units=config['experiments'].get('airdde_rnn_units', 64),
            rnn_num_layers=config['experiments'].get('airdde_rnn_num_layers', 1),
            agcn_cheb_k=config['experiments'].get('airdde_agcn_cheb_k', 3),
            mem_num=config['experiments'].get('airdde_mem_num', 20),
            mem_dim=config['experiments'].get('airdde_mem_dim', 64),
            local_mem_tau=config['experiments'].get('airdde_local_mem_tau', 3),
            local_mem_k=config['experiments'].get('airdde_local_mem_k', 8),
            ode_gcn_hidden_dim=config['experiments'].get('airdde_ode_gcn_hidden_dim', 64),
            ode_cheb_k=config['experiments'].get('airdde_ode_cheb_k', 3),
            ode_num_layers=config['experiments'].get('airdde_ode_num_layers', 3),
            ode_method=config['experiments'].get('airdde_ode_method', 'dopri5'),
            ode_rtol=config['experiments'].get('airdde_ode_rtol', 1e-2),
            ode_atol=config['experiments'].get('airdde_ode_atol', 1e-2),
            ode_adjoint=config['experiments'].get('airdde_ode_adjoint', True),
        )
    elif exp_model == 'AirPhyNet':
        # Neural-ODE model - noticeably slower per batch than the other
        # benchmarks here (adjoint backward re-solves the ODE): ~6s forward
        # / ~70s backward for a batch of 2 at this repo's real scale
        # (184 nodes, hist_len=24, pred_len=8, CPU). Budget accordingly for
        # exp_repeat x epochs. Set airphynet_ode_adjoint: False to try the
        # non-adjoint solver instead (more memory, potentially faster
        # backward for a state this small - not yet benchmarked here).
        return AirPhyNetPM25(
            hist_len, pred_len, in_dim, city_num, batch_size, device,
            graph.edge_index, graph.edge_attr, wind_mean, wind_std,
            rnn_units=config['experiments'].get('airphynet_rnn_units', 64),
            latent_dim=config['experiments'].get('airphynet_latent_dim', 4),
            gcn_step=config['experiments'].get('airphynet_gcn_step', 2),
            diff_coeff=config['experiments'].get('airphynet_diff_coeff', 0.1),
            n_traj_samples=config['experiments'].get('airphynet_n_traj_samples', 1),
            ode_method=config['experiments'].get('airphynet_ode_method', 'dopri5'),
            ode_rtol=config['experiments'].get('airphynet_ode_rtol', 1e-3),
            ode_atol=config['experiments'].get('airphynet_ode_atol', 1e-4),
            ode_adjoint=config['experiments'].get('airphynet_ode_adjoint', True),
            filter_type=config['experiments'].get('airphynet_filter_type', 'diff_adv'),
            max_deriv=config['experiments'].get('airphynet_max_deriv', 10.0),
        )
    elif exp_model == 'AirDualODE':
        return AirDualODEPM25(
            hist_len, pred_len, in_dim, city_num, batch_size, device,
            graph.edge_index, graph.edge_attr, wind_mean, wind_std,
            phy_latent_dim=config['experiments'].get('airdualode_phy_latent_dim', 8),
            unk_latent_dim=config['experiments'].get('airdualode_unk_latent_dim', 8),
            fusion_dim=config['experiments'].get('airdualode_fusion_dim', 16),
            gcn_step=config['experiments'].get('airdualode_gcn_step', 2),
            rnn_units=config['experiments'].get('airdualode_rnn_units', 32),
            attn_heads=config['experiments'].get('airdualode_attn_heads', 2),
            estimate_coeff=config['experiments'].get('airdualode_estimate_coeff', False),
            ode_method=config['experiments'].get('airdualode_ode_method', 'dopri5'),
            ode_rtol=config['experiments'].get('airdualode_ode_rtol', 1e-3),
            ode_atol=config['experiments'].get('airdualode_ode_atol', 1e-4),
            ode_adjoint=config['experiments'].get('airdualode_ode_adjoint', True),
        )
    elif exp_model == 'AirFormer':
        return AirFormerPM25(
            hist_len, pred_len, in_dim, city_num, batch_size, device,
            graph.edge_index, graph.edge_attr, wind_mean, wind_std,
            station_coords=coords,
            hidden_channels=config['experiments'].get('airformer_hidden_channels', 32),
            end_channels=config['experiments'].get('airformer_end_channels', 512),
            blocks=config['experiments'].get('airformer_blocks', 4),
            num_heads=config['experiments'].get('airformer_num_heads', 2),
            mlp_expansion=config['experiments'].get('airformer_mlp_expansion', 2),
            depth=config['experiments'].get('airformer_depth', 1),
            dropout=config['experiments'].get('airformer_dropout', 0.3),
            spatial_flag=config['experiments'].get('airformer_spatial_flag', True),
            stochastic_flag=config['experiments'].get('airformer_stochastic_flag', True),
        )
    elif exp_model == 'AirLapse':
        return AirLapse(
            hist_len, pred_len, in_dim, city_num, batch_size, device,
            graph.edge_index, graph.edge_attr, wind_mean, wind_std,
            station_coords=coords,
            station_elevation=altitude,
            feature_mean=train_data.feature_mean,
            feature_std=train_data.feature_std,
            # Defaults below updated to the winning configuration from
            # tune_airlapse.py's Optuna search (results/
            # airlapse_hparam_search_*.txt), with the found continuous
            # values rounded to clean numbers (e.g. sigma_d 299.62 -> 300,
            # dropout 0.2436 -> 0.25) - none of the roundings move far
            # enough from the found optimum to matter. Two of the search's
            # bounds were hit or nearly hit (gru_max_lag at its upper limit
            # of 10, gru_sigma_d within 0.4 of its upper limit of 300) -
            # worth a follow-up search with those two ranges widened
            # (e.g. max_lag up to 14-16, sigma_d up to 400) to check
            # whether an even better configuration exists just past them.
            hidden_dim=config['experiments'].get('gru_hidden_dim', 64),
            latent_dim=config['experiments'].get('gru_latent_dim', 8),
            attn_dim=config['experiments'].get('gru_attn_dim', 32),
            num_layers=config['experiments'].get('gru_num_layers', 1),
            dropout=config['experiments'].get('gru_dropout', 0.25),
            logvar_clamp=config['experiments'].get('gru_logvar_clamp', 10.0),
            spatial_mix_mode=config['experiments'].get('gru_spatial_mix_mode', 'per_step'),
            max_lag=config['experiments'].get('gru_max_lag', 10),
            dist_threshold_km=config['experiments'].get('gru_dist_threshold_km', 375.0),
            sigma_d=config['experiments'].get('gru_sigma_d', 300.0),
            sigma_h=config['experiments'].get('gru_sigma_h', 1750.0),
            sigma_tau_init_h=config['experiments'].get('gru_sigma_tau_init_h', 2.5),
            # AirLapse's advection-diffusion timing (lag-matching Gaussian,
            # Green's-function transport estimate) treats each history step
            # as dt_hours real hours elapsed - it MUST match the active
            # dataset's actual step spacing (config's freq_hours; 3 for
            # KnowAir 1-3, 1 for dataset 4's hourly sensor network) or every
            # timing computation is silently off by that ratio. Falls back
            # to _ds_cfg's freq_hours (itself defaulting to 3, matching
            # KnowAir) rather than a bare 3.0, so this stays correct for any
            # dataset_num - not just the ones known about when this default
            # was tuned - unless gru_dt_hours is set explicitly to override.
            dt_hours=config['experiments'].get('gru_dt_hours', _ds_cfg.get('freq_hours', 3.0)),
            diffusivity_km2_per_hour_init=config['experiments'].get('gru_diffusivity_km2_per_hour_init', 50.0),
            t_eps_hours=config['experiments'].get('gru_t_eps_hours', 0.25),
        )
    elif exp_model == 'AirLapseV2':
        # Own airlapsev2_* prefix throughout (not gru_*) even though most
        # of these mirror AirLapse V1's own defaults 1:1 - keeps a future
        # Optuna search over one from ever silently perturbing the other,
        # matching every other benchmark model's own-prefix convention
        # here (airdde_*, airphynet_*, ...). Only diff_hidden_dim/
        # diffusivity_along_init/diffusivity_cross_init are genuinely new
        # (AdaptivePhysicsTransport2D's context-adaptive diffusivity MLP -
        # see model/airlapse_v2.py's module docstring for why).
        return AirLapseV2(
            hist_len, pred_len, in_dim, city_num, batch_size, device,
            graph.edge_index, graph.edge_attr, wind_mean, wind_std,
            station_coords=coords,
            station_elevation=altitude,
            feature_mean=train_data.feature_mean,
            feature_std=train_data.feature_std,
            # Defaults below reverted to AirLapse V1's own tuned values
            # (see its branch above), not V2's own Optuna search result -
            # that search (tune_airlapse_v2.py) showed only negligible
            # improvement over untuned defaults, not enough to justify
            # keeping a divergent, V2-specific set of values over the ones
            # already validated for V1. Every value is still overridable
            # via its airlapsev2_* config.yaml key.
            hidden_dim=config['experiments'].get('airlapsev2_hidden_dim', 64),
            latent_dim=config['experiments'].get('airlapsev2_latent_dim', 8),
            attn_dim=config['experiments'].get('airlapsev2_attn_dim', 32),
            num_layers=config['experiments'].get('airlapsev2_num_layers', 1),
            dropout=config['experiments'].get('airlapsev2_dropout', 0.25),
            # No logvar_clamp here (unlike V1's branch above): AirLapseV2
            # dropped the VAE stochastic latent entirely - negligible
            # improvement from tuning didn't justify the added training
            # noise/KL loss (see model/airlapse_v2.py's docstrings).
            # Dataset-family-aware, same reasoning as max_lag below:
            # 'per_step' is V1's own tuned value, but every test of it on
            # dataset 4 in this repo's history (including a 3-epoch check
            # right after the max_lag fix below was added) shows the same
            # "train_loss much larger than val_loss" spike signature that
            # caused the original divergence - milder since the transport/
            # pm25_lag clamps in model/airlapse_v2.py's forward() bound the
            # quantity that blew up, but not gone. 'bottleneck' has been
            # consistently clean AND scored better (RMSE ~2.94-2.97 at
            # full/near-full budget) every time it's been tried on this
            # dataset. Keep 'per_step' for KnowAir, where it's V1's own
            # validated choice and this instability was never observed.
            spatial_mix_mode=config['experiments'].get(
                'airlapsev2_spatial_mix_mode', 'per_step' if _ds_cfg.get('family', 'knowair') == 'knowair' else 'bottleneck'),
            # Dataset-family-aware, like dt_hours below - NOT just carried
            # over from V1's tuned 10. EDA (neighbor lag correlation
            # matched against wind-implied travel time, redone directly on
            # SensorAir.npy - see the conversation/PR that added this)
            # found a real, positive wind-conditioned transport signal on
            # dataset 4's graph, but concentrated in ~1-3 hours and gone by
            # ~8h - a much shorter useful window than KnowAir's own EDA
            # table (reports/eda/tables/sec4_5_wind_implied_transport_time.csv)
            # shows there (positive lift persisting to 21h+), consistent
            # with this network's much denser station spacing. Separately,
            # tune_airlapse_v2.py's Optuna search had already found
            # max_lag=4 as its best value for dataset 4 - two independent
            # signals pointing the same way.
            max_lag=config['experiments'].get(
                'airlapsev2_max_lag', 10 if _ds_cfg.get('family', 'knowair') == 'knowair' else 4),
            dist_threshold_km=config['experiments'].get('airlapsev2_dist_threshold_km', 375.0),
            sigma_d=config['experiments'].get('airlapsev2_sigma_d', 300.0),
            sigma_h=config['experiments'].get('airlapsev2_sigma_h', 1750.0),
            sigma_tau_init_h=config['experiments'].get('airlapsev2_sigma_tau_init_h', 2.5),
            # Same freq_hours-aware default as AirLapse V1 (see its branch
            # above) - avoids reintroducing the same 3-hour-cadence-
            # assumed-everywhere bug for V2 from day one.
            dt_hours=config['experiments'].get('airlapsev2_dt_hours', _ds_cfg.get('freq_hours', 3.0)),
            diff_hidden_dim=config['experiments'].get('airlapsev2_diff_hidden_dim', 16),
            # V1 has one isotropic diffusivity_km2_per_hour_init (50.0) -
            # no along/cross split to revert to, so both start at that
            # same value (a temporarily-isotropic starting point) rather
            # than either V2's own original asymmetric default (50/20) or
            # the tuned one (140/80).
            diffusivity_along_init=config['experiments'].get('airlapsev2_diffusivity_along_init', 50.0),
            diffusivity_cross_init=config['experiments'].get('airlapsev2_diffusivity_cross_init', 50.0),
            t_eps_hours=config['experiments'].get('airlapsev2_t_eps_hours', 0.25),
            # "Option B" joint spatio-temporal encoder (see
            # TemporalGraphEncoder's docstring in model/airlapse_v2.py) -
            # only takes effect in 'bottleneck' mode. 1 layer by default;
            # 0 disables it for an apples-to-apples ablation against the
            # per-node-independent encoding this file used before.
            st_encoder_layers=config['experiments'].get('airlapsev2_st_encoder_layers', 1),
            # 'softmax_lag' (default) keeps the original, already-tuned
            # weighted-average-over-lag + peak-reach transport estimate;
            # 'sum_integral' switches to literal double summation over
            # (tau, j), dt_hours-scaled - see AdaptivePhysicsTransport2D's
            # docstring ("TRANSPORT AGGREGATION MODES") for the tradeoff.
            # Left at the original default until A/B'd on dataset 4.
            transport_agg=config['experiments'].get('airlapsev2_transport_agg', 'softmax_lag'),
        )
    else:
        raise Exception('Wrong model name!')


def train(train_loader, model, optimizer):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        pm25, feature, time_arr = data
        pm25 = pm25.to(device)
        feature = feature.to(device)
        pm25_label = pm25[:, hist_len:]
        pm25_hist = pm25[:, :hist_len]
        pm25_pred = model(pm25_hist, feature)
        loss = criterion(pm25_pred, pm25_label)
        # Optional auxiliary regularization terms (VAE-style KL / dual-branch
        # alignment). val()/test() deliberately do NOT include these - the
        # validation loss is used for early stopping / best-model selection
        # and should reflect pure predictive accuracy, not training-time
        # regularizers.
        kl = getattr(model, 'last_kl_loss', None)
        if kl is not None:
            loss = loss + kl_weight * kl
        alignment = getattr(model, 'last_alignment_loss', None)
        if alignment is not None:
            loss = loss + alignment_weight * alignment
        # MegaCRN's memory-bank separate/compact losses (see model/megacrn.py) -
        # already weighted internally (memory_lamb/memory_lamb1), unlike
        # kl_weight/alignment_weight above which are applied externally here.
        memory_loss = getattr(model, 'last_memory_loss', None)
        if memory_loss is not None:
            loss = loss + memory_loss
        moe_diversity = getattr(model, 'last_moe_diversity_loss', None)
        if moe_diversity is not None:
            loss = loss + moe_diversity_weight * moe_diversity
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= batch_idx + 1
    return train_loss


def val(val_loader, model):
    model.eval()
    val_loss = 0
    for batch_idx, data in enumerate(val_loader):
        pm25, feature, time_arr = data
        pm25 = pm25.to(device)
        feature = feature.to(device)
        pm25_label = pm25[:, hist_len:]
        pm25_hist = pm25[:, :hist_len]
        pm25_pred = model(pm25_hist, feature)
        loss = criterion(pm25_pred, pm25_label)
        val_loss += loss.item()

    val_loss /= batch_idx + 1
    return val_loss


def test(test_loader, model):
    model.eval()
    predict_list = []
    label_list = []
    time_list = []
    test_loss = 0
    for batch_idx, data in enumerate(test_loader):
        pm25, feature, time_arr = data
        pm25 = pm25.to(device)
        feature = feature.to(device)
        pm25_label = pm25[:, hist_len:]
        pm25_hist = pm25[:, :hist_len]
        pm25_pred = model(pm25_hist, feature)
        loss = criterion(pm25_pred, pm25_label)
        test_loss += loss.item()

        pm25_pred_val = np.concatenate([pm25_hist.cpu().detach().numpy(), pm25_pred.cpu().detach().numpy()], axis=1) * pm25_std + pm25_mean
        pm25_label_val = pm25.cpu().detach().numpy() * pm25_std + pm25_mean
        predict_list.append(pm25_pred_val)
        label_list.append(pm25_label_val)
        time_list.append(time_arr.cpu().detach().numpy())

    test_loss /= batch_idx + 1

    predict_epoch = np.concatenate(predict_list, axis=0)
    label_epoch = np.concatenate(label_list, axis=0)
    time_epoch = np.concatenate(time_list, axis=0)
    predict_epoch[predict_epoch < 0] = 0

    return test_loss, predict_epoch, label_epoch, time_epoch


def get_mean_std(data_list):
    data = np.asarray(data_list)
    return data.mean(), data.std()


def reset_peak_memory():
    """Call once before a repeat's training starts. On GPU, torch tracks
    the true peak itself from here on (see peak_memory_mb). On CPU there's
    no equivalent - the best we can do without a background sampling
    thread is return the current RSS baseline, and let the caller take a
    running max via peak_memory_mb(..., running_max) at a few points
    during training (this file samples after every epoch, see main())."""
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)
        return 0.0
    elif _psutil_process is not None:
        return _psutil_process.memory_info().rss / 1e6
    return float('nan')


def peak_memory_mb(running_max=0.0):
    """GPU: exact peak since the last reset_peak_memory() call. CPU: max of
    running_max and the current RSS sample (see reset_peak_memory's note -
    this is a point-in-time approximation of peak, not a true one)."""
    if use_cuda:
        return torch.cuda.max_memory_allocated(device) / 1e6
    elif _psutil_process is not None:
        return max(running_max, _psutil_process.memory_info().rss / 1e6)
    return float('nan')


def measure_inference_latency(test_loader, model):
    """Dedicated no_grad timing pass, separate from test()'s accuracy-focused
    evaluation (which also builds up predict/label arrays and doesn't use
    no_grad) - this is purely for the ms/sample efficiency number."""
    model.eval()
    n_samples = 0
    t0 = time.time()
    with torch.no_grad():
        for data in test_loader:
            pm25, feature, time_arr = data
            pm25 = pm25.to(device)
            feature = feature.to(device)
            pm25_hist = pm25[:, :hist_len]
            model(pm25_hist, feature)
            n_samples += pm25.shape[0]
    elapsed = time.time() - t0
    return (elapsed / n_samples * 1000) if n_samples else float('nan')


def main():
    exp_info = get_exp_info()
    # Colab (and other captured/non-TTY runners) has a per-cell output-line
    # cap (~5000 lines) - the old per-batch tqdm bars and per-epoch dumps
    # blew past that on any real run and killed the cell outright. Console
    # output is trimmed to this one line; the full exp_info/model summary/
    # metrics (everything that used to scroll by above) still gets written
    # to metric_fp below exactly as before, so nothing is actually lost -
    # it just isn't echoed to stdout during training.
    print('Model: %s | Dataset: %s | hist_len: %s | pred_len: %s' % (exp_model, dataset_num, hist_len, pred_len))

    exp_time = arrow.now().format('YYYYMMDDHHmmss')

    train_loss_list, val_loss_list, test_loss_list, rmse_list, mae_list, mape_list, csi_list, pod_list, far_list = [], [], [], [], [], [], [], [], []
    # Efficiency metrics, commonly reported alongside accuracy in papers:
    # model size, wall-clock training cost, inference latency, peak memory.
    param_count_list, epoch_time_list, inference_time_list, peak_memory_list = [], [], [], []

    for exp_idx in range(exp_repeat):
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, drop_last=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True)

        model = get_model()
        model = model.to(device)
        model_name = type(model).__name__
        param_count = sum(p.numel() for p in model.parameters())

        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)

        exp_model_dir = os.path.join(results_dir, '%s_%s' % (hist_len, pred_len), str(dataset_num), model_name, str(exp_time), '%02d' % exp_idx)
        if not os.path.exists(exp_model_dir):
            os.makedirs(exp_model_dir)
        model_fp = os.path.join(exp_model_dir, 'model.pth')

        val_loss_min = 100000
        best_epoch = 0

        train_loss_, val_loss_ = 0, 0
        epoch_times = []
        running_peak_mb = reset_peak_memory()

        for epoch in range(epochs):
            t0 = time.time()
            train_loss = train(train_loader, model, optimizer)
            epoch_times.append(time.time() - t0)
            val_loss = val(val_loader, model)
            running_peak_mb = peak_memory_mb(running_peak_mb)

            if epoch - best_epoch > early_stop:
                break

            if val_loss < val_loss_min:
                val_loss_min = val_loss
                best_epoch = epoch
                torch.save(model.state_dict(), model_fp)

                test_loss, predict_epoch, label_epoch, time_epoch = test(test_loader, model)
                train_loss_, val_loss_ = train_loss, val_loss
                rmse, mae, mape, csi, pod, far = get_metric(predict_epoch, label_epoch)

                if save_npy:
                    np.save(os.path.join(exp_model_dir, 'predict.npy'), predict_epoch)
                    np.save(os.path.join(exp_model_dir, 'label.npy'), label_epoch)
                    np.save(os.path.join(exp_model_dir, 'time.npy'), time_epoch)

        inference_time_ms = measure_inference_latency(test_loader, model)
        running_peak_mb = peak_memory_mb(running_peak_mb)

        train_loss_list.append(train_loss_)
        val_loss_list.append(val_loss_)
        test_loss_list.append(test_loss)
        rmse_list.append(rmse)
        mae_list.append(mae)
        mape_list.append(mape)
        csi_list.append(csi)
        pod_list.append(pod)
        far_list.append(far)
        param_count_list.append(param_count)
        epoch_time_list.append(np.mean(epoch_times) if epoch_times else float('nan'))
        inference_time_list.append(inference_time_ms)
        peak_memory_list.append(running_peak_mb)

    exp_metric_str = '---------------------------------------\n' + \
                     'train_loss | mean: %0.4f std: %0.4f\n' % (get_mean_std(train_loss_list)) + \
                     'val_loss   | mean: %0.4f std: %0.4f\n' % (get_mean_std(val_loss_list)) + \
                     'test_loss  | mean: %0.4f std: %0.4f\n' % (get_mean_std(test_loss_list)) + \
                     'RMSE       | mean: %0.4f std: %0.4f\n' % (get_mean_std(rmse_list)) + \
                     'MAE        | mean: %0.4f std: %0.4f\n' % (get_mean_std(mae_list)) + \
                     'MAPE       | mean: %0.4f%% std: %0.4f%%\n' % (get_mean_std(mape_list)) + \
                     'CSI        | mean: %0.4f std: %0.4f\n' % (get_mean_std(csi_list)) + \
                     'POD        | mean: %0.4f std: %0.4f\n' % (get_mean_std(pod_list)) + \
                     'FAR        | mean: %0.4f std: %0.4f\n' % (get_mean_std(far_list)) + \
                     '---------------------------------------\n' + \
                     'Efficiency (mean +/- std over %d repeat(s)):\n' % exp_repeat + \
                     'param_count       | %d\n' % param_count_list[0] + \
                     'epoch_train_time  | mean: %0.2fs std: %0.2fs\n' % (get_mean_std(epoch_time_list)) + \
                     'inference_latency | mean: %0.3fms/sample std: %0.3fms/sample\n' % (get_mean_std(inference_time_list)) + \
                     'peak_memory       | mean: %0.1fMB std: %0.1fMB\n' % (get_mean_std(peak_memory_list))

    # "{order}. {name} {year}_{hist_len}_{pred_len}_{dataset_num}.txt" - a
    # plain "metric.txt" loses all identifying info the moment it's copied
    # out of its (already quite deep) results_dir/{hist_len}_{pred_len}/
    # {dataset_num}/{model}/{exp_time}/ folder; this makes a flat pile of
    # these files still self-describing, AND keeps them consistent with
    # config.yaml's numbered/year-annotated model catalog. Built from
    # exp_model (the config.yaml dispatch string, e.g. 'Informer'), not
    # model_name (the model class's own __name__, e.g. 'InformerPM25' for
    # several benchmarks here) - exp_model is what actually matches the
    # catalog. No collision risk across runs - exp_time already makes the
    # parent directory unique per invocation.
    metric_fp = os.path.join(os.path.dirname(exp_model_dir), '%s_%s_%s_%s.txt' % (_catalog_label(exp_model), hist_len, pred_len, dataset_num))
    with open(metric_fp, 'w') as f:
        f.write(exp_info)
        f.write(str(model))
        f.write(exp_metric_str)

    # One more line, not a wall of them - full results (exp_info, model
    # summary, all metrics) live in this file, same as always.
    print(metric_fp)


if __name__ == '__main__':
    main()