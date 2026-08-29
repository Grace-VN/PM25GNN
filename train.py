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
from model.mgsfformer import MGSFformerPM25
from model.timexer import TimeXerPM25
from model.agcrn import AGCRNPM25
from model.megacrn import MegaCRNPM25

import arrow
import torch
from torch import nn
from tqdm import tqdm
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

torch.set_num_threads(1)
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

graph = Graph()
city_num = graph.node_num

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
    if exp_model == 'MLP':
        return MLP(hist_len, pred_len, in_dim)
    elif exp_model == 'LSTM':
        return LSTM(hist_len, pred_len, in_dim, city_num, batch_size, device)
    elif exp_model == 'GRU':
        return GRU(hist_len, pred_len, in_dim, city_num, batch_size, device)
    elif exp_model == 'nodesFC_GRU':
        return nodesFC_GRU(hist_len, pred_len, in_dim, city_num, batch_size, device)
    elif exp_model == 'GC_LSTM':
        return GC_LSTM(hist_len, pred_len, in_dim, city_num, batch_size, device, graph.edge_index)
    elif exp_model == 'PM25_GNN':
        return PM25_GNN(hist_len, pred_len, in_dim, city_num, batch_size, device, graph.edge_index, graph.edge_attr, wind_mean, wind_std)
    elif exp_model == 'PM25_GNN_nosub':
        return PM25_GNN_nosub(hist_len, pred_len, in_dim, city_num, batch_size, device, graph.edge_index, graph.edge_attr, wind_mean, wind_std)
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
    elif exp_model == 'AirLapse':
        return AirLapse(
            hist_len, pred_len, in_dim, city_num, batch_size, device,
            graph.edge_index, graph.edge_attr, wind_mean, wind_std,
            station_coords=coords,
            station_elevation=altitude,
            feature_mean=train_data.feature_mean,
            feature_std=train_data.feature_std,
            hidden_dim=config['experiments'].get('gru_hidden_dim', 64),
            latent_dim=config['experiments'].get('gru_latent_dim', 16),
            attn_dim=config['experiments'].get('gru_attn_dim', 32),
            num_layers=config['experiments'].get('gru_num_layers', 1),
            dropout=config['experiments'].get('gru_dropout', 0.1),
            logvar_clamp=config['experiments'].get('gru_logvar_clamp', 10.0),
            spatial_mix_mode=config['experiments'].get('gru_spatial_mix_mode', 'bottleneck'),
            max_lag=config['experiments'].get('gru_max_lag', 6),
            dist_threshold_km=config['experiments'].get('gru_dist_threshold_km', 300.0),
            sigma_d=config['experiments'].get('gru_sigma_d', 200.0),
            sigma_h=config['experiments'].get('gru_sigma_h', 1200.0),
            sigma_tau_init_h=config['experiments'].get('gru_sigma_tau_init_h', 3.0),
            dt_hours=config['experiments'].get('gru_dt_hours', 3.0),
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
    elif exp_model == 'AGCRN':
        # History-only baseline like the three above: AGCRN's forward()
        # only ever takes the historical sequence - see model/agcrn.py's
        # docstring. Unlike the other three, it also has no fixed graph at
        # all - node_embeddings and the adjacency they imply are learned
        # end-to-end from data, in contrast to every physics/geography-
        # based graph model in this repo (including AirLapse itself).
        return AGCRNPM25(
            hist_len, pred_len, in_dim, city_num, batch_size, device,
            rnn_units=config['experiments'].get('agcrn_rnn_units', 64),
            num_layers=config['experiments'].get('agcrn_num_layers', 2),
            cheb_k=config['experiments'].get('agcrn_cheb_k', 2),
            embed_dim=config['experiments'].get('agcrn_embed_dim', 10),
        )
    elif exp_model == 'MegaCRN':
        # Unlike MGSFformer/TimeXer/AGCRN above, this one DOES use future-
        # known weather (as decoder y_cov) - see model/megacrn.py's
        # docstring. Its graph is also learned (like AGCRN's) but from a
        # shared memory bank rather than free per-node embeddings, and is
        # asymmetric (two directional graphs, not one symmetric adjacency).
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
    else:
        raise Exception('Wrong model name!')


def train(train_loader, model, optimizer):
    model.train()
    train_loss = 0
    for batch_idx, data in tqdm(enumerate(train_loader)):
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
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= batch_idx + 1
    return train_loss


def val(val_loader, model):
    model.eval()
    val_loss = 0
    for batch_idx, data in tqdm(enumerate(val_loader)):
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
    print(exp_info)

    exp_time = arrow.now().format('YYYYMMDDHHmmss')

    train_loss_list, val_loss_list, test_loss_list, rmse_list, mae_list, mape_list, csi_list, pod_list, far_list = [], [], [], [], [], [], [], [], []
    # Efficiency metrics, commonly reported alongside accuracy in papers:
    # model size, wall-clock training cost, inference latency, peak memory.
    param_count_list, epoch_time_list, inference_time_list, peak_memory_list = [], [], [], []

    for exp_idx in range(exp_repeat):
        print('\nNo.%2d experiment ~~~' % exp_idx)

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, drop_last=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True)

        model = get_model()
        model = model.to(device)
        model_name = type(model).__name__
        param_count = sum(p.numel() for p in model.parameters())

        print(str(model))
        print('param_count: %d' % param_count)

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
            print('\nTrain epoch %s:' % (epoch))

            t0 = time.time()
            train_loss = train(train_loader, model, optimizer)
            epoch_times.append(time.time() - t0)
            val_loss = val(val_loader, model)
            running_peak_mb = peak_memory_mb(running_peak_mb)

            print('train_loss: %.4f' % train_loss)
            print('val_loss: %.4f' % val_loss)

            if epoch - best_epoch > early_stop:
                break

            if val_loss < val_loss_min:
                val_loss_min = val_loss
                best_epoch = epoch
                print('Minimum val loss!!!')
                torch.save(model.state_dict(), model_fp)
                print('Save model: %s' % model_fp)

                test_loss, predict_epoch, label_epoch, time_epoch = test(test_loader, model)
                train_loss_, val_loss_ = train_loss, val_loss
                rmse, mae, mape, csi, pod, far = get_metric(predict_epoch, label_epoch)
                print('Train loss: %0.4f, Val loss: %0.4f, Test loss: %0.4f, RMSE: %0.2f, MAE: %0.2f, MAPE: %0.2f%%, CSI: %0.4f, POD: %0.4f, FAR: %0.4f' % (train_loss_, val_loss_, test_loss, rmse, mae, mape, csi, pod, far))

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

        print('\nNo.%2d experiment results:' % exp_idx)
        print(
            'Train loss: %0.4f, Val loss: %0.4f, Test loss: %0.4f, RMSE: %0.2f, MAE: %0.2f, MAPE: %0.2f%%, CSI: %0.4f, POD: %0.4f, FAR: %0.4f' % (
            train_loss_, val_loss_, test_loss, rmse, mae, mape, csi, pod, far))
        print(
            'param_count: %d, avg epoch train time: %0.2fs, inference: %0.3fms/sample, peak memory: %0.1fMB' % (
            param_count, epoch_time_list[-1], inference_time_ms, running_peak_mb))

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

    # {model}_{hist_len}_{pred_len}_{dataset_num}.txt - a plain "metric.txt"
    # loses all identifying info the moment it's copied out of its (already
    # quite deep) results_dir/{hist_len}_{pred_len}/{dataset_num}/{model}/
    # {exp_time}/ folder; this makes a flat pile of these files still
    # self-describing. No collision risk across runs - exp_time already
    # makes the parent directory unique per invocation.
    metric_fp = os.path.join(os.path.dirname(exp_model_dir), '%s_%s_%s_%s.txt' % (model_name, hist_len, pred_len, dataset_num))
    with open(metric_fp, 'w') as f:
        f.write(exp_info)
        f.write(str(model))
        f.write(exp_metric_str)

    print('=========================\n')
    print(exp_info)
    print(exp_metric_str)
    print(str(model))
    print(metric_fp)


if __name__ == '__main__':
    main()