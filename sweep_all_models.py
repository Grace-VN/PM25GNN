"""
Run every model in this repo on ONE fixed (dataset_num, hist_len, pred_len)
config, sequentially, in a single process - so a Kaggle/Colab/local run
doesn't need to re-launch train.py (and re-edit config.yaml) once per model.

For each model, trains --exp_repeat times (default from config.yaml's
train.exp_repeat), averages the metrics, and writes:

    <output_dir>/<MODEL>_<hist_len>_<pred_len>_<dataset_num>.txt - one detailed report per model
    <output_dir>/summary.csv                      - one row per model, side by side

Usage:
    python sweep_all_models.py --dataset_num 3 --hist_len 24 --pred_len 8
    python sweep_all_models.py --dataset_num 3 --hist_len 24 --pred_len 8 \
        --models MLP,GRU,Informer --epochs 5 --exp_repeat 2
    python sweep_all_models.py --dataset_num 3 --hist_len 24 --pred_len 8 \
        --output_dir /kaggle/working/sweep_output

Respects the same KNOWAIR_FP / RESULTS_DIR env vars as train.py/util.py -
see README's "Running on Kaggle" section. --output_dir is separate from
RESULTS_DIR (checkpoints, if --save_checkpoints, still go under
RESULTS_DIR; the per-model reports and summary.csv go under --output_dir).

All models this script can name are listed in ALL_MODELS below - kept in
sync with train.py's get_model() dispatcher, which this file's get_model()
mirrors (same construction calls, same config.yaml-driven hyperparameter
overrides).
"""

import argparse
import csv
import os
import sys
import time
import traceback

try:
    import psutil
    _psutil_process = psutil.Process()
except ImportError:
    # peak_memory_mb reports NaN on CPU runs without it; GPU runs are
    # unaffected (torch.cuda's own memory stats don't need psutil).
    psutil = None
    _psutil_process = None

proj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(proj_dir)

import arrow
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

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
from model.airlapse2 import AirLapse2
from model.airlapse3 import AirLapse3
from model.airlapse4 import AirLapse4
from model.airlapse5 import AirLapse5
from model.airlapse6 import AirLapse6
from model.airlapse7 import AirLapse7
from model.mgsfformer import MGSFformerPM25
from model.timexer import TimeXerPM25
from model.agcrn import AGCRNPM25
from model.megacrn import MegaCRNPM25

ALL_MODELS = [
    'MLP', 'LSTM', 'GRU', 'GC_LSTM', 'nodesFC_GRU',
    'PM25_GNN', 'PM25_GNN_nosub', 'AirFormer', 'Informer', 'Autoformer',
    'PatchTST', 'STAEformer', 'AirDDE', 'AirPhyNet', 'AirDualODE',
    'AirLapse', 'AirLapse2', 'AirLapse3', 'AirLapse4', 'AirLapse5', 'AirLapse6', 'AirLapse7',
    'MGSFformer', 'TimeXer', 'AGCRN', 'MegaCRN',
]

ACCURACY_KEYS = ['train_loss', 'val_loss', 'test_loss', 'rmse', 'mae', 'mape', 'csi', 'pod', 'far']
EFFICIENCY_KEYS = ['param_count', 'epoch_train_time_sec', 'inference_latency_ms', 'peak_memory_mb']
METRIC_KEYS = ACCURACY_KEYS + EFFICIENCY_KEYS
METRIC_UNITS = {'mape': '%', 'epoch_train_time_sec': 's', 'inference_latency_ms': 'ms/sample', 'peak_memory_mb': 'MB'}


def _fmt_metric(k, mean, std):
    if k == 'param_count':
        return f'{k:22s} | {mean:.0f}'
    unit = METRIC_UNITS.get(k, '')
    return f'{k:22s} | mean: {mean:.4f}{unit}  std: {std:.4f}{unit}'


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset_num', type=int, required=True, choices=(1, 2, 3))
    p.add_argument('--hist_len', type=int, required=True)
    p.add_argument('--pred_len', type=int, required=True)
    p.add_argument('--models', type=str, default='all',
                    help="Comma-separated model names, or 'all' (default) for every model in ALL_MODELS.")
    p.add_argument('--exp_repeat', type=int, default=config['train']['exp_repeat'])
    p.add_argument('--epochs', type=int, default=config['train']['epochs'])
    p.add_argument('--batch_size', type=int, default=config['train']['batch_size'])
    p.add_argument('--early_stop', type=int, default=config['train']['early_stop'])
    p.add_argument('--lr', type=float, default=config['train']['lr'])
    p.add_argument('--weight_decay', type=float, default=config['train']['weight_decay'])
    p.add_argument('--save_npy', action='store_true', help='Save predict/label/time .npy per repeat (off by default - adds up fast across many models).')
    p.add_argument('--save_checkpoints', action='store_true', help='Save model.pth per repeat under RESULTS_DIR (off by default).')
    p.add_argument('--output_dir', type=str, default=os.path.join(proj_dir, 'sweep_output'))
    args = p.parse_args()
    args.models = ALL_MODELS if args.models.strip().lower() == 'all' else [m.strip() for m in args.models.split(',') if m.strip()]
    unknown = [m for m in args.models if m not in ALL_MODELS]
    if unknown:
        p.error(f'Unknown model name(s): {unknown}. Valid options: {ALL_MODELS}')
    return args


def get_model(exp_model, hist_len, pred_len, in_dim, city_num, batch_size, device,
              graph, wind_mean, wind_std, coords, altitude, train_data):
    # Mirrors train.py's get_model() exactly (same construction calls / config.yaml
    # overrides) - kept as its own copy here since this script builds a fresh
    # dataset per CLI config instead of once at train.py's module-import time.
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
            station_coords=coords, station_elevation=altitude,
            feature_mean=train_data.feature_mean, feature_std=train_data.feature_std,
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
    elif exp_model == 'AirLapse2':
        # AirLapse with a trend-augmented key/value in its spatial attention
        # (model/airlapse2.py) - identical elsewhere, so it reuses AirLapse's
        # gru_* config keys under a gru2_* prefix (independently tunable, same
        # defaults) rather than inventing a parallel set of names.
        return AirLapse2(
            hist_len, pred_len, in_dim, city_num, batch_size, device,
            graph.edge_index, graph.edge_attr, wind_mean, wind_std,
            station_coords=coords, station_elevation=altitude,
            feature_mean=train_data.feature_mean, feature_std=train_data.feature_std,
            hidden_dim=config['experiments'].get('gru2_hidden_dim', 64),
            latent_dim=config['experiments'].get('gru2_latent_dim', 16),
            attn_dim=config['experiments'].get('gru2_attn_dim', 32),
            num_layers=config['experiments'].get('gru2_num_layers', 1),
            dropout=config['experiments'].get('gru2_dropout', 0.1),
            logvar_clamp=config['experiments'].get('gru2_logvar_clamp', 10.0),
            spatial_mix_mode=config['experiments'].get('gru2_spatial_mix_mode', 'bottleneck'),
            max_lag=config['experiments'].get('gru2_max_lag', 6),
            dist_threshold_km=config['experiments'].get('gru2_dist_threshold_km', 300.0),
            sigma_d=config['experiments'].get('gru2_sigma_d', 200.0),
            sigma_h=config['experiments'].get('gru2_sigma_h', 1200.0),
            sigma_tau_init_h=config['experiments'].get('gru2_sigma_tau_init_h', 3.0),
            dt_hours=config['experiments'].get('gru2_dt_hours', 3.0),
        )
    elif exp_model == 'AirLapse3':
        # AirLapse with an explicit, non-learned "transported pollution
        # from neighbors" estimate added alongside its spatial attention
        # (model/airlapse3.py) - identical elsewhere, so it reuses
        # AirLapse's gru_* config keys under a gru3_* prefix (independently
        # tunable, same defaults) rather than inventing a parallel set of
        # names.
        return AirLapse3(
            hist_len, pred_len, in_dim, city_num, batch_size, device,
            graph.edge_index, graph.edge_attr, wind_mean, wind_std,
            station_coords=coords, station_elevation=altitude,
            feature_mean=train_data.feature_mean, feature_std=train_data.feature_std,
            hidden_dim=config['experiments'].get('gru3_hidden_dim', 64),
            latent_dim=config['experiments'].get('gru3_latent_dim', 16),
            attn_dim=config['experiments'].get('gru3_attn_dim', 32),
            num_layers=config['experiments'].get('gru3_num_layers', 1),
            dropout=config['experiments'].get('gru3_dropout', 0.1),
            logvar_clamp=config['experiments'].get('gru3_logvar_clamp', 10.0),
            spatial_mix_mode=config['experiments'].get('gru3_spatial_mix_mode', 'bottleneck'),
            max_lag=config['experiments'].get('gru3_max_lag', 6),
            dist_threshold_km=config['experiments'].get('gru3_dist_threshold_km', 300.0),
            sigma_d=config['experiments'].get('gru3_sigma_d', 200.0),
            sigma_h=config['experiments'].get('gru3_sigma_h', 1200.0),
            sigma_tau_init_h=config['experiments'].get('gru3_sigma_tau_init_h', 3.0),
            dt_hours=config['experiments'].get('gru3_dt_hours', 3.0),
        )
    elif exp_model == 'AirLapse4':
        # AirLapse3 with its explicit transport estimate upgraded to the 1D
        # advection-diffusion Green's function (model/airlapse4.py) -
        # identical elsewhere, so it reuses AirLapse3's gru_* config keys
        # under a gru4_* prefix (independently tunable, same defaults)
        # plus the two new diffusion hyperparameters.
        return AirLapse4(
            hist_len, pred_len, in_dim, city_num, batch_size, device,
            graph.edge_index, graph.edge_attr, wind_mean, wind_std,
            station_coords=coords, station_elevation=altitude,
            feature_mean=train_data.feature_mean, feature_std=train_data.feature_std,
            hidden_dim=config['experiments'].get('gru4_hidden_dim', 64),
            latent_dim=config['experiments'].get('gru4_latent_dim', 16),
            attn_dim=config['experiments'].get('gru4_attn_dim', 32),
            num_layers=config['experiments'].get('gru4_num_layers', 1),
            dropout=config['experiments'].get('gru4_dropout', 0.1),
            logvar_clamp=config['experiments'].get('gru4_logvar_clamp', 10.0),
            spatial_mix_mode=config['experiments'].get('gru4_spatial_mix_mode', 'bottleneck'),
            max_lag=config['experiments'].get('gru4_max_lag', 6),
            dist_threshold_km=config['experiments'].get('gru4_dist_threshold_km', 300.0),
            sigma_d=config['experiments'].get('gru4_sigma_d', 200.0),
            sigma_h=config['experiments'].get('gru4_sigma_h', 1200.0),
            sigma_tau_init_h=config['experiments'].get('gru4_sigma_tau_init_h', 3.0),
            dt_hours=config['experiments'].get('gru4_dt_hours', 3.0),
            diffusivity_km2_per_hour_init=config['experiments'].get('gru4_diffusivity_km2_per_hour_init', 50.0),
            t_eps_hours=config['experiments'].get('gru4_t_eps_hours', 0.25),
        )
    elif exp_model == 'AirLapse6':
        # AirLapse4 with its transport estimate upgraded to a full 2D
        # downwind/crosswind advection-diffusion decomposition, instead of
        # AirLapse4's 1D projection onto the fixed source-receiver line
        # (model/airlapse6.py). The learned attention (including lag_bias)
        # is unchanged from AirLapse4. Reuses AirLapse4's gru_* config keys
        # under a gru6_* prefix, with one isotropic diffusivity replaced by
        # two (downwind/crosswind).
        return AirLapse6(
            hist_len, pred_len, in_dim, city_num, batch_size, device,
            graph.edge_index, graph.edge_attr, wind_mean, wind_std,
            station_coords=coords, station_elevation=altitude,
            feature_mean=train_data.feature_mean, feature_std=train_data.feature_std,
            hidden_dim=config['experiments'].get('gru6_hidden_dim', 64),
            latent_dim=config['experiments'].get('gru6_latent_dim', 16),
            attn_dim=config['experiments'].get('gru6_attn_dim', 32),
            num_layers=config['experiments'].get('gru6_num_layers', 1),
            dropout=config['experiments'].get('gru6_dropout', 0.1),
            logvar_clamp=config['experiments'].get('gru6_logvar_clamp', 10.0),
            spatial_mix_mode=config['experiments'].get('gru6_spatial_mix_mode', 'bottleneck'),
            max_lag=config['experiments'].get('gru6_max_lag', 6),
            dist_threshold_km=config['experiments'].get('gru6_dist_threshold_km', 300.0),
            sigma_d=config['experiments'].get('gru6_sigma_d', 200.0),
            sigma_h=config['experiments'].get('gru6_sigma_h', 1200.0),
            sigma_tau_init_h=config['experiments'].get('gru6_sigma_tau_init_h', 3.0),
            dt_hours=config['experiments'].get('gru6_dt_hours', 3.0),
            diffusivity_downwind_km2_per_hour_init=config['experiments'].get(
                'gru6_diffusivity_downwind_km2_per_hour_init', 50.0),
            diffusivity_crosswind_km2_per_hour_init=config['experiments'].get(
                'gru6_diffusivity_crosswind_km2_per_hour_init', 50.0),
            t_eps_hours=config['experiments'].get('gru6_t_eps_hours', 0.25),
        )
    elif exp_model == 'AirLapse7':
        # AirLapse4 with the explicit transport estimate's cross-neighbor
        # aggregation replaced by a second softmax stage instead of an
        # additive sum - an ablation testing whether a competing-budget
        # aggregation across sources beats the physically-motivated
        # additive one (model/airlapse7.py). Reuses AirLapse4's gru_*
        # config keys under a gru7_* prefix.
        return AirLapse7(
            hist_len, pred_len, in_dim, city_num, batch_size, device,
            graph.edge_index, graph.edge_attr, wind_mean, wind_std,
            station_coords=coords, station_elevation=altitude,
            feature_mean=train_data.feature_mean, feature_std=train_data.feature_std,
            hidden_dim=config['experiments'].get('gru7_hidden_dim', 64),
            latent_dim=config['experiments'].get('gru7_latent_dim', 16),
            attn_dim=config['experiments'].get('gru7_attn_dim', 32),
            num_layers=config['experiments'].get('gru7_num_layers', 1),
            dropout=config['experiments'].get('gru7_dropout', 0.1),
            logvar_clamp=config['experiments'].get('gru7_logvar_clamp', 10.0),
            spatial_mix_mode=config['experiments'].get('gru7_spatial_mix_mode', 'bottleneck'),
            max_lag=config['experiments'].get('gru7_max_lag', 6),
            dist_threshold_km=config['experiments'].get('gru7_dist_threshold_km', 300.0),
            sigma_d=config['experiments'].get('gru7_sigma_d', 200.0),
            sigma_h=config['experiments'].get('gru7_sigma_h', 1200.0),
            sigma_tau_init_h=config['experiments'].get('gru7_sigma_tau_init_h', 3.0),
            dt_hours=config['experiments'].get('gru7_dt_hours', 3.0),
            diffusivity_km2_per_hour_init=config['experiments'].get('gru7_diffusivity_km2_per_hour_init', 50.0),
            t_eps_hours=config['experiments'].get('gru7_t_eps_hours', 0.25),
        )
    elif exp_model == 'AirLapse5':
        # AirLapse4 with the learned attention's lag_bias score term (and
        # its w_lag/sigma_tau/speed_floor_kmh apparatus) removed, since the
        # diffusion-based transport estimate already covers "how much and
        # when" more accurately (model/airlapse5.py). Reuses AirLapse4's
        # gru_* config keys under a gru5_* prefix - no sigma_tau_init_h
        # here, that knob no longer exists.
        return AirLapse5(
            hist_len, pred_len, in_dim, city_num, batch_size, device,
            graph.edge_index, graph.edge_attr, wind_mean, wind_std,
            station_coords=coords, station_elevation=altitude,
            feature_mean=train_data.feature_mean, feature_std=train_data.feature_std,
            hidden_dim=config['experiments'].get('gru5_hidden_dim', 64),
            latent_dim=config['experiments'].get('gru5_latent_dim', 16),
            attn_dim=config['experiments'].get('gru5_attn_dim', 32),
            num_layers=config['experiments'].get('gru5_num_layers', 1),
            dropout=config['experiments'].get('gru5_dropout', 0.1),
            logvar_clamp=config['experiments'].get('gru5_logvar_clamp', 10.0),
            spatial_mix_mode=config['experiments'].get('gru5_spatial_mix_mode', 'bottleneck'),
            max_lag=config['experiments'].get('gru5_max_lag', 6),
            dist_threshold_km=config['experiments'].get('gru5_dist_threshold_km', 300.0),
            sigma_d=config['experiments'].get('gru5_sigma_d', 200.0),
            sigma_h=config['experiments'].get('gru5_sigma_h', 1200.0),
            dt_hours=config['experiments'].get('gru5_dt_hours', 3.0),
            diffusivity_km2_per_hour_init=config['experiments'].get('gru5_diffusivity_km2_per_hour_init', 50.0),
            t_eps_hours=config['experiments'].get('gru5_t_eps_hours', 0.25),
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
    predict = predict_epoch[:, :, :, 0].transpose((0, 2, 1))
    label = label_epoch[:, :, :, 0].transpose((0, 2, 1))
    predict = predict.reshape((-1, predict.shape[-1]))
    label = label.reshape((-1, label.shape[-1]))
    mae = np.mean(np.mean(np.abs(predict - label), axis=1))
    rmse = np.mean(np.sqrt(np.mean(np.square(predict - label), axis=1)))
    mape = get_mape(predict, label)
    return rmse, mae, mape, csi, pod, far


def get_mape(predict, label, eps_threshold=1.0):
    """Mean Absolute Percentage Error, masking out |label| < eps_threshold
    (ug/m3) - see train.py's get_mape for why (near-zero PM2.5 readings
    make the percentage denominator meaningless, not a real error signal)."""
    mask = np.abs(label) >= eps_threshold
    if not np.any(mask):
        return float('nan')
    return float(np.mean(np.abs((predict[mask] - label[mask]) / label[mask])) * 100)


def reset_peak_memory(device):
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        return 0.0
    elif _psutil_process is not None:
        return _psutil_process.memory_info().rss / 1e6
    return float('nan')


def peak_memory_mb(device, running_max=0.0):
    if device.type == 'cuda':
        return torch.cuda.max_memory_allocated(device) / 1e6
    elif _psutil_process is not None:
        return max(running_max, _psutil_process.memory_info().rss / 1e6)
    return float('nan')


def measure_inference_latency(test_loader, model, hist_len, device):
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


def train_epoch(train_loader, model, optimizer, hist_len, device, criterion, kl_weight, alignment_weight):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(tqdm(train_loader, leave=False)):
        optimizer.zero_grad()
        pm25, feature, time_arr = data
        pm25 = pm25.to(device)
        feature = feature.to(device)
        pm25_label = pm25[:, hist_len:]
        pm25_hist = pm25[:, :hist_len]
        pm25_pred = model(pm25_hist, feature)
        loss = criterion(pm25_pred, pm25_label)
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
    return train_loss / (batch_idx + 1)


def val_epoch(val_loader, model, hist_len, device, criterion):
    model.eval()
    val_loss = 0
    for batch_idx, data in enumerate(tqdm(val_loader, leave=False)):
        pm25, feature, time_arr = data
        pm25 = pm25.to(device)
        feature = feature.to(device)
        pm25_label = pm25[:, hist_len:]
        pm25_hist = pm25[:, :hist_len]
        pm25_pred = model(pm25_hist, feature)
        loss = criterion(pm25_pred, pm25_label)
        val_loss += loss.item()
    return val_loss / (batch_idx + 1)


def test_epoch(test_loader, model, hist_len, device, criterion, pm25_mean, pm25_std):
    model.eval()
    predict_list, label_list, time_list = [], [], []
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


def run_single_repeat(model_name, model, optimizer, train_loader, val_loader, test_loader,
                       args, device, criterion, kl_weight, alignment_weight, pm25_mean, pm25_std, repeat_dir):
    val_loss_min = float('inf')
    best_epoch = 0
    best = None
    train_loss = val_loss = None
    param_count = sum(p.numel() for p in model.parameters())
    epoch_times = []
    running_peak_mb = reset_peak_memory(device)

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = train_epoch(train_loader, model, optimizer, args.hist_len, device, criterion, kl_weight, alignment_weight)
        epoch_times.append(time.time() - t0)
        val_loss = val_epoch(val_loader, model, args.hist_len, device, criterion)
        running_peak_mb = peak_memory_mb(device, running_peak_mb)

        if epoch - best_epoch > args.early_stop:
            break

        if val_loss < val_loss_min:
            val_loss_min = val_loss
            best_epoch = epoch
            test_loss, predict_epoch, label_epoch, time_epoch = test_epoch(test_loader, model, args.hist_len, device, criterion, pm25_mean, pm25_std)
            rmse, mae, mape, csi, pod, far = get_metric(predict_epoch, label_epoch)
            best = dict(train_loss=train_loss, val_loss=val_loss, test_loss=test_loss,
                        rmse=rmse, mae=mae, mape=mape, csi=csi, pod=pod, far=far, epoch=epoch)

            if args.save_checkpoints or args.save_npy:
                os.makedirs(repeat_dir, exist_ok=True)
            if args.save_checkpoints:
                torch.save(model.state_dict(), os.path.join(repeat_dir, 'model.pth'))
            if args.save_npy:
                np.save(os.path.join(repeat_dir, 'predict.npy'), predict_epoch)
                np.save(os.path.join(repeat_dir, 'label.npy'), label_epoch)
                np.save(os.path.join(repeat_dir, 'time.npy'), time_epoch)

    if best is None:
        test_loss, predict_epoch, label_epoch, time_epoch = test_epoch(test_loader, model, args.hist_len, device, criterion, pm25_mean, pm25_std)
        rmse, mae, mape, csi, pod, far = get_metric(predict_epoch, label_epoch)
        best = dict(train_loss=train_loss, val_loss=val_loss, test_loss=test_loss,
                    rmse=rmse, mae=mae, mape=mape, csi=csi, pod=pod, far=far, epoch=epoch)

    best['param_count'] = param_count
    best['epoch_train_time_sec'] = np.mean(epoch_times) if epoch_times else float('nan')
    best['inference_latency_ms'] = measure_inference_latency(test_loader, model, args.hist_len, device)
    best['peak_memory_mb'] = peak_memory_mb(device, running_peak_mb)
    return best


def run_for_model(model_name, args, ctx):
    # {model}_{hist_len}_{pred_len}_{dataset_num} - same convention as
    # train.py's metric report filename, for consistency across both scripts.
    tag = f'{model_name}_{args.hist_len}_{args.pred_len}_{args.dataset_num}'
    print(f'\n=== {tag} ===')
    combo_dir = os.path.join(args.output_dir, 'artifacts', tag)
    per_repeat = {k: [] for k in METRIC_KEYS}
    n_ok = 0
    model_repr = ''

    for exp_idx in range(args.exp_repeat):
        print(f'  repeat {exp_idx}...', end=' ', flush=True)
        t0 = time.time()
        train_loader = torch.utils.data.DataLoader(ctx['train_data'], batch_size=args.batch_size, shuffle=True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(ctx['val_data'], batch_size=args.batch_size, shuffle=False, drop_last=True)
        test_loader = torch.utils.data.DataLoader(ctx['test_data'], batch_size=args.batch_size, shuffle=False, drop_last=True)
        try:
            model = get_model(model_name, args.hist_len, args.pred_len, ctx['in_dim'], ctx['city_num'],
                               args.batch_size, ctx['device'], ctx['graph'], ctx['wind_mean'], ctx['wind_std'],
                               ctx['coords'], ctx['altitude'], ctx['train_data']).to(ctx['device'])
            model_repr = str(model)
            optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            repeat_dir = os.path.join(combo_dir, f'rep{exp_idx:02d}')
            result = run_single_repeat(model_name, model, optimizer, train_loader, val_loader, test_loader,
                                        args, ctx['device'], ctx['criterion'], ctx['kl_weight'], ctx['alignment_weight'],
                                        ctx['pm25_mean'], ctx['pm25_std'], repeat_dir)
            for k in METRIC_KEYS:
                per_repeat[k].append(result[k])
            n_ok += 1
            print(f'test RMSE={result["rmse"]:.3f} MAE={result["mae"]:.3f} MAPE={result["mape"]:.1f}% '
                  f'params={result["param_count"]:.0f} ({time.time() - t0:.0f}s)')
        except Exception as e:
            print(f'FAILED: {type(e).__name__}: {e}')
            traceback.print_exc(limit=3)

    summary = dict(model=model_name, dataset_num=args.dataset_num, hist_len=args.hist_len, pred_len=args.pred_len,
                    status='ok' if n_ok else 'all_repeats_failed', n_repeats_ok=n_ok)
    for k in METRIC_KEYS:
        mean, std = get_mean_std(per_repeat[k]) if per_repeat[k] else (float('nan'), float('nan'))
        summary[f'{k}_mean'] = mean
        summary[f'{k}_std'] = std

    train_data, val_data, test_data = ctx['train_data'], ctx['val_data'], ctx['test_data']
    report = (
        '============== Model comparison report ==============\n'
        f'Model: {model_name}\n'
        f'dataset_num: {args.dataset_num}  hist_len: {args.hist_len}  pred_len: {args.pred_len}\n'
        f'Train: {train_data.start_time} --> {train_data.end_time}\n'
        f'Val:   {val_data.start_time} --> {val_data.end_time}\n'
        f'Test:  {test_data.start_time} --> {test_data.end_time}\n'
        f'City number: {ctx["city_num"]}\n'
        f'batch_size: {args.batch_size}  epochs(max): {args.epochs}  early_stop: {args.early_stop}  lr: {args.lr}\n'
        f'exp_repeat requested: {args.exp_repeat}  succeeded: {n_ok}\n'
        '=======================================================\n'
        + '\n'.join(_fmt_metric(k, summary[f'{k}_mean'], summary[f'{k}_std']) for k in ACCURACY_KEYS) + '\n'
        '-------------------------------------------------------\n'
        'Efficiency:\n'
        + '\n'.join(_fmt_metric(k, summary[f'{k}_mean'], summary[f'{k}_std']) for k in EFFICIENCY_KEYS) + '\n'
        f'=======================================================\n{model_repr}\n'
    )
    print(report)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, f'{tag}.txt'), 'w') as f:
        f.write(report)
    return summary


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    torch.set_num_threads(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    print(f'{len(args.models)} model(s) x {args.exp_repeat} repeat(s) = {len(args.models) * args.exp_repeat} total training run(s)')
    print('output dir:', args.output_dir)

    graph = Graph()
    city_num = graph.node_num
    criterion = nn.MSELoss()
    kl_weight = config['train'].get('kl_weight', 0.01)
    alignment_weight = config['train'].get('alignment_weight', 0.1)

    idx, citys, lons, lats = graph.traverse_graph()
    coords = torch.tensor(np.stack([lats, lons], axis=1), dtype=torch.float32)
    altitude = torch.tensor(graph.node_attr[:, 0], dtype=torch.float32)

    try:
        train_data = HazeData(graph, args.hist_len, args.pred_len, args.dataset_num, flag='Train')
        val_data = HazeData(graph, args.hist_len, args.pred_len, args.dataset_num, flag='Val')
        test_data = HazeData(graph, args.hist_len, args.pred_len, args.dataset_num, flag='Test')
    except AssertionError as e:
        sys.exit(
            f'dataset_num={args.dataset_num} is too short to window at hist_len={args.hist_len}+pred_len={args.pred_len}. '
            f'Lower --pred_len/--hist_len or pick a different --dataset_num. ({e})'
        )

    counts = {'train': len(train_data) // args.batch_size, 'val': len(val_data) // args.batch_size, 'test': len(test_data) // args.batch_size}
    print('batches/epoch:', counts, '(need >=1 in each)')
    if min(counts.values()) < 1:
        sys.exit(
            f'Not enough data for batch_size={args.batch_size} at dataset_num={args.dataset_num}, '
            f'hist_len={args.hist_len}, pred_len={args.pred_len} (batches/epoch: {counts}). '
            f'Lower --batch_size/--pred_len/--hist_len or pick a different --dataset_num.'
        )

    in_dim = train_data.feature.shape[-1] + train_data.pm25.shape[-1]
    wind_mean, wind_std = train_data.wind_mean, train_data.wind_std
    pm25_mean, pm25_std = test_data.pm25_mean, test_data.pm25_std
    print('in_dim:', in_dim, '| train/val/test sizes:', len(train_data), len(val_data), len(test_data))
    print(f'Train: {train_data.start_time} --> {train_data.end_time}')
    print(f'Val:   {val_data.start_time} --> {val_data.end_time}')
    print(f'Test:  {test_data.start_time} --> {test_data.end_time}')

    ctx = dict(train_data=train_data, val_data=val_data, test_data=test_data, in_dim=in_dim,
               city_num=city_num, device=device, graph=graph, wind_mean=wind_mean, wind_std=wind_std,
               coords=coords, altitude=altitude, criterion=criterion, kl_weight=kl_weight,
               alignment_weight=alignment_weight, pm25_mean=pm25_mean, pm25_std=pm25_std)

    summary_fp = os.path.join(args.output_dir, 'summary.csv')
    fieldnames = ['model', 'dataset_num', 'hist_len', 'pred_len', 'status', 'n_repeats_ok']
    for k in METRIC_KEYS:
        fieldnames += [f'{k}_mean', f'{k}_std']

    all_summaries = []
    with open(summary_fp, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for model_name in args.models:
            try:
                summary = run_for_model(model_name, args, ctx)
            except Exception as e:
                print(f'{model_name} FAILED entirely: {type(e).__name__}: {e}')
                traceback.print_exc(limit=3)
                summary = dict(model=model_name, dataset_num=args.dataset_num, hist_len=args.hist_len,
                                pred_len=args.pred_len, status='error', n_repeats_ok=0)
            all_summaries.append(summary)
            writer.writerow({k: summary.get(k, '') for k in fieldnames})
            f.flush()

    failed = [s for s in all_summaries if s['status'] != 'ok']
    if failed:
        with open(os.path.join(args.output_dir, 'failed.txt'), 'w') as f:
            for s in failed:
                f.write(f"{s['model']}: {s['status']}\n")

    print(f'\nDone. {len(all_summaries) - len(failed)}/{len(all_summaries)} models completed.')
    print('summary:', summary_fp)
    if failed:
        print('failed models logged in', os.path.join(args.output_dir, 'failed.txt'), ':', [(s['model'], s['status']) for s in failed])


if __name__ == '__main__':
    main()
