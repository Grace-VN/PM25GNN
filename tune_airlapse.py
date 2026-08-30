"""
Optuna hyperparameter search for AirLapse (this repo's proposed model),
designed to run standalone on Google Colab (or any machine with this repo
checked out).

Usage (Colab):
    !pip install -q optuna
    !python tune_airlapse.py --n_trials 30 --search_epochs 10 --final_epochs 50

hist_len/pred_len/dataset_num/batch_size all come from config.yaml's
existing `train:`/`experiments:` sections, exactly as a normal train.py
run would use them - set those there first. This script forces
`model: AirLapse` for the search regardless of whatever config.yaml's
`model:` field currently says (so you don't need to edit that yourself).

WHAT IT DOES
  1. Imports train.py as a module - this triggers its one-time dataset/
     graph loading (Graph, HazeData x3), exactly like running train.py
     directly, just without executing its own main(). Every function
     train.py defines (get_model, train, val, test, get_metric, main, ...)
     reads its hyperparameters from train.py's own module-level globals
     (exp_model, config, lr, weight_decay, epochs, ...) FRESH at call
     time, not at import time - so overwriting those attributes on the
     imported module (`train.exp_model = 'AirLapse'`, etc.) is enough to
     drive it through this script without duplicating any of its dataset/
     training/reporting logic.
  2. Runs an Optuna TPE search over AirLapse's architecture and
     optimizer hyperparameters (see SEARCH_SPACE_KEYS below), each trial
     training for --search_epochs epochs (much less than a full run) and
     reporting validation loss, with median pruning so clearly-bad trials
     are abandoned early instead of run to completion. The search itself
     is persisted to a local SQLite file (airlapse_tuning.db) via
     `load_if_exists=True`, so re-running this script after a Colab
     disconnect resumes the same study instead of starting over.
  3. Prints a ranked summary of all trials and saves the full trials
     table (study.trials_dataframe()) plus a short text report to
     results_dir (the same directory train.py's own reports go to).
  4. Unless --final_epochs 0, retrains the best-found configuration for
     --final_epochs epochs via train.py's own main() - so the final
     result gets the SAME properly-named, fully-featured report file
     (RMSE/MAE/MAPE/CSI/POD/FAR, param_count, timing, saved under
     train.py's own "N. AirLapse YYYY_hist_pred_dataset.txt" naming) every
     other benchmark in this repo gets, rather than a search-only number.

train.py's own console output is already trimmed to one line per run
("Model: ... | Dataset: ... | hist_len: ... | pred_len: ..."), so
n_trials * search_epochs worth of trials stays readable on Colab without
any extra silencing here - Optuna's own "Trial N finished..." line is
the only per-trial output.
"""
import argparse
import os
import sys

proj_dir = os.path.dirname(os.path.abspath(__file__))
if proj_dir not in sys.path:
    sys.path.append(proj_dir)

import arrow
import numpy as np
import torch

try:
    import optuna
except ImportError:
    raise SystemExit(
        "optuna is required for this script - install it first:\n"
        "    pip install optuna\n"
        "(kept out of requirements.txt since it's only needed for this "
        "hyperparameter-search script, not for training/benchmarking any "
        "single model configuration.)"
    )

import train as T  # noqa: E402 - reuses train.py's dataset/graph loading, get_model(), train()/val()/main()

# Every key here is looked up via config['experiments'].get(key, default)
# inside AirLapse's get_model() branch in train.py - see that file for the
# authoritative defaults these ranges are centered around.
SEARCH_RANGES = {
    'gru_hidden_dim': [32, 64, 96, 128],
    'gru_latent_dim': [8, 16, 32],
    'gru_attn_dim': [16, 32, 48],
    'gru_max_lag': (3, 10),                       # int
    'gru_dropout': (0.0, 0.3),                    # float
    'gru_dist_threshold_km': (150.0, 400.0),      # float
    'gru_sigma_d': (100.0, 300.0),                # float
    'gru_sigma_h': (500.0, 2000.0),                # float
    'gru_sigma_tau_init_h': (1.0, 6.0),           # float
    'gru_diffusivity_km2_per_hour_init': (10.0, 150.0),  # float
    'lr': (1e-4, 3e-3),                            # float, log-scale
    'weight_decay': (1e-5, 1e-3),                  # float, log-scale
}


def _suggest_params(trial):
    spatial_mix_mode = trial.suggest_categorical('gru_spatial_mix_mode', ['bottleneck', 'per_step'])
    experiment_overrides = {
        'gru_hidden_dim': trial.suggest_categorical('gru_hidden_dim', SEARCH_RANGES['gru_hidden_dim']),
        'gru_latent_dim': trial.suggest_categorical('gru_latent_dim', SEARCH_RANGES['gru_latent_dim']),
        'gru_attn_dim': trial.suggest_categorical('gru_attn_dim', SEARCH_RANGES['gru_attn_dim']),
        'gru_dropout': trial.suggest_float('gru_dropout', *SEARCH_RANGES['gru_dropout']),
        'gru_spatial_mix_mode': spatial_mix_mode,
        # per_step unrolls a single-layer GRUCell manually - num_layers > 1
        # isn't valid there (AirLapse's own constructor raises on it), so
        # only search it when bottleneck mode was chosen this trial.
        'gru_num_layers': 1 if spatial_mix_mode == 'per_step' else trial.suggest_int('gru_num_layers', 1, 2),
        'gru_max_lag': trial.suggest_int('gru_max_lag', *SEARCH_RANGES['gru_max_lag']),
        'gru_dist_threshold_km': trial.suggest_float('gru_dist_threshold_km', *SEARCH_RANGES['gru_dist_threshold_km']),
        'gru_sigma_d': trial.suggest_float('gru_sigma_d', *SEARCH_RANGES['gru_sigma_d']),
        'gru_sigma_h': trial.suggest_float('gru_sigma_h', *SEARCH_RANGES['gru_sigma_h']),
        'gru_sigma_tau_init_h': trial.suggest_float('gru_sigma_tau_init_h', *SEARCH_RANGES['gru_sigma_tau_init_h']),
        'gru_diffusivity_km2_per_hour_init': trial.suggest_float(
            'gru_diffusivity_km2_per_hour_init', *SEARCH_RANGES['gru_diffusivity_km2_per_hour_init']),
    }
    lr = trial.suggest_float('lr', *SEARCH_RANGES['lr'], log=True)
    weight_decay = trial.suggest_float('weight_decay', *SEARCH_RANGES['weight_decay'], log=True)
    return experiment_overrides, lr, weight_decay


def run_trial(trial, search_epochs, search_early_stop):
    T.exp_model = 'AirLapse'
    experiment_overrides, lr, weight_decay = _suggest_params(trial)
    T.config['experiments'].update(experiment_overrides)

    model = T.get_model().to(T.device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = torch.utils.data.DataLoader(T.train_data, batch_size=T.batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(T.val_data, batch_size=T.batch_size, shuffle=False, drop_last=True)

    best_val = float('inf')
    best_epoch = 0
    for epoch in range(search_epochs):
        T.train(train_loader, model, optimizer)
        val_loss = T.val(val_loader, model)
        if not np.isfinite(val_loss):
            # a diverged/NaN trial is unambiguously bad - report it as such
            # rather than letting Optuna keep exploring nearby (it would
            # otherwise see a crash instead of a comparable number).
            raise optuna.TrialPruned()
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
        if epoch - best_epoch > search_early_stop:
            break

    return best_val


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n_trials', type=int, default=30)
    ap.add_argument('--search_epochs', type=int, default=50,
                     help='epoch budget per trial during the search (kept short - not a full training run)')
    ap.add_argument('--search_early_stop', type=int, default=3,
                     help='per-trial early-stop patience during the search')
    ap.add_argument('--final_epochs', type=int, default=50,
                     help='epochs for the final retrain of the best config via train.py\'s own main() - 0 to skip it')
    ap.add_argument('--study_name', type=str, default='airlapse_tuning')
    ap.add_argument('--storage', type=str, default=os.path.join(proj_dir, 'airlapse_tuning.db'),
                     help='SQLite file the study is persisted to - re-running with the same path resumes it')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=3)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=f'sqlite:///{args.storage}',
        load_if_exists=True,
        direction='minimize',
        sampler=sampler,
        pruner=pruner,
    )
    already_done = len(study.trials)
    if already_done:
        print(f'Resuming study "{args.study_name}" from {args.storage} - {already_done} trial(s) already recorded.')

    study.optimize(
        lambda trial: run_trial(trial, args.search_epochs, args.search_early_stop),
        n_trials=args.n_trials,
    )

    print('\n' + '=' * 70)
    print(f'Hyperparameter search complete - {len(study.trials)} total trial(s) recorded')
    print('=' * 70)
    completed = [t for t in study.trials if t.value is not None]
    if not completed:
        raise SystemExit('No trial completed successfully - nothing to report or retrain.')
    print('Best val_loss: %.4f' % study.best_value)
    print('Best params:')
    for k, v in study.best_trial.params.items():
        print(f'  {k}: {v}')

    os.makedirs(T.results_dir, exist_ok=True)
    stamp = arrow.now().format('YYYYMMDDHHmmss')
    trials_df = study.trials_dataframe().sort_values('value')

    csv_fp = os.path.join(T.results_dir, f'airlapse_hparam_search_{stamp}.csv')
    trials_df.to_csv(csv_fp, index=False)
    print(f'\nFull trial results saved to: {csv_fp}')

    report_fp = os.path.join(T.results_dir, f'airlapse_hparam_search_{stamp}.txt')
    with open(report_fp, 'w') as f:
        f.write('AirLapse hyperparameter search report\n')
        f.write('=' * 70 + '\n')
        f.write(f'hist_len: {T.hist_len}  pred_len: {T.pred_len}  dataset_num: {T.dataset_num}\n')
        f.write(f'trials recorded: {len(study.trials)}  search_epochs: {args.search_epochs}\n')
        f.write(f'best val_loss: {study.best_value:.4f}\n\nbest params:\n')
        for k, v in study.best_trial.params.items():
            f.write(f'  {k}: {v}\n')
        f.write('\ntop 10 trials (by val_loss):\n')
        f.write(trials_df.head(10).to_string(index=False))
        f.write('\n')
    print(f'Text summary saved to: {report_fp}')

    if args.final_epochs > 0:
        print('\n' + '=' * 70)
        print(f'Retraining the best config for {args.final_epochs} epoch(s) '
              f'(full report, same format as every other benchmark)...')
        print('=' * 70)
        best_params = dict(study.best_trial.params)
        lr = best_params.pop('lr')
        weight_decay = best_params.pop('weight_decay')
        T.config['experiments'].update(best_params)
        T.exp_model = 'AirLapse'
        T.epochs = args.final_epochs
        T.lr = lr
        T.weight_decay = weight_decay
        T.exp_repeat = 1
        T.main()


if __name__ == '__main__':
    main()
