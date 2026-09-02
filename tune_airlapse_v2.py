"""
Optuna hyperparameter search for AirLapseV2 (model/airlapse_v2.py),
designed to run standalone on Google Colab/Kaggle (or any machine with
this repo checked out) - a direct sibling of tune_airlapse.py, adapted
for AirLapseV2's own airlapsev2_* config prefix and its new
context-adaptive-diffusivity hyperparameters. See that file for the
fuller explanation of the overall approach; only what differs for V2 is
called out below.

Usage (Colab/Kaggle):
    !pip install -q optuna
    !python tune_airlapse_v2.py --n_trials 30 --search_epochs 10 --final_epochs 50

hist_len/pred_len/dataset_num/batch_size all come from config.yaml's
existing `train:`/`experiments:` sections, exactly as a normal train.py
run would use them - set those there first. This script forces
`model: AirLapseV2` for the search regardless of whatever config.yaml's
`model:` field currently says.

WHAT'S DIFFERENT FROM tune_airlapse.py (V1's search)
  - Every override key uses the airlapsev2_* prefix (matching train.py's
    get_model() AirLapseV2 branch) instead of gru_* - a deliberate
    separation (see that branch's comment) so this search can never
    silently perturb V1's own already-tuned gru_* defaults, and vice
    versa.
  - gru_diffusivity_km2_per_hour_init (one scalar) is replaced by THREE
    new knobs specific to AdaptivePhysicsTransport2D's diffusivity MLP:
    airlapsev2_diff_hidden_dim (the MLP's hidden width) and separate
    airlapsev2_diffusivity_along_init / _cross_init (the along-wind and
    cross-wind starting points the MLP's output is biased toward before
    training adapts it - see model/airlapse_v2.py's class docstring for
    why these start unequal, along > cross, by physical default).
  - Otherwise the search space mirrors V1's ranges as closely as the two
    models' actual constructor signatures allow (hidden_dim, latent_dim,
    attn_dim, max_lag, dropout, spatial_mix_mode, num_layers,
    dist_threshold_km, sigma_d, sigma_h, sigma_tau_init_h, lr,
    weight_decay) - centered on the SAME defaults train.py's AirLapseV2
    branch falls back to when a key isn't in config['experiments'].
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
# inside AirLapseV2's get_model() branch in train.py - see that file for
# the authoritative defaults these ranges are centered around.
SEARCH_RANGES = {
    'airlapsev2_hidden_dim': [32, 64, 96, 128],
    'airlapsev2_latent_dim': [8, 16, 32],
    'airlapsev2_attn_dim': [16, 32, 48],
    'airlapsev2_max_lag': (3, 10),                       # int
    'airlapsev2_dropout': (0.0, 0.3),                    # float
    'airlapsev2_dist_threshold_km': (150.0, 400.0),      # float
    'airlapsev2_sigma_d': (100.0, 300.0),                # float
    'airlapsev2_sigma_h': (500.0, 2000.0),                # float
    'airlapsev2_sigma_tau_init_h': (1.0, 6.0),           # float
    'airlapsev2_diff_hidden_dim': [8, 16, 32],
    'airlapsev2_diffusivity_along_init': (10.0, 150.0),  # float
    'airlapsev2_diffusivity_cross_init': (5.0, 100.0),   # float
    'lr': (1e-4, 3e-3),                            # float, log-scale
    'weight_decay': (1e-5, 1e-3),                  # float, log-scale
}


def _suggest_params(trial):
    spatial_mix_mode = trial.suggest_categorical('airlapsev2_spatial_mix_mode', ['bottleneck', 'per_step'])
    experiment_overrides = {
        'airlapsev2_hidden_dim': trial.suggest_categorical(
            'airlapsev2_hidden_dim', SEARCH_RANGES['airlapsev2_hidden_dim']),
        'airlapsev2_latent_dim': trial.suggest_categorical(
            'airlapsev2_latent_dim', SEARCH_RANGES['airlapsev2_latent_dim']),
        'airlapsev2_attn_dim': trial.suggest_categorical(
            'airlapsev2_attn_dim', SEARCH_RANGES['airlapsev2_attn_dim']),
        'airlapsev2_dropout': trial.suggest_float('airlapsev2_dropout', *SEARCH_RANGES['airlapsev2_dropout']),
        'airlapsev2_spatial_mix_mode': spatial_mix_mode,
        # per_step unrolls a single-layer GRUCell manually - num_layers > 1
        # isn't valid there (AirLapseV2's own constructor raises on it), so
        # only search it when bottleneck mode was chosen this trial.
        'airlapsev2_num_layers': 1 if spatial_mix_mode == 'per_step' else trial.suggest_int(
            'airlapsev2_num_layers', 1, 2),
        'airlapsev2_max_lag': trial.suggest_int('airlapsev2_max_lag', *SEARCH_RANGES['airlapsev2_max_lag']),
        'airlapsev2_dist_threshold_km': trial.suggest_float(
            'airlapsev2_dist_threshold_km', *SEARCH_RANGES['airlapsev2_dist_threshold_km']),
        'airlapsev2_sigma_d': trial.suggest_float('airlapsev2_sigma_d', *SEARCH_RANGES['airlapsev2_sigma_d']),
        'airlapsev2_sigma_h': trial.suggest_float('airlapsev2_sigma_h', *SEARCH_RANGES['airlapsev2_sigma_h']),
        'airlapsev2_sigma_tau_init_h': trial.suggest_float(
            'airlapsev2_sigma_tau_init_h', *SEARCH_RANGES['airlapsev2_sigma_tau_init_h']),
        'airlapsev2_diff_hidden_dim': trial.suggest_categorical(
            'airlapsev2_diff_hidden_dim', SEARCH_RANGES['airlapsev2_diff_hidden_dim']),
        'airlapsev2_diffusivity_along_init': trial.suggest_float(
            'airlapsev2_diffusivity_along_init', *SEARCH_RANGES['airlapsev2_diffusivity_along_init']),
        'airlapsev2_diffusivity_cross_init': trial.suggest_float(
            'airlapsev2_diffusivity_cross_init', *SEARCH_RANGES['airlapsev2_diffusivity_cross_init']),
    }
    lr = trial.suggest_float('lr', *SEARCH_RANGES['lr'], log=True)
    weight_decay = trial.suggest_float('weight_decay', *SEARCH_RANGES['weight_decay'], log=True)
    return experiment_overrides, lr, weight_decay


def run_trial(trial, search_epochs, search_early_stop):
    T.exp_model = 'AirLapseV2'
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
    ap.add_argument('--study_name', type=str, default='airlapsev2_tuning')
    ap.add_argument('--storage', type=str, default=os.path.join(proj_dir, 'airlapsev2_tuning.db'),
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

    csv_fp = os.path.join(T.results_dir, f'airlapsev2_hparam_search_{stamp}.csv')
    trials_df.to_csv(csv_fp, index=False)
    print(f'\nFull trial results saved to: {csv_fp}')

    report_fp = os.path.join(T.results_dir, f'airlapsev2_hparam_search_{stamp}.txt')
    with open(report_fp, 'w') as f:
        f.write('AirLapseV2 hyperparameter search report\n')
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
        T.exp_model = 'AirLapseV2'
        T.epochs = args.final_epochs
        T.lr = lr
        T.weight_decay = weight_decay
        T.exp_repeat = 1
        T.main()


if __name__ == '__main__':
    main()
