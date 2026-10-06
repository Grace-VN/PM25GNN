"""Run the BC smoke pilot: each model x {5 nofire, 6 fire_iso, 7 fire_aniso}.

Drives train.py unchanged, via a generated config (CONFIG_FP - config.yaml
itself is never rewritten, so its comments survive) and RESULTS_DIR =
pilot/results. After each run, reloads the saved best-epoch predict/label
arrays and scores them the way a smoke paper would: RMSE overall and on
smoke hours, plus CSI/POD/FAR at the US AQI "unhealthy for sensitive
groups" (35.5) and "unhealthy" (55.5) PM2.5 thresholds - train.py's own
CSI/POD/FAR use KnowAir's 75 ug/m3 haze threshold.

Usage: python pilot/run_pilot.py [--epochs 30] [--repeats 3] [--models AirLapseV2 GRU]
Writes pilot/results/pilot_summary.csv and prints a table.
"""
import argparse
import glob
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
# PILOT_RESULTS_DIR lets Colab write straight to Google Drive (a runtime's
# local disk is wiped when it disconnects).
RESULTS = os.environ.get("PILOT_RESULTS_DIR", os.path.join(HERE, "results"))
DATASETS = {5: "no fire", 6: "fire, distance only", 7: "fire, wind-transported",
            8: "2y, PM2.5 + weather"}
THRESHOLDS = (35.5, 55.5)


def smoke_scores(pred, label):
    """pred/label: [S, pred_len, N, 1] in ug/m3."""
    p, y = pred[..., 0], label[..., 0]
    out = dict(RMSE=float(np.sqrt(np.mean((p - y) ** 2))),
               MAE=float(np.mean(np.abs(p - y))))
    smoky = y > THRESHOLDS[0]
    out["RMSE_smoke"] = float(np.sqrt(np.mean((p[smoky] - y[smoky]) ** 2))) if smoky.any() else np.nan
    for thr in THRESHOLDS:
        ph, yh = p >= thr, y >= thr
        hit, miss, fa = (ph & yh).sum(), (~ph & yh).sum(), (ph & ~yh).sum()
        out[f"CSI@{thr}"] = hit / max(hit + miss + fa, 1)
        out[f"POD@{thr}"] = hit / max(hit + miss, 1)
        out[f"FAR@{thr}"] = fa / max(hit + fa, 1)
    return out


def forecast_part(pred, label, H):
    """train.py's saved predict/label both span hist_len + pred_len, with the
    history hours copied verbatim into predict (see train.py's test()) -
    score only the forecast hours."""
    return pred[:, H:], label[:, H:]


def persistence(label, H):
    """Last observed hour repeated over the horizon, on the same test windows."""
    return np.repeat(label[:, H - 1:H], label.shape[1] - H, axis=1)


def run(model, ds_num, args, base_cfg):
    cfg = yaml.safe_load(yaml.safe_dump(base_cfg))
    cfg["experiments"].update(model=model, dataset_num=ds_num, save_npy=True,
                              airlapsev2_ode_transport=False)  # ODE branch: too slow for a CPU pilot
    cfg["train"].update(epochs=args.epochs, exp_repeat=args.repeats, early_stop=args.early_stop)
    cfg_fp = os.path.join(HERE, "cache", f"config_{model}_{ds_num}.yaml")
    os.makedirs(os.path.dirname(cfg_fp), exist_ok=True)
    yaml.safe_dump(cfg, open(cfg_fp, "w"), sort_keys=False)
    env = dict(os.environ, CONFIG_FP=cfg_fp, RESULTS_DIR=RESULTS)
    r = subprocess.run([sys.executable, "train.py"], cwd=REPO, env=env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-3000:])
        raise RuntimeError(f"{model} on dataset {ds_num} failed")
    metric_fp = r.stdout.strip().splitlines()[-1]
    return cfg, os.path.dirname(metric_fp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["AirLapseV2", "GRU"])
    ap.add_argument("--datasets", nargs="+", type=int, default=list(DATASETS))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--early_stop", type=int, default=8)
    args = ap.parse_args()
    base_cfg = yaml.safe_load(open(os.path.join(REPO, "config.yaml"), encoding="utf-8"))

    rows = []
    for model in args.models:
        for ds_num in args.datasets:
            print(f"running {model} on dataset {ds_num} ({DATASETS[ds_num]}) ...", flush=True)
            cfg, run_dir = run(model, ds_num, args, base_cfg)
            for rep_dir in sorted(glob.glob(os.path.join(run_dir, "[0-9][0-9]"))):
                H = cfg["train"]["hist_len"]
                pred_full = np.load(os.path.join(rep_dir, "predict.npy"))
                label_full = np.load(os.path.join(rep_dir, "label.npy"))
                pred, label = forecast_part(pred_full, label_full, H)
                rows.append(dict(model=model, dataset=ds_num, fire=DATASETS[ds_num],
                                 repeat=int(os.path.basename(rep_dir)), **smoke_scores(pred, label)))
                if model == args.models[0] and ds_num == args.datasets[0] and rep_dir.endswith("00"):
                    rows.append(dict(model="Persistence", dataset=ds_num, fire="-", repeat=0,
                                     **smoke_scores(persistence(label_full, H), label)))
            pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "pilot_summary.csv"), index=False)

    df = pd.DataFrame(rows)
    cols = ["RMSE", "RMSE_smoke", "CSI@35.5", "POD@35.5", "FAR@35.5", "CSI@55.5"]
    table = df.groupby(["model", "fire"], sort=False)[cols].agg(["mean", "std"])
    pd.set_option("display.width", 200)
    print(table.round(3).to_string())


if __name__ == "__main__":
    main()
