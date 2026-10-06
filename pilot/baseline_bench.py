"""Baseline benchmark on one BC smoke dataset, reusing pilot/run_pilot.py.

Each model runs in isolation so one failure doesn't kill the rest, and
results append to <results>/baseline_bench_ds<N>.csv (re-running a model
replaces its rows), so an interrupted run resumes with just the missing
models. Scored like run_pilot.py: forecast hours only, RMSE overall and on
smoke hours, CSI/POD/FAR at 35.5 and 55.5 ug/m3, plus persistence.

Usage: python pilot/baseline_bench.py [--dataset 10] [--epochs 10] [--repeats 1]
                                      [--models AirLapseV2 GRU ...]
Results dir: pilot/results, or env PILOT_RESULTS_DIR (e.g. Google Drive on Colab).
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import run_pilot as rp  # noqa: E402

DEFAULT_MODELS = ["AirLapseV2", "LSTM", "GRU", "PM25_GNN", "Informer", "PatchTST"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=int, default=7)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--early_stop", type=int, default=5)
    args = ap.parse_args()

    os.makedirs(rp.RESULTS, exist_ok=True)
    base = yaml.safe_load(open(os.path.join(REPO, "config.yaml"), encoding="utf-8"))
    out = os.path.join(rp.RESULTS, f"baseline_bench_ds{args.dataset}.csv")
    rows = pd.read_csv(out).to_dict("records") if os.path.exists(out) else []
    rows = [r for r in rows if r["model"] not in args.models]

    for m in args.models:
        t0 = time.time()
        print(f"== {m} on dataset {args.dataset}", flush=True)
        try:
            cfg, run_dir = rp.run(m, args.dataset, args, base)
        except Exception as e:
            print(f"   FAILED: {e}", flush=True)
            rows.append(dict(model=m, status="failed"))
            pd.DataFrame(rows).to_csv(out, index=False)
            continue
        H = cfg["train"]["hist_len"]
        for rep in sorted(glob.glob(os.path.join(run_dir, "[0-9][0-9]"))):
            pf = np.load(os.path.join(rep, "predict.npy"))
            lf = np.load(os.path.join(rep, "label.npy"))
            p, l = rp.forecast_part(pf, lf, H)
            rows.append(dict(model=m, status="ok", repeat=int(os.path.basename(rep)),
                             minutes=round((time.time() - t0) / 60, 1), **rp.smoke_scores(p, l)))
            if not any(r["model"] == "Persistence" for r in rows):
                rows.append(dict(model="Persistence", status="ok", repeat=0,
                                 **rp.smoke_scores(rp.persistence(lf, H), l)))
        print(f"   done in {(time.time() - t0) / 60:.1f} min", flush=True)
        pd.DataFrame(rows).to_csv(out, index=False)

    df = pd.DataFrame(rows)
    ok = df[df.status == "ok"]
    cols = ["RMSE", "RMSE_smoke", "CSI@35.5", "POD@35.5", "FAR@35.5", "CSI@55.5"]
    pd.set_option("display.width", 200)
    print(ok.groupby("model", sort=False)[cols].agg(["mean", "std"]).round(3).to_string())
    failed = df[df.status == "failed"].model.tolist()
    if failed:
        print("failed:", failed)


if __name__ == "__main__":
    main()
