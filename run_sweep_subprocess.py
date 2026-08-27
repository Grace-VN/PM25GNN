"""
Loops train.py (as a subprocess, one model per process) over a list of
models at one fixed (dataset_num, pred_len) config - based on the original
Colab driver script, hardened against two problems that showed up running
Autoformer on a CPU-only Colab instance:

1. OOM (SIGKILL, no traceback): Informer/Autoformer/PatchTST/STAEformer all
   fold the 184-city dimension into the batch dimension internally, so
   config.yaml's batch_size becomes an effective batch of batch_size*184.
   At the default batch_size=32, that's 5,888 - measured locally, one
   Autoformer train step at pred_len=24 with batch_size=32 spiked resident
   memory by ~5GB; at batch_size=4 it stayed under ~3.5GB total. This
   script overrides batch_size down for just those four models (see
   FOLDED_BATCH_MODELS / FOLDED_BATCH_SIZE below) - everything else keeps
   config.yaml's original batch_size.
2. `subprocess.run(..., check=True)` raises on the first non-zero exit and
   aborts the whole loop - so one OOM'd model previously took the rest of
   the sweep down with it. Each model now runs in a try/except so a crash
   is logged and the loop moves on to the next model.

Keeping each model in its own subprocess (rather than one long-lived
Python process looping in-memory) is deliberate: it's what makes recovery
from a SIGKILL possible at all - the OS can only kill the process actually
holding the oversized allocation, not python objects inside a shared
process from the outside.
"""

import subprocess

import yaml

models = [
    "Autoformer",
]

# Experiment settings
dataset_num = 3
pred_len = 24

# Models whose forward pass folds city_num (184 for KnowAir) into the batch
# dimension - config.yaml's batch_size means batch_size*184 real sequences
# per step for these. Lower batch_size keeps memory manageable on a
# CPU-only / free-tier instance. If you still see SIGKILL/OOM at this
# value, try 2 next.
FOLDED_BATCH_MODELS = {"Informer", "Autoformer", "PatchTST", "STAEformer"}
FOLDED_BATCH_SIZE = 4

results = {}

for model in models:

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Change only these parameters
    config["experiments"]["model"] = model
    config["experiments"]["dataset_num"] = dataset_num
    config["train"]["pred_len"] = pred_len

    original_batch_size = config["train"]["batch_size"]
    if model in FOLDED_BATCH_MODELS:
        config["train"]["batch_size"] = FOLDED_BATCH_SIZE

    with open("config.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    print("=" * 70)
    print(f"Model: {model}")
    print(f"Dataset: {dataset_num}")
    print(f"History length: {config['train']['hist_len']}")
    print(f"Prediction length: {pred_len}")
    print(f"Batch size: {config['train']['batch_size']}"
          + (f" (overridden from {original_batch_size} - folds city dim into batch)"
             if model in FOLDED_BATCH_MODELS else ""))
    print("=" * 70)

    try:
        subprocess.run(["python", "train.py"], check=True)
        results[model] = "ok"
    except subprocess.CalledProcessError as e:
        # SIGKILL (OOM) shows up here as a negative returncode (-9); anything
        # else train.py itself raised shows up as a positive non-zero code.
        reason = "SIGKILL (likely OOM)" if e.returncode == -9 else f"exit code {e.returncode}"
        print(f"\n!!! {model} FAILED: {reason} - continuing with remaining models !!!\n")
        results[model] = reason

print("\n" + "=" * 70)
print("Sweep summary:")
for model, status in results.items():
    print(f"  {model}: {status}")
print("=" * 70)
