"""Fetch training summaries for all experiments from wandb run files."""
import json
import os
import glob

experiments = [
    "exp04_interleave_noclean",
    "exp05_interleave_clean",
    "exp06_rslora_interleaved",
]

keys = {"train_loss", "train_runtime", "train/loss", "train/epoch",
        "train/global_step", "train/grad_norm", "train_samples_per_second",
        "eval/loss", "eval/runtime"}

for exp in experiments:
    config_path = f"models/{exp}/train_config.json"
    if not os.path.exists(config_path):
        print(f"=== {exp} — model not found ===\n")
        continue

    run_id = json.load(open(config_path)).get("wandb_run_id", "")
    matches = glob.glob(f"wandb/*{run_id}*/files/wandb-summary.json")

    if not matches:
        print(f"=== {exp} — wandb run {run_id} not found ===\n")
        continue

    summary = json.load(open(matches[0]))
    print(f"=== {exp} ===")
    for k, v in summary.items():
        if k in keys:
            print(f"  {k}: {v}")
    print()
