"""Fetch training summaries for all experiments and save to logs/."""
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

os.makedirs("logs", exist_ok=True)

for exp in experiments:
    config_path = f"models/{exp}/train_config.json"
    if not os.path.exists(config_path):
        print(f"=== {exp} — model not found ===")
        continue

    config = json.load(open(config_path))
    run_id = config.get("wandb_run_id", "")
    matches = glob.glob(f"wandb/*{run_id}*/files/wandb-summary.json")

    if not matches:
        print(f"=== {exp} — wandb run {run_id} not found ===")
        continue

    summary = json.load(open(matches[0]))
    filtered = {k: v for k, v in summary.items() if k in keys}

    lines = [f"## Config"]
    lines.append(f"lora_r: {config.get('lora_r')}, lr: {config.get('learning_rate')}, "
                 f"epochs: {config.get('epochs')}, mix: {config.get('mix')}, "
                 f"mix_ratio: {config.get('mix_ratio')}, rslora: {config.get('rslora')}")
    lines.append("")
    lines.append("## Training Metrics")
    for k, v in filtered.items():
        lines.append(f"{k}: {v}")

    log_path = f"logs/{exp}_train.md"
    with open(log_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved {log_path}")

