"""Run inference on all experiments and save eval logs."""
import json
import os
import subprocess
import sys

experiments = [
    ("exp04_interleave_noclean",    "results_exp04.json"),
    ("exp05_interleave_clean",      "results_exp05.json"),
    ("exp06_rslora_interleaved",    "results_exp06.json"),
    ("exp05b_interleave_clean_r32", "results_exp05b.json"),
]

dataset = "cpt_val_dataset_50.jsonl"
num_samples = 50
wandb_project = "cpt-smollm135m"

os.makedirs("logs", exist_ok=True)

for exp, results_file in experiments:
    model_path = f"models/{exp}/final"
    if not os.path.exists(model_path):
        print(f"Skipping {exp} — model not found")
        continue

    log_path = f"logs/{exp}_eval.md"
    if os.path.exists(log_path):
        print(f"Skipping {exp} — eval log already exists")
        continue

    print(f"=== Running inference for {exp} ===")
    cmd = [
        sys.executable, "inference.py",
        "--models", model_path,
        "--dataset", dataset,
        "--num_samples", str(num_samples),
        "--output_results", results_file,
        "--wandb_project", wandb_project,
        "--wandb_run_name", exp,
    ]
    subprocess.run(cmd, check=True)

    if os.path.exists(results_file):
        results = json.load(open(results_file))
        with open(log_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved {log_path}")
    else:
        print(f"Warning: {results_file} not found after inference")
