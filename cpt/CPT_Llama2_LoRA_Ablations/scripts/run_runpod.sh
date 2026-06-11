#!/usr/bin/env bash
# One-paste runner for a fresh RunPod CUDA pod (PyTorch template, RTX 4090).
#
#   export GH_TOKEN=ghp_...        # GitHub token (repo scope) — commits results to the branch
#   export HF_TOKEN=hf_...         # alternative: upload results to an HF dataset repo
#   export HF_RESULTS_REPO=...     # optional; default jaydeepraijada/cpt-lora-ablations-results
#   export WANDB_API_KEY=...       # optional; omit to run CSV-only
#   export SEEDS="0 1 2"           # optional; default below
#   curl -sL https://raw.githubusercontent.com/jaydeepraijada/post-training-experiments/add-cpt-llama2-lora-ablations/cpt/CPT_Llama2_LoRA_Ablations/scripts/run_runpod.sh | bash
#
# Tiers: smoke test (10 steps, one config) runs first and aborts the sweep on
# failure, so an integration bug costs minutes, not the budget. The pod stops
# itself when done (AUTO_STOP=0 to disable) — results survive via git + W&B.
set -euo pipefail

REPO="jaydeepraijada/post-training-experiments"
BRANCH="add-cpt-llama2-lora-ablations"
EXP_DIR="cpt/CPT_Llama2_LoRA_Ablations"
SEEDS="${SEEDS:-0 1 2}"
AUTO_STOP="${AUTO_STOP:-1}"

echo "=== GPU ==="
nvidia-smi || { echo "No GPU visible — wrong pod template?"; exit 1; }

echo "=== Install stack ==="
pip install -q "unsloth @ git+https://github.com/unslothai/unsloth.git"
pip install -q --no-deps xformers trl peft accelerate bitsandbytes
pip install -q pyyaml wandb

echo "=== Clone branch ==="
WORKDIR="${WORKDIR:-/workspace}"
cd "$WORKDIR"
[ -d repo ] || git clone --branch "$BRANCH" "https://github.com/$REPO.git" repo
cd "repo/$EXP_DIR"

WANDB_ARGS="--no-wandb"
if [ -n "${WANDB_API_KEY:-}" ]; then
  wandb login "$WANDB_API_KEY" && WANDB_ARGS=""
fi

echo "=== Smoke test (10 steps, 01_baseline) ==="
python src/train.py --config configs/01_baseline.yaml --max-steps 10 --no-wandb

echo "=== Sweep: full ladder, seeds [$SEEDS] ==="
SEEDS="$SEEDS" EXTRA_ARGS="$WANDB_ARGS" bash scripts/run_all.sh
python scripts/make_table.py || true

echo "=== Persist results (pod disk is ephemeral) ==="
PERSISTED=0
if [ -n "${GH_TOKEN:-}" ]; then
  git config user.name  "Jaydeep"
  git config user.email "j.raijada25@gmail.com"
  git add results/
  git commit -m "Add GPU ablation results (runs.csv)" || echo "nothing to commit"
  git push "https://jaydeepraijada:${GH_TOKEN}@github.com/${REPO}.git" "HEAD:${BRANCH}"
  PERSISTED=1
fi
if [ -n "${HF_TOKEN:-}" ]; then
  HF_RESULTS_REPO="${HF_RESULTS_REPO:-jaydeepraijada/cpt-lora-ablations-results}"
  pip install -q -U huggingface_hub   # ensures the `hf` CLI entrypoint exists
  hf repos create "$HF_RESULTS_REPO" --type dataset --private --exist-ok
  hf upload "$HF_RESULTS_REPO" results . --type dataset \
    --exclude "*.gitkeep" --commit-message "GPU ablation results (runs.csv)"
  PERSISTED=1
fi
if [ "$PERSISTED" = "0" ]; then
  echo "WARNING: no GH_TOKEN or HF_TOKEN — results NOT persisted."
  echo "Copy results/runs.csv off the pod before it stops."
fi

echo "=== Done ==="
if [ "$AUTO_STOP" = "1" ] && [ -n "${RUNPOD_POD_ID:-}" ] && command -v runpodctl >/dev/null 2>&1; then
  echo "Stopping pod $RUNPOD_POD_ID to save credit."
  runpodctl stop pod "$RUNPOD_POD_ID"
fi
