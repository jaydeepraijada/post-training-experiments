#!/usr/bin/env bash
# Run the full ablation ladder. Optionally pass seeds to measure variance:
#   bash scripts/run_all.sh                 # single seed (3407, from config)
#   SEEDS="0 1 2" bash scripts/run_all.sh   # 3 seeds per config -> mean +/- std
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIGS=(
  configs/01_baseline.yaml
  configs/02_gate_proj.yaml
  configs/03_embed_lmhead.yaml
  configs/04_rslora.yaml
  configs/05_decoupled_lr.yaml
  configs/06_rslora_decoupled.yaml
)

SEEDS="${SEEDS:-3407}"

for cfg in "${CONFIGS[@]}"; do
  for seed in $SEEDS; do
    echo ">>> $cfg (seed $seed)"
    python src/train.py --config "$cfg" --seed "$seed"
  done
done

echo ">>> Sweep done. Regenerate the results table:"
echo "    python scripts/make_table.py"
