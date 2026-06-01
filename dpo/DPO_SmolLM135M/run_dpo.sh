#!/usr/bin/env bash
# End-to-end DPO/ORPO run for SmolLM-135M on RunPod (RTX 4090).
#
# Reuses the existing preference dataset (paperbd/paper_preference_150K-v1) —
# no data generation. Steps: baseline diversity (SFT) -> train -> post-train
# diversity (compare for mode collapse).
#
# Usage:
#   bash run_dpo.sh                      # DPO, defaults
#   METHOD=orpo RUN_ID=orpo_default bash run_dpo.sh
#   BATCH_SIZE=32 GRAD_ACCUM=4 bash run_dpo.sh   # if 24GB has room
#   RUN_BASELINE=0 bash run_dpo.sh       # skip the baseline (already done)
set -euo pipefail
cd "$(dirname "$0")"

# Reduce allocator fragmentation (helps the large fp32 logits upcast in DPO).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SFT_MODEL="${SFT_MODEL:-paperbd/smollm_135M_neuraltxt_v1}"
DATASET="${DATASET:-paperbd/paper_preference_150K-v1}"
METHOD="${METHOD:-dpo}"
RUN_ID="${RUN_ID:-dpo_default}"
# DPO computes a reference forward and upcasts full-vocab logits to fp32:
# ~ (2*batch) * seq * vocab * 4 bytes. On a 24GB card, batch 16 / seq 2048 OOMs.
# batch 8 + seq 1024 keeps that tensor ~3GB; grad_accum 16 keeps eff batch ~128.
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-768}"
DIVERSITY_SAMPLES="${DIVERSITY_SAMPLES:-100}"
TEMPS="0.3 0.5 0.7 1.0"

echo "=========================================="
echo " method=${METHOD} run_id=${RUN_ID}"
echo " base=${SFT_MODEL}"
echo " dataset=${DATASET}"
echo " batch=${BATCH_SIZE} grad_accum=${GRAD_ACCUM} (eff ~$((BATCH_SIZE*GRAD_ACCUM)))"
echo "=========================================="

echo "== uv sync =="
uv sync

# ── Step 0: baseline diversity on the SFT model (pre-DPO reference) ───────────
BASE_RESP="evals/baseline_${SFT_MODEL##*/}_n${DIVERSITY_SAMPLES}_r4.jsonl"
if [ "${RUN_BASELINE:-1}" = "1" ]; then
  echo "== Step 0: baseline diversity (SFT model) =="
  uv run python diversity_evals/generate_responses.py \
      -m "$SFT_MODEL" --num_samples "$DIVERSITY_SAMPLES" --n_responses 4 \
      --temperatures $TEMPS -bs 32 -o "$BASE_RESP"
  uv run python diversity_evals/diversity.py -i "$BASE_RESP"
else
  echo "== Step 0: skipped (RUN_BASELINE=0) =="
fi

# ── Step 1: preference training ──────────────────────────────────────────────
echo "== Step 1: ${METHOD} training =="
uv run python train_preference.py \
    --method "$METHOD" -o "$RUN_ID" \
    --base_model_id "$SFT_MODEL" --dataset "$DATASET" \
    --batch_size "$BATCH_SIZE" --grad_accum "$GRAD_ACCUM" \
    --max_seq_length "$MAX_SEQ_LEN" --max_prompt_length "$MAX_PROMPT_LEN"

# ── Step 2: post-train diversity on the merged model (collapse check) ─────────
echo "== Step 2: post-${METHOD} diversity (compare to baseline above) =="
DPO_RESP="evals/${RUN_ID}_n${DIVERSITY_SAMPLES}_r4.jsonl"
uv run python diversity_evals/generate_responses.py \
    -m "models/${RUN_ID}/merged" --num_samples "$DIVERSITY_SAMPLES" --n_responses 4 \
    --temperatures $TEMPS -bs 32 -o "$DPO_RESP"
uv run python diversity_evals/diversity.py -i "$DPO_RESP"

echo "=========================================="
echo " Done. Compare the two diversity summaries:"
echo "   baseline: ${BASE_RESP}"
echo "   ${RUN_ID}: ${DPO_RESP}"
echo " A large drop in SBERT/Vendi => mode collapse."
echo " Merged model: models/${RUN_ID}/merged"
echo "=========================================="
