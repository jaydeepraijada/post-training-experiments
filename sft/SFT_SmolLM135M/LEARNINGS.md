# Learnings & Observations

A running log of insights from each SFT experiment.

---

## Setup Notes

- **save_steps=50 is too frequent** — 227 checkpoint saves over 11,355 steps added ~2-3 hours of overhead on top of ~5-6 hours of actual compute. Always use `save_steps=500` going forward.

- **MLX in pyproject.toml breaks Linux pods** — `mlx` and `mlx-lm` are Mac-only. Having them in `pyproject.toml` caused `libmlx.so` import errors on the RunPod RTX 4090. Removed. Keep MLX imports lazy (inside `if mlx:` blocks only).

- **TextIteratorStreamer needs `skip_prompt=True`** — without it, the Gradio app streamed the full prompt (system message + user message) before the actual response. Classic streaming bug.

- **`train_loss=0.087` in Trainer summary is misleading** — this is the average NLL normalized over ALL tokens in each packed sequence, including masked input tokens. Since packing at 2048 tokens leaves only ~100-200 response tokens per sequence, the denominator is ~10-20x larger than the actual response token count. The true per-token NLL on responses is ~1.2 (visible in the step-level W&B logs). Do not use Trainer summary `train_loss` as a quality metric for response-only training.

---

## Exp-001: exp01_cpt_sft

### Training Dynamics (from W&B logs)

**Phase 1 — Rapid initial adaptation (steps 0–~500, epoch 0–0.13):**
- Loss dropped from ~3.0 → ~2.5 in the first 76 steps (from training terminal logs)
- Grad norm was high and variable early (0.53–0.89) — model was rapidly adjusting to ChatML format
- This fast initial drop reflects the model learning the instruction-following template, not factual content

**Phase 2 — Steady improvement (epoch 1.35–2.45, W&B run 1):**
- Step-level train loss: ~1.35 → ~1.23 (slow, steady decrease)
- Eval loss: 1.344 → 1.237 — consistent improvement, no sign of overfitting
- Grad norm stabilised to 0.15–0.21 — smooth convergence
- LR decaying linearly from ~1.13e-04

**Phase 3 — Plateau (epoch 2.45–3.0, W&B runs 2 and 3):**
- Eval loss: 1.237 → 1.222 — improvement rate slowed dramatically
- Step-level train loss: ~1.20 (flat)
- Grad norm: 0.18–0.20 (very stable)
- LR decayed to near 0 (~1e-07 at final step)
- Clear plateau — additional epochs unlikely to help without data or architecture changes

### Eval Loss Curve (full run)

| Epoch | Eval Loss | Notes |
|---|---|---|
| 0.013 | 2.774 | From terminal log, step 50 |
| 1.35 | 1.344 | Start of W&B run 1 (resumed) |
| 1.50 | 1.330 | |
| 2.00 | 1.267 | End of epoch 2 |
| 2.45 | 1.237 | Run 1 killed (Ctrl+C) |
| 2.79 | 1.224 | Run 2 killed |
| 3.00 | **1.222** | Final |

### Train/Eval Gap

Step-level train loss at epoch 3: ~1.19  
Eval loss at epoch 3: 1.222  
**Gap: ~0.03** — almost no overfitting. The model generalises as well as it trains, which at 135M params on 600K samples suggests the model is limited by capacity, not overfitting.

### Key Findings

**1. The model learned task structure, not facts**
Relevance (3.04/5) is the strongest judge dimension — the model consistently responds to the right kind of task. But correctness (1.98) and completeness (1.85) are weak, because 135M params cannot reliably recall and reproduce factual ML content. The SFT gave it the right *shape* of responses, not the right *content*.

**2. Train/eval gap is tiny — capacity, not data, is the ceiling**
With 600K training examples and only 9.7M trainable LoRA params, the model reached its capacity ceiling by epoch 3. The eval loss plateau at 1.22 is not a data problem — more data would not help much at this scale. A larger model (360M, 1B) would benefit more from the same dataset.

**3. Eval loss plateaus in epoch 3 — 3 epochs is the right choice**
Eval loss improved from 1.344 → 1.222 across the run, but the rate halved by epoch 2 and essentially stopped by epoch 2.8. Running 4-5 epochs would waste compute with <0.01 improvement. 3 epochs is the right default for this model/data size combination.

**4. Grad norm stability suggests good training health**
After initial warmup, grad norm settled to 0.15–0.21 throughout — no spikes, no instability. The LoRA r=32 config with lr=2e-4 and linear decay is stable for this setup.

**5. Response-only training loss normalisation artifact**
The Trainer reports `train_loss=0.087` in its summary, which looks like near-perfect learning. This is an artefact of normalising NLL over all packed tokens (including masked input tokens). The true per-response-token NLL is ~1.2. Always check step-level `train/loss` in W&B, not the Trainer summary.

**6. CPT base vs raw base — still unknown**
This experiment used the CPT-adapted model as the base. We don't yet know how much the CPT pre-training contributed to the SFT results. exp02 (SFT on raw SmolLM-135M base) will isolate this.

### Judge Scores

| Metric | Score (1–5) |
|---|---|
| Faithfulness | 2.70 |
| Answer Correctness | 1.98 |
| Relevance | **3.04** |
| Completeness | 1.85 |
| **Overall** | **2.39** |

Relevance > Faithfulness > Correctness ≈ Completeness. The model knows what kind of answer to give but struggles to make it accurate and complete.

### Cost

- Hardware: NVIDIA RTX 4090 (24GB, RunPod)
- Wall time: ~10 hours (5-6h actual training + ~3h checkpoint overhead from save_steps=50)
- Cost: ~$8
- Expected with save_steps=500: ~$3-4

---

## Hypotheses for Upcoming Experiments

### exp02 — SFT on raw SmolLM-135M (no CPT)
**Hypothesis:** The CPT stage meaningfully improves downstream SFT quality. Without CPT, the model lacks ML vocabulary and scientific writing style, so instruction-following on ML tasks will be weaker.  
**What to watch:** Judge scores, especially faithfulness and correctness. If scores are similar to exp01, CPT's contribution to SFT is marginal. If scores drop significantly, CPT is load-bearing.

### exp03 — LoRA r=16 vs r=32
**Hypothesis:** Based on CPT findings, rank doesn't matter much once the dataset is the bottleneck. r=16 should match r=32 at this data scale and train faster.

### exp04 — Custom synthetic data (data_prep pipeline)
**Hypothesis:** Training on data generated from the same 188 papers the CPT model was trained on (domain-specific) will improve domain task quality vs the generic 300K dataset. Risk: small dataset → overfitting.
