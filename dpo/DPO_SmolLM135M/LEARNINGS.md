# Learnings & Observations

A running log of insights from each DPO/ORPO experiment.

---

## Setup Notes

- **`save_steps=500` / `eval_steps=500`** — the upstream configs shipped with `50`, which added hours of checkpoint overhead in the SFT stage. Standardised to 500 for both DPO and ORPO.

- **MLX stays out of core deps** — `mlx` / `mlx-lm` are Mac-only and break Linux/CUDA pods (learned in SFT). They live in the `mlx` optional extra; the local MLX ranker (`generate_ranked_pairs.py`) also needs `mlx_vlm_batch_outlines` from source. The OpenAI/OpenRouter ranker (`generate_ranked_pairs_openai.py`) is the cross-platform default.

- **Run from inside the experiment folder** — imports are flat (`from chat_formatting import ...`) and output paths are relative (`evals/...`). Running from the repo root will not resolve them.

- **`SYSTEM_PROMPT` is vendored** — copied byte-for-byte from the SFT stage into `prompts.py`. It must match the string the model was trained on; do not edit it.

- **DPO OOMs on 24GB at batch 16 / seq 2048** — DPO runs a reference forward and upcasts full-vocab logits to fp32 (`accelerate.convert_to_fp32` → `tensor.float()`), a single `~(2*batch)*seq*vocab*4` byte allocation (~9.8GB here). It dies on step 1. Fix: shrink batch **and** sequence — `batch 8 / grad_accum 16 / max_seq_length 1024 / max_prompt_length 768` keeps that tensor ~3GB and fits a 3090/4090 with margin (effective batch stays 128). Also set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. The model loading + tokenization caches, so an OOM retry is cheap (~2 min).

---

## Findings

### dpo_default — DPO works at 135M and doesn't collapse diversity (2026-06-01)

First preference-tuning run. DPO (β=0.1, LoRA r=32) on `paper_preference_150K-v1`, 3 epochs, ~6h17m on a 3090.

- **It learned the preference.** Held-out reward accuracy 0.5 → **0.72**, eval loss 0.577 → **0.457**, margins 0.46 → 1.65. Steady, healthy curves.
- **No mode collapse.** EAD/SBERT/Vendi all flat vs the SFT baseline (within +0.3–2.6%). The big worry with DPO — narrowing the output distribution — did not happen at β=0.1. Diversity is *not* the binding constraint here.
- **Mild overfitting is the real limit.** Train reward accuracy hit ~0.85 vs eval 0.72; eval loss plateaued by epoch ~2.7. So **more epochs would overfit, not help** — 3 is right. The lever for more quality is a **lower β** (let the policy move further from the SFT ref), not longer training.
- **Reward drift is normal.** Both chosen and rejected rewards go increasingly negative (KL drift from the frozen ref); the *margin* growing is the signal that matters.
- **Open question:** eval reward accuracy (0.72) measures agreement with the LLM-judge that *made* the pairs — it's not an independent quality signal. To know if outputs are actually better, train the reward model or run a win-rate judge. (Win-rate is partly circular since the same judge ranked the data.)

## Hypotheses for Upcoming Experiments

### baseline_diversity — diversity of the SFT model (pre-DPO)
**Hypothesis:** At 135M params the SFT model already produces fairly templated responses; diversity may be low even before DPO. Need this reference to interpret any post-DPO drop.
**What to watch:** EAD / SBERT / Vendi across the temperature sweep. If diversity is already low at temp 1.0, the preference pairs mined from sampling will be weak.

### dpo_default — DPO, beta=0.1
**Hypothesis:** DPO on LLM-ranked pairs nudges the model toward higher-quality (more correct/complete) responses without retraining from scratch. Given the SFT ceiling was *capacity*, not data, gains may be modest — DPO sharpens preference, it doesn't add knowledge.
**What to watch:** judge win-rate vs SFT, and whether diversity collapses relative to baseline.

### orpo_default — ORPO (reference-free)
**Hypothesis:** ORPO folds the SFT and preference objectives into one loss with no reference model, so it may be more stable / cheaper at this scale. Compare final quality and diversity head-to-head with DPO.

### dpo_beta_sweep — KL strength
**Hypothesis:** Lower beta (weaker KL anchor) lets the policy move further from the SFT model — higher win-rate but more diversity loss. Higher beta stays closer and safer. Expect a quality↔diversity trade-off curve.

### reward_model — does the preference signal generalise?
**Hypothesis:** A reward model trained on the same pairs should exceed ~chance pairwise accuracy on held-out pairs. Low accuracy would mean the LLM-judge rankings are noisy and the DPO signal is weak.
