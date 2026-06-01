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
