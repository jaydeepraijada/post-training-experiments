# Learnings & Hypotheses

A running log of insights, surprises, and reasoning behind each experiment decision.

---

## Setup Decisions

- **Early stopping removed** — for CPT, eval loss plateaus in the 3rd decimal place due to noise, not true convergence. With patience=3 at eval_steps=20, training was stopping at epoch ~5.6 due to 0.002-level fluctuations. Since train/eval gap is small (not overfitting), better to run full epochs and save best checkpoint manually.

- **No HuggingFace push until best model identified** — pushing every checkpoint is noisy. Only the winner of all experiments goes to HF.

---

## Experiment Insights

### exp03_lora_r32 (Baseline — LoRA r=32, 10 epochs)

**Results:**
| Metric | Value |
|---|---|
| Perplexity | 18.36 |
| Cross-Entropy | 2.91 |
| ROUGE-1 | 0.2135 |
| ROUGE-L | 0.1434 |
| BERTScore F1 | 0.7532 |

**Observations:**
- Training completed in ~14 min total (8 min first run + 6 min resumed)
- Eval loss curve dropped fast early (3.049 → 2.87 in first 2 epochs) then plateaued around 2.845-2.851
- Train loss reached ~2.86 at final step, avg train loss 1.14 (packing artifact — loss per token vs per sequence)
- ROUGE scores are low (0.21 / 0.14) but expected for open-ended scientific text generation
- BERTScore 0.75 is reasonable for a 135M model

**What this tells us:**
- CPT is working — perplexity dropped 20% vs base model (22.97 → 18.36)
- Every metric improved over base: ROUGE-1 +0.036, BERTScore +0.017, Cross-Entropy -0.224
- The plateau around 2.845 eval loss may be the capacity ceiling for r=32 LoRA on this dataset size
- Low ROUGE may just reflect the nature of scientific text generation (high vocabulary diversity)

---

### base_model (HuggingFaceTB/SmolLM-135M — no fine-tuning)

| Metric | Value |
|---|---|
| Perplexity | 22.97 |
| Cross-Entropy | 3.134 |
| ROUGE-1 | 0.1778 |
| ROUGE-L | 0.1136 |
| BERTScore F1 | 0.7361 |

This is the reference point for all experiments. Any run that doesn't beat this across all metrics is a failure.

---

### exp01_full_ft (Full fine-tuning — BROKEN, 4-bit)

| Metric | Value |
|---|---|
| Perplexity | 22.91 |
| Cross-Entropy | 3.131 |
| ROUGE-1 | 0.1775 |
| BERTScore F1 | 0.7344 |

**Result: Essentially identical to base model — CPT did nothing.**

**Root cause:** Unsloth's `full_finetuning=True` with `load_in_4bit=True` corrupts weight updates — gradients flow through quantized weights but updates get re-quantized after each step, severely limiting learning. Train runtime was 43 min but model barely moved.

**Fix:** Load in bf16 when `--full_training` is set (`load_in_4bit = not args.full_training`). Rerunning as `exp01_full_ft_bf16`.

---

## Hypotheses for Upcoming Experiments

### exp01_full_ft_bf16 (Full fine-tuning — fixed)
**Hypothesis:** With bf16 (no quantization), weight updates are lossless. Expect significantly lower perplexity than LoRA r=32. Risk: slower training (~45 min), higher VRAM, and potential catastrophic forgetting since all 135M params are updated.
- lr=1e-5, NEFTune noise=5, max_grad_norm=0.7

### exp03_lora_r8 / r16 (LoRA rank sweep)
**Hypothesis:** Lower rank = more regularisation = potentially better generalisation on small dataset (138 papers). r=8 might outperform r=32 if r=32 is slightly overfitting to the training distribution.

### exp04 / exp05 (Interleaved dataset, 20% custom + 80% HF scientific_papers)
**Hypothesis:** 138 papers is a very small dataset. Mixing in the large HF scientific_papers corpus (80%) should give the model much broader scientific language coverage while still adapting to our domain. Risk: our domain signal gets diluted.

### exp05 vs exp04 (Cleaned vs uncleaned)
**Hypothesis:** Removing reference sections reduces noise (bibliography text is not useful for generation). Cleaning should improve metrics slightly.

### exp06 (rsLoRA)
**Hypothesis:** Rank-stabilised LoRA normalises updates by sqrt(r), keeping gradient scale stable. Should help at r=16 where standard LoRA updates might be slightly too large.

---

## Catastrophic Forgetting Testing Plan

After exp01_full_ft_bf16 completes:

1. **Run lm-evaluation-harness** on 3 models:
   - `HuggingFaceTB/SmolLM-135M` (base — reference)
   - `models/exp01_full_ft_bf16/final` (full FT)
   - `models/exp01_full_ft_bf16_ties/` (TIES merged)

2. **Tasks:** HellaSwag (commonsense), ARC-Easy (general knowledge), PIQA (physical intuition)

3. **What to look for:**
   - If full FT drops benchmark scores vs base → catastrophic forgetting confirmed
   - If TIES-merged model recovers benchmark scores while keeping domain gains → TIES is working
   - Ideal outcome: TIES model has lower perplexity than base AND similar benchmark scores to base

4. **Command:**
```bash
lm_eval --model hf --model_args pretrained=models/exp01_full_ft_bf16/final \
    --tasks hellaswag,arc_easy,piqa --device cuda --output_path results/lm_eval_exp01_bf16.json
```

---

## Questions to Answer as Results Come In

1. ~~How much does CPT help vs base model?~~ **Answered: 20% perplexity drop, all metrics improve.**
2. Does full fine-tuning (bf16) beat LoRA? (running now as exp01_full_ft_bf16)
3. Does TIES reduce catastrophic forgetting while preserving domain gains?
4. Is the eval loss plateau a dataset size problem or a model capacity problem? (exp04/05 will test)
5. Do ROUGE/BERTScore correlate with perplexity improvements, or diverge?
6. Does lower LoRA rank (r=8/16) generalise better on this small dataset?
