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

### exp01_full_ft_bf16 (Full fine-tuning — bf16, fixed)

| Metric | Value |
|---|---|
| Perplexity | 22.88 |
| Cross-Entropy | 3.130 |
| ROUGE-1 | 0.1704 |
| ROUGE-L | 0.1105 |
| BERTScore F1 | 0.7372 |

**Result: Worse than LoRA r=32 on every metric. ROUGE-1 even dropped below base model (0.170 vs 0.178).**

**What this tells us:**
- Full fine-tuning on 138 papers = overfitting. 135M params, tiny dataset — the model memorises training distribution instead of generalising.
- High grad_norm (1.18 vs 0.25 for LoRA) confirms unstable optimization — too many params chasing too little data.
- LoRA's regularisation (only 9.7M trainable params) is exactly what's needed at this data scale.
- Full fine-tuning may work at scale — **revisit after interleaved experiments (exp04/05) with much larger effective dataset.**

**Revised plan:** Skip catastrophic forgetting eval for now. Do full FT + TIES + forgetting eval as a final experiment after finding the best config on the large interleaved dataset.

---

## Hypotheses for Upcoming Experiments

### exp03_lora_r16 (LoRA rank 16)

| Metric | Value |
|---|---|
| Perplexity | 18.39 |
| Cross-Entropy | 2.912 |
| ROUGE-1 | 0.2070 |
| ROUGE-L | 0.1382 |
| BERTScore F1 | 0.7522 |
| Train runtime | ~13 min |

**Result: Essentially identical to r=32 across all metrics. r=32 edges it by a tiny margin.**

**What this tells us:**
- The performance difference between r=16 and r=32 is within noise — neither is meaningfully better
- r=16 uses half the parameters and trains in the same time — arguably more efficient
- Diminishing returns above r=16 for this dataset size
- r=8 will tell us if there's a meaningful floor — expecting slight degradation

---

### exp03_lora_r8 (LoRA rank sweep — pending)
**Hypothesis:** Lower rank = more regularisation = potentially better generalisation on small dataset (138 papers). r=8 might outperform r=32 if r=32 is slightly overfitting to the training distribution.

### exp04 / exp05 (Interleaved dataset, 20% custom + 80% HF scientific_papers)
**Hypothesis:** 138 papers is a very small dataset. Mixing in the large HF scientific_papers corpus (80%) should give the model much broader scientific language coverage while still adapting to our domain. Risk: our domain signal gets diluted.

### exp05 vs exp04 (Cleaned vs uncleaned)
**Hypothesis:** Removing reference sections reduces noise (bibliography text is not useful for generation). Cleaning should improve metrics slightly.

### exp06 (rsLoRA on interleaved dataset, best rank)
**Decision:** Dropped small-dataset rsLoRA comparisons — the small dataset plateaus at ~2.845 regardless of config, so any differences would be noise not signal. rsLoRA will be tested on the interleaved dataset at the best rank from exp04/05, where differences can actually manifest.

---

## Catastrophic Forgetting Testing Plan (deferred)

**Decision:** Skip for now. Full FT on 138 papers doesn't learn enough to cause meaningful forgetting anyway. Will revisit after finding best config on large interleaved dataset.

**Final experiment plan (post all LoRA/interleaved runs):**
1. Full fine-tuning (bf16) on best interleaved dataset config
2. TIES merge sweep (density 0.2/0.3/0.5)
3. lm-eval on HellaSwag + ARC-Easy + PIQA: base vs full FT vs TIES
4. Push winner to HuggingFace

---

## Questions to Answer as Results Come In

1. ~~How much does CPT help vs base model?~~ **Answered: 20% perplexity drop, all metrics improve.**
2. ~~Does full fine-tuning (bf16) beat LoRA on small data?~~ **Answered: No — LoRA wins on 138 papers. Full FT overfits.**
3. Does lower LoRA rank (r=8/16) generalise better than r=32 on small dataset?
4. Is the eval loss plateau a dataset size problem or model capacity problem? (exp04/05 will test)
5. Does the interleaved large dataset unlock full fine-tuning's potential?
6. Does TIES reduce catastrophic forgetting while preserving domain gains? (final experiment)
