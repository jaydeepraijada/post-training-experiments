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

### exp03_lora_r8 (LoRA rank 8)

| Metric | Value |
|---|---|
| Perplexity | 18.42 |
| Cross-Entropy | 2.914 |
| ROUGE-1 | 0.2093 |
| ROUGE-L | 0.1433 |
| BERTScore F1 | 0.7477 |
| Train runtime | ~13 min |
| grad_norm | 0.187 (lowest of all ranks) |

**Result: Marginally worse than r=16 and r=32, but differences are within noise.**

**Rank sweep conclusion:**

| Rank | PPL | ROUGE-1 | BERTScore | Params trained |
|---|---|---|---|---|
| r=8 | 18.42 | 0.209 | 0.748 | ~4.9M |
| r=16 | 18.39 | 0.207 | 0.752 | ~9.7M |
| r=32 | 18.36 | 0.213 | 0.753 | ~19.4M |

**The dataset is the bottleneck, not the rank.** All three converge to the same ~2.845 eval loss plateau. For interleaved experiments use **r=16** — best efficiency/performance tradeoff.

### exp04 / exp05 (Interleaved dataset, 20% custom + 80% HF scientific_papers)
**Hypothesis:** 138 papers is a very small dataset. Mixing in the large HF scientific_papers corpus (80%) should give the model much broader scientific language coverage while still adapting to our domain. Risk: our domain signal gets diluted.

### exp05 vs exp04 (Cleaned vs uncleaned)
**Hypothesis:** Removing reference sections reduces noise (bibliography text is not useful for generation). Cleaning should improve metrics slightly.

### exp06 (rsLoRA on interleaved dataset, best rank)
**Decision:** Dropped small-dataset rsLoRA comparisons — the small dataset plateaus at ~2.845 regardless of config, so any differences would be noise not signal. rsLoRA will be tested on the interleaved dataset at the best rank from exp04/05, where differences can actually manifest.

---

### exp04/05/06 (Interleaved dataset)

| Exp | PPL | ROUGE-1 | BERTScore F1 |
|---|---|---|---|
| exp04 (uncleaned, r=16) | 18.74 | 0.203 | 0.755 |
| exp05 (cleaned, r=16) | 18.40 | 0.204 | 0.753 |
| exp06 (rsLoRA r=16) | 18.58 | 0.213 | 0.756 |

**Result: Interleaved dataset didn't clearly beat small-paper LoRA (r=32: PPL 18.36, ROUGE 0.213).**

**Root cause:** `stopping_strategy="first_exhausted"` in `interleave_datasets` caused training to stop when the custom dataset (5,200 chunks) was exhausted. At 20% sampling rate, this means only ~26,000 total samples per "epoch" — of which only ~20,800 were from HF (118k papers). We barely scratched the surface of the big dataset.

- exp04 ran 1720 steps (uncleaned = more chunks = runs longer)
- exp05/06 ran only 820 steps (cleaned = fewer chunks = exhausts faster)

**Fix for exp05b:** Switch to `stopping_strategy="all_exhausted"` + `--max_steps 3000`. This forces training to continue deep into the HF dataset, seeing ~192,000 sequences total (vs ~52,000 before). At 80% HF ratio = ~153,000 HF sequences — 37× more than before.

**exp06 rsLoRA observation:** grad_norm shot up to 0.557 (vs 0.219 for standard LoRA). rsLoRA at r=16 seems to destabilise training rather than help. Would need tuning or higher rank to benefit.

---

## Catastrophic Forgetting — DROPPED

**Decision:** Dataset (188 papers) is too small to cause meaningful forgetting. Full FT bf16 results were essentially identical to base model — if 43 minutes of full FT on 188 papers doesn't shift the weights meaningfully, there's nothing to forget. TIES and lm-eval benchmarks would just be noise.

**Would revisit if:** running on 1000+ papers with full fine-tuning.

## Final Experiment Plan

1. ✅ Rank sweep (r=8/16/32) — done
2. ✅ Full FT (4-bit broken, bf16 worse than LoRA) — done
3. ✅ Interleaved exp04/05/06 — done
4. ⏳ exp05b: cleaned + mixed + r=32 (reference best config)
5. ⏳ Qualitative comparison: `compare.py` — base vs worst vs best on same prompts
6. ⏳ Push best model to HuggingFace

---

## Questions to Answer as Results Come In

1. ~~How much does CPT help vs base model?~~ **Answered: 20% perplexity drop, all metrics improve.**
2. ~~Does full fine-tuning (bf16) beat LoRA on small data?~~ **Answered: No — LoRA wins on 138 papers. Full FT overfits.**
3. ~~Does lower LoRA rank (r=8/16) generalise better than r=32 on small dataset?~~ **Answered: No difference — dataset is the bottleneck. Use r=16 for efficiency.**
4. ~~Is the eval loss plateau a dataset size problem?~~ **Root cause found: `stopping_strategy="first_exhausted"` caused training to stop after ~4,000 HF papers/epoch — barely leveraging the big dataset. Fixed with `all_exhausted` + `--max_steps 3000` for exp05b.**
5. Does the interleaved large dataset unlock full fine-tuning's potential?
6. Does TIES reduce catastrophic forgetting while preserving domain gains? (final experiment)
