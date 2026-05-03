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

**Open question:** No base model baseline yet — can't tell how much CPT actually helped until `results_base_model.json` is compared.

**What this tells us:**
- The model is learning something — perplexity of 18.36 on held-out ML papers
- The plateau around 2.845 eval loss may be the capacity ceiling for r=32 LoRA on this dataset size
- Low ROUGE may just reflect the nature of scientific text generation (high vocabulary diversity)

---

## Hypotheses for Upcoming Experiments

### exp01_full_ft (Full fine-tuning)
**Hypothesis:** Training all 135M parameters (vs ~9.7M with LoRA r=32) should allow the model to absorb domain knowledge more deeply. Expect lower perplexity but risk of more catastrophic forgetting of general language ability.
- lr=1e-5 (vs 2e-4 for LoRA) to be conservative
- NEFTune noise added for regularisation

### exp03_lora_r8 / r16 (LoRA rank sweep)
**Hypothesis:** Lower rank = more regularisation = potentially better generalisation on small dataset (138 papers). r=8 might outperform r=32 if r=32 is slightly overfitting to the training distribution.

### exp04 / exp05 (Interleaved dataset, 20% custom + 80% HF scientific_papers)
**Hypothesis:** 138 papers is a very small dataset. Mixing in the large HF scientific_papers corpus (80%) should give the model much broader scientific language coverage while still adapting to our domain. Risk: our domain signal gets diluted.

### exp05 vs exp04 (Cleaned vs uncleaned)
**Hypothesis:** Removing reference sections reduces noise (bibliography text is not useful for generation). Cleaning should improve metrics slightly.

### exp06 (rsLoRA)
**Hypothesis:** Rank-stabilised LoRA normalises updates by sqrt(r), keeping gradient scale stable. Should help at r=16 where standard LoRA updates might be slightly too large.

---

## Questions to Answer as Results Come In

1. How much does CPT help vs base model? (waiting on base model baseline)
2. Does full fine-tuning beat LoRA, and at what cost (forgetting)?
3. Is the eval loss plateau a dataset size problem or a model capacity problem?
4. Do ROUGE/BERTScore correlate with perplexity improvements, or diverge?
