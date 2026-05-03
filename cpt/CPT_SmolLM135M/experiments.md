# Experiment Log

## Planned Experiments

| # | Run ID | Description | Key Flags | Status |
|---|--------|-------------|-----------|--------|
| 1a | `exp01_full_ft` | Full fine-tuning, 4-bit (broken) | `--full_training` | ❌ Broken |
| 1b | `exp01_full_ft_bf16` | Full fine-tuning, bf16 (fixed) | `--full_training` | ✅ Done — worse than LoRA |
| 7 | `exp07_full_ft_interleaved` | Full FT bf16 on best interleaved dataset config | `--full_training` | ⏳ Deferred to end |
| 8 | `exp07_ties` | TIES merge of exp07 back into base | `merge.py --density 0.2/0.3/0.5` | ⏳ Deferred to end |
| 9 | Catastrophic forgetting eval | lm-eval HellaSwag + ARC-Easy + PIQA: base vs exp07 vs exp07_ties | `lm_eval` | ⏳ Deferred to end |
| 3a | `exp03_lora_r8` | LoRA rank 8 | `--lora_r 8` | ✅ Done — within noise of r=16/32 |
| 3b | `exp03_lora_r16` | LoRA rank 16 | `--lora_r 16` | ✅ Done — ~same as r=32 |
| 3c | `exp03_lora_r32` | LoRA rank 32 (baseline) | `--lora_r 32` | ✅ Done |
| 4 | `exp04_interleave_noclean` | 20% custom + 80% HF scientific_papers, no cleaning, LoRA r=16 | `--mix --mix_ratio 0.2 --lora_r 16`, dataset built with `--no_clean` | ⏳ Pending |
| 5 | `exp05_interleave_clean` | Same mix, cleaned dataset (refs removed, appendix kept), LoRA r=16 | `--mix --mix_ratio 0.2 --lora_r 16` | ⏳ Pending |
| 6 | `exp06_rslora_interleaved` | rsLoRA at best rank on interleaved dataset — compare against exp04/05 winner | `--mix --mix_ratio 0.2 --rslora --lora_r <best>` | ⏳ Pending after exp04/05 |

### Notes on each experiment

**Exp 1a — Full fine-tuning (broken)**
Used `load_in_4bit=True` with `full_finetuning=True` in Unsloth — weight updates get re-quantized after each step, model barely learned. Results identical to base model.

**Exp 1b — Full fine-tuning bf16 (fixed)**
Load in bf16 (no quantization) so weight updates are lossless. Real full fine-tuning of all 135M parameters.

**Exp 2 — TIES merge + catastrophic forgetting eval**
After exp01_bf16 trains:
1. Run `merge.py` to TIES-merge back toward base (sweep density 0.2/0.3/0.5)
2. Run `lm_eval` on HellaSwag + ARC-Easy on: base model, exp01_bf16, and TIES-merged version
3. Goal: quantify forgetting and test if TIES reduces it while preserving domain gains

**Exp 3 — LoRA rank sweep**
Isolate the effect of rank. Lower rank = fewer parameters, more regularisation. Compare r=8/16/32 on identical data and LR.

**Exp 4 — Interleaved, uncleaned**
Reverse the mix ratio (20% custom, 80% HF) to test whether large general scientific text helps more than domain-specific small data. Build the dataset with `python data_prep/get_dataset.py papers/ --no_clean` to skip reference cleaning.

**Exp 5 — Interleaved, cleaned**
Same mix as exp04 but use the default cleaned dataset (references stripped, appendix preserved). Isolates the effect of cleaning.

**Exp 6 — rsLoRA on interleaved dataset**
Run after exp04/05 to identify the best rank. Then run rsLoRA at that rank on the same interleaved dataset for a direct comparison. Only meaningful comparison is on the big dataset where differences can actually show up.

<!-- AUTO-GENERATED: do not edit below this line -->

## Results

| Run | Date | Hypothesis | PPL | ROUGE-1 | ROUGE-L | BERTScore F1 | Notes |
|-----|------|------------|-----|---------|---------|--------------|-------|

---

*No experiments logged yet. Run `log_experiment.py` after each eval to populate this table.*
