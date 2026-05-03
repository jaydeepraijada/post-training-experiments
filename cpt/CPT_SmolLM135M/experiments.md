# Experiment Log

## Planned Experiments

| # | Run ID | Description | Key Flags | Status |
|---|--------|-------------|-----------|--------|
| 1a | `exp01_full_ft` | Full fine-tuning, 4-bit (broken) | `--full_training` | ❌ Broken |
| 1b | `exp01_full_ft_bf16` | Full fine-tuning, bf16 (fixed) | `--full_training` | ✅ Done — worse than LoRA |
| 3a | `exp03_lora_r8` | LoRA rank 8 | `--lora_r 8` | ✅ Done — within noise of r=16/32 |
| 3b | `exp03_lora_r16` | LoRA rank 16 | `--lora_r 16` | ✅ Done — ~same as r=32 |
| 3c | `exp03_lora_r32` | LoRA rank 32 (baseline) | `--lora_r 32` | ✅ Done |
| 4 | `exp04_interleave_noclean` | 20% custom + 80% HF mix, no cleaning, LoRA r=16 | `--mix --mix_ratio 0.2 --lora_r 16` | ✅ Done |
| 5 | `exp05_interleave_clean` | Same mix, cleaned dataset, LoRA r=16 | `--mix --mix_ratio 0.2 --lora_r 16` | ✅ Done |
| 6 | `exp06_rslora_interleaved` | rsLoRA r=16 on interleaved cleaned dataset | `--mix --mix_ratio 0.2 --rslora --lora_r 16` | ✅ Done |
| 5b | `exp05b_interleave_clean_r32` | Same as exp05 but r=32 + all_exhausted + max_steps=3000 | `--mix --mix_ratio 0.2 --lora_r 32 --max_steps 3000` | ❌ Worse than base — all_exhausted diluted domain signal |
| F | Qualitative comparison | Generate text samples: base vs worst vs best model on same prompts | `compare.py` | ⏳ After exp05b |
| F | Push to HuggingFace | Push best model | `huggingface-cli upload` | ⏳ Final step |

### Notes on each experiment

**Exp 1a — Full fine-tuning (broken)**
Used `load_in_4bit=True` with `full_finetuning=True` in Unsloth — weight updates get re-quantized after each step, model barely learned. Results identical to base model.

**Exp 1b — Full fine-tuning bf16 (fixed)**
Load in bf16 (no quantization) so weight updates are lossless. Real full fine-tuning of all 135M parameters.

**TIES + Catastrophic forgetting — DROPPED**
Dataset (188 papers) is too small to cause meaningful forgetting. Full FT results were essentially identical to base model — nothing to forget. Would revisit on a much larger dataset (1000+ papers).

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
