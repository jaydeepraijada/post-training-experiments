# Experiment Log

## Planned Experiments

| # | Run ID | Description | Key Flags |
|---|--------|-------------|-----------|
| 1 | `exp01_full_ft` | Full fine-tuning, no LoRA | `--full_training` (lr=1e-5, NEFTune noise=5) |
| 2 | `exp01_full_ft_ties` | TIES merge of exp01 back into base | `merge.py --density 0.3 --weight 1.0` |
| 3a | `exp03_lora_r8` | LoRA rank 8 | `--lora_r 8` |
| 3b | `exp03_lora_r16` | LoRA rank 16 | `--lora_r 16` |
| 3c | `exp03_lora_r32` | LoRA rank 32 (baseline) | `--lora_r 32` |
| 4 | `exp04_interleave_noclean` | 20% custom + 80% HF scientific_papers, no cleaning, LoRA r=16 | `--mix --mix_ratio 0.2 --lora_r 16`, dataset built with `--no_clean` |
| 5 | `exp05_interleave_clean` | Same mix, cleaned dataset (refs removed, appendix kept), LoRA r=16 | `--mix --mix_ratio 0.2 --lora_r 16`, dataset built without `--no_clean` |
| 6 | `exp06_interleave_rslora` | Same as exp05 + rsLoRA | `--mix --mix_ratio 0.2 --lora_r 16 --rslora` |

### Notes on each experiment

**Exp 1 — Full fine-tuning**
Train all 135M parameters. Uses lower LR (1e-5 vs 2e-4) and NEFTune noise (α=5) to stabilise. Expect higher memory usage and longer training.

**Exp 2 — TIES merge**
After exp01 trains, run `merge.py` to blend the fine-tuned weights back toward the base via task-vector sparsification. Goal: reduce domain drift while keeping domain knowledge. Sweep `--density` (0.2, 0.3, 0.5) and `--weight` (0.5, 0.75, 1.0).

**Exp 3 — LoRA rank sweep**
Isolate the effect of rank. Lower rank = fewer parameters, more regularisation. Compare r=8/16/32 on identical data and LR.

**Exp 4 — Interleaved, uncleaned**
Reverse the mix ratio (20% custom, 80% HF) to test whether large general scientific text helps more than domain-specific small data. Build the dataset with `python data_prep/get_dataset.py papers/ --no_clean` to skip reference cleaning.

**Exp 5 — Interleaved, cleaned**
Same mix as exp04 but use the default cleaned dataset (references stripped, appendix preserved). Isolates the effect of cleaning.

**Exp 6 — rsLoRA**
Rank-stabilised LoRA normalises the LoRA update by `sqrt(r)`, which keeps gradient scale stable at higher ranks. Compare directly against exp05.

<!-- AUTO-GENERATED: do not edit below this line -->

## Results

| Run | Date | Hypothesis | PPL | ROUGE-1 | ROUGE-L | BERTScore F1 | Notes |
|-----|------|------------|-----|---------|---------|--------------|-------|

---

*No experiments logged yet. Run `log_experiment.py` after each eval to populate this table.*
