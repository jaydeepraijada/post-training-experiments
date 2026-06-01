# DPO Experiment Log

Base model for all runs: `paperbd/smollm_135M_neuraltxt_v1` (the SFT'd assistant).
Preference dataset: `paperbd/paper_preference_150K-v1` unless noted.

## Experiment Plan

| # | Run ID | Description | Key Flags | Status |
|---|--------|-------------|-----------|--------|
| 0 | `baseline_diversity` | Diversity of the SFT model's sampled responses (pre-DPO reference) | `generate_responses.py` → `diversity.py` | ⏳ Planned |
| 1 | `dpo_default` | DPO, beta=0.1, LoRA r=32 | defaults | ⏳ Planned |
| 2 | `orpo_default` | ORPO (reference-free), same data | `--method orpo` | ⏳ Planned |
| 3 | `dpo_beta_sweep` | Effect of KL strength: beta 0.1 / 0.3 / 0.5 | `--beta` | ⏳ Planned |
| 4 | `reward_model` | Train + eval a reward model on the pairs | `train_reward.py` / `evaluate_reward_model.py` | ⏳ Planned |

---

## Results

| Run | Date | Method | Base | Dataset | beta | r | Epochs | Eval Loss | Reward Acc | EAD | SBERT | Vendi | Judge Win % vs SFT | Notes |
|-----|------|--------|------|---------|------|---|--------|-----------|------------|-----|-------|-------|--------------------|-------|
| _none yet_ | | | | | | | | | | | | | | |

---

## Eval metrics

- **Eval loss** — DPO/ORPO loss on the 2% held-out split
- **Reward accuracy** — pairwise chosen>rejected agreement (`evaluate_reward_model.py`)
- **Diversity** — EAD / SBERT / Vendi on sampled responses; compare to `baseline_diversity` to detect mode collapse
- **Judge win %** — LLM-judge head-to-head win rate of the DPO model vs the SFT model

---

<!-- Per-experiment template:

## Exp-NNN: <run_id>

**Date:**
**Hypothesis:**
**Method / Base / Dataset:**
**Hardware / Training time:**

**Config:**
| Parameter | Value |
|---|---|
| Method | dpo / orpo |
| beta | 0.1 |
| LoRA rank | 32 |
| Batch size | 32 |
| Grad accumulation | 4 |
| Learning rate | 2e-4 |
| Epochs | 3 |
| Max seq length | 2048 |

**Results:** eval loss, diversity (vs baseline), judge win %, reward acc

**Takeaway:**

**Next experiment:**

**HuggingFace / W&B:**
-->
