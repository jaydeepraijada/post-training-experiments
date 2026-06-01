# DPO Experiment Log

Base model for all runs: `paperbd/smollm_135M_neuraltxt_v1` (the SFT'd assistant).
Preference dataset: `paperbd/paper_preference_150K-v1` unless noted.

## Experiment Plan

| # | Run ID | Description | Key Flags | Status |
|---|--------|-------------|-----------|--------|
| 0 | `baseline_diversity` | Diversity of the SFT model's sampled responses (pre-DPO reference) | `generate_responses.py` → `diversity.py` | ✅ Done |
| 1 | `dpo_default` | DPO, beta=0.1, LoRA r=32 | defaults | ✅ Done |
| 2 | `orpo_default` | ORPO (reference-free), same data | `--method orpo` | ⏳ Planned |
| 3 | `dpo_beta_sweep` | Effect of KL strength: beta 0.1 / 0.3 / 0.5 | `--beta` | ⏳ Planned |
| 4 | `reward_model` | Train + eval a reward model on the pairs | `train_reward.py` / `evaluate_reward_model.py` | ⏳ Planned |

---

## Results

| Run | Date | Method | Base | Dataset | beta | r | Epochs | Eval Loss | Reward Acc | EAD | SBERT | Vendi | Judge Win % vs SFT | Notes |
|-----|------|--------|------|---------|------|---|--------|-----------|------------|-----|-------|-------|--------------------|-------|
| baseline_diversity | 2026-06-01 | — (SFT) | neuraltxt_v1 | — | — | — | — | — | 0.1173 | 0.2263 | 2.7327 | — | Pre-DPO reference |
| dpo_default | 2026-06-01 | DPO | neuraltxt_v1 | paper_preference_150K-v1 | 0.1 | 32 | 3 | 0.4573 | 0.723 | 0.1193 | 0.2322 | 2.7410 | (deferred) | No collapse; 3090 ~6h; batch8×ga16 seq1024 |

---

## Eval metrics

- **Eval loss** — DPO/ORPO loss on the 2% held-out split
- **Reward accuracy** — pairwise chosen>rejected agreement (`evaluate_reward_model.py`)
- **Diversity** — EAD / SBERT / Vendi on sampled responses; compare to `baseline_diversity` to detect mode collapse
- **Judge win %** — LLM-judge head-to-head win rate of the DPO model vs the SFT model

---

## Exp-001: dpo_default

**Date:** 2026-06-01
**Hypothesis:** DPO on LLM-ranked pairs nudges the SFT model toward higher-quality responses. Since the SFT ceiling was capacity (not data), gains may be modest — DPO sharpens preference, it doesn't add knowledge. Watch for diversity collapse.
**Method / Base / Dataset:** DPO · `paperbd/smollm_135M_neuraltxt_v1` · `paperbd/paper_preference_150K-v1` (117,600 train pairs after filtering, 2,400 eval)
**Hardware / Training time:** RTX 3090 (24GB) · 6h17m · 2,757 steps (full 3 epochs, early-stopping did not trigger)

**Config:**
| Parameter | Value |
|---|---|
| Method | DPO |
| beta | 0.1 |
| LoRA rank / alpha | 32 / 32 |
| Per-device batch | 8 |
| Grad accumulation | 16 (effective batch 128) |
| Learning rate | 2e-4 (linear decay) |
| Epochs | 3 |
| Max seq / prompt length | 1024 / 768 |
| Trainable params | 9.77M (6.77%) |

**Eval-loss curve:**
| Epoch | eval_loss | reward acc | margins |
|---|---|---|---|
| 0.54 | 0.5771 | 0.625 | 0.46 |
| 1.09 | 0.5218 | 0.671 | 0.88 |
| 1.63 | 0.4785 | 0.708 | 1.15 |
| 2.18 | 0.4653 | 0.713 | 1.56 |
| 2.72 | **0.4573** | 0.722 | 1.62 |
| 3.00 | 0.4576 | 0.723 | 1.65 |

**Diversity (vs baseline) — collapse check:**
| Metric | Baseline (SFT) | dpo_default | Δ |
|---|---|---|---|
| EAD | 0.1173 | 0.1193 | +1.7% |
| SBERT | 0.2263 | 0.2322 | +2.6% |
| Vendi | 2.7327 | 2.7410 | +0.3% |

**Takeaway:**
- **It worked and didn't collapse.** Held-out reward accuracy reached **0.72** (from 0.5 chance), eval loss 0.577 → 0.457, margins 0.46 → 1.65 — the model learned the preference.
- **Diversity fully preserved** (all three metrics flat / marginally up). The gentle β=0.1 update sharpened preference without narrowing the output distribution. No mode collapse.
- Eval loss plateaued by **epoch ~2.7**; 3 epochs is the right length here (epoch-3 tick-up is noise; best checkpoint loaded via `load_best_model_at_end`).
- Both chosen and rejected rewards drift negative (KL drift from ref); margin growth is what matters and it's healthy.
- **Mild overfitting:** train reward accuracy ended ~0.85 vs eval 0.72 (train loss ~0.25 vs eval 0.457). A real but moderate gap — eval still improved to the end, so not harmful, but it caps the benefit of training longer.

**Next experiment:** `reward_model` for an independent quality signal, or `orpo_default` / `dpo_beta_sweep`. The diversity headroom suggests safety isn't the constraint; the train/eval gap is — so **more epochs would overfit, not help**. A **lower β** (let the policy move further) is the lever to try for more quality, watching the gap. Win-rate judge still deferred (redundant with the ranking judge that made the data).

**HuggingFace:** `paperbd/smollm_135M_neuraltxt_dpo_v1` (merged 16-bit)
**W&B:** none (report_to="none")

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
