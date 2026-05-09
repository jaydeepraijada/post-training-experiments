# SFT Experiment Log

## Experiment Plan

| # | Run ID | Description | Key Flags | Status |
|---|--------|-------------|-----------|--------|
| 1 | `exp01_cpt_sft` | SFT on CPT-adapted SmolLM-135M, default config | defaults | ✅ Done |
| 2 | `exp02_base_sft` | SFT on raw SmolLM-135M base (no CPT) | `--base_model_id HuggingFaceTB/SmolLM-135M` | ⏳ Planned |
| 3 | `exp03_lora_r16` | LoRA r=16 vs r=32 | `--lora_r 16` | ⏳ Planned |
| 4 | `exp04_custom_data` | SFT on custom synthetic data (data_prep pipeline) | `--dataset custom` | ⏳ Planned |

---

## Results

| Run | Date | Base Model | Dataset | LoRA r | Epochs | Train Loss | Eval Loss | Faithfulness | Answer Correctness | Relevance | Completeness | Overall | Notes |
|-----|------|------------|---------|--------|--------|------------|-----------|--------------|-------------------|-----------|--------------|---------|-------|
| exp01_cpt_sft | 2026-05-09 | paperbd/smollm_135M_arxiv_cpt | paperbd/paper_instructions_300K-v1 | 32 | 3 | 0.087 | 1.222 | 2.70 | 1.98 | 3.04 | 1.85 | 2.39 | First SFT run, ~10hr on RTX 4090 |

---

## Exp-001: exp01_cpt_sft

**Date:** 2026-05-09
**Hypothesis:** SFT on the CPT-adapted model teaches instruction-following for ML paper research tasks
**Base model:** `paperbd/smollm_135M_arxiv_cpt` (CPT LoRA r=32, merged)
**Dataset:** `paperbd/paper_instructions_300K-v1` (300K synthetic instruction pairs, variations=2 → 600K effective)
**Hardware:** NVIDIA RTX 4090 (24GB)
**Training time:** ~10 hours (including frequent checkpointing overhead at save_steps=50)

**Config:**
| Parameter | Value |
|---|---|
| LoRA rank | 32 |
| LoRA alpha | 32 |
| Batch size | 32 |
| Gradient accumulation | 4 (effective batch: 128) |
| Learning rate | 2e-4 (linear decay) |
| Warmup ratio | 0.03 |
| Epochs | 3 |
| Total steps | 11,355 |
| Max sequence length | 2048 |
| Packing | True |
| Optimizer | adamw_8bit |

**Results:**
| Metric | Value |
|---|---|
| Train loss | 0.087 |
| Eval loss | 1.222 |
| Faithfulness | 2.70 / 5 |
| Answer Correctness | 1.98 / 5 |
| Relevance | 3.04 / 5 |
| Completeness | 1.85 / 5 |
| Overall | 2.39 / 5 |

**Judge model:** grok-3-mini via xAI API, 1000 test samples

**Takeaway:**
- Model learned instruction format well (train loss 0.087) but generalisation is limited (eval loss 1.222 — large gap)
- Relevance (3.04) is the strongest dimension — model stays on topic
- Answer correctness (1.98) and completeness (1.85) are weak — 135M params struggle to recall and reproduce factual content accurately
- No comparison against base model yet — needed to quantify SFT contribution

**Next experiment:** Run exp02 (SFT on raw base model) to isolate CPT contribution. Run exp03 (LoRA r=16) to check if lower rank generalises better.

**HuggingFace:** `JaydeepR/SmolLM-135M-SFT-exp01`
**W&B run:** `exp01_cpt_sft` in project `sft-smollm135m`
