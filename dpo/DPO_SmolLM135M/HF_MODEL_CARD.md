---
license: apache-2.0
base_model: paperbd/smollm_135M_neuraltxt_v1
datasets:
  - paperbd/paper_preference_150K-v1
library_name: transformers
pipeline_tag: text-generation
tags:
  - dpo
  - preference-optimization
  - trl
  - unsloth
  - smollm
---

# SmolLM-135M-neuraltxt-dpo-v1

Preference-tuned (DPO) version of the SFT'd SmolLM-135M ML-paper research assistant. This is **stage 3** of a CPT → SFT → DPO pipeline on a 135M-parameter model.

- **Base (SFT):** [`paperbd/smollm_135M_neuraltxt_v1`](https://huggingface.co/paperbd/smollm_135M_neuraltxt_v1)
- **Preference data:** [`paperbd/paper_preference_150K-v1`](https://huggingface.co/datasets/paperbd/paper_preference_150K-v1) — pairs mined by sampling the SFT model and ranking responses with an LLM judge.
- **Method:** DPO (Direct Preference Optimization), β=0.1, LoRA r=32, merged to 16-bit.

## Training

| | |
|---|---|
| Method | DPO (TRL + Unsloth) |
| beta | 0.1 |
| LoRA rank / alpha | 32 / 32 |
| Effective batch | 128 (8 × grad_accum 16) |
| Max seq / prompt length | 1024 / 768 |
| Learning rate | 2e-4, linear decay |
| Epochs | 3 (2,757 steps) |
| Hardware | 1× RTX 3090, ~6h17m |

## Evaluation

Held-out 2% split + diversity on 100 sampled prompts (4 responses × 4 temperatures).

| Metric | SFT baseline | This model (DPO) |
|---|---|---|
| Eval loss | — | **0.457** |
| Reward accuracy (held-out) | 0.50 (chance) | **0.72** |
| Reward margin | — | 1.65 |
| Diversity — EAD | 0.1173 | 0.1193 |
| Diversity — SBERT | 0.2263 | 0.2322 |
| Diversity — Vendi | 2.7327 | 2.7410 |

**Takeaways:** the model learned the preference (reward accuracy 0.50 → 0.72) while **preserving output diversity** (no mode collapse — all diversity metrics flat vs the SFT baseline). Training shows mild overfitting (train reward accuracy ~0.85 vs eval 0.72), so 3 epochs is the right length.

## Intended use & limitations

- Same scope as the SFT base: a **structured ML-paper research assistant**, not a general chatbot. Best used via the `PaperResearcher` task API from the SFT stage.
- At 135M parameters the model is **capacity-limited** — it learns task *shape* and preference, not deep factual recall. DPO sharpens which response style is preferred; it does not add knowledge.
- The reward/eval accuracy measures agreement with the LLM judge that created the preference data, so it is **not a fully independent quality signal**.

## Reproduce

See [`dpo/DPO_SmolLM135M`](https://github.com/jaydeepraijada/post-training-experiments/tree/main/dpo/DPO_SmolLM135M) (`run_dpo.sh`, `experiments.md`, `LEARNINGS.md`).
