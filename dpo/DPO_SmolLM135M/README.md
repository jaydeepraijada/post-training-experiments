# DPO_SmolLM135M

Preference optimization (DPO / ORPO) of the SFT'd SmolLM-135M paper-research assistant. Aligns the model on pairwise preferences ranked by an LLM judge, and tracks output diversity to watch for mode collapse.

This is stage 3 of the post-training loop: **CPT → SFT → DPO → (RLVR)**.

## Pipeline Overview

```
SFT model  (paperbd/smollm_135M_neuraltxt_v1)
       │
       ▼
generate_responses.py     sample K responses per prompt (temperature sweep)
       │                   out: {id, question, responses:[...], ground_truth}
       │
       ├────────► diversity.py        EAD / SBERT / Vendi  ── DIAGNOSTIC (is there
       │                              enough variation to mine preferences?)
       ▼
generate_ranked_pairs_openai.py   LLM judge ranks best→worst (+ garbage filter)
       │                   out: + {ranking_list, ignore flags}
       ▼
build_dpo_dataset.py      4-way ranking → pairs (1v4, 2v4, 1v3, 1v2)
       │                   out: {prompt, chosen, rejected}
       │                   (== paperbd/paper_preference_150K-v1)
       ▼
train_preference.py       DPO or ORPO, LoRA r=32, Unsloth + TRL
       │
       ▼
CHECK   re-run diversity.py + LLM judge on the DPO'd model vs the SFT model
```

Optional reward-model branch (for analysis / future RLHF):
`train_reward.py` → `evaluate_reward_model.py` (pairwise preference accuracy).

## Why diversity eval matters here

DPO only learns from the *gap* between `chosen` and `rejected`. Two roles:

1. **Before** — if the K sampled responses are near-identical, the pairs are degenerate and there's no signal. Diversity eval gates the sampling temperature.
2. **After** — DPO is prone to reducing diversity / mode collapse. Re-run the same eval on the tuned model to confirm we didn't over-optimize, even if judge win-rate improved.

## Stack

`PyTorch` · `Transformers` · `Unsloth` · `TRL` · `PEFT` · `sentence-transformers` · `Outlines` · `Python 3.12`

## Quickstart

Run all commands **from inside this folder** (`dpo/DPO_SmolLM135M/`).

```bash
# Install
uv sync                       # add --extra mlx on Apple Silicon

# 0. (baseline) sample responses + score diversity on the SFT model
uv run python diversity_evals/generate_responses.py -m paperbd/smollm_135M_neuraltxt_v1 \
    --num_samples 100 --n_responses 4 --temperatures 0.3 0.5 0.7 1.0
uv run python diversity_evals/diversity.py -i evals/<responses>.jsonl

# 1. rank candidate responses into preferences (needs OPENROUTER_API_KEY)
uv run python generate_ranked_pairs_openai.py -i evals/pref_dataset/train_4r_temp0.5.jsonl

# 2. build pairwise DPO dataset
uv run python build_dpo_dataset.py -i evals/pref_dataset/train_4r_temp0.5_ranked.jsonl

# 3. train (DPO or ORPO)
uv run python train_preference.py --method dpo -o dpo_default \
    --dataset paperbd/paper_preference_150K-v1

# 4. check: diversity + reward-model accuracy
uv run python diversity_evals/diversity.py -i evals/<dpo_responses>.jsonl
uv run python evaluate_reward_model.py --model models/preference/<reward>/final
```

## Tracking

- `experiments.md` — planned experiments + results
- `LEARNINGS.md` — running log of insights
- Best models published under `paperbd/` on the HuggingFace Hub
