# RL_SmolLM135M

GRPO on SmolLM-135M: teach the reasoning base model to produce better `<think>…</think>` + answer responses, scored by a learned reward model plus format, schema, doom-loop, and length checks. Ported from [finetuning_recipes](https://github.com/avbiswas/finetuning_recipes).

## Layout

| Path | What |
|---|---|
| `reasoning/env.py`, `schema_match.py` | Reward function shared by training and evals |
| `reasoning/grpo/` | GRPO trainer, rollouts, loss, and YAML configs |
| `reasoning/eval_*.py`, `*_head_to_head.py` | Checkpoint evals |
| `reasoning/EVALUATION.md` | Reward design, diagnosis, and results from upstream |
| `reward_models/` | Reward model training and `REWARD_MODEL_FINDINGS.md` |
| `data_prep/` | Reasoning data, rollouts, confounds, and reward benchmarks |
| `merge_adapter.py`, `mlx_export.py` | Merge a checkpoint and convert it to MLX |

## Quickstart

```bash
uv sync                                  # GRPO only; add --extra warmup/dataprep/mlx as needed
.venv/bin/python -m unittest tests/reasoning/test_*.py reasoning/tests/test_schema_match.py

# Tiny end-to-end check
.venv/bin/python reasoning/grpo/train.py reasoning/grpo/smoke.yaml -o smoke

# Train on a shared machine (memory cap + watchdog)
./run_guarded.sh reasoning/grpo/dgx_spark.yaml grpo_dgx_v1
```

Run everything from this folder. Checkpoints and logs go to `models/<output_model_id>/`: `llm` (latest), `llm_best_<n>` (best eval reward so far), `logs/train_metrics.jsonl`, and `logs/eval_generations_*.json`.

## Changes from upstream

- Configs use the public `paperbd/neuraltxt-v1-135M-reasoning-base-v2` and `paperbd/paper_instructions_300K-v1`; the `-v2` dataset is not publicly accessible.
- `train.py` falls back to `sdpa` when flash-attn isn't installed, honours `max_gpu_memory_gb`, and saves a final checkpoint.
- `reasoning/grpo/dgx_spark.yaml` and `run_guarded.sh` are new, for the shared DGX Spark.
- Some upstream reward datasets (for example `reward_model_dataset_clean.jsonl`) aren't published, so `data_prep/eval_reward_models.py` can't run as-is.
