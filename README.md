# post-training-experiments

Experiments in post-training small LLMs. Each subfolder is a self-contained experiment.

## Current Experiments

### CPT SmolLM-135M
Domain-adapting [SmolLM-135M](https://huggingface.co/HuggingFaceTB/SmolLM-135M) on arXiv ML papers via continued pre-training.

**Best model:** [`jaydeepraijada/SmolLM-135M-CPT-LoRA-r32`](https://huggingface.co/jaydeepraijada/SmolLM-135M-CPT-LoRA-r32)

## The Experiment Loop

Every experiment follows this loop:

```
Set up Dataset + Eval Harness
         ↓
    Run Experiment
         ↓
   Track Metrics
         ↓
Hypothesize & Plan Next
         ↑_____________|
```

## Results Summary

| Model | Perplexity | ROUGE-1 | BERTScore F1 | vs Base |
|---|---|---|---|---|
| SmolLM-135M (Base) | 22.97 | 0.178 | 0.736 | — |
| Full Fine-Tuning (bf16) | 22.88 | 0.170 | 0.737 | ❌ worse on ROUGE |
| LoRA r=8 | 18.42 | 0.209 | 0.748 | +17.4% ROUGE |
| LoRA r=16 | 18.39 | 0.207 | 0.752 | +16.3% ROUGE |
| **LoRA r=32** | **18.36** | **0.213** | **0.753** | **+19.7% ROUGE ✦ best** |
| Interleaved + LoRA r=16 | 18.40 | 0.204 | 0.753 | +14.6% ROUGE |
| Interleaved + rsLoRA r=16 | 18.58 | 0.213 | 0.756 | +19.7% ROUGE |

## Key Learnings

1. **LoRA beats full fine-tuning on small datasets** — regularisation matters more than capacity at 138 papers
2. **Rank doesn't matter much** (r=8/16/32 all plateau at the same loss) — data is the bottleneck
3. **Interleaving with large HF dataset didn't help** — `stopping_strategy="first_exhausted"` limited exposure to only ~4k HF papers/epoch
4. **`all_exhausted` killed domain signal** — 80% HF data + 3000 steps → model forgot domain, worse than base
5. **Next iteration:** 500-1000 papers with the best config (LoRA r=32)

## Quickstart

```bash
# 1. Download papers
python data_prep/download_arxiv_files.py

# 2. Build dataset
python data_prep/get_dataset.py papers/

# 3. Train
python train.py -d cpt_train_dataset_138.jsonl -td cpt_val_dataset_50.jsonl \
    -o my_run --lora_r 32 --wandb_project cpt-smollm135m

# 4. Eval
python inference.py --models models/my_run/final \
    --dataset cpt_val_dataset_50.jsonl --num_samples 50 \
    --output_results results.json

# 5. Log
python log_experiment.py --results results.json --run_name my_run \
    --hypothesis "what I tested" --notes "what I observed"

# 6. Qualitative comparison
python compare.py --models HuggingFaceTB/SmolLM-135M models/my_run/final \
    --names "Base" "CPT" --dataset cpt_val_dataset_50.jsonl
```

## Stack

`PyTorch` · `Transformers` · `Unsloth` · `TRL` · `PEFT` · `W&B` · `Python 3.12`

## Adding New Experiments

Add as a sibling folder: `cpt/CPT_SmolLM360M/` or a new top-level dir like `sft/`.
