# post-training-experiments

Experiments in post-training small LLMs. Each subfolder is a self-contained experiment with its own `pyproject.toml`.

## Structure

```
post-training-experiments/
└── cpt/
    └── CPT_SmolLM135M/   — continued pre-training on SmolLM-135M
```

## CPT_SmolLM135M

**Goal:** Domain-adapt SmolLM-135M (135M params) on arxiv ML papers via continued pre-training.

**Stack:** PyTorch, HuggingFace Transformers, Unsloth, TRL, PEFT

**Training config:**
- LoRA r=32, 4-bit quantization
- batch=32, grad_accum=2, seq_len=512, epochs=10
- Chunking: 256-word chunks with 20% overlap, packed to 512 tokens
- Estimated runtime on RTX 4090: ~10-20 min for ~150 papers

**Workflow:**
```bash
# 1. Download papers
python data_prep/download_arxiv_files.py

# 2. Build train/test splits (cleaned by default; use --no_clean to skip)
python data_prep/get_dataset.py papers/
python data_prep/get_dataset.py papers/ --no_clean   # for uncleaned variant

# 3. Train
python train.py -d cpt_train_dataset_*.jsonl -td cpt_val_dataset_50.jsonl -o my_run

# 4. Inference + evals
python inference.py --models models/my_run/final --dataset cpt_val_dataset_50.jsonl \
    --output_results results.json --wandb_project cpt-smollm135m --wandb_run_id <id>

# 5. Log experiment
python log_experiment.py --results results.json --run_name my_run \
    --hypothesis "what I was testing" --notes "what I observed"

# 6. (Optional) TIES merge after full fine-tuning
python merge.py --finetuned models/my_run/final --output models/my_run_ties \
    --density 0.3 --weight 1.0
```

**Eval metrics:** perplexity, cross-entropy, ROUGE-1/2/L, BLEU, BERTScore F1

**Qualitative comparison (final step):**
```bash
python compare.py \
    --models HuggingFaceTB/SmolLM-135M models/exp01_full_ft_bf16/final models/<best>/final \
    --names base_model worst_model best_model \
    --dataset cpt_val_dataset_50.jsonl --num_prompts 3 \
    --output comparison_results.json
```

**Key flags:**
- `--full_training` / `-ft`: disable LoRA, train all weights (uses lr=1e-5, NEFTune)
- `--lora_r`: LoRA rank (8 / 16 / 32)
- `--rslora`: rank-stabilised LoRA
- `--mix`: interleave HF scientific_papers dataset
- `--mix_ratio 0.2`: 20% custom, 80% HF (default 0.8 = 80% custom)
- `--no_wandb`: skip W&B logging

**Experiment tracking:**
- `experiments.md` — human-readable planned + results log
- `experiments.json` — machine-readable completed runs
- W&B project: `cpt-smollm135m`

## Adding new experiments

Add as a sibling folder, e.g. `cpt/CPT_SmolLM360M/` or a new top-level dir like `sft/`.
