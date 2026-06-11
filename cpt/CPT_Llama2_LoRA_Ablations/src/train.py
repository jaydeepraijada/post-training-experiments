"""Single parametrized CPT-LoRA training run.

    python src/train.py --config configs/06_rslora_decoupled.yaml --seed 3407

Everything that defines the experiment lives in the YAML config. This script is
identical for every ablation; the only thing that changes is the config passed in.
That is the whole point — one code path, many configs.
"""
from __future__ import annotations

import os
# Reduce CUDA fragmentation on tight-VRAM GPUs (e.g. 14.5 GB T4). Must be set
# before the CUDA context initialises, i.e. before importing torch/unsloth.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Unsloth MUST be imported before transformers/trl so its patches apply.
from unsloth import (
    FastLanguageModel,
    UnslothTrainer,
    UnslothTrainingArguments,
    is_bfloat16_supported,
)

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from typing import Any

import torch
from transformers import TrainingArguments
from trl import SFTTrainer

sys.path.insert(0, os.path.dirname(__file__))
from config import RunConfig  # noqa: E402
from data import build_datasets  # noqa: E402
from eval import evaluate_perplexity  # noqa: E402

RESULTS_CSV = os.path.join(os.path.dirname(__file__), "..", "results", "runs.csv")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to an experiment YAML config")
    p.add_argument("--seed", type=int, default=None, help="Override config seed (for multi-seed runs)")
    p.add_argument("--max-steps", type=int, default=None, help="Override max training steps (budget knob)")
    p.add_argument("--epochs", type=float, default=None, help="Train by epochs instead of steps (overrides --max-steps)")
    p.add_argument("--wandb-project", default="cpt-lora-ablations")
    p.add_argument("--no-wandb", action="store_true", help="Disable W&B (offline/CSV-only)")
    p.add_argument("--output-dir", default="outputs")
    return p.parse_args()


def tokens_per_step(cfg: RunConfig) -> int:
    """Upper-bound tokens seen per optimizer step. With packing=True sequences are
    filled to max_seq_length, so this is accurate; without packing it's an upper bound."""
    return (
        cfg.per_device_train_batch_size
        * cfg.gradient_accumulation_steps
        * cfg.max_seq_length
    )


def build_trainer(cfg: RunConfig, model, tokenizer, train_ds, val_ds, report_to, run_name,
                  output_dir, max_steps, num_train_epochs):
    common: dict[str, Any] = dict(
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        max_steps=max_steps,                  # -1 when training by epochs
        num_train_epochs=num_train_epochs,    # ignored when max_steps > 0
        warmup_steps=cfg.warmup_steps,
        learning_rate=cfg.learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim=cfg.optim,
        weight_decay=cfg.weight_decay,
        lr_scheduler_type=cfg.lr_scheduler_type,
        seed=cfg.seed,
        output_dir=output_dir,
        report_to=report_to,
        run_name=run_name,
        # We evaluate once at the very end (see evaluate_perplexity); no mid-run eval.
        eval_strategy="no",
        # No checkpoints: the run's artifact is the runs.csv row, not adapter
        # weights. Also avoids pickling unsloth's patched SFTConfig on save.
        save_strategy="no",
        per_device_eval_batch_size=cfg.per_device_train_batch_size,
    )

    trainer_kwargs = dict(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="text",
        max_seq_length=cfg.max_seq_length,
        dataset_num_proc=8,
        packing=cfg.packing,
    )

    if cfg.uses_decoupled_lr:
        # UnslothTrainer adds the decoupled embedding_learning_rate knob.
        args = UnslothTrainingArguments(embedding_learning_rate=cfg.embedding_learning_rate, **common)
        return UnslothTrainer(args=args, **trainer_kwargs)
    else:
        args = TrainingArguments(**common)
        return SFTTrainer(args=args, **trainer_kwargs)


def append_result(row: dict):
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    new_file = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            w.writeheader()
        w.writerow(row)


def main():
    args = parse_args()
    cfg = RunConfig.from_file(args.config, seed=args.seed, max_steps=args.max_steps)

    # Budget knob: train by epochs (max_steps=-1) or by a fixed step count.
    if args.epochs is not None:
        num_train_epochs = args.epochs
        max_steps = -1
    else:
        num_train_epochs = 1  # unused when max_steps > 0
        max_steps = cfg.max_steps

    run_name = f"{cfg.name}_seed{cfg.seed}"
    output_dir = os.path.join(args.output_dir, run_name)

    report_to = "none" if args.no_wandb else "wandb"
    if not args.no_wandb:
        import wandb

        wandb.init(project=args.wandb_project, name=run_name, config=vars(cfg))

    print(f"=== Run: {run_name} ===")
    print(json.dumps(vars(cfg), indent=2, default=str))

    # --- Model + LoRA ---
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model_name,
        max_seq_length=cfg.max_seq_length,
        dtype=None,
        load_in_4bit=cfg.load_in_4bit,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.r,
        target_modules=cfg.target_modules,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        use_gradient_checkpointing=cfg.use_gradient_checkpointing,
        random_state=cfg.random_state,
        use_rslora=cfg.use_rslora,
        loftq_config=None,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # --- Data ---
    train_ds, val_ds = build_datasets(cfg.dataset_name, tokenizer, cfg.val_size, cfg.seed)

    # --- Train ---
    trainer = build_trainer(cfg, model, tokenizer, train_ds, val_ds, report_to, run_name,
                            output_dir, max_steps, num_train_epochs)
    train_stats = trainer.train()

    # --- Held-out eval (the real metric) ---
    eval_metrics = evaluate_perplexity(trainer)
    peak_mem = round(torch.cuda.max_memory_reserved() / 1e9, 3) if torch.cuda.is_available() else 0.0

    # Tokens actually seen during training (steps reported by the trainer × tokens/step).
    steps_done = int(train_stats.global_step) if hasattr(train_stats, "global_step") else max_steps
    tokens_trained = steps_done * tokens_per_step(cfg)

    row = {
        "name": cfg.name,
        "seed": cfg.seed,
        "trainable_params": trainable,
        "steps": steps_done,
        "tokens_trained": tokens_trained,
        "train_loss": round(float(train_stats.metrics.get("train_loss", float("nan"))), 4),
        "eval_loss": round(eval_metrics["eval_loss"], 4),
        "eval_perplexity": round(eval_metrics["eval_perplexity"], 4),
        "train_runtime_s": round(float(train_stats.metrics.get("train_runtime", 0.0)), 1),
        "peak_vram_gb": peak_mem,
        "use_rslora": cfg.use_rslora,
        "learning_rate": cfg.learning_rate,
        "embedding_learning_rate": cfg.embedding_learning_rate,
        "n_target_modules": len(cfg.target_modules),
        "packing": cfg.packing,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    print(f"Tokens trained: {tokens_trained:,} ({steps_done} steps x {tokens_per_step(cfg):,} tok/step)")
    append_result(row)
    print("=== RESULT ===")
    print(json.dumps(row, indent=2))

    if not args.no_wandb:
        import wandb

        wandb.log({k: row[k] for k in ("eval_loss", "eval_perplexity", "trainable_params", "peak_vram_gb")})
        wandb.finish()


if __name__ == "__main__":
    main()
