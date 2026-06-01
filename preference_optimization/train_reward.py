"""
Reward model training with TRL + PEFT.

Works on CUDA, Apple Silicon (MPS), and CPU. Auto-detects device.

Usage:
    uv run python preference_optimization/train_reward.py

    uv run python preference_optimization/train_reward.py \
        --base_model_id paperbd/smollm_135M_neuraltxt_v1 \
        --output_model_id reward_model_v1 \
        --epochs 3
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from preference_optimization.chat_formatting import build_reward_conversations
    from preference_optimization.trl_compat import patch_trl_optional_dependency_checks
except ModuleNotFoundError:
    from chat_formatting import build_reward_conversations
    from trl_compat import patch_trl_optional_dependency_checks

from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer
from trl import RewardConfig, RewardTrainer

SEED = 3407


def detect_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"


def parse_args() -> argparse.Namespace:
    default_device = detect_device()

    p = argparse.ArgumentParser(description="Reward model training with TRL + PEFT.")
    p.add_argument(
        "--base_model_id",
        type=str,
        default="paperbd/smollm_135M_neuraltxt_dpo_v1",
    )
    p.add_argument(
        "--output_model_id",
        "-o",
        type=str,
        default="reward_model_v1",
    )
    p.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="paperbd/paper_preference_150K-v1",
        help="Local JSONL file or HF dataset name.",
    )
    p.add_argument("--max_seq_length", type=int, default=1024)
    p.add_argument("--batch_size", "-bs", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--epochs", "-e", type=int, default=3)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--learning_rate", "-lr", type=float, default=1e-4)
    p.add_argument(
        "--device",
        type=str,
        default=default_device,
        choices=["auto", "cuda", "mps", "cpu"],
    )
    p.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Cap dataset size for quick tests.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    patch_trl_optional_dependency_checks()

    device = detect_device()
    dtype = torch.bfloat16

    print(f"Device: {device}  Dtype: {dtype}")
    print(f"Base model: {args.base_model_id}")
    print(f"Dataset: {args.dataset}")
    print(f"LoRA r={args.lora_r}  alpha={args.lora_alpha}  lr={args.learning_rate}")
    print(f"Batch size: {args.batch_size}  Grad accum: {args.grad_accum}")

    # ── Dataset ──────────────────────────────────────────────────────────

    dataset = load_dataset(args.dataset, split="train")

    split = dataset.train_test_split(test_size=0.05, seed=SEED)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    if args.max_samples is not None:
        train_dataset = train_dataset.select(
            range(min(args.max_samples, len(train_dataset)))
        )
        eval_dataset = eval_dataset.select(
            range(min(args.max_samples // 10, len(eval_dataset)))
        )

    print(f"Train: {len(train_dataset)}  Eval: {len(eval_dataset)}")

    # ── Pre-tokenize (collator expects plain token-id lists) ──────────────

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id)

    def pre_tokenize(example):
        chosen, rejected = build_reward_conversations(example)
        chosen_ids = tokenizer.apply_chat_template(
            chosen,
            tokenize=True,
            return_dict=False,
        )
        rejected_ids = tokenizer.apply_chat_template(
            rejected,
            tokenize=True,
            return_dict=False,
        )
        return {"chosen_input_ids": chosen_ids, "rejected_input_ids": rejected_ids}

    train_dataset = train_dataset.map(pre_tokenize)
    eval_dataset = eval_dataset.map(pre_tokenize)

    # ── PEFT config ──────────────────────────────────────────────────────

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        modules_to_save=["score"],
    )

    # ── Training config ──────────────────────────────────────────────────

    output_dir = f"models/preference/{args.output_model_id}"

    model_init_kwargs = {
        "torch_dtype": dtype,
        "device_map": device,
    }

    config = RewardConfig(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=0.03,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        logging_steps=10,
        weight_decay=0.001,
        lr_scheduler_type="linear",
        report_to="none",
        seed=SEED,
        bf16=(device == "cuda"),
        fp16=False,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        max_length=args.max_seq_length,
        remove_unused_columns=False,
        model_init_kwargs=model_init_kwargs,
        center_rewards_coefficient=0.01,
        disable_dropout=True,
    )

    # ── Trainer ──────────────────────────────────────────────────────────

    trainer = RewardTrainer(
        model=args.base_model_id,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )

    trainer.train()

    # ── Save ─────────────────────────────────────────────────────────────

    final_dir = f"{output_dir}/final"
    trainer.save_model(final_dir)
    print(f"Model saved to {final_dir}")


if __name__ == "__main__":
    main()
