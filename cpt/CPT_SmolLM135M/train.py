try:
    from unsloth import FastLanguageModel
    from unsloth.trainer import UnslothTrainer, UnslothTrainingArguments
except:
    print("cant import unsloth")
import argparse
import json
import os
import torch
import wandb
from datasets import load_dataset, interleave_datasets
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig

SEED = 3407

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a model using Unsloth (if CUDA) or standard HF (if not).")
    parser.add_argument("--base_model_id", "-i", type=str, default="HuggingFaceTB/SmolLM-135M", help="Base model ID from Hugging Face.")
    parser.add_argument("--output_model_id", "-o", type=str, default="cpt_arxiv", help="ID for the new fine-tuned model.")
    parser.add_argument("--dataset_path", "-d", type=str, required=True, help="Path to the dataset in JSONL format.")
    parser.add_argument("--test_dataset_path", "-td", type=str, required=True, help="Path to the test dataset in JSONL format for evaluation.")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length.")
    parser.add_argument("--load_in_4bit", action="store_true", default=True, help="Load model in 4-bit precision (CUDA only).")
    parser.add_argument("--full_training", "-ft", action="store_true", help="Enable full training mode (no LoRA/PEFT).")
    parser.add_argument("--split_by_words", type=float, default=0.5, help="Word level split ratio of max_seq_length. Default 0.5.")
    parser.add_argument("--batch_size", "-bs", type=int, default=32)
    parser.add_argument("--epochs", "-e", type=int, default=10)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--learning_rate", "-lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--mix", action="store_true", help="Mix in scientific_papers arxiv data.")
    parser.add_argument("--mix_ratio", type=float, default=0.8, help="Fraction of custom data in the mix (default 0.8 = 80%% custom, 20%% HF).")
    parser.add_argument("--rslora", action="store_true", help="Use rank-stabilized LoRA (rsLoRA).")
    parser.add_argument("--wandb_project", type=str, default="cpt-smollm135m", help="W&B project name.")
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging.")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint in output_dir.")
    args = parser.parse_args()

    train_dataset = load_dataset("json", data_files=args.dataset_path, split="train")
    eval_dataset = load_dataset("json", data_files=args.test_dataset_path, split="train")

    if args.mix:
        print("Mixing in 20% scientific_papers arxiv data...")
        general_data = load_dataset("allenai/peS2o", split="train", trust_remote_code=True)

        def normalize_schema(example):
            return {"text": example.get("text", "") or example.get("article", "")}

        general_data = general_data.map(normalize_schema, remove_columns=general_data.column_names)
        hf_ratio = 1.0 - args.mix_ratio
        train_dataset = interleave_datasets(
            [train_dataset, general_data],
            probabilities=[args.mix_ratio, hf_ratio],
            seed=SEED,
        )
        print(f"Mixed dataset: {args.mix_ratio*100:.0f}% custom, {hf_ratio*100:.0f}% HF scientific_papers")

    if args.split_by_words > 0:
        chunk_size = int(args.max_seq_length * args.split_by_words)
        overlap_ratio = 0.2
        step_size = int(chunk_size * (1 - overlap_ratio))
        print(f"Chunking: chunk_size={chunk_size} words, step={step_size} words...")

        def chunk_text(examples):
            all_chunks = []
            for text in examples["text"]:
                words = text.split()
                for i in range(0, len(words), step_size):
                    chunk = words[i : i + chunk_size]
                    if len(chunk) > 10:
                        all_chunks.append(" ".join(chunk))
            return {"text": all_chunks}

        train_dataset = train_dataset.map(chunk_text, batched=True, remove_columns=train_dataset.column_names)
        eval_dataset = eval_dataset.map(chunk_text, batched=True, remove_columns=eval_dataset.column_names)
        print(f"Train: {len(train_dataset)} chunks, Eval: {len(eval_dataset)} chunks")

    base_lora_config = dict(
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_r,
        lora_dropout=0,
        bias="none",
    )

    if args.full_training:
        lr = args.learning_rate or 1e-5
        max_grad_norm = 0.7
        neftune_noise_alpha = 5
    else:
        lr = args.learning_rate or 2e-4
        max_grad_norm = 1.0
        neftune_noise_alpha = None

    print(f"LR={lr}, grad_norm={max_grad_norm}, NEFTune={neftune_noise_alpha}")

    train_config = {
        "base_model": args.base_model_id,
        "lora_r": 0 if args.full_training else args.lora_r,
        "rslora": args.rslora,
        "max_seq_length": args.max_seq_length,
        "batch_size": args.batch_size,
        "grad_accum": 2,
        "warmup_steps": 100,
        "epochs": args.epochs,
        "learning_rate": lr,
        "full_training": args.full_training,
        "mix": args.mix,
        "mix_ratio": args.mix_ratio if args.mix else None,
        "max_grad_norm": max_grad_norm,
    }

    wandb_enabled = not args.no_wandb
    wandb_run_id = None
    if wandb_enabled:
        existing_config_path = f"models/{args.output_model_id}/train_config.json"
        existing_run_id = None
        if args.resume and os.path.exists(existing_config_path):
            with open(existing_config_path) as f:
                existing_run_id = json.load(f).get("wandb_run_id")
        run = wandb.init(
            project=args.wandb_project,
            name=args.output_model_id,
            config=train_config,
            job_type="train",
            id=existing_run_id,
            resume="must" if existing_run_id else None,
        )
        wandb_run_id = run.id
        train_config["wandb_run_id"] = wandb_run_id

    common_training_args = dict(
        output_dir=f"models/{args.output_model_id}",
        save_total_limit=10,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        save_steps=20,
        gradient_accumulation_steps=2,
        warmup_steps=100,
        max_length=args.max_seq_length,
        learning_rate=lr,
        packing=True,
        dataset_num_proc=2,
        dataset_text_field="text",
        seed=SEED,
        logging_steps=1,
        max_grad_norm=max_grad_norm,
        neftune_noise_alpha=neftune_noise_alpha,
        eval_strategy="steps",
        eval_steps=20,
        save_strategy="steps",
        per_device_eval_batch_size=args.batch_size,
        report_to="wandb" if wandb_enabled else "none",
        run_name=args.output_model_id,
    )

    if torch.cuda.is_available():
        print("CUDA detected — using Unsloth.")

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.base_model_id,
            max_seq_length=args.max_seq_length,
            load_in_4bit=not args.full_training,
            full_finetuning=args.full_training,
        )

        if not args.full_training:
            model = FastLanguageModel.get_peft_model(
                model,
                **base_lora_config,
                use_gradient_checkpointing="unsloth",
                random_state=SEED,
                use_rslora=args.rslora,
                loftq_config=None,
            )

        training_args = UnslothTrainingArguments(
            **common_training_args,
            embedding_learning_rate=lr * 0.1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
        )
        trainer = UnslothTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=None,
            args=training_args,
        )
    else:
        print("No CUDA — using standard HuggingFace.")

        tokenizer = AutoTokenizer.from_pretrained(args.base_model_id)
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            args.base_model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

        peft_config = None
        if not args.full_training:
            peft_config = LoraConfig(**base_lora_config, task_type="CAUSAL_LM", use_rslora=args.rslora)

        training_args = SFTConfig(
            **common_training_args,
            eos_token=tokenizer.eos_token,
            pad_token=tokenizer.pad_token,
        )
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_config,
            args=training_args,
        )

    trainer.train(resume_from_checkpoint=args.resume or None)

    if torch.cuda.is_available():
        model.save_pretrained(f"models/{args.output_model_id}/final")
        tokenizer.save_pretrained(f"models/{args.output_model_id}/final")
    else:
        trainer.model.save_pretrained(f"models/{args.output_model_id}/final")
        tokenizer.save_pretrained(f"models/{args.output_model_id}/final")

    os.makedirs(f"models/{args.output_model_id}", exist_ok=True)
    with open(f"models/{args.output_model_id}/train_config.json", "w") as f:
        json.dump(train_config, f, indent=2)

    if wandb_enabled:
        wandb.finish()

    print(f"Saved to models/{args.output_model_id}/final")


if __name__ == "__main__":
    main()
