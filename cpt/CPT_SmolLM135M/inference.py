import argparse
import json
import math
import torch
import random
import os
import glob
from rich.console import Console
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def load_model(model_path, base_model_id="HuggingFaceTB/SmolLM-135M", load_in_4bit=True):
    print(f"Loading model from: {model_path}...")

    if torch.cuda.is_available():
        from unsloth import FastLanguageModel
        print("CUDA detected — using Unsloth.")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=load_in_4bit,
        )
        FastLanguageModel.for_inference(model)
    else:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {device}")

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        except:
            tokenizer = AutoTokenizer.from_pretrained(base_model_id)

        is_adapter = os.path.exists(os.path.join(model_path, "adapter_config.json"))
        if is_adapter:
            print(f"Adapter found — loading base model {base_model_id} first...")
            model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                device_map=device,
                torch_dtype=torch.float16 if device == "mps" else torch.float32,
            )
            model = PeftModel.from_pretrained(model, model_path)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device,
                torch_dtype=torch.float16 if device == "mps" else torch.float32,
            )

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def generate_batch(model, tokenizer, prompts, max_new_tokens=64, batch_size=4, repetition_penalty=1.2):
    all_outputs = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(model.device)
        input_length = inputs.input_ids.shape[1]
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                repetition_penalty=repetition_penalty,
            )
        generated_tokens = outputs[:, input_length:]
        decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        all_outputs.extend([t.strip() for t in decoded])
    return all_outputs


def compute_perplexity(model, tokenizer, prefixes, ground_truths):
    total_loss = 0
    total_tokens = 0
    for prefix, gt in zip(prefixes, ground_truths):
        full_text = prefix + " " + gt
        prefix_ids = tokenizer(prefix, return_tensors="pt").input_ids
        full_ids = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=2048).input_ids.to(model.device)
        prefix_len = prefix_ids.shape[1]
        labels = full_ids.clone()
        labels[0, :prefix_len] = -100
        with torch.no_grad():
            outputs = model(input_ids=full_ids, labels=labels)
        num_gt_tokens = (labels != -100).sum().item()
        total_loss += outputs.loss.item() * num_gt_tokens
        total_tokens += num_gt_tokens
    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss), avg_loss


def main():
    parser = argparse.ArgumentParser(description="Batch inference and evaluation for CPT models.")
    parser.add_argument("--models", nargs='+', required=True, help="Model paths or HF IDs")
    parser.add_argument("--base_model", type=str, default="HuggingFaceTB/SmolLM-135M")
    parser.add_argument("--dataset", type=str, required=True, help="Path to JSONL dataset")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--prefix_len", type=int, default=20, help="Words used as input prefix")
    parser.add_argument("--predict_len", type=int, default=50, help="Words to predict")
    parser.add_argument("--output_json", type=str, default="inference_generations.json")
    parser.add_argument("--output_results", type=str, default=None, help="If set, run evals and save here")
    parser.add_argument("--wandb_project", type=str, default=None, help="W&B project to log eval metrics to.")
    parser.add_argument("--wandb_run_id", type=str, default=None, help="Resume an existing W&B run by ID.")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Name for a new W&B eval run.")
    args = parser.parse_args()

    console = Console()

    console.rule("[bold blue]Preparing Dataset")
    with open(args.dataset, 'r') as f:
        lines = f.readlines()

    random.seed(42)
    random.shuffle(lines)

    all_texts = []
    for line in lines:
        data = json.loads(line)
        words = data.get('text', '').split()
        words = words[:int(len(words) * 0.9)]
        if len(words) >= args.prefix_len + args.predict_len:
            all_texts.append(words)

    samples, prompts, ground_truths = [], [], []
    sample_id = 0
    while len(samples) < args.num_samples and all_texts:
        text_words = random.choice(all_texts)
        max_start = len(text_words) - (args.prefix_len + args.predict_len)
        start_idx = random.randint(0, max_start)
        prefix_words = text_words[start_idx : start_idx + args.prefix_len]
        gt_words = text_words[start_idx + args.prefix_len : start_idx + args.prefix_len + args.predict_len]
        samples.append({"id": sample_id, "prefix": " ".join(prefix_words), "ground_truth": " ".join(gt_words), "predictions": {}})
        prompts.append(" ".join(prefix_words))
        ground_truths.append(" ".join(gt_words))
        sample_id += 1

    console.print(f"Prepared {len(samples)} samples from {len(all_texts)} texts.")

    expanded_model_paths = []
    for pattern in args.models:
        matches = glob.glob(pattern, recursive=True)
        expanded_model_paths.extend(matches if matches else [pattern])
    unique_paths = list(dict.fromkeys(expanded_model_paths))

    perplexities = {}
    cross_entropies = {}

    for model_path in unique_paths:
        console.rule(f"[bold blue]{model_path}")
        if os.path.exists(model_path):
            parts = model_path.rstrip(os.sep).split(os.sep)
            display_name = f"{parts[-2]}/{parts[-1]}" if len(parts) >= 2 else parts[-1]
        else:
            display_name = model_path

        try:
            model, tokenizer = load_model(model_path, args.base_model)
            max_tokens = int(args.predict_len * 1.5)
            predictions = generate_batch(model, tokenizer, prompts, max_new_tokens=max_tokens, batch_size=args.batch_size)
            for sample, pred in zip(samples, predictions):
                sample["predictions"][display_name] = pred

            console.print("Computing perplexity and cross-entropy...")
            ppl, ce = compute_perplexity(model, tokenizer, prompts, ground_truths)
            perplexities[display_name] = ppl
            cross_entropies[display_name] = ce
            console.print(f"Perplexity: {ppl:.4f} | Cross-Entropy: {ce:.4f}")

            del model, tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            console.print(f"[bold red]Error: {e}[/bold red]")
            import traceback
            traceback.print_exc()

    console.rule("[bold blue]Saving Results")
    with open(args.output_json, 'w') as f:
        json.dump(samples, f, indent=4)
    console.print(f"[bold green]Generations saved to {args.output_json}[/bold green]")

    if args.output_results:
        from evals import calculate_metrics
        console.rule("[bold blue]Evaluations")
        model_names = list(samples[0]["predictions"].keys())

        from rich.table import Table
        from rich import box
        table = Table(title="Results", box=box.ROUNDED)
        table.add_column("Model", style="cyan")
        table.add_column("Perplexity", style="blue")
        table.add_column("Cross-Entropy", style="blue")
        table.add_column("ROUGE-1", style="green")
        table.add_column("ROUGE-L", style="green")
        table.add_column("BERTScore F1", style="magenta")

        final_scores = {}
        for model_name in model_names:
            preds = [s["predictions"][model_name] for s in samples if model_name in s["predictions"]]
            refs = [s["ground_truth"] for s in samples if model_name in s["predictions"]]
            scores = calculate_metrics(preds, refs)
            scores['perplexity'] = perplexities.get(model_name)
            scores['cross_entropy'] = cross_entropies.get(model_name)
            final_scores[model_name] = scores
            table.add_row(
                model_name,
                f"{scores['perplexity']:.4f}" if scores['perplexity'] else "N/A",
                f"{scores['cross_entropy']:.4f}" if scores['cross_entropy'] else "N/A",
                f"{scores['rouge1']:.4f}",
                f"{scores['rougeL']:.4f}",
                f"{scores.get('bertscore_f1', 0):.4f}",
            )
        console.print(table)
        with open(args.output_results, 'w') as f:
            json.dump(final_scores, f, indent=4)
        console.print(f"[bold green]Eval results saved to {args.output_results}[/bold green]")

        if args.wandb_project:
            import wandb
            if args.wandb_run_id:
                wandb.init(project=args.wandb_project, id=args.wandb_run_id, resume="must")
            else:
                wandb.init(
                    project=args.wandb_project,
                    name=args.wandb_run_name or "eval",
                    job_type="eval",
                )
            for model_name, scores in final_scores.items():
                prefix = model_name.replace("/", "_")
                wandb.log({f"eval/{prefix}/{k}": v for k, v in scores.items() if v is not None})
            wandb.finish()
            console.print("[bold green]Eval metrics logged to W&B[/bold green]")


if __name__ == "__main__":
    main()
