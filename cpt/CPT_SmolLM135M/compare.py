"""Qualitative comparison: generate text from multiple models on the same prompts."""
import argparse
import json
import random
import torch
from rich.console import Console
from rich.table import Table
from rich import box
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from unsloth import FastLanguageModel
    _unsloth = True
except:
    _unsloth = False


def load_model(path, base_model="HuggingFaceTB/SmolLM-135M"):
    if torch.cuda.is_available() and _unsloth:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=path, max_seq_length=512, dtype=None, load_in_4bit=True
        )
        FastLanguageModel.for_inference(model)
    else:
        from peft import PeftModel
        import os
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        try:
            tokenizer = AutoTokenizer.from_pretrained(path)
        except:
            tokenizer = AutoTokenizer.from_pretrained(base_model)
        is_adapter = os.path.exists(os.path.join(path, "adapter_config.json"))
        if is_adapter:
            model = AutoModelForCausalLM.from_pretrained(base_model, device_map=device, torch_dtype=torch.float16)
            model = PeftModel.from_pretrained(model, path)
        else:
            model = AutoModelForCausalLM.from_pretrained(path, device_map=device, torch_dtype=torch.float16)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def generate(model, tokenizer, prompt, max_new_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=1.2,
        )
    return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description="Qualitative comparison across models.")
    parser.add_argument("--models", nargs="+", required=True, help="Model paths or HF IDs.")
    parser.add_argument("--names", nargs="+", default=None, help="Display names for each model.")
    parser.add_argument("--dataset", type=str, default=None, help="JSONL dataset to sample prompts from.")
    parser.add_argument("--prompts", nargs="+", default=None, help="Custom prompts to use.")
    parser.add_argument("--num_prompts", type=int, default=3, help="Number of prompts to sample from dataset.")
    parser.add_argument("--prefix_len", type=int, default=30, help="Words to use as prompt prefix.")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--output", type=str, default="comparison_results.json")
    parser.add_argument("--base_model", type=str, default="HuggingFaceTB/SmolLM-135M")
    args = parser.parse_args()

    console = Console()
    names = args.names or [m.replace("/", "_").replace("models/", "").replace("/final", "") for m in args.models]

    # Build prompts
    if args.prompts:
        prompts = args.prompts
    elif args.dataset:
        random.seed(42)
        with open(args.dataset) as f:
            lines = f.readlines()
        random.shuffle(lines)
        prompts = []
        for line in lines:
            words = json.loads(line)["text"].split()
            if len(words) >= args.prefix_len + 50:
                prompts.append(" ".join(words[:args.prefix_len]))
            if len(prompts) >= args.num_prompts:
                break
    else:
        prompts = [
            "We propose a novel attention mechanism that",
            "The results demonstrate that our method achieves state-of-the-art performance on",
            "In this paper, we introduce a new approach to training large language models by",
        ]

    results = []
    all_generations = {name: [] for name in names}

    for model_path, name in zip(args.models, names):
        console.rule(f"[bold blue]Loading {name}")
        model, tokenizer = load_model(model_path, args.base_model)

        for prompt in prompts:
            gen = generate(model, tokenizer, prompt, args.max_new_tokens)
            all_generations[name].append(gen)

        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Display results
    for i, prompt in enumerate(prompts):
        console.rule(f"[bold yellow]Prompt {i+1}")
        console.print(f"[bold]Prompt:[/bold] {prompt}\n")
        table = Table(box=box.ROUNDED, show_lines=True)
        table.add_column("Model", style="cyan", width=25)
        table.add_column("Generation", style="white")
        for name in names:
            table.add_row(name, all_generations[name][i])
        console.print(table)
        results.append({"prompt": prompt, "generations": {n: all_generations[n][i] for n in names}})

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"\n[bold green]Saved to {args.output}[/bold green]")


if __name__ == "__main__":
    main()
