"""
Generate multiple parallel responses per prompt at different temperatures
for diversity analysis.

Uses Hugging Face transformers for GPU inference, or mlx_lm for
Apple-Silicon inference.

Output JSONL format (input to diversity.py):
  {"id": 0, "question": "...", "responses": ["r1", "r2", ...]}

Usage (transformers):
    uv run preference_optimization/generate_responses.py \\
        -m paperbd/smollm_135M_neuraltxt_v1 \\
        --num_samples 100 --n_responses 4 --temperatures 0.3 0.5 0.7 1.0

Usage (MLX on Mac):
    uv run preference_optimization/generate_responses.py \\
        -m paperbd/smollm_135M_neuraltxt_v1 \\
        --mlx \\
        --num_samples 100 --n_responses 4 --temperatures 0.3 0.5 0.7 1.0
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any

from datasets import load_dataset

SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.
You are an expert in AI, deep learning, and machine learning research and its applications.
Your answers are concise and helps directly solve any user query truthfully.
If you do not know the answer, you will inform the user that you do not know instead of making answers up.
    """

parser = argparse.ArgumentParser(
    description="Generate N responses per prompt across temperatures for diversity eval."
)
parser.add_argument(
    "--model_path",
    "-m",
    type=str,
    required=True,
)
parser.add_argument(
    "--num_samples",
    "-n",
    type=int,
    default=100,
    help="Number of prompts to sample from the test split.",
)
parser.add_argument(
    "--n_responses",
    type=int,
    default=4,
    help="Number of responses to generate *per temperature*.",
)
parser.add_argument(
    "--temperatures",
    nargs="+",
    type=float,
    default=[0.3, 0.5, 0.7, 1.0],
    help="List of temperatures to sweep. Total responses = len(temperatures) * n_responses.",
)
parser.add_argument(
    "--max_new_tokens",
    type=int,
    default=512,
)
parser.add_argument(
    "--batch_size",
    "-bs",
    type=int,
    default=1,
    help="Batch size for generation. Only used for transformers path; MLX path processes one prompt at a time.",
)
parser.add_argument(
    "--seed",
    type=int,
    default=3407,
)
parser.add_argument(
    "--output_file",
    "-o",
    type=str,
    default=None,
)
parser.add_argument(
    "--mlx",
    action="store_true",
    help="Use MLX inference (Apple Silicon).",
)
args = parser.parse_args()


# ── Dataset ──────────────────────────────────────────────────────────────────

dataset = load_dataset("paperbd/paper_instructions_300K-v1", split="test")
dataset = dataset.shuffle(seed=args.seed).select(range(args.num_samples))
examples = list(dataset)

os.makedirs("preference_optimization/evals", exist_ok=True)

if args.output_file is None:
    model_name = os.path.basename(os.path.normpath(args.model_path))
    out_path = (
        f"preference_optimization/evals/"
        f"{model_name}_diversity_n{args.num_samples}_r{args.n_responses}.jsonl"
    )
else:
    out_path = args.output_file


def build_messages(example: dict[str, Any]) -> tuple[str, list[dict[str, str]]]:
    instruction = example["instruction"]
    inp = example.get("input", "")
    question = instruction if not inp else f"{instruction}\n\n{inp}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return question, messages


# ── Generation helpers ───────────────────────────────────────────────────────


def generate_transformers(
    model,
    tokenizer,
    prompts: list[str],
    temperatures: list[float],
    n_per_temp: int,
    max_new_tokens: int,
) -> list[list[str]]:
    """
    For each prompt, generate (len(temperatures) * n_per_temp) responses.
    Temperatures are processed sequentially; each temperature group is batched.
    """
    import torch

    responses_per_prompt: list[list[str]] = []
    for prompt_text in prompts:
        all_texts: list[str] = []
        for temp in temperatures:
            for g_start in range(0, n_per_temp, args.batch_size):
                g_size = min(args.batch_size, n_per_temp - g_start)
                batch = [prompt_text] * g_size
                encoded = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1024,
                ).to(model.device)
                with torch.no_grad():
                    output_ids = model.generate(
                        **encoded,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=temp,
                        top_p=0.95,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                input_len = encoded["input_ids"].shape[1]
                for j in range(g_size):
                    generated = tokenizer.decode(
                        output_ids[j][input_len:],
                        skip_special_tokens=True,
                    )
                    all_texts.append(generated)
        responses_per_prompt.append(all_texts)
    return responses_per_prompt


def generate_mlx(
    model,
    tokenizer,
    prompt_texts: list[str],
    temperatures: list[float],
    n_per_temp: int,
    max_new_tokens: int,
) -> list[list[str]]:
    """
    MLX path. mlx_lm.generate takes a single string prompt, so we loop.
    """
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm import generate as mlx_generate

    responses_per_prompt: list[list[str]] = []
    for prompt_text in prompt_texts:
        per_prompt: list[str] = []
        for temp in temperatures:
            sampler = make_sampler(temp=temp)
            for _ in range(n_per_temp):
                text = mlx_generate(
                    model,
                    tokenizer,
                    prompt=prompt_text,
                    max_tokens=max_new_tokens,
                    sampler=sampler,
                    verbose=False,
                )
                per_prompt.append(text)
        responses_per_prompt.append(per_prompt)
    return responses_per_prompt


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    # Load tokenizer *before* building prompt texts so that the chat template is available.
    if args.mlx:
        from mlx_lm import load

        model, tokenizer = load(args.model_path)
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model.eval()

    questions: list[str] = []
    prompt_texts: list[str] = []
    ground_truths: list[str] = []
    for ex in examples:
        q, messages = build_messages(ex)
        questions.append(q)
        prompt_texts.append(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )
        ground_truths.append(ex["output"])

    if args.mlx:
        all_responses = generate_mlx(
            model,
            tokenizer,
            prompt_texts,
            args.temperatures,
            args.n_responses,
            args.max_new_tokens,
        )
    else:
        all_responses = generate_transformers(
            model,
            tokenizer,
            prompt_texts,
            args.temperatures,
            args.n_responses,
            args.max_new_tokens,
        )

    total = len(examples)
    with open(out_path, "w") as f:
        for idx, (q, gt, responses) in enumerate(zip(questions, ground_truths, all_responses)):
            rec = {
                "id": idx,
                "question": q,
                "responses": responses,
                "ground_truth": gt,
            }
            f.write(json.dumps(rec) + "\n")
            print(
                f"[{idx + 1}/{total}] {len(responses)} responses  "
                f"avg_len={sum(len(r) for r in responses) // len(responses)} chars"
            )
            if idx < 2:
                # Show preview of first two prompts
                for t_idx, r in enumerate(responses):
                    temp = args.temperatures[t_idx // args.n_responses]
                    print(f"    t={temp}  {r[:80].replace(chr(10), ' ')}...")

    print(f"\nSaved {total} records to {out_path}")


if __name__ == "__main__":
    main()
