"""Head-to-head benchmark: two models scored by NeuralTxtReward on the test split (MLX backend)."""
import argparse
import json
import math
import sys
import time
from pathlib import Path

from reasoning.env import (
    SEED,
    SYSTEM_PROMPT as REASONING_SYSTEM_PROMPT,
    _extract_think_content,
    format_row,
)


SFT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.
You are an expert in AI, deep learning, and machine learning research and its applications.
Your answers are concise and helps directly solve any user query truthfully.
If you do not know the answer, you will inform the user that you do not know instead of making answers up.
    """


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark two models on the test split using NeuralTxtReward (MLX)."
    )
    parser.add_argument(
        "--model_a",
        default="models/mlx/neuraltxt-135M-reasoning-base-v2",
        help="Reasoning model (MLX, uses <think> prompt format)",
    )
    parser.add_argument(
        "--model_b",
        default="paperbd/neuraltxt-v1-135M-mlx",
        help="SFT model (MLX, uses standard chat prompt format)",
    )
    parser.add_argument(
        "--dataset_name",
        default="paperbd/paper_instructions_300K-v1",
    )
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output_dir", type=str, default="reasoning/evals/benchmark")
    return parser.parse_args()


def short_name(path: str) -> str:
    return Path(path).name


def load_and_generate_mlx(model_path, examples, use_reasoning_prompt, args, output_dir, label):
    from mlx_lm import load, batch_generate
    from mlx_lm.sample_utils import make_sampler

    name = short_name(model_path)
    print(f"\n=== {label} ({name}) ===", flush=True)
    print(f"Loading: {model_path}", flush=True)

    model, tokenizer = load(model_path)
    print(f"  model loaded", flush=True)

    prompts = []
    references = []
    print(f"  building prompts for {len(examples)} examples...", flush=True)
    for ex in examples:
        if use_reasoning_prompt:
            formatted = format_row(ex)
            prompt_text = tokenizer.apply_chat_template(
                formatted["prompt"],
                tokenize=False,
                add_generation_prompt=True,
            )
            ref = str(formatted["answer"])
        else:
            instruction = ex["instruction"]
            inp = ex.get("input", "")
            question = instruction if not inp else f"{instruction}\n\n{inp}"
            messages = [
                {"role": "system", "content": SFT_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ]
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            ref = str(ex["output"])
        prompts.append(tokenizer.encode(prompt_text))
        references.append(ref)

    gen_path = output_dir / f"generations_{name}.jsonl"
    completions = []
    sampler = make_sampler(temp=args.temperature) if args.temperature > 0 else make_sampler(temp=0.0)
    print(f"  generating {len(prompts)} responses (batch_size={args.batch_size}, max_new_tokens={args.max_new_tokens})...", flush=True)
    t_start = time.time()

    for start in range(0, len(prompts), args.batch_size):
        t_batch = time.time()
        batch_prompts = prompts[start : start + args.batch_size]
        result = batch_generate(
            model, tokenizer,
            prompts=batch_prompts,
            max_tokens=args.max_new_tokens,
            sampler=sampler,
            verbose=False,
        )
        batch_texts = [t.strip() for t in result.texts]
        completions.extend(batch_texts)

        n_done = len(completions)
        elapsed = time.time() - t_start
        batch_time = time.time() - t_batch
        eta = (elapsed / n_done) * (len(prompts) - n_done)
        print(f"  {n_done}/{len(prompts)} | batch {batch_time:.1f}s | elapsed {elapsed:.0f}s | ETA {eta:.0f}s", flush=True)

    with gen_path.open("w", encoding="utf-8") as f:
        for i, (ref, comp) in enumerate(zip(references, completions)):
            f.write(json.dumps({"id": i, "reference": ref, "completion": comp}, ensure_ascii=True) + "\n")
    print(f"  Generations saved to {gen_path}", flush=True)

    return name, completions, references


def main():
    from datasets import load_dataset
    from neuraltxt import NeuralTxtReward

    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model A (reasoning/MLX): {args.model_a}", flush=True)
    print(f"Model B (sft/MLX):      {args.model_b}", flush=True)
    print(f"Loading test split: {args.dataset_name} (test, n={args.num_samples})", flush=True)

    dataset = load_dataset(args.dataset_name, split="test")
    dataset = dataset.shuffle(seed=args.seed).select(range(args.num_samples))
    examples = list(dataset)

    name_a, completions_a, references_a = load_and_generate_mlx(
        args.model_a, examples, use_reasoning_prompt=True,
        args=args, output_dir=output_dir, label="Model A",
    )
    name_b, completions_b, references_b = load_and_generate_mlx(
        args.model_b, examples, use_reasoning_prompt=False,
        args=args, output_dir=output_dir, label="Model B",
    )

    references = references_a

    # --- Score with NeuralTxtReward (answer only for reasoning model) ---
    print("\n=== Scoring with NeuralTxtReward (HF backend) ===", flush=True)
    reward_model = NeuralTxtReward(backend="hf")

    answers_a = []
    for comp in completions_a:
        _, response, _ = _extract_think_content(comp)
        answers_a.append(response if response else comp)
    answers_b = completions_b

    scores_a = reward_model.batch_score(answers_a, references, batch_size=8)
    scores_b = reward_model.batch_score(answers_b, references, batch_size=8)

    # --- Compare ---
    wins_a = 0
    wins_b = 0
    ties = 0
    diffs = []
    results = []

    for i, (ref, comp_a, comp_b, ans_a, s_a, s_b) in enumerate(
        zip(references, completions_a, completions_b, answers_a, scores_a, scores_b)
    ):
        s_a_f = float(s_a)
        s_b_f = float(s_b)
        diff = s_a_f - s_b_f

        if diff > 0.001:
            wins_a += 1
        elif diff < -0.001:
            wins_b += 1
        else:
            ties += 1
        diffs.append(diff)

        results.append({
            "id": i,
            "reference": ref,
            f"{name_a}_completion": comp_a,
            f"{name_a}_answer": ans_a,
            f"{name_b}_completion": comp_b,
            f"{name_a}_score": s_a_f,
            f"{name_b}_score": s_b_f,
            "delta": diff,
            "winner": name_a if diff > 0.001 else (name_b if diff < -0.001 else "tie"),
        })

    mean_diff = sum(diffs) / len(diffs) if diffs else 0.0
    std_diff = math.sqrt(
        sum((d - mean_diff) ** 2 for d in diffs) / len(diffs)
    ) if diffs else 0.0

    summary = {
        "model_a": name_a,
        "model_a_path": args.model_a,
        "model_b": name_b,
        "model_b_path": args.model_b,
        "dataset": args.dataset_name,
        "num_samples": args.num_samples,
        "temperature": args.temperature,
        "seed": args.seed,
        "model_a_mean_score": sum(scores_a) / len(scores_a),
        "model_b_mean_score": sum(scores_b) / len(scores_b),
        "mean_delta": mean_diff,
        "std_delta": std_diff,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        "win_rate_a": wins_a / args.num_samples,
        "win_rate_b": wins_b / args.num_samples,
        "tie_rate": ties / args.num_samples,
    }

    results_path = output_dir / "head_to_head_results.jsonl"
    summary_path = output_dir / "head_to_head_summary.json"

    with results_path.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    print(f"\n=== Results ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"\nPer-example results: {results_path}", flush=True)
    print(f"Summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
