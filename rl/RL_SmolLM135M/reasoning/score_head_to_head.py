"""Score two rollout JSONL files with NeuralTxtReward and compare head-to-head."""
import argparse
import json
import math
import sys
from pathlib import Path

from reasoning.env import _extract_think_content


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare two rollout JSONL files using NeuralTxtReward."
    )
    parser.add_argument("--file_a", "-a", required=True, help="Reasoning model rollouts (may have <think> tags)")
    parser.add_argument("--file_b", "-b", required=True, help="SFT model rollouts")
    parser.add_argument("--name_a", default=None, help="Label for model A")
    parser.add_argument("--name_b", default=None, help="Label for model B")
    parser.add_argument("--output_dir", default="reasoning/evals/benchmark")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def load_jsonl(path, limit=None):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def extract_answer(completion: str) -> str:
    """Extract just the answer from a reasoning completion (after </think>)."""
    _, response, _ = _extract_think_content(completion)
    return response if response else completion


def main():
    from neuraltxt import NeuralTxtReward

    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    name_a = args.name_a or Path(args.file_a).stem
    name_b = args.name_b or Path(args.file_b).stem

    print(f"Loading A: {args.file_a}", flush=True)
    rows_a = load_jsonl(args.file_a, args.limit)
    print(f"Loading B: {args.file_b}", flush=True)
    rows_b = load_jsonl(args.file_b, args.limit)

    if args.limit:
        rows_a = rows_a[:args.limit]
        rows_b = rows_b[:args.limit]

    assert len(rows_a) == len(rows_b), f"File length mismatch: {len(rows_a)} vs {len(rows_b)}"
    n = len(rows_a)
    print(f"Comparing {n} examples", flush=True)

    answers_a = [extract_answer(r["response"]) for r in rows_a]
    answers_b = [r["response"] for r in rows_b]
    references = [r["ground_truth"] for r in rows_a]

    print("Scoring with NeuralTxtReward...", flush=True)
    reward_model = NeuralTxtReward(backend="hf")
    scores_a = reward_model.batch_score(answers_a, references, batch_size=8)
    scores_b = reward_model.batch_score(answers_b, references, batch_size=8)

    wins_a = 0
    wins_b = 0
    ties = 0
    diffs = []
    results = []

    for i, (ref, comp_a, comp_b, ans_a, s_a, s_b) in enumerate(
        zip(references, rows_a, rows_b, answers_a, scores_a, scores_b)
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
            "id": comp_a.get("id", i),
            "question": comp_a.get("question", ""),
            "reference": ref,
            f"{name_a}_completion": comp_a["response"],
            f"{name_a}_answer": ans_a,
            f"{name_b}_completion": comp_b["response"],
            f"{name_a}_score": s_a_f,
            f"{name_b}_score": s_b_f,
            "delta": diff,
            "winner": name_a if diff > 0.001 else (name_b if diff < -0.001 else "tie"),
        })

    mean_diff = sum(diffs) / n if diffs else 0.0
    std_diff = math.sqrt(sum((d - mean_diff) ** 2 for d in diffs) / n) if diffs else 0.0

    summary = {
        "model_a": name_a,
        "model_a_file": args.file_a,
        "model_b": name_b,
        "model_b_file": args.file_b,
        "num_samples": n,
        "model_a_mean_score": sum(scores_a) / n,
        "model_b_mean_score": sum(scores_b) / n,
        "mean_delta": mean_diff,
        "std_delta": std_diff,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        "win_rate_a": wins_a / n,
        "win_rate_b": wins_b / n,
        "tie_rate": ties / n,
    }

    results_path = output_dir / "head_to_head_results.jsonl"
    summary_path = output_dir / "head_to_head_summary.json"

    with results_path.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    print(f"\n=== NeuralTxtReward Head-to-Head ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"\nResults: {results_path}", flush=True)
    print(f"Summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
