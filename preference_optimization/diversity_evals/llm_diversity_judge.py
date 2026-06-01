"""
Async LLM judge for diversity-evaluation JSONL files.

Given a JSONL where each record contains:
  {"id": int, "question": str, "responses": [str, ...], "ground_truth": str}

This script expands each record into N judge calls (one per response),
scoring both correctness and coherence with an external LLM.

Output keeps the same top-level record shape but adds:
  "response_scores": [{"correctness": 1-5, "coherence": 1-5, "reasoning": str}, ...]
  "avg_correctness": float
  "avg_coherence": float

Usage:
    uv run preference_optimization/llm_diversity_judge.py \\
        -i preference_optimization/evals/..._diversity.jsonl \\
        --limit 20
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from typing import Any
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
import outlines

JUDGE_MODEL = "openai/gpt-5.4-mini"
SEMAPHORE = 30  # lower concurrency because N responses per prompt multiplies cost

JUDGE_PROMPT = """\
You are an expert evaluator for AI assistant responses.

Given a question (which includes the passage / task context) and a generated response,
score the response on **correctness** and **coherence**.

Scoring guide (1-5):
  1 = Very poor   2 = Poor   3 = Acceptable   4 = Good   5 = Excellent

Dimensions:
- correctness:  How factually accurate and well-supported is the response given the passage / instruction?
                  Penalise hallucinations, contradictions with the provided context, and unsupported claims heavily.
                  For open-ended tasks (e.g., continuation, rephrasing, generating valid questions),
                  accept any plausible, well-supported answer rather than penalising for deviating from a single reference.
- coherence:      How well-structured, fluent, and internally consistent is the response?
                  Does it follow a logical flow without random jumps or garbled text?

Provide a brief reasoning (1-2 sentences) before the scores.

---
QUESTION:
{question}

MODEL RESPONSE:
{response}
---
"""


class ResponseScore(BaseModel):
    reasoning: str = Field(..., description="Brief explanation of the scores.")
    correctness: int = Field(..., ge=1, le=5)
    coherence: int = Field(..., ge=1, le=5)


parser = argparse.ArgumentParser(
    description="Async LLM judge for diversity response sets."
)
parser.add_argument("--input_file", "-i", type=str, required=True)
parser.add_argument("--output_file", "-o", type=str, default=None)
parser.add_argument("--api_key", type=str, default=os.environ.get("OPENROUTER_API_KEY"))
parser.add_argument("--model", type=str, default=JUDGE_MODEL)
parser.add_argument("--semaphore", type=int, default=SEMAPHORE)
parser.add_argument("--limit", "-n", type=int, default=None)
args = parser.parse_args()

if not args.api_key:
    raise ValueError("Set OPENROUTER_API_KEY or pass --api_key")

if args.output_file is None:
    base, ext = os.path.splitext(args.input_file)
    args.output_file = f"{base}_judged{ext}"

client = AsyncOpenAI(
    api_key=args.api_key,
    base_url="https://openrouter.ai/api/v1",
)
model = outlines.from_openai(client, args.model)


async def judge_response(
    sem: asyncio.Semaphore,
    idx: int,
    total: int,
    rid: int,
    response_idx: int,
    question: str,
    response: str,
) -> dict:
    prompt = JUDGE_PROMPT.format(
        question=question,
        response=response,
    )
    async with sem:
        raw = await model(prompt, ResponseScore, max_tokens=512)
        score = ResponseScore.model_validate_json(raw) if isinstance(raw, str) else raw

    print(
        f"[{idx}/{total}] response[{response_idx}] "
        f"C={score.correctness} CO={score.coherence} | "
        f"{response[:60].replace(chr(10), ' ')}..."
    )
    return {
        "response_idx": response_idx,
        "correctness": score.correctness,
        "coherence": score.coherence,
        "reasoning": score.reasoning,
    }


async def main():
    with open(args.input_file) as f:
        records = [json.loads(line) for line in f]

    if args.limit:
        records = records[: args.limit]

    # Flatten all (record, response) pairs for async judging
    tasks_info: list[tuple[Any, int]] = []  # (record, response_idx)
    for rec in records:
        for i in range(len(rec.get("responses", []))):
            tasks_info.append((rec, i))

    total_flat = len(tasks_info)
    sem = asyncio.Semaphore(args.semaphore)

    print(f"Judging {len(records)} prompts × {total_flat // len(records)} responses = {total_flat} total calls ...")

    tasks = []
    for flat_idx, (rec, resp_idx) in enumerate(tasks_info):
        tasks.append(
            judge_response(
                sem,
                flat_idx + 1,
                total_flat,
                rec["id"],
                resp_idx,
                rec["question"],
                rec["responses"][resp_idx],
            )
        )

    flat_results = await asyncio.gather(*tasks)

    # Map flat results back per record
    # Actually iterate in order
    it = iter(flat_results)
    scored_records: list[dict] = []
    for rec in records:
        n = len(rec.get("responses", []))
        resp_scores = [next(it) for _ in range(n)]
        avg_corr = sum(s["correctness"] for s in resp_scores) / n
        avg_cohe = sum(s["coherence"] for s in resp_scores) / n
        scored_records.append(
            {
                **rec,
                "response_scores": [
                    {
                        "correctness": s["correctness"],
                        "coherence": s["coherence"],
                        "reasoning": s["reasoning"],
                    }
                    for s in resp_scores
                ],
                "avg_correctness": avg_corr,
                "avg_coherence": avg_cohe,
            }
        )

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, "w") as out:
        for r in scored_records:
            out.write(json.dumps(r) + "\n")

    # Summary
    all_corr = [r["avg_correctness"] for r in scored_records]
    all_cohe = [r["avg_coherence"] for r in scored_records]
    all_overall = [(c + h) / 2 for c, h in zip(all_corr, all_cohe)]

    print("\n--- summary ---")
    print(f"  {'correctness':15s}  avg={sum(all_corr)/len(all_corr):.2f}  min={min(all_corr):.2f}  max={max(all_corr):.2f}")
    print(f"  {'coherence':15s}  avg={sum(all_cohe)/len(all_cohe):.2f}  min={min(all_cohe):.2f}  max={max(all_cohe):.2f}")
    print(f"  {'overall':15s}  avg={sum(all_overall)/len(all_overall):.2f}  min={min(all_overall):.2f}  max={max(all_overall):.2f}")
    print(f"\nSaved to {args.output_file}")


asyncio.run(main())
