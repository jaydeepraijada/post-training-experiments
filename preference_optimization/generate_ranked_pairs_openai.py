from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

import os
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
import outlines
import openai

# ── Hardcoded endpoint config ────────────────────────────────────────────────
MODEL = "openai/gpt-5.4-nano"
BASE_URL = "https://openrouter.ai/api/v1"

API_KEY = os.getenv("OPENROUTER_API_KEY")

RANKING_PROMPT = """\
You are an expert evaluator for AI assistant responses on research and academic questions.
You are ranking the responses of a 100M parameter small model. Your goal is to find the best response and the worst response to the question.
Given a question and {n} candidate responses, rank the responses from **best** (1st) to **worst** ({n}th) based on overall quality: correctness, relevance, completeness, and clarity.

Also evaluate the example as a whole:
- ignore_example: set to true if the question / prompt itself is malformed, nonsensical, or otherwise unsuitable and should not be part of the dataset.
- ignore_all_samples: set to true if **every** candidate response is incoherent, garbled, contains repetition loops, or is structurally unusable. Factual incorrectness alone is acceptable; only flag this when the outputs are genuinely unreadable.

During ranking, pick in this order:
- correctness (how correct the response is given the specific prompt that was provided to the model)
- completeness (how complete the response is given the specific prompt that was provided to the model)
- relevance (how well the response answers the question)
- conciseness (who got the most correct information in the shortest amount of tokens)


Ignoring all samples is only allowed for extreme cases where the model response appears completely broken.

Question:
{question}

{responses_block}

Provide a brief reasoning for your ranking, then output the structured result with:
- reasoning: str  (1-2 sentences)
- ranking_list: list[int]  (response indices 0-{n_minus_1} ordered from best to worst; every index must appear exactly once)
- ignore_example: bool (True if the question/prompt itself is malformed or unsuitable.)
- ignore_all_samples: bool (True if every response is incoherent / garbled / repetitive. Minor factual errors can be tolerated.)
"""


class RankingOutput(BaseModel):
    ranking_list: list[int]
    ignore_example: bool
    ignore_all_samples: bool


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate ranked pairs for preference data via a local OpenAI-compatible endpoint."
    )
    p.add_argument(
        "--input_file",
        "-i",
        type=str,
        default="preference_optimization/evals/pref_dataset/train_4r_temp0.5.jsonl",
    )
    p.add_argument(
        "--output_file",
        "-o",
        type=str,
        default=None,
        help="Defaults to <input>_ranked.jsonl",
    )
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--limit", "-n", type=int, default=None)
    p.add_argument(
        "--semaphore", type=int, default=50, help="Max concurrent API requests."
    )
    p.add_argument("--max_retries", type=int, default=2)
    p.add_argument(
        "--resume_idx",
        type=int,
        default=0,
        help="Skip this many records from the start.",
    )
    return p.parse_args()


def build_ranking_prompt_text(question: str, responses: list[str]) -> str:
    n = len(responses)
    responses_block = "\n\n".join(
        f"Response {i}:\n{r}" for i, r in enumerate(responses)
    )
    return RANKING_PROMPT.format(
        question=question,
        n=n,
        n_minus_1=n - 1,
        responses_block=responses_block,
    )


def validate_ranking(ranking: list[int], expected_len: int) -> bool:
    if len(ranking) != expected_len:
        return False
    return set(ranking) == set(range(expected_len))


def _extract_result(raw: Any) -> dict[str, Any] | None:
    """Normalise outlines output to a plain dict."""
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            return RankingOutput.model_validate_json(raw).model_dump()
        except Exception:
            return None
    if hasattr(raw, "model_dump"):
        return raw.model_dump()
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(str(raw))
    except json.JSONDecodeError:
        return None


async def rank_one(
    sem: asyncio.Semaphore,
    model,
    rec: dict[str, Any],
    max_retries: int,
    max_tokens: int,
    global_idx: int,
) -> dict[str, Any] | None:
    """
    Rank a single preference record.
    Returns None if generation fails completely after retries — caller should drop it.
    """
    n_resp = len(rec["responses"])
    prompt = build_ranking_prompt_text(rec["question"], rec["responses"])

    async with sem:
        for attempt in range(1, max_retries + 1):
            raw = await model(prompt, RankingOutput)
            print(raw)

            data = _extract_result(raw)
            if data is None:
                continue

            candidate = data.get("ranking_list", [])
            if validate_ranking(candidate, n_resp):
                return {
                    **rec,
                    "ranking_list": candidate,
                    "ranking_reasoning": data.get("reasoning", ""),
                    "ignore_example": data.get("ignore_example", False),
                    "ignore_all_samples": data.get("ignore_all_samples", False),
                }
            else:
                print(
                    f"  [{global_idx+1}] attempt {attempt} invalid ranking: {candidate}"
                )

    print(f"  [{global_idx+1}] Dropped — all retries failed")
    return None


async def main() -> None:
    args = parse_args()

    if args.output_file is None:
        args.output_file = args.input_file.replace(".jsonl", "_ranked.jsonl")
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
    model = outlines.from_openai(client, MODEL)
    print(f"Using endpoint model: {MODEL} @ {BASE_URL}\n")

    with open(args.input_file, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    # Slice by user-supplied resume / limit
    start = max(0, args.resume_idx)
    end = len(records) if args.limit is None else start + args.limit
    records = records[start:end]

    total = len(records)
    print(
        f"Processing {total} records (start={start}, limit={args.limit}) with max {args.semaphore} concurrent requests …"
    )

    t0 = time.time()
    sem = asyncio.Semaphore(args.semaphore)

    tasks = [
        asyncio.create_task(
            rank_one(
                sem,
                model,
                rec,
                args.max_retries,
                args.max_tokens,
                start + idx,
            )
        )
        for idx, rec in enumerate(records)
    ]

    written = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for completed in asyncio.as_completed(tasks):
            out_rec = await completed
            if out_rec is None:
                continue
            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            written += 1

            global_idx = start + written
            if (global_idx) % 10 == 0:
                elapsed = time.time() - t0
                rate = written / max(elapsed, 1e-6)
                print(
                    f"[{global_idx}/{start+total}] "
                    f"ranking={out_rec['ranking_list']} | {rate:.2f} rec/s | {elapsed/60:.1f} min elapsed"
                )

    elapsed = time.time() - t0
    print(
        f"\nDone. {written} records written to {out_path} ({total - written} dropped)"
    )
    print(f"Time: {elapsed/60:.1f} min")


if __name__ == "__main__":
    asyncio.run(main())
