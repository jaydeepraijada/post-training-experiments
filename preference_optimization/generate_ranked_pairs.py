from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from mlx_vlm_batch_outlines import batch_chat, load
from pydantic import BaseModel, Field

DEFAULT_MODEL = "mlx-community/Qwen3.5-4B-MLX-4bit"

SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.
You are an expert in AI, deep learning, and machine learning research and its applications.
Your answers are concise and helps directly solve any user query truthfully.
If you do not know the answer, you will inform the user that you do not know instead of making answers up.
"""

RANKING_PROMPT = """\
You are an expert evaluator for AI assistant responses on research and academic questions.

Given a question and {n} candidate responses, rank the responses from **best** (1st) to **worst** ({n}th) based on overall quality: correctness, relevance, completeness, and clarity.

Question:
{question}

{responses_block}

Provide a brief reasoning for your ranking, then output the ranking as a JSON object with:
- reasoning: str  (1-2 sentences)
- ranking_list: list[int]  (response indices 0-{n_minus_1} ordered from best to worst; every index must appear exactly once)
"""


class RankingOutput(BaseModel):
    reasoning: str = Field(..., description="Brief explanation for the ranking.")
    ranking_list: list[int] = Field(
        ...,
        description="Permutation of [0..n-1] ordered from best to worst.",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate ranked pairs for preference data using mlx-vlm-batch-outlines batched structured generation."
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
    p.add_argument("--model_path", "-m", type=str, default=DEFAULT_MODEL)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--limit", "-n", type=int, default=None)
    p.add_argument("--start_index", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_retries", type=int, default=2)
    return p.parse_args()


def count_valid_lines(path: Path) -> int:
    if not path.exists():
        return 0
    valid = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                json.loads(line)
                valid += 1
            except json.JSONDecodeError:
                break
    return valid


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
    """Normalise batch_chat output to a plain dict."""
    if raw is None:
        return None
    if hasattr(raw, "model_dump"):
        return raw.model_dump()
    if isinstance(raw, dict):
        return raw
    # Some older versions wrap text in an object with a .text attribute
    if hasattr(raw, "text"):
        try:
            return json.loads(raw.text)
        except json.JSONDecodeError:
            return None
    try:
        return json.loads(str(raw))
    except json.JSONDecodeError:
        return None


def main() -> None:
    args = parse_args()

    if args.output_file is None:
        args.output_file = args.input_file.replace(".jsonl", "_ranked.jsonl")
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.model_path} …")
    model, processor = load(args.model_path)
    print("Model loaded.\n")

    with open(args.input_file, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    if args.limit is not None:
        records = records[: args.limit]

    existing = count_valid_lines(out_path)
    if existing > 0:
        print(f"Resuming: {existing}/{len(records)} records already in output.")
    records = records[existing:]

    total = len(records)
    print(f"Processing {total} records in batches of {args.batch_size} …")

    t0 = time.time()
    written = existing

    for batch_start in range(0, total, args.batch_size):
        batch = records[batch_start : batch_start + args.batch_size]
        batch_size = len(batch)

        # Build text-only conversations for this batch
        conversations: list[list[dict[str, str]]] = []
        for rec in batch:
            user_text = build_ranking_prompt_text(rec["question"], rec["responses"])
            conversations.append(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_text},
                ]
            )

        # --- Batch generation ------------------------------------------------
        batch_results: list[Any] = [None] * batch_size
        try:
            raw_batch = batch_chat(
                model,
                processor,
                conversations,
                output_type=RankingOutput,
                max_tokens=args.max_tokens,
            )
            # Normalise to a plain list
            if batch_size == 1 and not isinstance(raw_batch, list):
                batch_results = [raw_batch]
            elif isinstance(raw_batch, list):
                batch_results = raw_batch
            else:
                # The library may return a list-like wrapper; try iterating
                batch_results = list(raw_batch)
        except Exception as e:
            print(f"  Batch {batch_start // args.batch_size + 1} failed: {e}")

        # --- Write / retry per record ----------------------------------------
        for idx_in_batch, (rec, raw) in enumerate(zip(batch, batch_results)):
            global_idx = existing + batch_start + idx_in_batch
            n_resp = len(rec["responses"])

            ranking: list[int] | None = None
            reasoning = ""

            # 1. Try batch result
            data = _extract_result(raw)
            if data is not None:
                candidate = data.get("ranking_list", [])
                if validate_ranking(candidate, n_resp):
                    ranking = candidate
                    reasoning = data.get("reasoning", "")

            # 2. Retry individually if batch result was missing / invalid
            for attempt in range(args.max_retries):
                if ranking is not None:
                    break
                try:
                    single_raw = batch_chat(
                        model,
                        processor,
                        [
                            [
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {
                                    "role": "user",
                                    "content": build_ranking_prompt_text(
                                        rec["question"], rec["responses"]
                                    ),
                                },
                            ]
                        ],
                        output_type=RankingOutput,
                        max_tokens=args.max_tokens,
                    )
                    # normalise single-item wrapper
                    if not isinstance(single_raw, list):
                        single_raw = [single_raw]
                    data = _extract_result(single_raw[0]) if single_raw else None
                    if data is not None:
                        candidate = data.get("ranking_list", [])
                        if validate_ranking(candidate, n_resp):
                            ranking = candidate
                            reasoning = data.get("reasoning", "")
                            break
                        else:
                            print(
                                f"  [{global_idx+1}] attempt {attempt+1} invalid ranking: {candidate}"
                            )
                except Exception as e:
                    print(f"  [{global_idx+1}] attempt {attempt+1} error: {e}")

            if ranking is None:
                print(f"  [{global_idx+1}] WARNING: all retries failed, using fallback ranking.")
                ranking = list(range(n_resp))
                reasoning = "Fallback ranking due to generation failure."

            out_rec = {
                **rec,
                "ranking_list": ranking,
                "ranking_reasoning": reasoning,
            }
            with open(out_path, "a", encoding="utf-8") as fout:
                fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            written += 1

            elapsed = time.time() - t0
            rate = (written - existing) / max(elapsed, 1e-6)
            print(
                f"[{global_idx+1}/{existing+total}] "
                f"ranking={ranking} | {rate:.2f} rec/s | {elapsed/60:.1f} min elapsed"
            )

    elapsed = time.time() - t0
    print(f"\nDone. {written} total records written to {out_path}")
    print(f"Time: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
