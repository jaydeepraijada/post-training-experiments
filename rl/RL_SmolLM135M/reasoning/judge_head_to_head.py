"""
Blind LLM judge between two sets of eval outputs on the same prompts. Each pair is
shown in a random A/B order with the full task, so the judge can check the answers
against the source passage. Reasoning text is stripped: only final answers are judged.

Usage:
    python reasoning/judge_head_to_head.py \
        --baseline eval_generations_5.json --candidate eval_generations_1515.json \
        --output judge_1515_vs_5.jsonl
"""
import argparse
import asyncio
import json
import math
import os
import random
from pathlib import Path

from openai import AsyncOpenAI

PROMPT = """You are judging two answers to the same task. Decide which is better.

Judge by, in order of importance:
1. Accuracy: every claim must be supported by the passage in the task. Made-up or contradicted details are serious errors.
2. Task completion: it does what was asked, in the requested format.
3. Usefulness: it is complete and clear. Length alone is not a virtue.

<task>
{task}
</task>

<answer_a>
{answer_a}
</answer_a>

<answer_b>
{answer_b}
</answer_b>

Reply with only a JSON object: {{"winner": "A" | "B" | "tie", "reason": "<one sentence>"}}"""

parser = argparse.ArgumentParser()
parser.add_argument("--baseline", required=True)
parser.add_argument("--candidate", required=True)
parser.add_argument("--output", "-o", required=True)
parser.add_argument("--num", "-n", type=int, default=None)
parser.add_argument("--model", default="deepseek-v4-pro")
parser.add_argument("--semaphore", "-s", type=int, default=16)
parser.add_argument("--seed", type=int, default=3407)
args = parser.parse_args()

client = AsyncOpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com"
)


def final_answer(row):
    return row["response"].split("</think>")[-1].strip()


async def judge(index, baseline, candidate, semaphore):
    task = baseline["question"][-1]["content"]
    if task != candidate["question"][-1]["content"]:
        raise ValueError(f"Prompts differ at row {index}")
    baseline_answer = final_answer(baseline)
    candidate_answer = final_answer(candidate)
    result = {"index": index, "datatype": baseline["reference_datatype"]}
    if baseline_answer == candidate_answer:
        return {**result, "outcome": "tie", "reason": "identical answers", "usage": None}

    candidate_is_a = random.Random(f"{args.seed}-{index}").random() < 0.5
    answer_a, answer_b = (
        (candidate_answer, baseline_answer)
        if candidate_is_a
        else (baseline_answer, candidate_answer)
    )
    prompt = PROMPT.format(
        task=task, answer_a=answer_a or "(empty)", answer_b=answer_b or "(empty)"
    )
    async with semaphore:
        for attempt in range(3):
            try:
                response = await client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=8192,
                    response_format={"type": "json_object"},
                )
                verdict = json.loads(response.choices[0].message.content)
                winner = verdict["winner"]
                if winner not in ("A", "B", "tie"):
                    raise ValueError(f"Bad winner: {winner}")
                break
            except Exception as error:
                if attempt == 2:
                    return {**result, "outcome": "error", "reason": str(error), "usage": None}
                await asyncio.sleep(2**attempt)
    if winner == "tie":
        outcome = "tie"
    else:
        outcome = "candidate" if (winner == "A") == candidate_is_a else "baseline"
    return {
        **result,
        "outcome": outcome,
        "candidate_is_a": candidate_is_a,
        "reason": verdict.get("reason", ""),
        "usage": response.usage.model_dump(),
    }


async def main():
    baseline = json.loads(Path(args.baseline).read_text())
    candidate = json.loads(Path(args.candidate).read_text())
    pairs = list(zip(baseline, candidate))[: args.num]
    semaphore = asyncio.Semaphore(args.semaphore)
    results = await asyncio.gather(
        *(judge(i, b, c, semaphore) for i, (b, c) in enumerate(pairs))
    )
    with open(args.output, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    counts = {k: 0 for k in ("candidate", "baseline", "tie", "error")}
    for result in results:
        counts[result["outcome"]] += 1
    decided = counts["candidate"] + counts["baseline"] + counts["tie"]
    # Ties count as half a win.
    score = (counts["candidate"] + counts["tie"] / 2) / max(decided, 1)
    margin = 1.96 * math.sqrt(score * (1 - score) / max(decided, 1))
    usages = [r["usage"] for r in results if r["usage"]]
    print(counts)
    print(f"candidate score: {score:.1%} +- {margin:.1%}")
    print(
        "tokens:",
        sum(u["prompt_tokens"] for u in usages),
        "in,",
        sum(u["completion_tokens"] for u in usages),
        "out over",
        len(usages),
        "calls",
    )


asyncio.run(main())
