"""
Augment low-quality training data by asking DeepSeek to rewrite 1-3★
responses as 5★ versions with minimal changes.
"""

import argparse
import json
import asyncio
import os
import random
from datasets import load_dataset
from openai import AsyncOpenAI

PROMPT = """You are correcting a low-quality AI response to make it score 5/5.

Below is:
- The correct reference answer (ground truth)
- The AI's low-quality response (scored {score}/5)

Rewrite the response to score 5/5 with MINIMAL changes:
1. Fix ONLY the factual errors and missing information
2. Keep the exact same response format (JSON stays JSON, markdown stays markdown, Q&A stays Q&A)
3. Keep the same sentence structure and vocabulary where correct
4. Do NOT make it longer — be as concise as the original
5. Do NOT copy the reference verbatim — keep the AI's voice

Return ONLY the corrected response. No explanation, no markdown fences.

Reference (ground truth): {reference}
Low-quality response ({score}/5): {response}
"""

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset", "-d", type=str, default="paperbd/paper_answers_reward"
)
parser.add_argument("--output", "-o", type=str, default="data/augmented_5star.jsonl")
parser.add_argument("--num", "-n", type=int, default=1000)
parser.add_argument("--semaphore", "-s", type=int, default=50)
args = parser.parse_args()

client = AsyncOpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api/v1"
)


async def generate(reference, response, score):
    prompt = PROMPT.format(
        reference=reference, score=round(score, 1), response=response
    )
    for attempt in range(3):
        try:
            resp = await client.chat.completions.create(
                model="deepseek/deepseek-v4-flash",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.4,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            if attempt == 2:
                return None
            await asyncio.sleep(1)


async def main():
    # Load from HF
    ds = load_dataset(args.dataset, split="train")

    def to_str(x):
        if isinstance(x, str):
            return x
        if isinstance(x, (list, dict)):
            return json.dumps(x)
        return str(x)

    low = [
        r
        for r in ds
        if 1.0 <= r["orig_score"] <= 3.0 and r["orig_score"] not in (2.5, 3.0)
    ]
    rng = random.Random(42)
    rng.shuffle(low)
    sample = low[: args.num]
    print(f"Low-quality records available: {len(low)}, selected: {len(sample)}")

    # Resume
    done = set()
    if os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                done.add(json.loads(line)["orig_response"])

    pending = [r for r in sample if to_str(r["orig_response"]) not in done]
    print(f"Already done: {len(done)}, pending: {len(pending)}")

    sem = asyncio.Semaphore(args.semaphore)
    out_file = open(args.output, "a")

    async def process_one(r):
        ref = to_str(r["orig_reference_answer"])
        resp = to_str(r["orig_response"])
        async with sem:
            improved = await generate(ref, resp, r["orig_score"])
        if improved and improved != resp:
            out_file.write(
                json.dumps(
                    {
                        "orig_reference_answer": ref,
                        "orig_response": improved,
                        "orig_score": 5.0,
                    }
                )
                + "\n"
            )
            out_file.flush()

    tasks = [process_one(r) for r in pending]
    for i, coro in enumerate(asyncio.as_completed(tasks)):
        await coro
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(pending)}")

    out_file.close()
    print(f"Done → {args.output}")


asyncio.run(main())
