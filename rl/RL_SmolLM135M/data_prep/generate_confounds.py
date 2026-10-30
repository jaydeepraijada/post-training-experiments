import argparse
import json
import asyncio
import os
import random
from openai import AsyncOpenAI

parser = argparse.ArgumentParser()
parser.add_argument("--input", "-i", type=str, required=True, help="JSONL file with responses to confound")
parser.add_argument("--output", "-o", type=str, default=None)
parser.add_argument("--num_per_ref", "-n", type=int, default=3, help="Confounds per reference")
parser.add_argument("--total", "-t", type=int, default=None, help="Total confounds to generate")
parser.add_argument("--semaphore", "-s", type=int, default=100)
args = parser.parse_args()

if args.output is None:
    args.output = args.input.replace(".jsonl", "_confounds.jsonl")

MODEL = "deepseek/deepseek-v4-flash"

PROMPT = """You are helping create training data for a reward model.

Below is a reference answer (correct). Generate {n} CONFOUNDING responses that share vocabulary with the reference but are FACTUALLY WRONG.

Make the errors OBVIOUS and DETECTABLE — do NOT be subtle. Target:
- KEYWORDS: swap critical terms (e.g. "combines" → "replaces", "increases" → "decreases")
- KEY RELATIONSHIPS: negate or invert the main claim
- NUMBERS: change key metrics or values
- NAMES: swap model/dataset/method names

A reader should immediately notice the error upon comparing with the reference.

Return ONLY a JSON array of exactly {n} strings. No explanation, no markdown.

Reference: {reference}
"""

client = AsyncOpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)


async def generate(reference, n):
    for attempt in range(3):
        try:
            resp = await client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": PROMPT.format(n=n, reference=reference)}],
                max_tokens=2048,
                temperature=0.9,
            )
            text = resp.choices[0].message.content.strip()
            if text.startswith("```"):
                parts = text.split("\n", 1)
                text = parts[1] if len(parts) > 1 else text
                if text.endswith("```"):
                    text = text[:-3]
            confounds = json.loads(text.strip())
            if isinstance(confounds, list) and len(confounds) >= 1:
                return confounds[:n]
            return None
        except json.JSONDecodeError:
            if attempt == 2:
                return None
            await asyncio.sleep(1)
        except Exception as e:
            if attempt == 2:
                return None
            await asyncio.sleep(2)
    return None


async def main():
    # Load inputs
    with open(args.input) as f:
        records = [json.loads(l) for l in f]

    # Extract unique references (responses from the model)
    refs = list(set(r.get("response", r.get("orig_response", "")) for r in records if r.get("response", "") or r.get("orig_response", "")))
    # Filter short/empty
    refs = [r for r in refs if len(r.split()) > 5]

    # Limit total
    rng = random.Random(42)
    rng.shuffle(refs)
    if args.total:
        refs = refs[:max(1, args.total // args.num_per_ref)]

    # Resume
    done = set()
    if os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                done.add(json.loads(line)["orig_reference_answer"])

    pending = [r for r in refs if r not in done]
    print(f"References: {len(refs)}, already done: {len(done)}, pending: {len(pending)}")

    sem = asyncio.Semaphore(args.semaphore)
    out_file = open(args.output, "a")

    async def process_one(ref):
        async with sem:
            confounds = await generate(ref, args.num_per_ref)
        if confounds:
            for bad in confounds:
                out_file.write(json.dumps({
                    "orig_reference_answer": ref,
                    "orig_response": bad,
                    "orig_score": 3.0,
                }) + "\n")
            out_file.flush()

    tasks = [process_one(r) for r in pending]
    for i, coro in enumerate(asyncio.as_completed(tasks)):
        await coro
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(pending)}")

    out_file.close()
    print(f"Done → {args.output}")


asyncio.run(main())
