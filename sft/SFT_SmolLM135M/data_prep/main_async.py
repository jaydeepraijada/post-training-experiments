import argparse
import asyncio
import os
import random

from text_albumentations import OutlinesModel, arun_augmentation, save_dataset
from text_albumentations.tasks import (
    bullet_augmentation,
    comparison_augmentation,
    continuation_augmentation,
    qa_pair_augmentation,
    rephrase_augmentation,
    retrieval_augmentation,
    triplet_augmentation,
)
from main import PROB_TO_RUN_REPHRASE, PROB_TO_RUN_STEP, chunk_text, load_texts_from_jsonl, truncate_dataset

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_jsonl")
    parser.add_argument("output_jsonl")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--model-name", type=str, default=os.environ.get("TEXT_ALBUMENTATIONS_MODEL", "gpt-4.1-nano"))
    parser.add_argument("--provider", choices=["openai", "openrouter"], default=os.environ.get("LLM_PROVIDER", "openai"))
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--total-concurrent-calls", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_client_config(args):
    if args.provider == "openrouter":
        return {"api_key": args.api_key or os.environ.get("OPENROUTER_API_KEY"), "base_url": args.base_url or OPENROUTER_BASE_URL}
    return {"api_key": args.api_key or os.environ.get("OPENAI_API_KEY"), "base_url": args.base_url or os.environ.get("OPENAI_BASE_URL")}


def build_async_runtime(args):
    import openai
    import outlines
    client_kwargs = {k: v for k, v in resolve_client_config(args).items() if v}
    client = openai.AsyncOpenAI(**client_kwargs)
    model = outlines.from_openai(client, args.model_name)
    return OutlinesModel(model, async_mode=True, total_concurrent_calls=args.total_concurrent_calls, max_tokens_parameter="max_completion_tokens")


async def generate_examples_for_chunks_async(chunks: list[str], runtime):
    tasks = []
    single_chunk_tasks = [
        ("bullets", bullet_augmentation, PROB_TO_RUN_STEP),
        ("qa_pairs", qa_pair_augmentation, PROB_TO_RUN_STEP),
        ("rephrase", rephrase_augmentation, PROB_TO_RUN_REPHRASE),
        ("continuation", continuation_augmentation, PROB_TO_RUN_STEP),
        ("triplets", triplet_augmentation, PROB_TO_RUN_STEP),
    ]
    for task_name, augmentation, probability in single_chunk_tasks:
        selected = [c for c in chunks if random.random() < probability]
        if not selected:
            continue
        print(f"Scheduling {task_name} on {len(selected)} chunks")
        for chunk in selected:
            tasks.append(arun_augmentation(chunk, augmentation, runtime))

    if not tasks:
        return []

    results = await asyncio.gather(*tasks, return_exceptions=True)
    dataset = []
    for result in results:
        if isinstance(result, KeyboardInterrupt):
            raise result
        if isinstance(result, Exception):
            print(f"Skipping: {result}")
            continue
        dataset.extend(result)
    return dataset


async def generate_cross_chunk_examples_async(chunks: list[str], runtime):
    tasks = []
    if len(chunks) >= 2:
        if random.random() < PROB_TO_RUN_STEP:
            tasks.append(arun_augmentation(chunks, retrieval_augmentation, runtime))
        if random.random() < PROB_TO_RUN_REPHRASE:
            left_idx, right_idx = random.sample(range(len(chunks)), 2)
            tasks.append(arun_augmentation([chunks[left_idx], chunks[right_idx]], comparison_augmentation, runtime))

    if not tasks:
        return []

    results = await asyncio.gather(*tasks, return_exceptions=True)
    dataset = []
    for result in results:
        if isinstance(result, KeyboardInterrupt):
            raise result
        if isinstance(result, Exception):
            print(f"Skipping: {result}")
            continue
        dataset.extend(result)
    return dataset


async def amain():
    args = parse_args()
    random.seed(args.seed)
    runtime = build_async_runtime(args)
    texts = load_texts_from_jsonl(args.input_jsonl)
    texts = texts[args.start_index:]
    print(f"Loaded {len(texts)} texts | provider={args.provider} | model={args.model_name}")

    total_examples = 0
    for text_idx, text in enumerate(texts, start=1):
        print(f"Processing text {text_idx}/{len(texts)}")
        chunks = chunk_text(text, args.chunk_size, args.chunk_overlap)
        if not chunks:
            continue
        chunks = chunks[: int(len(chunks) // 2)]

        dataset = await generate_examples_for_chunks_async(chunks, runtime)
        if args.max_rows is not None:
            remaining = args.max_rows - total_examples
            if remaining <= 0:
                break
            dataset = truncate_dataset(dataset, remaining)
        total_examples += len(dataset)
        save_dataset(dataset, args.output_jsonl)
        print(f"Total saved: {total_examples}")

        if args.max_rows is not None and total_examples >= args.max_rows:
            break

        cross_dataset = await generate_cross_chunk_examples_async(chunks, runtime)
        if args.max_rows is not None:
            remaining = args.max_rows - total_examples
            if remaining <= 0:
                break
            cross_dataset = truncate_dataset(cross_dataset, remaining)
        total_examples += len(cross_dataset)
        save_dataset(cross_dataset, args.output_jsonl)
        print(f"Total saved: {total_examples}")

        if args.max_rows is not None and total_examples >= args.max_rows:
            break

    print(f"Done. Generated {total_examples} examples.")


if __name__ == "__main__":
    asyncio.run(amain())
