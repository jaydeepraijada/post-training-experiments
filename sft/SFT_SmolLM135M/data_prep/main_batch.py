import argparse
import random
from functools import lru_cache

import outlines
from text_albumentations import OutlinesModel, run_augmentation, run_batch_augmentation
from transformers import AutoModelForCausalLM, AutoTokenizer
from main import (
    PROB_TO_RUN_REPHRASE,
    PROB_TO_RUN_STEP,
    bullet_augmentation,
    chunk_text,
    comparison_augmentation,
    continuation_augmentation,
    load_texts_from_jsonl,
    qa_pair_augmentation,
    rephrase_augmentation,
    retrieval_augmentation,
    save_dataset,
    try_generate,
    triplet_augmentation,
    truncate_dataset,
)

MODEL_NAME = "google/gemma-3-1b-it"


def batched(items, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


@lru_cache(maxsize=1)
def get_batch_runtime():
    print(f"Loading {MODEL_NAME}...")
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="auto", device_map="auto")
    hf_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = outlines.from_transformers(hf_model, hf_tokenizer)
    return OutlinesModel(model, max_tokens_parameter="max_new_tokens")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_jsonl")
    parser.add_argument("output_jsonl")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def generate_examples_batch(chunks: list[str], batch_size: int):
    runtime = get_batch_runtime()
    dataset = []
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
        print(f"Batch {task_name} on {len(selected)} chunks")
        for chunk_batch in batched(selected, batch_size):
            dataset.extend(try_generate(
                f"{task_name} batch",
                lambda a=augmentation, b=chunk_batch: run_batch_augmentation(b, a, runtime),
            ))
    return dataset


def generate_cross_chunk_examples(chunks: list[str]):
    runtime = get_batch_runtime()
    dataset = []
    if len(chunks) >= 2:
        if random.random() < PROB_TO_RUN_STEP:
            dataset.extend(try_generate("retrieval", lambda: run_augmentation(chunks, retrieval_augmentation, runtime)))
        if random.random() < PROB_TO_RUN_REPHRASE:
            left_idx, right_idx = random.sample(range(len(chunks)), 2)
            dataset.extend(try_generate("comparison", lambda: run_augmentation([chunks[left_idx], chunks[right_idx]], comparison_augmentation, runtime)))
    return dataset


def main():
    args = parse_args()
    random.seed(args.seed)
    texts = load_texts_from_jsonl(args.input_jsonl)
    texts = texts[args.start_index:]
    print(f"Loaded {len(texts)} texts | batch_model={MODEL_NAME}")

    total_examples = 0
    for text_idx, text in enumerate(texts, start=1):
        print(f"Processing text {text_idx}/{len(texts)}")
        chunks = chunk_text(text, args.chunk_size, args.chunk_overlap)
        if not chunks:
            continue
        chunks = chunks[: int(len(chunks) // 2)]

        dataset = generate_examples_batch(chunks, args.batch_size)
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

        cross_dataset = generate_cross_chunk_examples(chunks)
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
    main()
