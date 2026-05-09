import argparse
import json
import random
from text_albumentations import (
    get_default_outlines_runtime,
    run_augmentation,
    save_dataset,
)
from text_albumentations.tasks import (
    bullet_augmentation,
    comparison_augmentation,
    continuation_augmentation,
    qa_pair_augmentation,
    rephrase_augmentation,
    retrieval_augmentation,
    triplet_augmentation,
)

PROB_TO_RUN_STEP = 0.25
PROB_TO_RUN_REPHRASE = 0.1
RUNTIME = None


def get_runtime():
    global RUNTIME
    if RUNTIME is None:
        RUNTIME = get_default_outlines_runtime()
    return RUNTIME


def chunk_text(text: str, chunk_size_words: int = 500, overlap_words: int = 100) -> list[str]:
    words = text.split()
    if not words:
        return []
    if overlap_words >= chunk_size_words:
        raise ValueError("overlap_words must be smaller than chunk_size_words")
    step = chunk_size_words - overlap_words
    return [
        " ".join(words[idx : idx + chunk_size_words])
        for idx in range(0, len(words), step)
    ]


def load_texts_from_jsonl(path: str) -> list[str]:
    texts = []
    with open(path) as f:
        for line_number, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            row = json.loads(raw)
            text = row.get("text", "")
            if not isinstance(text, str):
                raise ValueError(f"Expected 'text' to be a string on line {line_number}")
            text = text.strip()
            if text:
                texts.append(text)
    return texts


def try_generate(label: str, fn):
    try:
        return fn()
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(f"Skipping {label}: {exc}")
        return []


def generate_examples_for_chunk(chunk: str):
    dataset = []
    if random.random() < PROB_TO_RUN_STEP:
        dataset.extend(try_generate("bullets", lambda: run_augmentation(chunk, bullet_augmentation, get_runtime())))
    if random.random() < PROB_TO_RUN_STEP:
        dataset.extend(try_generate("qa_pairs", lambda: run_augmentation(chunk, qa_pair_augmentation, get_runtime())))
    if random.random() < PROB_TO_RUN_REPHRASE:
        dataset.extend(try_generate("rephrase", lambda: run_augmentation(chunk, rephrase_augmentation, get_runtime())))
    if random.random() < PROB_TO_RUN_STEP:
        dataset.extend(try_generate("continuation", lambda: run_augmentation(chunk, continuation_augmentation, get_runtime())))
    if random.random() < PROB_TO_RUN_STEP:
        dataset.extend(try_generate("triplets", lambda: run_augmentation(chunk, triplet_augmentation, get_runtime())))
    return dataset


def generate_cross_chunk_examples(chunks: list[str]):
    dataset = []
    if len(chunks) >= 2:
        if random.random() < PROB_TO_RUN_STEP:
            dataset.extend(try_generate("retrieval", lambda: run_augmentation(chunks, retrieval_augmentation, get_runtime())))
        if random.random() < PROB_TO_RUN_REPHRASE:
            left_idx, right_idx = random.sample(range(len(chunks)), 2)
            dataset.extend(try_generate("comparison", lambda: run_augmentation([chunks[left_idx], chunks[right_idx]], comparison_augmentation, get_runtime())))
    return dataset


def truncate_dataset(dataset, remaining_rows):
    if remaining_rows is None:
        return dataset
    return dataset[:remaining_rows]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_jsonl")
    parser.add_argument("output_jsonl")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    texts = load_texts_from_jsonl(args.input_jsonl)
    print(f"Loaded {len(texts)} texts from {args.input_jsonl}")

    total_chunks = 0
    total_examples = 0
    texts = texts[args.start_index:]

    for text_idx, text in enumerate(texts, start=1):
        print(f"Processing text {text_idx}/{len(texts)}")
        chunks = chunk_text(text, args.chunk_size, args.chunk_overlap)
        if not chunks:
            continue

        total_chunks += len(chunks)
        chunks = chunks[: int(len(chunks) // 2)]

        for chunk_idx, chunk in enumerate(chunks, start=1):
            dataset = []
            try:
                dataset = generate_examples_for_chunk(chunk)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                print(f"Errored out for chunk {chunk_idx}: {exc}")

            if args.max_rows is not None:
                remaining_rows = args.max_rows - total_examples
                if remaining_rows <= 0:
                    break
                dataset = truncate_dataset(dataset, remaining_rows)

            total_examples += len(dataset)
            save_dataset(dataset, args.output_jsonl)

            if args.max_rows is not None and total_examples >= args.max_rows:
                break

        if args.max_rows is not None and total_examples >= args.max_rows:
            break

        dataset = generate_cross_chunk_examples(chunks)
        if args.max_rows is not None:
            remaining_rows = args.max_rows - total_examples
            if remaining_rows <= 0:
                break
            dataset = truncate_dataset(dataset, remaining_rows)

        total_examples += len(dataset)
        save_dataset(dataset, args.output_jsonl)

        if args.max_rows is not None and total_examples >= args.max_rows:
            break

    print(f"Generated {total_examples} examples from {total_chunks} chunks.")


if __name__ == "__main__":
    main()
