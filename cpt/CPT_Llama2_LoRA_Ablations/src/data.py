"""Dataset loading + formatting, shared identically by every run.

A single held-out split (seeded) gives us the validation set used for the
perplexity metric — the actual outcome we compare across ablations. Because
this lives in one place, every experiment sees byte-identical train/val data.
"""
from __future__ import annotations

from datasets import Dataset, load_dataset

# CPT format: raw "instruction / response" text. We are continuing pretraining,
# not instruction-tuning, so the format is plain text with an EOS terminator.
MAGICODER_PROMPT = """instruction: {}

response:
{}"""


def build_datasets(dataset_name: str, tokenizer, val_size: int, seed: int) -> tuple[Dataset, Dataset]:
    eos = tokenizer.eos_token

    def format_fn(examples):
        texts = [
            MAGICODER_PROMPT.format(instr, resp) + eos
            for instr, resp in zip(examples["instruction"], examples["response"])
        ]
        return {"text": texts}

    ds = load_dataset(dataset_name, split="train")
    ds = ds.map(format_fn, batched=True, num_proc=8)

    # Held-out validation split — fixed by `seed` so it is identical across runs.
    split = ds.train_test_split(test_size=val_size, seed=seed)
    return split["train"], split["test"]
