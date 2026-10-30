from pathlib import Path
import sys

from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from reasoning.env import format_row
except ModuleNotFoundError:
    from env import format_row


SEED = 3407


class PaperInstructionDataset(Dataset):
    def __init__(
        self,
        dataset_name_or_path,
        split="train",
        tokenizer=None,
        data_size=None,
        seed=SEED,
    ):
        self.data = _load_dataset(dataset_name_or_path, split)
        if seed is not None:
            self.data = self.data.shuffle(seed=seed)
        if data_size is not None:
            self.data = self.data.select(range(min(data_size, len(self.data))))
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        row = format_row(self.data[i])
        data = {
            "prompt": row["prompt"],
            "task_prompt": [
                message for message in row["prompt"]
                if message.get("role") != "system"
            ],
            "answer": row["answer"],
            "item": row,
        }
        if self.tokenizer is not None:
            tokenized = self.tokenizer(
                self.tokenizer.apply_chat_template(
                    data["prompt"],
                    tokenize=False,
                    add_generation_prompt=True,
                ),
                return_tensors="pt",
            )
            data["input_ids"] = tokenized["input_ids"]
            data["attention_mask"] = tokenized["attention_mask"]
        return data


def _load_dataset(dataset_name_or_path, split):
    path = Path(dataset_name_or_path)
    if path.exists():
        if path.is_file():
            return load_dataset("json", data_files=str(path), split="train")
        split_path = path / f"{split}.jsonl"
        if split_path.exists():
            return load_dataset("json", data_files=str(split_path), split="train")
    return load_dataset(dataset_name_or_path, split=split)


def collate_fn(batch, pad_token_id):
    return {
        "prompt": [item["prompt"] for item in batch],
        "task_prompt": [item["task_prompt"] for item in batch],
        "answer": [item["answer"] for item in batch],
        "item": [item["item"] for item in batch],
        "input_ids": pad_sequence(
            [item["input_ids"][0] for item in batch],
            batch_first=True,
            padding_value=pad_token_id,
            padding_side="left",
        ),
        "attention_mask": pad_sequence(
            [item["attention_mask"][0] for item in batch],
            batch_first=True,
            padding_value=0,
            padding_side="left",
        ),
    }


def get_dataloader(
    dataset_name_or_path,
    batch_size=32,
    tokenizer=None,
    split="train",
    data_size=None,
    seed=SEED,
    **kwargs,
):
    del kwargs
    dataset = PaperInstructionDataset(
        dataset_name_or_path,
        split=split,
        tokenizer=tokenizer,
        data_size=data_size,
        seed=seed,
    )
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.eos_token_id),
    )
