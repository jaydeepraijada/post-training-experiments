from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from preference_optimization.chat_formatting import (  # noqa: E402
        build_reward_conversations,
        normalize_explicit_preference_example,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(PROJECT_ROOT / "preference_optimization"))
    from chat_formatting import (  # type: ignore  # noqa: E402
        build_reward_conversations,
        normalize_explicit_preference_example,
    )

SEED = 3407


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a reward model on preference pairs.")
    parser.add_argument("--model", default="models/preference/reward2/final")
    parser.add_argument("--dataset", default="paperbd/paper_preference_150K-v1")
    parser.add_argument("--num-examples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument(
        "--split",
        choices=["dpo_test", "reward_test"],
        default="dpo_test",
        help="Use the DPO/ORPO 2%% test split or reward trainer 5%% eval split.",
    )
    parser.add_argument(
        "--output",
        default="preference_optimization/evals/reward2_final_dpo_test_100.jsonl",
    )
    return parser.parse_args()


def score_batch(model, tokenizer, conversations, max_length: int, device: torch.device) -> torch.Tensor:
    texts = [
        tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False,
        )
        for conversation in conversations
    ]
    batch = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    batch = {key: value.to(device) for key, value in batch.items()}
    with torch.inference_mode():
        logits = model(**batch).logits
    return logits.reshape(-1).detach().cpu()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    adapter_config_path = model_path / "adapter_config.json"
    is_adapter = adapter_config_path.exists()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_adapter:
        with adapter_config_path.open() as f:
            adapter_config = json.load(f)
        base_model_id = adapter_config["base_model_name_or_path"]
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_id,
            num_labels=1,
            torch_dtype=dtype,
        )
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        base_model_id = str(model_path)
        config = AutoConfig.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=getattr(config, "num_labels", 1),
            torch_dtype=dtype,
        )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.eval()

    dataset = load_dataset(args.dataset, split="train")
    dataset = dataset.map(
        normalize_explicit_preference_example,
        remove_columns=[
            column
            for column in dataset.column_names
            if column not in {"prompt", "chosen", "rejected"}
        ],
    )
    test_size = 0.02 if args.split == "dpo_test" else 0.05
    test_dataset = dataset.train_test_split(test_size=test_size, seed=SEED)["test"]
    test_dataset = test_dataset.select(range(min(args.num_examples, len(test_dataset))))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    correct = 0
    ties = 0
    rows: list[dict] = []
    for start in range(0, len(test_dataset), args.batch_size):
        examples = test_dataset.select(range(start, min(start + args.batch_size, len(test_dataset))))
        chosen_conversations = []
        rejected_conversations = []
        for example in examples:
            chosen, rejected = build_reward_conversations(example)
            chosen_conversations.append(chosen)
            rejected_conversations.append(rejected)

        chosen_scores = score_batch(
            model, tokenizer, chosen_conversations, args.max_length, device
        )
        rejected_scores = score_batch(
            model, tokenizer, rejected_conversations, args.max_length, device
        )

        for offset, (example, chosen_score, rejected_score) in enumerate(
            zip(examples, chosen_scores.tolist(), rejected_scores.tolist(), strict=True)
        ):
            margin = chosen_score - rejected_score
            is_correct = margin > 0
            is_tie = margin == 0
            correct += int(is_correct)
            ties += int(is_tie)
            rows.append(
                {
                    "index": start + offset,
                    "chosen_score": chosen_score,
                    "rejected_score": rejected_score,
                    "margin": margin,
                    "correct": is_correct,
                    "prompt": example["prompt"],
                    "chosen": example["chosen"],
                    "rejected": example["rejected"],
                }
            )
        print(f"scored {min(start + args.batch_size, len(test_dataset))}/{len(test_dataset)}")

    with output_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n = len(rows)
    accuracy = correct / n if n else 0.0
    avg_margin = sum(row["margin"] for row in rows) / n if n else 0.0
    print()
    print(f"Model: {args.model}")
    print(f"Base: {base_model_id}")
    print(f"Loaded as: {'adapter' if is_adapter else 'merged'}")
    print(f"Dataset split: {args.split}")
    print(f"Examples: {n}")
    print(f"Accuracy: {correct}/{n} ({accuracy:.1%})")
    print(f"Ties: {ties}")
    print(f"Average margin: {avg_margin:.4f}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
