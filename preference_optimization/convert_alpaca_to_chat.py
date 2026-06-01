"""Convert Alpaca-format JSONL to role/content message-list JSONL.

Usage:
    uv run python preference_optimization/convert_alpaca_to_chat.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert Alpaca-format JSONL to chat message-list JSONL."
    )
    p.add_argument(
        "--input_file",
        "-i",
        type=str,
        default="data/sft_data/test.jsonl",
    )
    p.add_argument(
        "--output_file",
        "-o",
        type=str,
        default="data_sft/data_sft.jsonl",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    in_path = Path(args.input_file)
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with open(in_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            rec = json.loads(line)
            instruction = rec.get("instruction", "")
            inp = rec.get("input", "")
            output = rec.get("output", "")

            prompt_text = instruction if not inp else f"{instruction}\n\n{inp}"

            chat_messages = [
                {"role": "system", "content": "You are a helpful assistant. Answer the user's query truthfully"},
                {"role": "user", "content": prompt_text},
                {"role": "assistant", "content": output},
            ]

            fout.write(json.dumps({"messages": chat_messages}, ensure_ascii=False) + "\n")
            written += 1

    print(f"Done. {written} records converted.")
    print(f"  Input : {in_path}")
    print(f"  Output: {out_path}")


if __name__ == "__main__":
    main()
