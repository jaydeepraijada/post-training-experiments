from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert ranked preference data into pairwise DPO-style preference dataset."
    )
    p.add_argument(
        "--input_file",
        "-i",
        type=str,
        default="data/pref_generation_data_ranked.jsonl",
    )
    p.add_argument(
        "--output_file",
        "-o",
        type=str,
        default=None,
        help="Defaults to <input>_dpo.jsonl",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.output_file is None:
        args.output_file = args.input_file.replace(".jsonl", "_dpo.jsonl")
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Pairs: 1st vs 4th, 2nd vs 4th, 1st vs 3rd, 1st vs 2nd
    # ranking_list is ordered best -> worst, so indices are:
    pair_offsets = [
        (0, 3),  # 1 vs 4
        (1, 3),  # 2 vs 4
        (0, 2),  # 1 vs 3
        (0, 1),  # 1 vs 2
    ]

    total_input = 0
    total_pairs = 0
    dropped = 0

    with open(args.input_file, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            rec = json.loads(line)
            total_input += 1

            if rec.get("ignore_example", False) or rec.get("ignore_all_samples", False):
                dropped += 1
                continue

            ranking = rec["ranking_list"]
            question = rec["question"]
            responses = rec["responses"]

            for chosen_idx, rejected_idx in pair_offsets:
                chosen_resp = responses[ranking[chosen_idx]]
                rejected_resp = responses[ranking[rejected_idx]]

                pref_example = {
                    "prompt": [{"role": "user", "content": question}],
                    "chosen": [{"role": "assistant", "content": chosen_resp}],
                    "rejected": [{"role": "assistant", "content": rejected_resp}],
                }
                fout.write(json.dumps(pref_example, ensure_ascii=False) + "\n")
                total_pairs += 1

    print(f"Done.")
    print(f"  Input records : {total_input}")
    print(f"  Dropped       : {dropped} (ignore_example or ignore_all_samples)")
    print(f"  Output pairs  : {total_pairs}")
    print(f"  Saved to      : {out_path}")


if __name__ == "__main__":
    main()
