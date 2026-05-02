import argparse
import json
import re
import random
from pathlib import Path

random.seed(42)


def process_text(text, filename):
    """Remove the references section, preserving the appendix if present."""
    ref_matches = list(re.finditer(r"(?i)\breferences\b", text))
    if not ref_matches:
        print(f"{filename.name}: Removed 0.00%")
        return text

    last_ref = ref_matches[-1]
    ref_start = last_ref.start()
    total_len = len(text)

    # Always look for appendix content after the references section.
    app_match = re.search(r"(?i)\bappendix\b", text[ref_start:])

    if app_match:
        app_start = ref_start + app_match.start()
        removed_len = app_start - ref_start
        if removed_len <= 0.3 * total_len:
            print(f"{filename.name}: Removed {removed_len / total_len * 100:.2f}% (kept appendix)")
            return text[:ref_start] + text[app_start:]

    # No appendix (or it was too large to trust) — remove from references onward.
    if ref_start > 0.7 * total_len:
        thrown_out_len = total_len - ref_start
        if thrown_out_len <= 0.3 * total_len:
            print(f"{filename.name}: Removed {thrown_out_len / total_len * 100:.2f}%")
            return text[:ref_start]

    print(f"{filename.name}: Removed 0.00%")
    return text


def main():
    parser = argparse.ArgumentParser(description="Build train/val JSONL splits from a directory of .txt papers.")
    parser.add_argument("directory", help="Directory containing .txt paper files.")
    parser.add_argument("--num_test", type=int, default=50, help="Number of papers reserved for validation.")
    parser.add_argument("--no_clean", action="store_true", help="Skip reference/appendix cleaning.")
    args = parser.parse_args()

    directory = Path(args.directory)
    all_files = list(directory.rglob("*.txt"))
    num_files = len(all_files)
    random.shuffle(all_files)
    num_test_files = args.num_test

    print(f"Train files: {num_files - num_test_files}, Test files: {num_test_files}")

    def get_text(path):
        raw = open(path).read()
        return raw if args.no_clean else process_text(raw, path)

    with open(f"cpt_val_dataset_{num_test_files}.jsonl", "a") as f:
        for a in all_files[:num_test_files]:
            f.write(json.dumps({"text": get_text(a)}) + "\n")

    with open(f"cpt_train_dataset_{num_files - num_test_files}.jsonl", "a") as f:
        for a in all_files[num_test_files:]:
            f.write(json.dumps({"text": get_text(a)}) + "\n")


if __name__ == "__main__":
    main()
