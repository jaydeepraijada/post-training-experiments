import argparse
import json
from pathlib import Path

CALC_SCORE_KEYS = ["faithfulness", "answer_correctness", "relevance", "completeness"]


def infer_model_name(path: Path) -> str:
    name = path.stem
    for suffix in ("_mlx_results_judged", "_results_judged", "_judged"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def load_summary(path: Path) -> dict:
    rows = []
    with path.open() as f:
        for line in f:
            record = json.loads(line)
            if "scores" in record:
                rows.append(record["scores"])

    if not rows:
        raise ValueError(f"No scored rows found in {path}")

    averages = {key: sum(row[key] for row in rows) / len(rows) for key in CALC_SCORE_KEYS}
    averages["overall"] = sum(averages[key] for key in CALC_SCORE_KEYS) / len(CALC_SCORE_KEYS)

    return {"model": infer_model_name(path), "file": path.name, "count": len(rows), **averages}


def main() -> None:
    parser = argparse.ArgumentParser(description="Print aggregate judge scores.")
    parser.add_argument("--eval-dir", type=Path, default=Path("evals"))
    args = parser.parse_args()

    files = sorted(args.eval_dir.glob("*_judged.jsonl"))
    if not files:
        raise SystemExit(f"No *_judged.jsonl files found in {args.eval_dir}")

    summaries = sorted([load_summary(p) for p in files], key=lambda r: r["overall"], reverse=True)

    headers = ["model", "count", "overall", *CALC_SCORE_KEYS, "file"]
    rows = [
        [s["model"], str(s["count"]), f'{s["overall"]:.2f}', *(f"{s[k]:.2f}" for k in CALC_SCORE_KEYS), s["file"]]
        for s in summaries
    ]

    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]

    def fmt(values):
        return "  ".join(v.ljust(widths[i]) for i, v in enumerate(values))

    print(fmt(headers))
    print(fmt(["-" * w for w in widths]))
    for row in rows:
        print(fmt(row))


if __name__ == "__main__":
    main()
