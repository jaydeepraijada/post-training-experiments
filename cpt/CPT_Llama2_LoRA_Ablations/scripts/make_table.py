"""Generate a markdown results table from results/runs.csv.

If multiple seeds were run per config, reports mean +/- std of eval perplexity.
Prints to stdout — paste into the README, or redirect to results/table.md.
"""
from __future__ import annotations

import csv
import os
import statistics
from collections import defaultdict

CSV = os.path.join(os.path.dirname(__file__), "..", "results", "runs.csv")


def main():
    if not os.path.exists(CSV):
        raise SystemExit(f"No results yet at {CSV}. Run the sweep first.")

    by_name: dict[str, list[dict]] = defaultdict(list)
    with open(CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_name[row["name"]].append(row)

    print("| Experiment | Trainable | Val perplexity | Eval loss | Peak VRAM | Seeds |")
    print("|---|---|---|---|---|---|")
    for name in sorted(by_name):
        rows = by_name[name]
        ppls = [float(r["eval_perplexity"]) for r in rows]
        losses = [float(r["eval_loss"]) for r in rows]
        params = int(rows[0]["trainable_params"])
        vram = max(float(r["peak_vram_gb"]) for r in rows)
        if len(ppls) > 1:
            ppl_str = f"{statistics.mean(ppls):.4f} ± {statistics.stdev(ppls):.4f}"
            loss_str = f"{statistics.mean(losses):.4f}"
        else:
            ppl_str = f"{ppls[0]:.4f}"
            loss_str = f"{losses[0]:.4f}"
        print(f"| {name} | {params/1e6:.1f}M | {ppl_str} | {loss_str} | {vram:.2f} GB | {len(rows)} |")


if __name__ == "__main__":
    main()
