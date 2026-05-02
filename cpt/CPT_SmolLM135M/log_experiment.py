"""Append an experiment run to experiments.json and regenerate experiments.md."""
import argparse
import json
import os
from datetime import datetime


def load_json(path):
    with open(path) as f:
        return json.load(f)


def _fmt(v, decimals=4):
    return f"{v:.{decimals}f}" if isinstance(v, float) else str(v)


_AUTO_MARKER = "<!-- AUTO-GENERATED: do not edit below this line -->"


def write_md(experiments, path="experiments.md"):
    # Preserve any hand-written content above the marker.
    header = ""
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            raw = f.read()
        if _AUTO_MARKER in raw:
            header = raw.split(_AUTO_MARKER)[0].rstrip() + "\n\n"

    lines = [
        _AUTO_MARKER,
        "",
        "## Results\n",
        "| Run | Date | Hypothesis | PPL | ROUGE-1 | ROUGE-L | BERTScore F1 | Notes |",
        "|-----|------|------------|-----|---------|---------|--------------|-------|",
    ]
    for e in experiments:
        metrics_flat = _flatten_metrics(e.get("metrics", {}))
        ppl = _fmt(metrics_flat.get("perplexity")) if metrics_flat.get("perplexity") is not None else "N/A"
        r1  = _fmt(metrics_flat.get("rouge1", 0.0))
        rl  = _fmt(metrics_flat.get("rougeL", 0.0))
        bs  = _fmt(metrics_flat.get("bertscore_f1", 0.0))
        hyp = e.get("hypothesis", "").replace("|", "\\|")
        notes = e.get("notes", "").replace("|", "\\|")
        lines.append(f"| {e['run_name']} | {e['date']} | {hyp} | {ppl} | {r1} | {rl} | {bs} | {notes} |")

    lines.append("")
    lines.append("---\n")

    for e in experiments:
        lines.append(f"## {e['run_name']}  ({e['date']})")
        lines.append(f"\n**Hypothesis:** {e.get('hypothesis', '')}\n")

        cfg = e.get("config", {})
        if cfg:
            lines.append("**Config:**\n")
            for k, v in cfg.items():
                if k != "wandb_run_id":
                    lines.append(f"- `{k}`: {v}")
            if cfg.get("wandb_run_id"):
                lines.append(f"- W&B run: `{cfg['wandb_run_id']}`")

        lines.append("\n**Metrics:**\n")
        for model_name, scores in e.get("metrics", {}).items():
            lines.append(f"*{model_name}*\n")
            for k, v in scores.items():
                if v is not None:
                    lines.append(f"- {k}: {_fmt(v)}")
            lines.append("")

        lines.append(f"**Notes:** {e.get('notes', '')}\n")
        lines.append("---\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(lines))


def _flatten_metrics(metrics_dict):
    """Merge all per-model metric dicts — last one wins for duplicate keys."""
    flat = {}
    for scores in metrics_dict.values():
        flat.update(scores)
    return flat


def main():
    parser = argparse.ArgumentParser(description="Log an experiment run to experiments.json and experiments.md.")
    parser.add_argument("--results", required=True, help="Path to eval results JSON (from inference.py).")
    parser.add_argument("--run_name", required=True, help="Unique name for this experiment run.")
    parser.add_argument("--hypothesis", default="", help="What you were testing in this run.")
    parser.add_argument("--notes", default="", help="Observations and conclusions after seeing results.")
    parser.add_argument("--train_config", default=None, help="Path to train_config.json. Auto-detected from models/<run_name>/train_config.json if omitted.")
    parser.add_argument("--log_file", default="experiments.json", help="Path to experiments.json.")
    parser.add_argument("--md_file", default="experiments.md", help="Path to experiments.md.")
    args = parser.parse_args()

    results = load_json(args.results)

    config_path = args.train_config or f"models/{args.run_name}/train_config.json"
    config = load_json(config_path) if os.path.exists(config_path) else {}

    entry = {
        "run_name": args.run_name,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "hypothesis": args.hypothesis,
        "config": config,
        "metrics": results,
        "notes": args.notes,
    }

    experiments = []
    if os.path.exists(args.log_file):
        experiments = load_json(args.log_file)

    updated = False
    for i, e in enumerate(experiments):
        if e["run_name"] == args.run_name:
            experiments[i] = entry
            updated = True
            break
    if not updated:
        experiments.append(entry)

    with open(args.log_file, "w") as f:
        json.dump(experiments, f, indent=2)

    write_md(experiments, args.md_file)

    action = "Updated" if updated else "Added"
    print(f"{action} '{args.run_name}' in {args.log_file} and {args.md_file}")


if __name__ == "__main__":
    main()
