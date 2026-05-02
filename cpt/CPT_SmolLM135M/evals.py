import argparse
import json
from rich.console import Console
from rich.table import Table
from rich import box
import evaluate

try:
    rouge = evaluate.load('rouge')
    bleu = evaluate.load('bleu')
    bertscore = evaluate.load('bertscore')
except Exception as e:
    print(f"Could not load metrics. Run: uv pip install evaluate rouge_score bert_score absl-py sacrebleu")
    print(f"Error: {e}")
    exit(1)


def calculate_metrics(predictions, references):
    results = {}

    rouge_res = rouge.compute(predictions=predictions, references=references)
    results['rouge1'] = rouge_res['rouge1']
    results['rouge2'] = rouge_res['rouge2']
    results['rougeL'] = rouge_res['rougeL']

    bleu_res = bleu.compute(predictions=predictions, references=references)
    results['bleu'] = bleu_res['bleu']

    bert_res = bertscore.compute(predictions=predictions, references=references, lang="en", model_type="distilbert-base-uncased")
    results['bertscore_f1'] = sum(bert_res['f1']) / len(bert_res['f1'])

    return results


def main():
    parser = argparse.ArgumentParser(description="Run evaluations on generated JSON.")
    parser.add_argument("--input_json", type=str, default="inference_generations.json")
    parser.add_argument("--output_results", type=str, default="eval_results.json")
    args = parser.parse_args()

    console = Console()

    with open(args.input_json, 'r') as f:
        data = json.load(f)

    if not data:
        console.print("[bold red]No data found.[/bold red]")
        return

    model_names = list(data[0]["predictions"].keys())
    console.print(f"Models: [cyan]{', '.join(model_names)}[/cyan]")

    model_data = {name: {"preds": [], "refs": []} for name in model_names}
    for item in data:
        for name in model_names:
            if name in item["predictions"]:
                model_data[name]["preds"].append(item["predictions"][name])
                model_data[name]["refs"].append(item["ground_truth"])

    console.rule("[bold blue]Running Evaluations")

    final_results = {"aggregate": {}, "instances": []}

    table = Table(title="Evaluation Results", box=box.ROUNDED)
    table.add_column("Model", style="cyan")
    table.add_column("ROUGE-1", style="green")
    table.add_column("ROUGE-L", style="green")
    table.add_column("BLEU", style="yellow")
    table.add_column("BERTScore F1", style="magenta")

    for model_name in model_names:
        preds = model_data[model_name]["preds"]
        refs = model_data[model_name]["refs"]
        if not preds:
            continue
        console.print(f"Evaluating {model_name}...")
        scores = calculate_metrics(preds, refs)
        final_results["aggregate"][model_name] = scores
        table.add_row(
            model_name,
            f"{scores['rouge1']:.4f}",
            f"{scores['rougeL']:.4f}",
            f"{scores['bleu']:.4f}",
            f"{scores.get('bertscore_f1', 0):.4f}",
        )

    console.print(table)

    console.print("\n[yellow]Per-instance metrics...[/yellow]")
    for item in data:
        instance_entry = {
            "id": item.get("id"),
            "prefix": item["prefix"],
            "ground_truth": item["ground_truth"],
            "predictions": item["predictions"],
            "metrics": {},
        }
        ref = item["ground_truth"]
        for model_name, pred in item["predictions"].items():
            r_res = rouge.compute(predictions=[pred], references=[ref])
            b_res = bleu.compute(predictions=[pred], references=[ref])
            instance_entry["metrics"][model_name] = {
                "rouge1": r_res['rouge1'],
                "rougeL": r_res['rougeL'],
                "bleu": b_res['bleu'],
            }
        final_results["instances"].append(instance_entry)

    with open(args.output_results, 'w') as f:
        json.dump(final_results, f, indent=4)

    console.print(f"\n[bold green]Saved to {args.output_results}[/bold green]")


if __name__ == "__main__":
    main()
