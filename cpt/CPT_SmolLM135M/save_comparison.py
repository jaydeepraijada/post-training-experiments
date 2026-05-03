"""Save comparison_results.json as a readable markdown file."""
import json

with open("comparison_results.json") as f:
    results = json.load(f)

lines = ["# Qualitative Comparison: Base vs Worst vs Best\n"]

for i, item in enumerate(results):
    lines.append(f"## Prompt {i+1}\n")
    lines.append(f"**Prompt:** {item['prompt']}\n")
    for model_name, generation in item["generations"].items():
        lines.append(f"### {model_name}")
        lines.append(f"{generation}\n")
    lines.append("---\n")

with open("comparison_results.md", "w") as f:
    f.write("\n".join(lines))

print("Saved to comparison_results.md")
