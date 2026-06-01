"""
Plot diversity vs correctness / coherence results.

Usage:
    uv run preference_optimization/plot_diversity_analysis.py
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

# Load data
records = []
with open("preference_optimization/evals/smollm_135M_neuraltxt_v1_diversity_n100_r4_final.jsonl") as f:
    for line in f:
        records.append(json.loads(line))

vendi = [r["diversity"]["vendi"] for r in records]
ead   = [r["diversity"]["ead"]   for r in records]
sbert = [r["diversity"]["sbert"] for r in records]
corr  = [r["avg_correctness"]    for r in records]
coh   = [r["avg_coherence"]     for r in records]

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle(
    "Diversity vs Quality — smollm_135M_neuraltxt_v1 (100 prompts, 16 responses each)",
    fontsize=14,
)

metrics = [("EAD", ead), ("SBert", sbert), ("Vendi", vendi)]
col_titles = ["EAD", "SBert diversity", "Vendi Score"]

for col, (label, vals) in enumerate(metrics):
    ax_top = axes[0, col]
    ax_bot = axes[1, col]

    # Top row: vs correctness
    ax_top.scatter(vals, corr, alpha=0.5, s=40, edgecolors="none")
    r, p = pearsonr(vals, corr)
    ax_top.set_title(f"{col_titles[col]}  vs  Correctness\nr={r:.3f}, p={p:.3g}")
    ax_top.set_xlabel(col_titles[col])
    ax_top.set_ylabel("Avg Correctness (1–5)")
    ax_top.set_ylim(0.5, 5.5)

    # Bottom row: vs coherence
    ax_bot.scatter(vals, coh, alpha=0.5, s=40, edgecolors="none", color="C1")
    r, p = pearsonr(vals, coh)
    ax_bot.set_title(f"{col_titles[col]}  vs  Coherence\nr={r:.3f}, p={p:.3g}")
    ax_bot.set_xlabel(col_titles[col])
    ax_bot.set_ylabel("Avg Coherence (1–5)")
    ax_bot.set_ylim(0.5, 5.5)

fig.tight_layout(rect=[0, 0, 1, 0.95])
out_png = "preference_optimization/evals/diversity_vs_quality.png"
fig.savefig(out_png, dpi=200)
print(f"Saved plot to {out_png}")

# Per-temperature bar chart
fig2, ax2 = plt.subplots(figsize=(8, 4))
from collections import defaultdict

temp_stats = defaultdict(lambda: {"corr_sum": 0, "coh_sum": 0, "count": 0})
for r in records:
    temps = [0.3, 0.5, 0.7, 1.0]
    n_per_temp = 4
    for t_idx in range(len(temps)):
        for j in range(n_per_temp):
            resp_idx = t_idx * n_per_temp + j
            s = r["response_scores"][resp_idx]
            temp_stats[temps[t_idx]]["corr_sum"] += s["correctness"]
            temp_stats[temps[t_idx]]["coh_sum"] += s["coherence"]
            temp_stats[temps[t_idx]]["count"] += 1

temps = sorted(temp_stats.keys())
corr_means = [temp_stats[t]["corr_sum"] / temp_stats[t]["count"] for t in temps]
coh_means  = [temp_stats[t]["coh_sum"]  / temp_stats[t]["count"] for t in temps]

x = np.arange(len(temps))
width = 0.35

bars1 = ax2.bar(x - width / 2, corr_means, width, label="Correctness")
bars2 = ax2.bar(x + width / 2, coh_means, width, label="Coherence")

ax2.set_ylabel("Average score (1–5)")
ax2.set_title("Average correctness & coherence by temperature (400 responses each)")
ax2.set_xticks(x)
ax2.set_xticklabels([str(t) for t in temps])
ax2.set_ylim(0, 5)
ax2.legend()
ax2.axhline(3, color="gray", linestyle="--", alpha=0.3)

fig2.tight_layout()
out_png2 = "preference_optimization/evals/temperature_vs_quality.png"
fig2.savefig(out_png2, dpi=200)
print(f"Saved plot to {out_png2}")
