"""
Headline evidence chart for the DPO run:
  (left)  held-out reward accuracy climbing 0.50 -> 0.72 over training
  (right) diversity (EAD/SBERT/Vendi) flat vs the SFT baseline -> no mode collapse

Requires: matplotlib  (uv run python make_results_chart.py)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "results_chart.png"

# ── Data (from experiments.md / the run log) ─────────────────────────────────
# Held-out reward accuracy by epoch (epoch 0 = random chance)
epochs = [0.0, 0.54, 1.09, 1.63, 2.18, 2.72, 3.0]
acc    = [0.50, 0.625, 0.671, 0.708, 0.713, 0.722, 0.723]

# Diversity: SFT baseline vs DPO
div_labels = ["EAD", "SBERT", "Vendi"]
base = [0.1173, 0.2263, 2.7327]
dpo  = [0.1193, 0.2322, 2.7410]

# ── Theme ────────────────────────────────────────────────────────────────────
BG, FG, GRID = "#0f0f14", "#d7d7d7", "#2d2d3c"
GREEN, ORANGE, DIM = "#64dc8c", "#ff966e", "#8a8a99"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": FG, "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG,
    "axes.edgecolor": GRID, "font.size": 12,
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# ── Panel 1: reward accuracy ─────────────────────────────────────────────────
ax1.plot(epochs, acc, "-o", color=GREEN, lw=2.5, ms=7, zorder=3)
ax1.axhline(0.50, ls="--", lw=1.2, color=DIM)
ax1.text(1.4, 0.51, "random  0.50", color=DIM, fontsize=10, va="bottom")
ax1.annotate("0.72", xy=(3.0, 0.723), xytext=(2.45, 0.745),
             color=GREEN, fontsize=14, fontweight="bold")
ax1.set_xlim(-0.1, 3.25); ax1.set_ylim(0.45, 0.78)
ax1.set_xlabel("epoch"); ax1.set_ylabel("held-out reward accuracy")
ax1.set_title("Learned the preference", color=FG, fontsize=14, fontweight="bold", pad=10)
ax1.grid(True, color=GRID, lw=0.6, alpha=0.6)
for s in ("top", "right"):
    ax1.spines[s].set_visible(False)

# ── Panel 2: diversity preserved (normalized to baseline = 1.0) ──────────────
import numpy as np
x = np.arange(len(div_labels)); w = 0.36
norm_base = [1.0, 1.0, 1.0]
norm_dpo = [d / b for d, b in zip(dpo, base)]
b1 = ax2.bar(x - w/2, norm_base, w, color=ORANGE, label="SFT baseline")
b2 = ax2.bar(x + w/2, norm_dpo, w, color=GREEN, label="DPO")
ax2.axhline(1.0, ls="--", lw=1.0, color=DIM, zorder=0)
for i, nd in enumerate(norm_dpo):
    ax2.text(x[i] + w/2, nd + 0.03, f"{(nd-1)*100:+.1f}%", ha="center",
             color=GREEN, fontsize=11, fontweight="bold")
ax2.set_xticks(x); ax2.set_xticklabels(div_labels)
ax2.set_ylim(0, 1.32); ax2.set_ylabel("diversity  (relative to SFT = 1.0)")
ax2.set_title("...without mode collapse", color=FG, fontsize=14, fontweight="bold", pad=10)
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)

fig.legend([b1, b2], ["SFT baseline", "DPO"], frameon=False, loc="lower center",
           ncol=2, bbox_to_anchor=(0.5, 0.0), fontsize=12)
fig.suptitle("DPO on SmolLM-135M (135M params):  preference accuracy 0.50 → 0.72,  diversity unchanged",
             color=FG, fontsize=15, fontweight="bold", y=0.99)
fig.tight_layout(rect=(0, 0.06, 1, 0.95))
fig.savefig(OUT, dpi=150)
print(f"Saved {OUT}")
