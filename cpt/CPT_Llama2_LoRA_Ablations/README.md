# Continued Pretraining with LoRA — A Controlled Ablation Study

A reproducible study of **which LoRA tricks actually matter for Continued Pretraining (CPT)**, reproducing and stress-testing the recommendations in Unsloth's [CPT guide](https://unsloth.ai/blog/contpretraining).

Llama‑2‑7B is continually pretrained on a code corpus (`Magicoder-Evol-Instruct-110K`). Starting from a paper baseline, each experiment adds **one** ingredient — extra LoRA target modules, rank-stabilized LoRA, a decoupled embedding learning rate — and measures the effect on **held-out validation perplexity**.

> One training script, six config files, one varied knob each. That identity is what makes it a controlled experiment rather than six scripts that drift apart.

---

## What is Continued Pretraining (CPT)?

A base model like Llama‑2‑7B is pretrained on trillions of general tokens but is weak at specific **domains** (law, medicine, code) or **languages**. CPT keeps training the base model on raw domain text so it absorbs new knowledge and vocabulary — pretraining round two, not instruction tuning. Doing it cheaply means **LoRA**. The catch: naive LoRA CPT underperforms, and the experiments below show exactly why and how to fix it.

---

## Repo layout

```
configs/        one YAML per experiment; each `extends: base.yaml` and overrides ONE knob
src/
  train.py      the single parametrized training run (model -> LoRA -> train -> eval -> log)
  data.py       dataset load/format + seeded held-out split (shared by all runs)
  eval.py       validation perplexity = exp(eval_loss)  -- the outcome metric
  config.py     config loader with `extends` inheritance
scripts/
  run_all.sh    runs the full ladder (optionally multi-seed)
  make_table.py results/runs.csv -> markdown table (mean ± std over seeds)
results/runs.csv  append-only experiment record (one row per run)
notebooks/      the original Colab notebooks this study grew out of (see Appendix)
```

## How to run

**Free T4 (Colab):** open `notebooks/run_colab_T4.ipynb`, set the runtime to T4 GPU, and run the cells (installs the stack, clones the repo, runs the sweep, logs to W&B).

**RunPod (RTX 4090, recommended):** launch a pod with the official PyTorch template, open its web terminal, and paste:
```bash
export GH_TOKEN=ghp_...        # GitHub token (repo scope) — pushes results back to the branch
export WANDB_API_KEY=...       # optional
curl -sL https://raw.githubusercontent.com/jaydeepraijada/post-training-experiments/add-cpt-llama2-lora-ablations/cpt/CPT_Llama2_LoRA_Ablations/scripts/run_runpod.sh | bash
```
This smoke-tests first (10 steps), then runs the full 7-config × 3-seed ladder, commits `results/runs.csv` back to the branch, and stops the pod when done (`AUTO_STOP=0` to keep it alive). ~11 h / ~$5 at $0.40–0.50/h.

**Any CUDA GPU (local / cloud):**
```bash
pip install -r requirements.txt
wandb login                                   # experiment tracking

# one experiment
python src/train.py --config configs/06_rslora_decoupled.yaml --seed 3407

# budget knobs (override the config without editing it)
python src/train.py --config configs/01_baseline.yaml --max-steps 500
python src/train.py --config configs/01_baseline.yaml --epochs 1

# the whole ladder, 3 seeds each (for variance)
SEEDS="0 1 2" bash scripts/run_all.sh
python scripts/make_table.py                  # regenerate the results table
```

Needs a CUDA GPU (4-bit Llama‑2‑7B at seq-len 4096 peaks ~14 GB). Each run reports **tokens trained** (`batch × seq × steps`) so you can report dataset coverage. Sequence **packing is on** (`packing: true` in `base.yaml`) so each step does useful work instead of padding. Runs were developed on a single T4 (~2 h / 50 steps).

---

## Experimental design

**Constant across every run:** model (`unsloth/llama-2-7b-bnb-4bit`), dataset, LoRA rank `r=256`, `alpha=32`, seq-len 4096, total batch 128 (2 × grad-accum 64), 50 steps, `lion_8bit`, cosine schedule, warmup 5, seed 3407. A fixed **1,000-example held-out split** is the validation set for perplexity.

**Varied — one knob per step:**

| Config | Change vs. previous | Target modules | rsLoRA | LR (main / embed) |
|---|---|---|---|---|
| `01_baseline` | paper setup | attn + up/down | off | 5e‑4 / — |
| `02_gate_proj` | + `gate_proj` | + gate | off | 5e‑4 / — |
| `03_embed_lmhead` | + `embed_tokens` + `lm_head`, **same LR** (deliberately naive) | all linear | off | 5e‑4 / — |
| `04_rslora` | turn on rsLoRA | all linear | **on** | 5e‑4 / — |
| `05_decoupled_lr` | decoupled embedding LR (no rsLoRA) | all linear | off | 5e‑4 / **5e‑5** |
| `06b_rslora_at_05_lrs` | + rsLoRA, LRs unchanged vs 05 | all linear | **on** | 5e‑4 / **5e‑5** |
| `06_rslora_decoupled` | full Unsloth recipe (lower LR pair) | all linear | **on** | 1e‑4 / **2.5e‑5** |

---

## Results

**Measured 2026-06-12** on a rented RTX 4090: full ladder × 3 data seeds × 50 steps (13.1M tokens/run), graded on held-out perplexity over a 1,000-example split. Raw rows in `results/runs.csv`; regenerate with `python scripts/make_table.py`. (A 10-step smoke-test row for `01_baseline` is in the CSV but excluded from all stats.)

| Experiment | Trainable | Val perplexity (mean ± std) | Verdict |
|---|---|---|---|
| 01_baseline | 515.9 M | 3.0070 ± 0.0205 | reference |
| 02_gate_proj | 639.6 M | 2.9887 ± 0.0211 | small win, all 3 seeds |
| 03_embed_lmhead (same LR) | 779.9 M | 2.9850 ± 0.0198 | **did NOT backfire** — beats baseline at every seed |
| 04_rslora | 779.9 M | 3.1779 ± 0.3220 | ⚠️ unstable: worse at every seed, 16× the variance |
| 05_decoupled_lr | 779.9 M | 2.9810 ± 0.0095 | solid, tight |
| 06b_rslora_at_05_lrs | 779.9 M | 3.3543 ± 0.5836 | ⚠️ worst row — rsLoRA instability at 5e-4 |
| 06_rslora_decoupled | 779.9 M | **2.9627 ± 0.0077** | ✅ best AND tightest |

### What the numbers say (vs. what the notebooks said)

1. **`gate_proj` helps a little (01→02, −0.018 ppl).** Wins the seed-matched comparison 3/3, but the effect is ~1σ — directionally as claimed, weakly.
2. **Naive embedding adaptation did NOT backfire (03).** It beats the baseline at every seed. The notebooks' "key failure mode" — the motivation for the whole recipe — does not replicate on held-out perplexity with the current stack.
3. **rsLoRA at the standard LR is the actual failure mode (04, 06b).** Worse than its non-rsLoRA twin at virtually every seed, with 16–60× the variance, and chronic loss spikes during training (see W&B). Mechanism: at `r=256, alpha=32`, rsLoRA scales adapters by 2.0 vs plain LoRA's 0.125 — a 16× stronger contribution fed the same 5e-4 Lion steps.
4. **The full recipe wins, but the rescue is the learning rate (06b→06).** 06 differs from 06b *only* by the lowered LR pair (1e-4 / 2.5e-5) and goes from worst row to best row. Unsloth's LR choice is load-bearing compensation for rsLoRA, not an independent nicety.
5. **Effect sizes are small.** Best vs baseline is −0.044 ppl (~1.5%); the largest effect in the table is rsLoRA *hurting* (+0.39). At this budget, avoiding the instability matters more than any single ingredient helping.

**Takeaway:** the recipe's net win is real but modest, and entirely contingent on the LR compensation; the dramatic notebook-era claims ("embeddings backfire", "rsLoRA fixes it") did not replicate when measured on unseen data with seeds.

> **Methodology notes:** (a) The notebooks measured final *training* loss on one seed with a 2024 stack at seq-len 4096; this sweep measures held-out perplexity on 3 seeds with a 2026 stack at seq-len 2048 — treat cross-era comparisons as directional only. (b) Configs with a decoupled embedding LR (05/06b/06) run via `UnslothTrainer`, the rest via `SFTTrainer`; visible side effects: lower VRAM (11.4 vs ~15 GB) and ~40% longer runtime. (c) Current unsloth implements `embed_tokens`/`lm_head` adaptation differently than the 2024 notebooks (779.9M trainable vs the notebooks' 901.8M — the arithmetic matches full-training `embed_tokens` + LoRA on `lm_head`).

---

## The blog claims under test

Unsloth's CPT guide makes four claims about [Biderman et al.](https://arxiv.org/abs/2405.09673)'s LoRA setup. Each maps to a rung of the ladder:

| # | Blog claim | Tested by | Status |
|---|---|---|---|
| 1 | The paper never trained `gate_proj` (p.3 footnote) — train it | `01 → 02` | factual; effect size TBD |
| 2 | Code underperformed *because* `embed_tokens`/`lm_head` weren't trained | `03`, `05` (ppl); needs HumanEval for the full causal claim | untested by Unsloth; our headline question |
| 3 | At r=256 you must use rsLoRA (α/√r not α/r) | `04`, `06b` vs `06` | early evidence: rsLoRA at 5e-4 is *unstable*; the recipe's lower LR may be load-bearing compensation |
| 4 | LoftQ/PiSSA/LoRA+/DoRA as advanced options | excluded (blog itself hedges); DoRA parked as RESEARCH.md B8 | out of scope |

Note the scaling arithmetic behind claim 3: with `alpha=32, r=256`, plain LoRA scales adapter output by 32/256 ≈ 0.125 while rsLoRA uses 32/√256 = 2.0 — a **16× stronger adapter contribution** at the same learning rate.

## The Unsloth ingredients in one line each

- **`gate_proj` / all linear layers** — more adapter capacity for domain shift.
- **`embed_tokens` + `lm_head`** — relearn token in/out representations; essential when vocabulary usage shifts, dangerous at a normal LR.
- **rsLoRA** — `use_rslora=True`; scales adapters by `alpha/√r`, keeping gradients stable at large rank.
- **decoupled `embedding_learning_rate`** — `UnslothTrainer` applies a separate, smaller LR to embeddings/head. The headline CPT feature.

---

## Appendix — original notebooks

This study grew out of six Colab notebooks (`notebooks/`), one per rung of the ladder, where the experiment was encoded in the **filename**. They are kept for provenance and for the raw loss/VRAM readings above. The `src/` + `configs/` harness supersedes them: it removes hidden notebook state, makes each experiment a reviewable two-line config diff, adds seed control and a held-out metric, and runs headless for multi-seed sweeps.
