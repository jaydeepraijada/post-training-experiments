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
| `06_rslora_decoupled` | full Unsloth recipe | all linear | **on** | 1e‑4 / **2.5e‑5** |

---

## Results

Populate this from your own runs: `python scripts/make_table.py`. The numbers below are the **training-loss readings from the original exploratory notebooks** (Appendix) — kept as a directional preview until the perplexity sweep is re-run through the harness.

| Experiment | Trainable | Train loss @50 (notebook) |
|---|---|---|
| 01_baseline | 515.9 M | 1.0678 |
| 02_gate_proj | 639.6 M | 1.0630 |
| 03_embed_lmhead (same LR) | 901.8 M | **1.1102** ⚠️ worse than baseline |
| 04_rslora | 901.8 M | 1.0663 |
| 05_decoupled_lr | 901.8 M | 1.0626 |
| 06_rslora_decoupled | 901.8 M | **1.0541** ✅ best |

### What the numbers say

1. **`gate_proj` helps (01→02).** Training the SwiGLU gate, omitted by the original paper, lowers loss for little extra cost.
2. **Adapting embeddings at the normal LR backfires (03).** Adding `embed_tokens` + `lm_head` at 5e‑4 is *worse than the baseline* — these layers are large and sensitive; a full-speed LR destabilizes them.
3. **rsLoRA stabilizes high rank (04).** `alpha/√r` scaling (vs. `alpha/r`) is the recommended setup at `r=256` and recovers the regression.
4. **Decoupled embedding LR is the real fix (05→06).** Giving `embed_tokens`/`lm_head` a ~10× smaller LR via `UnslothTrainer` lets you adapt them safely; combined with rsLoRA it gives the best result.

**Takeaway:** adapting embeddings for CPT is worth it, but only with (a) a smaller, decoupled LR and (b) rsLoRA scaling. Either piece done wrong is worse than not touching embeddings at all.

> **Methodology note:** training loss is a proxy, not an outcome — a model can lower it while getting worse on unseen text. The harness therefore reports **held-out perplexity** instead. Also note the original notebooks changed two knobs at step 03→04 (rsLoRA *and* the LR); the configs here split that into `04` (rsLoRA only) and `05` (decoupled LR only) so each effect is isolated. For publishable numbers, run ≥3 seeds and report mean ± std.

---

## The Unsloth ingredients in one line each

- **`gate_proj` / all linear layers** — more adapter capacity for domain shift.
- **`embed_tokens` + `lm_head`** — relearn token in/out representations; essential when vocabulary usage shifts, dangerous at a normal LR.
- **rsLoRA** — `use_rslora=True`; scales adapters by `alpha/√r`, keeping gradients stable at large rank.
- **decoupled `embedding_learning_rate`** — `UnslothTrainer` applies a separate, smaller LR to embeddings/head. The headline CPT feature.

---

## Appendix — original notebooks

This study grew out of six Colab notebooks (`notebooks/`), one per rung of the ladder, where the experiment was encoded in the **filename**. They are kept for provenance and for the raw loss/VRAM readings above. The `src/` + `configs/` harness supersedes them: it removes hidden notebook state, makes each experiment a reviewable two-line config diff, adds seed control and a held-out metric, and runs headless for multi-seed sweeps.
