# The Experiments, Explained Simply

This file explains what each of the 7 experiments in this study does and why it
exists, in plain language. (The technical setup is in `README.md`; the research
plan is in `RESEARCH.md`.)

---

## The big picture

**What we're doing:** teaching Llama-2-7B (a general-purpose model) to be
better at **code**, by continuing its pretraining on a corpus of programming
problems and solutions. This is called **Continued Pretraining (CPT)** —
think of it as sending the model back to school for one extra subject.

**The constraint:** retraining all 7 billion weights is expensive. So we use
**LoRA** — instead of changing the model's weights directly, we bolt small
trainable "adapter" matrices onto them and train only those. It's like editing
a document with sticky notes instead of rewriting the pages: far cheaper, and
you can peel them off.

**The problem:** done naively, LoRA underperforms full retraining for CPT.
The Unsloth team published a recipe of fixes — but only ever showed it lowers
*training loss* on a short demo. A rigorous academic study (Biderman et al.
2024) showed LoRA falling short of full fine-tuning — but didn't use *any* of
Unsloth's fixes.

**The question:** which of the recipe's ingredients actually matter, measured
properly? To find out, we start from a bare-bones setup and add **one
ingredient at a time**, measuring each step. That's what "ablation" means:
isolate each part's contribution by changing only one thing at a time.

---

## How we measure: held-out perplexity

Before training, we set aside 1,000 examples the model **never trains on**.
After training, we ask: *how surprised is the model by these unseen examples?*
That surprise score is **perplexity** (lower = better — the model finds
code text more predictable, i.e. it genuinely learned the domain).

Why not just use training loss? Because a model can get better at its training
examples while getting *worse* at everything else (memorizing answers vs.
learning the subject). Held-out perplexity is the honest test: an exam with
questions the student hasn't seen.

**Every experiment also runs 3 times with different random seeds.** The
differences between experiments are small, so a single run can't tell a real
effect from luck. Three runs give us an average ± spread; an effect is only
real if it's bigger than the spread.

---

## The 7 experiments (the "ladder")

Everything except the listed change is **identical** in every run: same model,
same data, same LoRA size (rank 256), same batch size, same 50 training steps,
same evaluation. That identity is the whole experiment design.

### 01_baseline — the starting point
LoRA adapters on the model's **attention layers and most of its MLP layers**,
all trained at one learning rate (5e-4). This mirrors the setup used in the
academic paper that found LoRA lacking. Every other experiment is judged
against this.

### 02_gate_proj — train the forgotten layer
Llama's MLP blocks have three parts; the baseline (following the paper) skips
one of them — the **gate**. This experiment simply adds it.
*Hypothesis:* a cheap, free win. The notebooks suggest yes.

### 03_embed_lmhead — adapt the vocabulary (the naive way)
Adds LoRA to the model's **embedding table** (how it reads tokens in) and
**output head** (how it predicts tokens out). The intuition: if the domain
uses vocabulary differently (code does!), the model's "ears" and "mouth"
should adapt too, not just its "brain".

**This one is designed to fail.** Both layers are trained at the same fast
learning rate as everything else, and they're huge, sensitive matrices.
In the notebooks this was *worse than the baseline*. It's the failure mode
the rest of the recipe exists to fix — included deliberately so we can
quantify it.

### 04_rslora — fix attempt #1: rescale the adapters
Same layers as 03, but turns on **rsLoRA** (rank-stabilized LoRA). One-line
explanation: at large adapter sizes like ours (rank 256), standard LoRA
scales its updates *too weakly* (it divides by 256 when it should divide by
√256 = 16). rsLoRA corrects the math, keeping training stable at high rank.
*Question:* does fixing the scaling alone recover 03's regression?

### 05_decoupled_lr — fix attempt #2: slow down the embeddings
Same layers as 03, rsLoRA **off**, but now the embedding table and output
head get their own **10× smaller learning rate** (5e-5 vs 5e-4). The analogy:
the model's core can study at full speed, but its vocabulary should change
slowly and carefully. *Question:* is the gentler learning rate alone the
real fix?

### 06b_rslora_at_05_lrs — both fixes together
Experiment 05's setup with rsLoRA turned **on** — both fixes, nothing else
changed. Comparing 05 → 06b isolates exactly what rsLoRA adds *on top of*
the gentler embedding learning rate. (This rung exists because the original
notebooks jumped from 05 to 06 changing two things at once — bad science.)

### 06_rslora_decoupled — the full published recipe
Unsloth's exact recommended settings: both fixes, plus overall lower learning
rates (1e-4 main / 2.5e-5 embeddings). Comparing 06b → 06 isolates whether
the *lower learning rates themselves* matter, separate from the two fixes.
This is the configuration Unsloth actually ships in their guide.

---

## How to read the results

When the sweep finishes, `python scripts/make_table.py` produces a table of
**mean perplexity ± spread** per experiment. The story to look for:

1. **02 < 01?** → training the gate layer is a free win.
2. **03 > 01?** → naive embedding training really does backfire (not just
   notebook noise).
3. **04 vs 05 vs 03** → which fix (rescaling vs gentler LR) repairs more of
   the damage on its own.
4. **06b < both 04 and 05?** → the fixes stack.
5. **06 vs 06b** → does Unsloth's final learning-rate choice add anything,
   or was it the two fixes all along?

And the overall headline: **does the best recipe beat the plain baseline
(01) by more than the seed spread?** If yes, the recipe is real. If no, the
recipe's benefits don't show up at this budget — which is also a finding.

---

## What this costs

Each run: 50 steps ≈ 13M tokens ≈ 25 minutes on an RTX 4090.
Full study: 7 experiments × 3 seeds = 21 runs ≈ 10 hours ≈ $7 of GPU rental.
The results land in `results/runs.csv`, on Weights & Biases, and on the
Hugging Face Hub.

---

## The results (measured 2026-06-12)

The sweep ran. Answering the five questions from "How to read the results":

1. **Is 02 better than 01?** Yes, slightly — training the gate layer won at
   all 3 seeds, but the margin is about the same size as the noise. A cheap,
   undramatic win.
2. **Did 03 backfire?** **No — and this is the first surprise.** The
   "deliberately naive" embedding experiment *beat the baseline at every
   seed*. The failure story that motivates the whole recipe didn't happen.
3. **Which fix repairs more?** Trick question, it turned out: there was
   nothing to repair (see 2), and one of the "fixes" was the thing that
   broke. The gentler-learning-rate arm (05) was solid and stable. The
   rsLoRA arm (04) went haywire — training loss spiking to 8× normal,
   results swinging wildly between seeds, worse on the final exam.
4. **Do the fixes stack (06b)?** No — combining them gave the *worst*
   result of the whole study, because rsLoRA's instability dominates at
   the standard learning rate.
5. **Does Unsloth's final learning-rate choice matter (06 vs 06b)?**
   **It's everything.** Lowering the learning rate ~5× took the same setup
   from worst place to first place (best score, smallest noise). That's
   the study's headline: the recipe works, but its quiet ingredient — the
   lowered learning rate — is doing the rescuing, and the celebrated
   ingredient (rsLoRA) is what it's rescuing things *from*.

**Bottom line:** the full recipe beat the plain baseline by about 1.5% —
real, consistent across seeds, but modest. The big lesson is a warning,
not a recipe: at high adapter rank, turning on rsLoRA without lowering
the learning rate makes things much worse, not better. Numbers in
`results/runs.csv`; next steps in `RESEARCH.md` §0.
