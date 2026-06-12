# Research Plan — CPT + LoRA: What Actually Matters

> **Status:** Phase 1 spine **measured** (2026-06-12, RTX 4090, ~$7) — see §0 for results
> and the re-prioritized roadmap. Sections 1–10 below are the original proposal, kept intact.
> **Purpose:** turn the Unsloth-recipe reproduction into a study with a real finding.

---

## 0. Status update — Phase 1 results & new directions (2026-06-12)

**Done:** 7-config ladder × 3 data seeds × 50 steps, held-out perplexity (A3 + A4 from the
plan). Full numbers in `results/runs.csv` and the README results table.

**What we found (summary):**
1. The full recipe (06) wins: 2.963 ± 0.008 vs baseline 3.007 ± 0.021 — real but small (−1.5%).
2. **Naive embedding adaptation did NOT backfire** (03 beats baseline at every seed) —
   the notebook-era motivation for the recipe does not replicate.
3. **rsLoRA at the standard LR is the actual failure mode** (04, 06b: chronic loss spikes,
   worse ppl, 16–60× variance). The 06b→06 comparison shows the recipe's lowered LR pair is
   load-bearing compensation for rsLoRA's α/√r scale-up (16× at r=256, α=32).

**The headline moved.** The plan below anchored on B1 (embedding-LR mechanism) because we
expected "embeddings backfire." They didn't. The strongest finding is the rsLoRA–LR
interaction, so the revised priority order is:

| Priority | What | Cost | Why now |
|---|---|---|---|
| P1 (free) | Write up Phase 1: perplexity bars + the W&B loss-spike figure | $0 | findings are committed; figure makes it publishable |
| P2 | **rsLoRA × LR grid** ({on, off} × {5e-4, 2.5e-4, 1e-4, 5e-5}, 3 seeds on the deciders) | ~$3–4 | quantifies the LR compensation rsLoRA needs at r=256 — the curve nobody published; subsumes B9, reuses 4 existing runs |
| P3 | Eval harness: HumanEval/MBPP pass@1 + forgetting slice (A1, A2) over saved adapters | build time + ~$2 | tests blog claim 2 causally; converts "ppl moved" into "it matters"; needs a `--save-final` flag in train.py |
| P4 | B3 embedding-LR ratio curve | ~$3 | folklore→curve; complements P2 |
| P5 | B2 shift axis (second, high-shift domain) | ~$10+ | most expensive; benefits from P3 harness existing first |

**Budget note:** ~$2–3 remains of the original $10 RunPod credit; P2 fits it. P3+ needs a top-up.

---

## 1. The one-paragraph picture

Unsloth's [continued-pretraining guide](https://unsloth.ai/blog/contpretraining) recommends a CPT recipe — train **all linear layers** (incl. `gate_proj`), add LoRA to **`embed_tokens` + `lm_head`**, use **rsLoRA**, and a **decoupled (smaller) embedding learning rate**. But they only ever show it **lowers training loss on a 50-step demo**. Meanwhile [Biderman et al., *LoRA Learns Less and Forgets Less* (TMLR 2024)](https://arxiv.org/abs/2405.09673) — the authoritative LoRA-vs-full-finetuning CPT study — found LoRA **can't match full fine-tuning on code CPT** (HumanEval **0.175 vs 0.263**, gap uncclosed even at rank 256)… but their LoRA used **only attention + MLP, with no gate, no embeddings/lm_head, no rsLoRA, and no decoupled LR.**

**Unsloth's recipe is exactly the set of components Biderman left out — and nobody has measured whether it closes the gap, on real benchmarks, with error bars.** That's the opening.

---

## 2. What the literature already settles (so we don't redo it)

| Finding | Source |
|---|---|
| LoRA underperforms full-FT on **code CPT**, gap not closed at r256; LoRA **forgets less** (controlled by rank). LoRA used attn+MLP only, **no gate / embeddings / rsLoRA**. | [Biderman 2024](https://arxiv.org/abs/2405.09673) |
| rsLoRA: α/√r scaling is provably optimal at high rank (tested on generic FT, **not** CPT gap-closing). | [Kalajdzievski 2023](https://arxiv.org/abs/2312.03732) |
| Replay mitigates forgetting in **full** CPT (up to ~50% for strong shifts); critical-mixture scaling laws exist. **Not** studied in the LoRA-recipe regime (where forgetting is already lower). | [Replay/CPT 2025](https://arxiv.org/html/2508.01908v1), [CMR law](https://aclanthology.org/2024.emnlp-main.903.pdf) |
| DoRA (weight-decomposed LoRA): merge-able, no inference overhead, but training overhead. | [Liu 2024](https://arxiv.org/html/2402.09353v4) |
| Decoupled embedding LR ("2–10× smaller"): **pure folklore — no systematic study of the ratio.** | Unsloth blog only |

---

## 3. The thesis

> **Biderman showed vanilla LoRA can't match full fine-tuning for code continued-pretraining. Does Unsloth's recipe — the exact components Biderman omitted — close that gap? And *why* do its key pieces work or fail?**
>
> Answered by (a) decomposing the recipe component-by-component with **real downstream evals**, (b) explaining the **mechanism** of its central trick (the embedding LR), and (c) showing **when** the recipe matters (distribution-shift magnitude).

Our existing config ladder (`01_baseline` ≈ Biderman's vanilla LoRA → `06` full recipe) already spans the right axis. It just needs **real evaluation instead of training loss.**

---

## 4. Honest limitation (state up front)

Biderman used **20B tokens**; a rented-4090 budget realistically allows **~26M–500M**. So this is **not** a 20B head-to-head — it's a **controlled, fixed-budget decomposition** of which recipe components matter, *motivated by* their gap. Framing it as anything more would be dishonest.

---

## 5. The full idea menu

Every idea raised, grouped by tier. `★` = recommended core.

### Tier A — The spine (turns the demo into a study)

| # | Idea | Why it matters / is novel | Cost |
|---|---|---|---|
| A1 ★ | **Downstream task eval** — HumanEval + MBPP pass@1 on base vs each config | Biderman's metric; Unsloth has **no task metric at all** (only loss) | Med (needs sandboxed code-exec harness) |
| A2 ★ | **Forgetting** — general-text perplexity + small MMLU/HellaSwag slice, base vs after | Quantifies the *cost* of CPT; Unsloth ignores the failure mode | Low (eval only) |
| A3 ★ | **Multi-seed significance** — ≥3 seeds, mean ± std | Effects are tiny (~0.01 loss); makes claims defensible. Most blog "ablations" skip it | Low–Med (3× runs) |
| A4 | **Held-out perplexity** (already built) | Honest outcome metric vs. training loss | Free |
| A5 | **Perplexity-vs-tokens learning curves** (token counter already exists) | Does the best recipe also learn *faster per token*? | Free (log eval at intervals) |

### Tier B — Genuinely novel extensions (where the contribution lives)

| # | Idea | Why it matters / is novel | Cost |
|---|---|---|---|
| B1 ★ | **Embedding-LR *mechanism* + a fix.** Measure embedding drift (cosine, before/after) vs LR → first drift-threshold curve. Then propose **selective embedding training** (only train high-shift token rows) and/or **scheduled embedding LR**. | Turns folklore into mechanism, **and proposes a method that could beat Unsloth's blanket 10×.** This is the strongest single contribution. | Med |
| B2 ★ | **Distribution-shift axis — "when does the recipe matter?"** Run the recipe on small shift (code) vs large shift (new/low-resource language). Hypothesis: `embed_tokens`/`lm_head` importance **scales with shift magnitude**; for small shifts the recipe is overkill or hurts. | Converts "here's a recipe" into "**here's *when* to use which components**" — a decision rule. Matches Unsloth's own stated motivation (languages). | Med–High (2nd domain/data) |
| B3 | **Embedding-LR ratio curve** — sweep 1×/2×/5×/10×/20× | Measures the curve nobody published; feeds B1 | Low (~5 runs) |
| B4 | **Replay frontier under LoRA** — mix 0/10/30% general data; plot domain-gain vs forgetting | Open Q: since LoRA forgets less, does the recipe need *less* replay than full-CPT's ~50%? Connects to your **SmolLM interleaving finding** | Med |
| B5 | **Where the recipe *backfires*** — map small-shift × small-budget × blanket-embeddings = net-negative boundary | Contrarian, honest; Unsloth never shows a losing regime (our `03` already hints at it) | Low (reuses runs) |
| B6 | **Separate `embed_tokens` vs `lm_head`** — train each alone vs both | Always lumped together; could halve cost if only one carries the shift | Low |
| B7 | **Iso-parameter: rank vs breadth** — fixed trainable-param budget, all-layers-low-rank vs few-layers-high-rank | Directly stress-tests Unsloth's "train all layers" claim under a fair budget | Med |
| B8 | **Method comparison** — LoRA vs rsLoRA vs **DoRA**; and LoRA-on-embeddings vs **full-tuning just the embeddings** | Embeddings are lookups — LoRA on them is questionable; DoRA is current SOTA-ish | Med |
| B9 | **rsLoRA × rank grid** — {r16, r64, r256} × {rsLoRA on/off} | Confirms/quantifies "rsLoRA helps *more* at high rank" | Med |

### Tier C — Moonshots / artifacts (high ceiling, higher risk)

| # | Idea | Why it matters | Cost |
|---|---|---|---|
| C1 | **Predictive recipe heuristic / mini scaling-law** — fit optimal embedding-LR ratio & replay fraction as a function of (shift, token budget) | Lets practitioners **set knobs without sweeping**; CMR-law idea applied to recipe knobs | High |
| C2 | **Auto-CPT self-tuning trainer** — sets embedding LR from the drift heuristic (B1) + replay from shift (B2) | **Product angle** for the post-training company — the recipe self-tunes | High |
| C3 | **"CPT-recipe bench"** — standardized eval of CPT methods across shift magnitudes | Owning a benchmark is leverage | High |
| C4 | **Full-FT ceiling** — one full fine-tune on a rented A100-80/H100 (won't fit 24GB) | The real upper bound in *our* setup (else cite Biderman's 0.263) | Low–Med ($) |
| C5 | **Data-vs-recipe dominance** — at fixed recipe, how much does data quality/quantity dominate the knobs? | Connects to your SmolLM "data is the bottleneck" finding; a humbling, honest headline | Med |
| C6 | **QLoRA (4-bit) vs LoRA (16-bit) base for CPT** — does quantization cap knowledge absorption? | Challenges the "QLoRA is free" assumption | Med |
| C7 | **Packing on/off audit** — equal-token comparison | Packing attends across doc boundaries; tests a default we ourselves rely on | Low |

---

## 6. Recommended scope

**Sharpest story for the effort:** anchor on **B1 (mechanism + selective-embedding method)** and **B2 (shift axis)** as the headline, with **Tier A (HumanEval, forgetting, multi-seed)** as the evaluation scaffolding they hang on, plus **B3/B4/B5** as cheap supporting analysis.

> *"We explain **why** blanket embedding adaptation fails in LoRA-CPT, propose a **targeted fix** that beats the standard recipe, and show recipe value **scales with distribution shift** — evaluated on real code benchmarks with error bars, not training loss."*

That's a workshop-paper / strong-blog contribution, not just "we ablated Unsloth." Everything is tractable on a rented 4090; C4 (full-FT ceiling) optionally on a one-off A100-80.

---

## 7. Phasing & rough budget (rented RTX 4090 @ ~$0.40/hr)

| Phase | Contents | ~Cost |
|---|---|---|
| 0 | **Eval harness** (HumanEval/MBPP + forgetting) — everything depends on this | build time |
| 1 | Spine: ladder × 3 seeds + Tier-A evals (A1–A5) | ~$10–20 |
| 2 | Headline: B1 (drift + selective method) + B2 (shift axis) + B3 | ~$15–25 |
| 3 | Supporting: B4 replay, B5 backfire, B6 split | ~$10 |
| 4 (opt) | C4 full-FT ceiling (A100-80) / C-tier | ~$5–15 |

---

## 8. Deliverables

- `RESEARCH.md` (this) → evolves into a short technical report with figures: gap-closing bars, embedding-drift curve, shift-axis plot, replay frontier.
- Reusable **eval harness** in-repo (HumanEval / MBPP / perplexity / forgetting).
- Updated `README.md` + `LEARNINGS.md`.
- Optional: HF model card for the best CPT model; a public writeup/blog.

---

## 9. Open decisions (need your call)

1. **Scope** — spine-only (Tier A) / **recommended (A + B1 + B2 + B3–B5)** / go big (add Tier C).
2. **Dataset** — keep **Magicoder** (faithful to Unsloth) for the spine; for B2 we need a **second, higher-shift corpus** (a low-resource language or far domain). Which shift?
3. **Token budget per run** — smoke (50 steps/26M) → directional (500/260M) → credible (1000+/520M).
4. **Full-FT ceiling (C4)** — run it ourselves on an 80GB GPU, or cite Biderman?
5. **GPU/host** — rent 4090 on RunPod/Vast (recommended), or other?

---

## 10. Idea index (for quick reference)

`A1` HumanEval/MBPP · `A2` forgetting · `A3` multi-seed · `A4` held-out ppl · `A5` ppl-vs-tokens ·
`B1` embedding-drift mechanism + selective/scheduled embedding LR · `B2` distribution-shift axis · `B3` embedding-LR ratio curve · `B4` replay frontier · `B5` backfire boundary · `B6` split embed/lm_head · `B7` iso-parameter rank-vs-breadth · `B8` LoRA/rsLoRA/DoRA + full-tune-embeddings · `B9` rsLoRA×rank grid ·
`C1` predictive recipe law · `C2` auto-CPT trainer · `C3` CPT-recipe bench · `C4` full-FT ceiling · `C5` data-vs-recipe · `C6` QLoRA-vs-LoRA · `C7` packing audit
