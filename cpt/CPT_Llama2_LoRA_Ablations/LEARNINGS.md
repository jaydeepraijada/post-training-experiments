# Learnings — CPT Llama-2-7B LoRA recipe ablations

Domain: code (`Magicoder-Evol-Instruct-110K`). Base: `unsloth/llama-2-7b-bnb-4bit`, LoRA `r=256`.
Goal: isolate *which* of Unsloth's CPT recommendations actually move held-out perplexity, one knob at a time.

> Status: **measured** (2026-06-12, rented RTX 4090, ladder × 3 data seeds × 50 steps,
> held-out perplexity over a seeded 1k split). The notebook-era directional findings are
> superseded; two of them did not survive measurement. Numbers in `results/runs.csv`
> and the README results table.

1. **Training `gate_proj` helps, weakly.** −0.018 ppl vs baseline, winning at all 3 seeds,
   but only ~1σ. Cheap, safe, undramatic.

2. **Naive embedding adaptation did NOT backfire** (contradicts the notebooks). `03` at a
   single 5e-4 LR *beat the baseline at every seed* (2.985 ± 0.020 vs 3.007 ± 0.021). The
   "key failure mode" that motivates the recipe does not replicate on held-out perplexity
   with the 2026 stack.

3. **rsLoRA at the standard LR is the real failure mode** (contradicts the notebooks).
   `04` and `06b` show chronic loss spikes, worse perplexity at virtually every seed, and
   16–60× the variance of their non-rsLoRA twins. Mechanism: at `r=256, alpha=32`, rsLoRA
   scales adapter output by α/√r = 2.0 vs plain α/r = 0.125 — a 16× stronger contribution
   fed to the same Lion 5e-4 steps.

4. **The recipe's rescue is the learning rate, not rsLoRA.** `06b → 06` changes only the
   LR pair (5e-4/5e-5 → 1e-4/2.5e-5) and moves from the worst row (3.354 ± 0.584) to the
   best (2.963 ± 0.008). Unsloth's lower LR is load-bearing compensation for rsLoRA's
   scale-up. Practical rule: if you turn on rsLoRA at high rank, drop the LR ~5×, or skip
   rsLoRA and keep it simple (05 is within 0.02 ppl of the full recipe).

5. **Effect sizes are small at this budget.** Best vs baseline: −0.044 ppl (~1.5%). The
   biggest lever in the table is avoiding rsLoRA instability (+0.39), not any ingredient
   helping. Echoes the SmolLM finding: data/budget dominate knob-tuning.

## Methodology notes (carried into the harness)

- **Outcome metric is held-out perplexity**, not training loss — a model can lower train loss
  while getting worse on unseen text. The harness holds out a seeded 1k split and reports `exp(eval_loss)`.
- **One knob per config.** The original notebooks changed two knobs at once (rsLoRA *and* the LR)
  at one rung; the configs here split that into `04_rslora` (rsLoRA only) and `05_decoupled_lr`
  (decoupled LR only) so each effect is attributable.
- **Variance matters at this scale.** The effects are small (~0.01 loss); claims need ≥3 seeds
  with mean ± std, not a single run.

## Config-audit notes (verified by diffing all rungs through the loader)

- The ladder is a chain except one **fork**: 04 (rsLoRA) and 05 (decoupled LR) are parallel
  arms off 03, each isolating one fix. Compare 04→03 and 05→03, not 05→04.
  The chain resumes 05 → 06b (adds rsLoRA only) → 06 (lowers the LR pair only).
- `--seed` varies the data split + shuffle order; LoRA init (`random_state: 3407`) is fixed.
  Report the spread as **data-seed variance**, not full reinit variance.
- The 1k val split is cut with the run seed, so each seed is a different "exam" — identical
  across configs at the same seed (cross-config comparisons clean), but the ± std mixes
  training noise with exam difficulty. Conservative, not biased.
- Mislabel tripwire: `trainable_params` must read 515.9M (01), ~639.6M (02), ~901.8M (03–06);
  it is computed from the live model, so it cross-checks the YAML independently.

## Next (re-prioritized after the sweep — details in RESEARCH.md §0)

1. Write up Phase 1 (perplexity bars + the W&B loss-spike figure).
2. rsLoRA × LR grid (~$3–4): quantify how much LR compensation rsLoRA needs at r=256.
3. HumanEval/MBPP + forgetting harness over saved adapters (needs `--save-final` in train.py);
   tests the blog's causal claim about embeddings and code performance.
4. Embedding-LR ratio curve (B3), then the high-shift second domain (B2).

## Ops learnings (RunPod, 2026-06-12 sweep)

- Fresh-pod installs are the failure zone: `unsloth @ git+` did not bring its dep tree
  (huggingface_hub now ships as `hf`, unsloth_zoo/transformers/safetensors missing).
  A clean `pip install --force-reinstall unsloth unsloth_zoo` resolved everything in one pass.
- Checkpoint saving crashes under unsloth's patched `SFTConfig` (pickling error) — the harness
  sets `save_strategy="no"`; runs produce measurements, not weights, by design.
- The 10-step smoke test caught every one of these for cents; never skip it.
- Sweep economics: 21 runs ≈ 9.6 h ≈ $7 at $0.69/h (4090). UnslothTrainer runs used less VRAM
  (11.4 vs ~15 GB) but ~40% more time than SFTTrainer runs.
