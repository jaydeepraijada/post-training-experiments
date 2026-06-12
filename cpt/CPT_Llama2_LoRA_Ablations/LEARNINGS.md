# Learnings — CPT Llama-2-7B LoRA recipe ablations

Domain: code (`Magicoder-Evol-Instruct-110K`). Base: `unsloth/llama-2-7b-bnb-4bit`, LoRA `r=256`.
Goal: isolate *which* of Unsloth's CPT recommendations actually move held-out perplexity, one knob at a time.

> Status: harness built; the held-out-perplexity sweep is pending a GPU run. The findings
> below are directional, read off **training loss** from the original exploratory notebooks
> (`notebooks/`). They will be replaced with validation-perplexity numbers (mean ± std over
> seeds) once the sweep runs — see the experiment `README.md` for the run commands.

1. **Training `gate_proj` helps for free-ish.** The original paper recipe omitted the SwiGLU
   gate; adding it lowered loss with no instability. Cheap win.

2. **Adapting `embed_tokens` + `lm_head` at the *normal* LR backfires.** At a single 5e-4 LR
   this was *worse than the baseline* — the embedding/head matrices are large and sensitive,
   and a full-speed LR destabilises them. This is the key failure mode the rest of the recipe fixes.

3. **rsLoRA stabilises high rank.** `alpha/√r` scaling (vs `alpha/r`) is the right setup at
   `r=256` and recovered the regression from (2).

4. **Decoupled embedding LR is the actual fix.** Giving `embed_tokens`/`lm_head` a ~10× smaller
   LR via `UnslothTrainer` lets you adapt them safely; combined with rsLoRA it gave the best loss
   of the sweep. Takeaway: adapt embeddings for CPT **only** with (a) a smaller decoupled LR and
   (b) rsLoRA — either alone done wrong is worse than not touching them.

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

## Next

- Run the sweep on GPU (T4 smoke test → A100 for 500-step, 3-seed numbers); regenerate the
  results table via `scripts/make_table.py`.
- Optional: add a downstream code benchmark (HumanEval/MBPP pass@1) — a real task metric on top
  of perplexity, since the domain is code.
