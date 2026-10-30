# Reasoning Model: Reward System, Diagnosis & Solutions

This documents the reward design, why it was reshaped, the concrete changes made, and
the current measured results.

## TL;DR — what was wrong and what we changed

**Diagnosis:** Under group-normalized advantages, the reward landscape made **length
the only consistently varying signal** within a rollout group, so GRPO minimized
length (and eroded format) instead of learning quality. The fixes below reshape the
reward so quality is the signal that reaches the gradient.

**Fixes implemented (this iteration):**

| Area | Before | After |
|---|---|---|
| Length penalty | `raw_score × (multiplier − 1)` — a continuous pull toward shorter, anti-correlated with quality | flat `−1.0` past a hard cap (`2× ref length`), decoupled from quality — a guardrail, not an objective |
| Advantage | std-normalized (amplifies noise in near-degenerate groups) | mean-centered only; std-norm opt-in via `std_normalize_advantages` (default off) |
| Loss aggregation | per-sequence `1/L` normalization (length bias) | `dr_grpo`: `sum / (num_responses × max_tokens)`, length-unbiased; shared by PG/KL/entropy so `kld_weight` stays scale-consistent |
| `max_tokens` / `lr` | implicit / `1e-6` | `256` (≈ mean completion length) / `5e-6` (compensates for std-norm removal) |
| Reward scoring | duplicated across `evaluate_base.py` + 3 eval scripts | single source of truth in `env.py` (`score_completions` / `score_rollouts`); `evaluate_base.py` deleted |
| Eval | one temp; eval every buffer; redundant `eval_stepwise` | `eval_temperature: 0.2` (vs train `0.5`); `inference()` gated to every Nth train event; `eval_stepwise` removed |
| Rollout speed | `generate` at small batch, full-batch `old_log_probs` forward | `rollout_batch_size: 32`, `sdpa`, LoRA merge for decode (~1.6×), chunked `old_log_probs` (fixes OOM), `max_new_tokens: 768` |

## Why the reward was reshaped (length vs quality)

GRPO normalizes advantages **within each rollout group**: `A = (r − group_mean) /
group_std`. Any reward component that is roughly *constant across the group*
contributes ~zero to the gradient. The binary format/doom components saturate (most
rollouts pass → they cancel), and the raw quality signal is noisy at 135M and largely
averages out — leaving the **length penalty** (deterministic, monotone in length, high
within-group variance) as the dominant directional signal.

Under the old *multiplicative* penalty this pushed the policy to **minimize length**
(and eroded format, which had no protective gradient while saturated) instead of
improving quality — the penalty even scaled with quality, pulling hardest on the
*best* long answers. The fixes below make length a flat guardrail and leave quality
(`neuraltxt_reward`) as the signal that actually reaches the gradient.

## Reward components (current)

The total reward is a weighted sum of 5 components. Format is a **gate**: a
completion that fails `<think>...</think>` formatting scores 0 on neuraltxt and the
length penalty, and −0.5 on output-format.

| Component | Range | Weight | Description |
|---|---|---|---|
| `think_format_reward` | {0, 1} | 1.0 | 1.0 if valid `<think>...</think>` format |
| `output_format_reward` | {−0.5, +0.5} | 1.0 | +0.5 if output type (JSON/text) matches reference |
| `doom_loop_reward` | {−1.0, 0} | 1.0 | −1.0 if 4+ consecutive repeated words |
| `neuraltxt_reward` | [0, 1] | 1.0 | **Raw** semantic quality (22M MiniLM reward model `paperbd/neuraltxt-reward-22M`, v7-1) — no length term baked in |
| `length_penalty_reward` | {−1.0, 0} | **0.5** | **Flat** `−1.0` once response exceeds the hard cap, else 0 — independent of quality |

- **Max reward:** `1.0 + 0.5 + 0.0 + 1.0 + 0.5×0 = 2.5` (perfect, concise).
- **Over-cap:** `2.5 + 0.5×(−1.0) = 2.0` (the cap is a fixed cliff, not a gradient).
- **Format fail:** `−0.5`.

**Hard length cap** (`env.py:_length_cap`): `max(ref_words × 2.0, ref_words + 30)`.
Past it → flat `LENGTH_OVERAGE_PENALTY = −1.0`; below → 0. No dependence on the raw
score. Default weights live in `DEFAULT_REWARD_WEIGHTS` and are merged *under* any
caller-supplied weights, so a partial weight dict can't silently reset the length
penalty to 1.0.

### Why a guardrail, not an objective
Length is now a *guardrail* that only fires on egregious bloat — it carries little
within-group variance, so it no longer dominates the gradient. Quality
(`neuraltxt_reward`) is the sole continuous, quality-bearing signal, and with
std-normalization off, its real magnitude reaches the gradient instead of being
flattened to unit variance alongside judge noise.

## GRPO training changes

- **Advantage:** mean-centered (`r − group_mean`); std-normalization is opt-in via
  `training.std_normalize_advantages` (default `false`). Removing `/group_std` is the
  Dr.GRPO recommendation — it stops over-weighting low-variance/hard groups and
  amplifying judge noise.
- **Loss aggregation** (`grpo_utils._aggregate_token_loss`): one reducer for the
  policy-gradient, KL, and entropy terms so they share a normalization scheme and
  `kld_weight`/`entropy_weight` stay on a consistent scale. Modes:
  - `grpo`: per-sequence mean (legacy; has a length bias).
  - `dr_grpo` (**default**): `sum / (num_responses × max_tokens)` — length-unbiased,
    consistent under gradient accumulation. The constant `max_tokens` is effectively
    an lr reparametrization.
  - `bnpo`: token-level mean over the batch.
- **`max_tokens: 256`** ≈ the mean think+answer completion length (see
  [measurements](#measured-data)), chosen so the loss sits at per-token scale.
- **`learning_rate: 5e-6`** — ~3–4× the old `1e-6`, compensating for the step-size
  drop from removing std-normalization. (Dr.GRPO's paper uses `1e-6` for a *full
  fine-tune* of a 7B; LoRA on 135M wants higher.)
- **KL** is the format-protection lever now (format gets no within-group gradient
  while saturated). Watch format pass-rate; raise `kld_weight` if it drifts.

## Eval / inference

- **Separate temperatures:** training rollouts at `temperature: 0.5` (needs variance
  for GRPO, but >0.5 loses format rewards); eval at `eval_temperature: 0.2`
  (measurement / checkpoint-selection wants low variance, not exploration).
- **`inference()` is gated** to run every `eval_every_train_step` train events
  (default 8) — skips the near-useless early evals (the model has barely moved) and
  cuts recurring eval cost. The lightweight, print-only `eval_stepwise` was redundant
  with `inference()` (same generation pass, no checkpoint/logging side effects) and
  was removed.
- **Single scoring entrypoint:** `env.py` owns prompt formatting *and* reward
  (`score_completions`) *and* offline scoring/diagnostics (`score_rollouts`,
  `summarize`, `diagnose_completion`). Training (`grpo/inference.py`) and offline eval
  use the same code, so they can never diverge. `evaluate_base.py` (which carried a
  duplicate, drifted reward) was deleted.

## Rollout speed (HF path)

- **`rollout_batch_size: 32`** — a 135M model on an 80GB GPU is otherwise ~99% idle.
- **`attn_implementation="sdpa"`** on all model loads.
- **LoRA merge for the decode loop** (`merge_adapter` → generate → `unmerge_adapter`,
  in both training rollouts and eval) — folds the adapter into the base so each decode
  step skips the `base + A@B` recompute. **Measured ~1.6× on Mac MPS** (251 → 404 tok/s).
- **Chunked `old_log_probs`** (`rollout.calculate_token_log_probs`): the full
  `B×n_rollouts` forward materializes a ~23GB `[B, T, vocab]` log-softmax and OOMs at
  batch 32. We chunk over the batch (`chunk_size = batch_size`) — exact, since
  sequences are independent given their masks.
- **`max_new_tokens: 768`** ≈ the p99 of generated tokens; the prior `1536` only
  chased the top 0.1% tail (which the reward already discourages).

### A note on `old_log_probs` (what we tried and reverted)
We tried deriving `old_log_probs` from the generation logits to skip the forward
pass. It's exact when the adapter is live on both sides (~1e-5), **but** the
production path is *merged*-bf16 generation vs *unmerged*-bf16 forward, which diverges
the step-0 importance ratio to ~[0.85, 1.13] per token. That noise isn't worth
skipping one (cheap, parallel) forward, so `old_log_probs` is computed via a forward
over the live adapter — the same code path and weights as the training forward, so
the ratio is exactly 1.0 at step 0.

## Measured data

**Completion length** (`<think>` + answer, model tokenizer, 11,758 examples from the
Qwen-3.5-4B reasoning set):

| metric | tokens |
|---|---|
| mean | 259 |
| median | 212 |
| p90 | 464 |
| p95 | 544 |
| p99 | 772 |
| max | 2162 |

→ informed `max_new_tokens: 768` (p99) and `max_tokens: 256` (≈ mean, the dr_grpo
loss constant — *not* a generation cap).

**Reward diversity at temperature** (`reasoning-base`, 30 prompts, `n_rollouts=4`):

| Metric | Temp 0.5 | Temp 0.8 |
|---|---|---|
| Mean total reward | **0.662** | 0.355 |
| Avg group std | 0.528 | 0.547 |
| **Zero-std groups** | **26.7%** | 33.3% |

Temp 0.5 has fewer zero-std (no-signal) groups and higher mean reward → chosen for
training rollouts. (The remaining zero-std groups are mostly all-format-fail.)

## Results

### Current harness (MLX, greedy temp=0.0, 200 held-out test samples)

All scored under the **same** harness (`eval_mlx_checkpoints.py` → NeuralTxtReward),
so these are directly comparable. Stable across 512 and 1024 max-token budgets.

### Run 4 checkpoint progression

| Checkpoint | Steps | Raw NeuralTxt | Composite | Format Pass | Schema Match | Resp Words | Valid/200 |
|---|---|---|---|---|---|---|---|
| reasoning-base | 0 | 0.308 | 0.302 | 61.0% | 59.0% | 44.1 | 173 |
| ckpt-384 | 384 | 0.364 | 0.358 | 77.5% | 76.5% | 45.8 | 184 |
| ckpt-768 | 768 | 0.415 | 0.409 | 90.0% | 88.5% | 44.0 | 190 |
| ckpt-960 | 960 | 0.419 | 0.414 | 89.0% | 87.0% | 41.5 | 186 |
| ckpt-1300 | 1,300 | 0.446 | 0.441 | 93.0% | 91.0% | 45.1 | 191 |
| ckpt-1500 | 1,500 | 0.465 | 0.463 | 95.5% | 94.5% | 43.2 | 195 |
| ckpt-2500 | 2,500 | 0.436 | 0.434 | 96.5% | 96.5% | 41.9 | 196 |
| ckpt-2700 | 2,700 | 0.454 | 0.453 | 96.0% | 95.0% | 44.1 | 198 |
| ckpt-3500 | 3,500 | 0.456 | 0.453 | 98.0% | 97.0% | 39.7 | 198 |
| ckpt-4000 | 4,000 | 0.481 | 0.480 | 98.5% | 98.5% | 39.3 | 198 |
| ckpt-5200 | 5,200 | 0.491 | 0.490 | 99.0% | 99.0% | 38.0 | 199 |
| ckpt-5700 | 5,700 | 0.507 | 0.507 | 99.5% | 98.5% | 36.0 | 200 |
| ckpt-6000 | 6,000 | 0.516 | 0.515 | 100% | 100% | 36.2 | 200 |
| ckpt-8000 | 8,000 | 0.517 | 0.517 | 99.5% | 98.0% | 34.5 | 200 |
| ckpt-8700 | 8,700 | 0.520 | 0.520 | 99.5% | 99.0% | 35.0 | 200 |
| ckpt-10000 | 10,000 | 0.524 | 0.523 | 99.5% | 98.5% | 36.4 | 200 |
| ckpt-10200 | 10,200 | 0.530 | 0.528 | 99.5% | 98.5% | 37.1 | 200 |
| ckpt-11000 | 11,000 | 0.534 | 0.533 | 99.5% | 98.5% | 36.5 | 200 |
| ckpt-13000 | 13,000 | 0.545 | 0.544 | 99.5% | 98.0% | 35.4 | 200 |
| ckpt-15500 | 15,500 | 0.563 | 0.562 | 100% | 97.5% | 35.5 | 200 |
| **ckpt-16500** | **16,500** | **0.601** | **0.598** | **100%** | **88.5%** | **34.2** | **200** |
| SFT (`neuraltxt-v1-135M`) | — | 0.604 | 0.603 | 100% | — | 51.6 | 200 |
| SFT (`smollm_135M_neuraltxt_v1`) | — | 0.510 | — | — | — | — | — |

### ckpt-16500 task breakdown

| Task | Count | ckpt-16500 | SFT | Gap |
|---|---|---|---|---|
| medium_answer | 90 (45%) | 0.661 | 0.716 | −0.055 |
| json_output | 42 (21%) | 0.541 | 0.561 | −0.020 |
| qa_answer | 23 (11.5%) | 0.584 | 0.525 | +0.059 (ckpt wins) |
| long_form | 18 (9%) | 0.567 | 0.323 | +0.244 (ckpt wins) |
| list_format | 17 (8.5%) | 0.521 | 0.522 | −0.001 |

**Key observations:**
- Raw quality improved **+95%** from base (0.308 → 0.601), statistically tied with SFT (0.604).
- Format adherence went from 61% → 100%, solved completely.
- Response length compressed from 44 → 34 words — concise without sacrificing quality.
- The model **beats SFT** on long-form answers (+0.244) and qa_answer (+0.059).
- JSON output gap collapsed from −0.154 (at ckpt-8000) to −0.020 (at ckpt-16500).
- Biggest remaining gap: medium_answer (−0.055) — these are the bulk of the eval set where SFT's supervised training gives it a factual knowledge edge.
- Training trajectory: steady climb from 0-6K, plateau at 0.48-0.53 through 10K, breakout starting at 13K, accelerating through 15.5K, reaching parity at 16.5K.
- Pure GRPO on a 135M model with zero supervised data closed a 0.30 gap to SFT entirely via reward signal.

## Run 5 (fork from run4 ckpt-16500, adjusted KL + schema penalty)

| Checkpoint | Steps | Raw NeuralTxt | Composite | Schema Match | Resp Words |
|---|---|---|---|---|---|
| 17000v2 | ~17,000 | 0.646 | 0.640 | 89.5% | 31.8 |
| 17200v2 | ~17,200 | 0.671 | 0.663 | 88.5% | 32.5 |
| **17500v2** | **~17,500** | **0.665** | **0.659** | **91.5%** | **34.7** |
| 18200v2 | ~18,200 | 0.772 | 0.763 | 87.0% | 32.4 |
| 18500v2 | ~18,500 | 0.780 | 0.773 | 87.0% | 29.8 |
| 20000v2 | ~20,000 | 0.792 | 0.786 | 87.0% | 30.7 |

**Key:** 17500v2 was the run5 sweet spot — 91.5% schema, best LLM judge score. Post-17500, reward hacking took over (NeuralTxtReward inflated to 0.79 while schema collapsed to 87%).

## Run 6 (fork from run4 ckpt-16000, higher schema penalty)

| Checkpoint | Steps | Raw NeuralTxt | Composite | Schema Match | Resp Words |
|---|---|---|---|---|---|
| 16000v3 | ~16,000 | 0.574 | 0.572 | 97.5% | 36.3 |
| **16500v3** | **~16,500** | **0.583** | **0.582** | **97.5%** | **36.5** |
| 17000v3 | ~17,000 | 0.557 | 0.556 | 98.0% | 36.1 |
| 17500v3 | ~17,500 | 0.564 | 0.563 | 98.5% | 35.4 |
| 18000v3 | ~18,000 | 0.578 | 0.576 | 97.0% | 33.6 |
| 18500v3 | ~18,500 | 0.579 | 0.578 | 94.0% | 33.2 |
| 20000v3 | ~20,000 | 0.577 | 0.574 | 89.0% | 31.8 |

**Key:** 16500v3 had the cleanest output of any model (97.5% schema, only 18% passage hack rate). Quality peaked early and declined as the schema penalty suppressed content generation.

## LLM Judge Results (DeepSeek v4 Pro, 100-sample head-to-head vs SFT)

| Model | Run | Judge Wins | Avg Score | Schema | Hack Rate | NeuralTxt |
|---|---|---|---|---|---|---|
| **run5 17500v2** | 5 | **52%** | 2.86 | 91.5% | 57% | 0.665 |
| run4 16800 | 4 | 50% | 2.71 | 87% | ~80% | 0.599 |
| run6 16500v3 | 6 | 45% | 2.71 | 97.5% | 18% | 0.583 |
| run5 17200v2 | 5 | 43% | 2.78 | 88.5% | ~58% | 0.680 |
| run4 16500 | 4 | 43% | 2.63 | 88.5% | ~70% | 0.601 |

### Head-to-Head: run5 17500v2 vs run6 16500v3

| | Wins | Avg Score |
|---|---|---|
| run5 17500v2 | **57%** | 2.86 |
| run6 16500v3 | 43% | 2.64 |

## Where run5 17500v2 excels (LLM judge task breakdown)

| Task | n | Win% vs SFT | Pattern |
|---|---|---|---|
| QA answering | 10 | **90%** | SFT outputs questions instead of answers |
| Long-form | 12 | **67%** | Concise summaries beat SFT rambling |
| JSON | 20 | 50% | Even split |
| Medium | 43 | 44% | Slight SFT edge |
| List/Steps | 10 | **20%** | Collapses lists to prose |

## Best Model: `neuraltxt-135M-reasoning-grpo-v1`

Selected **run5 checkpoint 17500v2** as the best overall model:
- Highest LLM judge win rate (52% vs SFT)
- Balanced schema (91.5%)
- Strong QA performance (90% win rate)
- Saved as merged PyTorch at `models/merged/neuraltxt-135M-reasoning-grpo-v1/`
- MLX version at `models/mlx/neuraltxt-135M-reasoning-grpo-v1/`

## Key findings across all runs

1. **Pure GRPO can match SFT on a 135M model.** Run4 ckpt-16800 reached 50% LLM judge wins vs SFT with zero supervised data — only a 22M reward model signal.

2. **The passage template is both the model's strength and its weakness.** "The passage defines/explains..." format produces excellent QA and summarization answers (90% QA win rate) but fails on structured outputs (20% win rate on lists). The LLM judge prefers this format, making it a genuine improvement rather than pure reward hacking.

3. **Reward hacking is inevitable with the current 22M reward model.** Every run eventually learns to exploit the reward model. Higher schema penalties delay but don't prevent it. The reward model ceiling is ~0.60 NeuralTxtReward for genuine quality; beyond that, reward and quality diverge.

4. **Schema penalties help structure but suppress content.** Run6 achieved 97.5% schema match at the cost of weaker content and lower LLM judge scores. The sweet spot is ~91% schema with moderate hack rates.

5. **The model cannot do open-domain QA.** When asked questions without a passage, it hallucinates passages ("The passage explicitly defines...") and gives generic answers. It's a passage comprehension machine, not a general knowledge model.

## Open levers (not yet implemented)

- **Faster rollouts via vLLM-family engine** — Unsloth `fast_inference` (already a
  dependency; vLLM under the hood with automatic LoRA weight-sync) or direct vLLM.
  5–20× from continuous batching + prefix-caching the long shared prompts. CUDA-only;
  keep the HF path as the Mac/dev fallback.
- **k3 token-level KL estimator** — the current KL is the *exact* `Σ_v p·log(p/p_ref)`,
  which is the sole reason the full `[B, T, vocab]` distribution (and a full ref-model
  forward) is materialized in training. The GRPO-standard k3 estimator
  (`exp(Δ) − Δ − 1` at the sampled token) needs only per-token log-probs → drops the
  full-vocab tensors for both policy and ref, roughly halving training peak memory.
- **Eval/checkpoint cadence review** — gating `inference()` as a whole block also
  gates `save_model()` + logging; configs that don't set `eval_every_train_step` (e.g.
  short/local runs) can finish without saving any checkpoint. Consider gating only the
  eval, keeping per-buffer saves/logs.
- **Align remaining offline eval scripts** (`eval_mlx_checkpoints.py`,
  `score_head_to_head.py`, `benchmark_head_to_head.py`) to `score_completions` — they
  still carry the old multiplicative scoring.

## Current GRPO config (`grpo/gpu_80gb.yaml`)

```yaml
model:
  name: "paperbd/neuraltxt-135M-reasoning-base"
  max_new_tokens: 768        # ~p99 of generated think+answer tokens
training:
  rollout_batch_size: 32     # saturate the GPU
  n_rollouts: 4
  temperature: 0.5           # training rollouts
  eval_temperature: 0.2      # eval / checkpoint selection
  batch_size: 16
  gradient_accumulation_steps: 4
  learning_rate: 0.000005
  num_repeats: 3
  buffer_size: 64
  eval_every_train_step: 8   # gates inference() per train event
  std_normalize_advantages: false
loss:
  kld_weight: 0.02
  entropy_weight: 0.0
  loss_implementation: dr_grpo
  max_tokens: 256            # ~mean completion length; dr_grpo denominator (not a gen cap)
```

## How to benchmark a new checkpoint (merge → MLX → rollouts → rewards)

This is the exact, reproducible procedure. A GRPO checkpoint is a LoRA adapter, so it
must be merged into the base and converted to MLX before scoring. Run everything from
the repo root with the project venv (`.venv/bin/python`).

```bash
# Set these two for your checkpoint
ADAPTER=models/reasoning_models/run_4/checkpoint-384   # the LoRA checkpoint dir
NAME=run4_checkpoint-384                                # short name for outputs

# 1. Merge the LoRA adapter into its base model
.venv/bin/python merge_adapter.py \
    --adapter_path "$ADAPTER" \
    --output_path "models/merged/$NAME"

# 2. Convert the merged model to MLX
#    (delete models/mlx/$NAME first if it already exists — convert refuses to overwrite)
.venv/bin/python -m mlx_lm convert \
    --hf-path "models/merged/$NAME" \
    --mlx-path "models/mlx/$NAME"

# 2b. ONE-TIME: convert the reasoning-base comparator to MLX (skip if already done)
.venv/bin/python -m mlx_lm convert \
    --hf-path paperbd/neuraltxt-135M-reasoning-base \
    --mlx-path models/mlx/neuraltxt-135M-reasoning-base

# 3. Generate rollouts on the held-out test split and score with NeuralTxtReward.
#    Greedy (temp 0.0) for a reproducible, low-variance comparison. Including
#    reasoning-base in --ckpt_dirs means it's scored under the SAME harness, so the
#    numbers are directly comparable (do NOT compare against numbers from other harnesses).
.venv/bin/python reasoning/eval_mlx_checkpoints.py \
    --num_samples 200 --batch_size 32 --temperature 0.0 \
    --ckpt_dirs "models/mlx/$NAME" models/mlx/neuraltxt-135M-reasoning-base \
    --sft_model paperbd/neuraltxt-v1-135M-mlx \
    --output_dir "reasoning/evals/$NAME"
```

**Reading the result:** the per-model summary prints `mean NeuralTxt (raw: ...)`,
`fmt_pass`, and `doom`; the full table (incl. `mean_response_words`,
`length_penalty_rate`) is in `reasoning/evals/$NAME/summary.json`, and per-example
generations/scores are in `generations_*.jsonl` / `scored_*.jsonl`.

**Is it improving over reasoning-base?** Compare the checkpoint's row to
`neuraltxt-135M-reasoning-base` in the same run on: **raw NeuralTxt** (semantic
quality), **format pass-rate**, and **mean response words** (a quality/format gain at
equal-or-greater length is real; a "gain" that's mostly fewer words is length-hacking,
not improvement). SFT (`neuraltxt-v1-135M`, ~0.60 raw) is the stretch bar.
