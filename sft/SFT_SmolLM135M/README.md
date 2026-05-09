# SFT_SmolLM135M

Supervised fine-tuning of SmolLM-135M into a structured ML paper research assistant. The fine-tuned model exposes a typed Python API (`PaperResearcher`) with 10 task-specific endpoints — not a general chatbot.

## Pipeline Overview

```
arXiv ML papers
       │
       ▼
text-albumentations (chunking + constrained synthetic generation via Outlines)
       │
       ▼
paperbd/paper_instructions_300K-v1  (300K instruction-response pairs)
       │
       ▼
SFT training (LoRA r=32, ChatML, train_on_responses_only, Unsloth + TRL)
       │
       ▼
SmolLM-135M-SFT  →  PaperResearcher API  →  Gradio demo
```

## Stack

- **Training:** PyTorch, HuggingFace Transformers, Unsloth, TRL, PEFT
- **Data generation:** `text-albumentations`, Outlines (constrained decoding)
- **Eval:** LLM-as-judge (grok-3-mini) — faithfulness, correctness, relevance, completeness
- **Demo:** Gradio (`app.py`) + MLX inference backend for Apple Silicon

## Quickstart

```bash
# Install
pip install uv && uv sync

# Train
python train.py --base_model_id paperbd/smollm_135M_arxiv_cpt -o my_run

# Inference
python inference.py --model_path models/my_run/final --output_file results.jsonl

# Evaluate (requires XAI_API_KEY)
python llm_judge.py --input_file evals/results.jsonl

# Demo
python app.py models/my_run/final --n 1
```

## Training Config (defaults)

| Parameter | Value |
|---|---|
| Base model | `paperbd/smollm_135M_arxiv_cpt` |
| Dataset | `paperbd/paper_instructions_300K-v1` |
| LoRA rank | 32 |
| Batch size | 32 (effective 128 with grad_accum=4) |
| Learning rate | 2e-4 |
| Epochs | 3 |
| Sequence length | 2048 (packed) |
| Chat template | ChatML |
| Hardware | RTX 4090, ~5-6 hours |

Key flags:
- `--lora_r` — LoRA rank (default 32)
- `--epochs` — number of epochs (default 3)
- `--no_wandb` — disable W&B logging
- `--resume_from_checkpoint` — resume from a checkpoint path

## Data Generation

The `data_prep/` scripts build instruction datasets from raw paper text:

```bash
# Sync version
python data_prep/main.py papers.jsonl output.jsonl --max-rows 1000

# Async version (faster, uses API concurrency)
python data_prep/main_async.py papers.jsonl output.jsonl

# Batch version (cheapest for large scale)
python data_prep/main_batch.py papers.jsonl output.jsonl
```

**How it works:**
1. Input text is chunked into 500-word overlapping windows (100-word overlap)
2. Each chunk is passed through 5-7 augmentation tasks stochastically (25% prob each)
3. Outputs are constrained via **Outlines** — structured decoding guarantees valid JSON, QA pairs, triplets etc.
4. Cross-chunk tasks (retrieval, comparison) run across pairs of chunks

**Augmentation tasks:**

| Task | Output |
|---|---|
| `bullet_augmentation` | Markdown bullet points |
| `qa_pair_augmentation` | `list[QAPair(question, answer)]` |
| `rephrase_augmentation` | Elaborated restatement |
| `continuation_augmentation` | Completed passage |
| `triplet_augmentation` | `list[Triplet(subject, relation, object)]` |
| `retrieval_augmentation` | Which passage answers a question |
| `comparison_augmentation` | Detailed comparison of two passages |

Default generation model: `mlx-community/Qwen3.5-4B-OptiQ-4bit` (Apple Silicon / MLX).

## Evaluation

**Step 1 — Generate responses:**
```bash
python inference.py --model_path models/my_run/final --num_samples 1000
```

**Step 2 — LLM judge:**
```bash
export XAI_API_KEY=your_key
python llm_judge.py --input_file evals/my_run_results.jsonl
```

Judge model: `grok-3-mini` via xAI API. Scores each response 1–5 on:
- **Faithfulness** — no hallucinations
- **Answer Correctness** — semantic match to ground truth
- **Relevance** — on-topic, no padding
- **Completeness** — covers key points

**Step 3 — Compare models head-to-head:**
```bash
python llm_comparison_judge.py \
    --inputs evals/model_a_results.jsonl evals/model_b_results.jsonl
```

**Step 4 — Print summary:**
```bash
python print_judge_scores.py evals/my_run_results_judged.jsonl
```

## PaperResearcher API

```python
from paper_researcher import PaperResearcher

researcher = PaperResearcher("JaydeepR/SmolLM-135M-SFT-exp01")
passage = "Attention mechanisms compute weighted sums..."

researcher.extract_bullets(passage)       # list[str]
researcher.generate_qa_pairs(passage)     # list[QAPair]
researcher.extract_triplets(passage)      # list[Triplet]
researcher.answer(question, passage)      # str
researcher.rephrase(passage)              # str
researcher.continue_from(passage)        # str
researcher.extract_fact(passage)          # str
researcher.generate_question(passage)     # str
researcher.compare(passage_a, passage_b)  # str
researcher.find_relevant(q, passages)     # RetrievalResult
```

> **Note:** The instruction strings in `paper_researcher/tasks.py` must match exactly what the model was trained on — do not change them.

## Experiment Results

| Run | Base Model | Train Loss | Eval Loss | Overall (1–5) | HF |
|---|---|---|---|---|---|
| exp01_cpt_sft | paperbd/smollm_135M_arxiv_cpt | 0.087 | 1.222 | 2.39 | [link](https://huggingface.co/JaydeepR/SmolLM-135M-SFT-exp01) |

Full experiment log: [`experiments.md`](experiments.md)

## Demo

Live Gradio demo: [JaydeepR/paper-researcher](https://huggingface.co/spaces/JaydeepR/paper-researcher)

Run locally:
```bash
python app.py models/my_run/final --n 1
# --n 1-4: number of parallel generations to show
# --mlx: use MLX backend (Apple Silicon)
# --temperature 0.4: generation temperature
```
