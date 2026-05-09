---
language:
- en
license: apache-2.0
base_model: paperbd/smollm_135M_arxiv_cpt
tags:
- sft
- instruction-tuning
- lora
- unsloth
- scientific
- arxiv
- nlp
- paper-researcher
datasets:
- paperbd/paper_instructions_300K-v1
---

# SmolLM-135M-SFT-exp01

Supervised fine-tuning of [SmolLM-135M-CPT-LoRA-r32](https://huggingface.co/JaydeepR/SmolLM-135M-CPT-LoRA-r32) on 300K synthetic ML paper instruction pairs. The result is a structured research assistant API for ML papers — not a general chatbot.

This is **exp01** in a series of SFT experiments on top of the CPT-adapted SmolLM-135M.

---

## Full Pipeline

```
arXiv ML papers (188)
        │
        ▼
   text-albumentations
   (chunking + constrained synthetic generation)
        │
        ▼
paperbd/paper_instructions_300K-v1
   (300K instruction-response pairs)
        │
        ▼
   SFT training (LoRA r=32, ChatML, train_on_responses_only)
        │
        ▼
   SmolLM-135M-SFT-exp01
        │
        ▼
   PaperResearcher API (10 structured tasks)
```

---

## Model Description

- **Base model:** `paperbd/smollm_135M_arxiv_cpt` — SmolLM-135M after continued pre-training on arXiv ML papers
- **Method:** Supervised Fine-Tuning (SFT) with LoRA + `train_on_responses_only`
- **Domain:** ML/arXiv paper research tasks
- **Design:** Restricted API — 10 fixed task types, not a general chatbot

---

## Data Generation Pipeline

The training dataset was built from raw arXiv ML papers using a synthetic data generation pipeline:

### 1. Chunking
Raw paper text is split into overlapping 500-word chunks (100-word overlap) to create manageable context windows for generation.

### 2. Augmentation with `text-albumentations`
Each chunk is passed through stochastic augmentation tasks. Each task runs with 25% probability per chunk, ensuring dataset diversity:

| Task | Description | Output type |
|---|---|---|
| `bullet_augmentation` | Extract key points as markdown bullets | `list[str]` |
| `qa_pair_augmentation` | Generate question-answer pairs | `list[QAPair]` |
| `rephrase_augmentation` | Elaborate and restate the passage | `str` |
| `continuation_augmentation` | Continue from a passage prefix | `str` |
| `triplet_augmentation` | Extract knowledge graph triplets | `list[Triplet]` |
| `retrieval_augmentation` | Cross-chunk: which passage answers a question | `RetrievalResult` |
| `comparison_augmentation` | Cross-chunk: compare two passages | `str` |

### 3. Constrained Decoding via Outlines
All generation during data prep uses **[Outlines](https://github.com/dottxt-ai/outlines)** for structured output — a constrained decoding library that guarantees the generator returns outputs matching a predefined schema (Pydantic model or regex). This ensures:
- QA pairs always have valid `question` / `answer` fields
- Triplets always follow `(subject, relation, object)` format
- Retrieval results always return a valid passage index

Default runtime: `mlx-community/Qwen3.5-4B-OptiQ-4bit` via MLX (Apple Silicon). Async and batch variants available for large-scale generation.

### 4. Dataset
The final dataset `paperbd/paper_instructions_300K-v1` contains **300K instruction-response pairs** across all task types, uploaded to HuggingFace for reuse.

---

## Training Details

| Parameter | Value |
|---|---|
| LoRA rank | 32 |
| LoRA alpha | 32 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Trainable params | ~9.7M / 144M (6.77%) |
| Quantization | 4-bit (QLoRA via Unsloth) |
| Batch size | 32 |
| Gradient accumulation | 4 (effective batch: 128) |
| Learning rate | 2e-4 (linear decay) |
| Warmup ratio | 0.03 |
| Epochs | 3 |
| Total steps | 11,355 |
| Sequence length | 2048 (packed) |
| Chat template | ChatML |
| Response-only training | Yes — loss on assistant turns only |
| Data variations | 2 (conversation extension) → ~600K effective examples |
| Hardware | NVIDIA RTX 4090 |
| Training time | ~10 hours |

---

## Evaluation

### Method
1000 samples drawn from the `paper_instructions_300K-v1` test split. The fine-tuned model generates responses, which are then scored by `grok-3-mini` as an LLM judge.

### Judge Prompt (4 dimensions, 1–5 scale)
- **Faithfulness** — Does the response contain only factually correct claims? Penalise hallucinations.
- **Answer Correctness** — How closely does the response match the ground truth semantically?
- **Relevance** — Does the response directly address what was asked, without padding or going off-topic?
- **Completeness** — Does the response cover the key points from the ground truth without omitting important details?

### Results

| Metric | Score (1–5) |
|---|---|
| Faithfulness | 2.70 |
| Answer Correctness | 1.98 |
| Relevance | **3.04** |
| Completeness | 1.85 |
| **Overall** | **2.39** |

**Interpretation:** Relevance is the strongest dimension — the model stays on topic. Answer correctness and completeness are limited by the 135M parameter count; the model understands task structure but struggles to recall and reproduce factual content precisely.

---

## PaperResearcher API

The model is designed to be used as a structured API, not a free-form chatbot. The `PaperResearcher` class exposes 10 typed methods, each using the exact instruction strings the model was trained on:

```python
from paper_researcher import PaperResearcher

researcher = PaperResearcher("JaydeepR/SmolLM-135M-SFT-exp01")
passage = "Attention mechanisms compute weighted sums of values..."

# Extract key points
bullets: list[str] = researcher.extract_bullets(passage)

# Generate Q&A pairs
pairs: list[QAPair] = researcher.generate_qa_pairs(passage)
# → [QAPair(question="What does attention compute?", answer="Weighted sums of values")]

# Extract knowledge graph triplets
triplets: list[Triplet] = researcher.extract_triplets(passage)
# → [Triplet(subject="attention", relation="computes", object="weighted sums")]

# Answer a question given a passage
answer: str = researcher.answer("What does attention compute?", passage)

# Rephrase and elaborate
rephrased: str = researcher.rephrase(passage)

# Continue a passage from its beginning
continuation: str = researcher.continue_from(passage[:200])

# Extract a single key fact
fact: str = researcher.extract_fact(passage)

# Generate a question from a passage
question: str = researcher.generate_question(passage)

# Compare two passages
comparison: str = researcher.compare(passage_a, passage_b)

# Retrieval: which passage answers the question?
result: RetrievalResult = researcher.find_relevant(question, [passage_a, passage_b])
# → RetrievalResult(index=0, reasoning="Passage 1 directly defines...")
```

### Return Types

| Method | Return Type | Description |
|---|---|---|
| `extract_bullets` | `list[str]` | Parsed bullet points |
| `generate_qa_pairs` | `list[QAPair]` | `.question` and `.answer` fields |
| `extract_triplets` | `list[Triplet]` | `.subject`, `.relation`, `.object` fields |
| `find_relevant` | `RetrievalResult` | `.index` (0-based), `.reasoning` |
| All others | `str` | Raw text response |

---

## Raw Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

adapter_id = "JaydeepR/SmolLM-135M-SFT-exp01"
base_model_id = "paperbd/smollm_135M_arxiv_cpt"

tokenizer = AutoTokenizer.from_pretrained(adapter_id)
model = AutoModelForCausalLM.from_pretrained(base_model_id)
model = PeftModel.from_pretrained(model, adapter_id)

messages = [
    {"role": "system", "content": "You are an expert in AI and ML research. Your answers are concise and helpful."},
    {"role": "user", "content": "Extract the important points from this passage as markdown bullet points.\n\nAttention mechanisms..."},
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256, repetition_penalty=1.1, no_repeat_ngram_size=4)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

---

## Limitations

- 135M parameter model — limited factual recall and reasoning capacity
- Trained on synthetic data — instruction format matters; use the exact prompts from `tasks.py`
- Relevance strongest (3.04/5); correctness and completeness weak (< 2/5)
- Best suited for structured extraction (bullets, triplets, QA) over open-ended generation
- No comparison against uninstructed base model yet — exp02 planned

---

## Related Models

| Model | Description |
|---|---|
| [JaydeepR/SmolLM-135M-CPT-LoRA-r32](https://huggingface.co/JaydeepR/SmolLM-135M-CPT-LoRA-r32) | CPT base (this model's starting point) |
| [HuggingFaceTB/SmolLM-135M](https://huggingface.co/HuggingFaceTB/SmolLM-135M) | Original base model |

---

## Citation

```
@misc{smollm135m-sft-exp01,
  author = {Jaydeep Raijada},
  title  = {SmolLM-135M SFT exp01 — Instruction Tuning on ML Paper Research Tasks},
  year   = {2026},
  url    = {https://huggingface.co/JaydeepR/SmolLM-135M-SFT-exp01}
}
```
