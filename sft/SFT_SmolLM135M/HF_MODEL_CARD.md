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

Supervised fine-tuning of [SmolLM-135M-CPT-LoRA-r32](https://huggingface.co/JaydeepR/SmolLM-135M-CPT-LoRA-r32) on 300K synthetic ML paper instruction pairs.

This is **exp01** in a series of SFT experiments on top of the CPT-adapted SmolLM-135M.

## Model Description

- **Base model:** `paperbd/smollm_135M_arxiv_cpt` (CPT-adapted SmolLM-135M, merged)
- **Method:** Supervised Fine-Tuning (SFT) with LoRA + `train_on_responses_only`
- **Domain:** ML/arXiv paper research tasks
- **Task:** Instruction following — bullets, QA, triplets, retrieval, comparison, etc.

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
| Hardware | NVIDIA RTX 4090 |
| Training time | ~10 hours |

## Training Data

- **Dataset:** `paperbd/paper_instructions_300K-v1` — 300K synthetic instruction-response pairs generated from arXiv ML papers
- **Variations:** 2 (conversation extension) → ~600K effective training examples
- **Train/val split:** 98% / 2%
- **Response-only training:** Loss computed only on assistant turns, not user prompts

## Evaluation Results

Evaluated on 1000 samples from the `paper_instructions_300K-v1` test split, judged by `grok-3-mini`:

| Metric | Score (1-5) |
|---|---|
| Faithfulness | 2.70 |
| Answer Correctness | 1.98 |
| Relevance | 3.04 |
| Completeness | 1.85 |
| **Overall** | **2.39** |

## How to Use

### As PaperResearcher API

```python
from paper_researcher import PaperResearcher

researcher = PaperResearcher("JaydeepR/SmolLM-135M-SFT-exp01")

passage = "Attention mechanisms compute weighted sums of values..."

bullets = researcher.extract_bullets(passage)
qa_pairs = researcher.generate_qa_pairs(passage)
triplets = researcher.extract_triplets(passage)
answer = researcher.answer("What does attention compute?", passage)
```

### Raw inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

adapter_id = "JaydeepR/SmolLM-135M-SFT-exp01"
base_model_id = "paperbd/smollm_135M_arxiv_cpt"

tokenizer = AutoTokenizer.from_pretrained(adapter_id)
model = AutoModelForCausalLM.from_pretrained(base_model_id)
model = PeftModel.from_pretrained(model, adapter_id)

messages = [
    {"role": "system", "content": "You are an expert in AI and ML research."},
    {"role": "user", "content": "Extract the key points from this passage as bullet points.\n\nAttention mechanisms..."},
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256, repetition_penalty=1.1)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

## Supported Tasks

| Task | Method |
|---|---|
| Extract bullet points | `researcher.extract_bullets(passage)` |
| Generate Q&A pairs | `researcher.generate_qa_pairs(passage)` |
| Generate a question | `researcher.generate_question(passage)` |
| Extract a fact | `researcher.extract_fact(passage)` |
| Answer a question | `researcher.answer(question, passage)` |
| Rephrase passage | `researcher.rephrase(passage)` |
| Continue passage | `researcher.continue_from(passage_start)` |
| Extract knowledge graph | `researcher.extract_triplets(passage)` |
| Compare two passages | `researcher.compare(passage_a, passage_b)` |
| Retrieval | `researcher.find_relevant(question, passages)` |

## Limitations

- 135M parameter model — limited factual recall and reasoning
- Trained on synthetic data — may not generalise to all instruction styles
- Relevance is the strongest dimension (3.04/5); correctness and completeness are weak (< 2/5)
- Best used for structured extraction tasks, not open-ended QA

## Citation

```
@misc{smollm135m-sft-exp01,
  author = {Jaydeep Raijada},
  title  = {SmolLM-135M SFT exp01 — Instruction Tuning on ML Paper Research Tasks},
  year   = {2026},
  url    = {https://huggingface.co/JaydeepR/SmolLM-135M-SFT-exp01}
}
```
