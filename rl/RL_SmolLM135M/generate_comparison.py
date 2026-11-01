"""
Generate base, SFT, DPO and RL answers to the same test prompts for make_comparison_gif.py.

Prompts are the first prose, markdown and JSON rows of the eval set that fit in
MAX_PROMPT_CHARS, so they are chosen by rule and not by how well a model did. Every
model decodes greedily with its own training system prompt. Only the final answer is
kept: the RL model's <think> block is dropped.

Usage: uv run python generate_comparison.py [--adapter models/grpo_modal_v1/llm_best_31]
"""
import argparse
import json

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from reasoning.env import format_row
from reasoning.schema_match import infer_output_schema

DATASET = "paperbd/paper_instructions_300K-v1"
EVAL_SEED = 16
EVAL_SIZE = 256
MAX_PROMPT_CHARS = 1500
MAX_NEW_TOKENS = 384
OUTPUT = "base_sft_dpo_rl.json"

RL_BASE = "paperbd/neuraltxt-v1-135M-reasoning-base-v2"
MODELS = {
    "base": "HuggingFaceTB/SmolLM-135M",
    "sft": "JaydeepR/SmolLM-135M-SFT-exp01",
    "dpo": "JaydeepR/SmolLM-135M-neuraltxt-dpo-v1",
}
SFT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.
You are an expert in AI, deep learning, and machine learning research and its applications.
Your answers are concise and helps directly solve any user query truthfully.
If you do not know the answer, you will inform the user that you do not know instead of making answers up.
    """


def select_scenes():
    rows = load_dataset(DATASET, split="test").shuffle(seed=EVAL_SEED).select(range(EVAL_SIZE))
    scenes = {}
    for row in map(format_row, rows):
        user_prompt = row["prompt"][-1]["content"]
        datatype = infer_output_schema(row["answer"]).kind
        if datatype in ("prose", "markdown", "json") and len(user_prompt) <= MAX_PROMPT_CHARS:
            scenes.setdefault(datatype, row)
    return [scenes[datatype] for datatype in ("prose", "markdown", "json")]


def generate(model, tokenizer, messages):
    if tokenizer.chat_template:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        # The pretraining-only base model has no chat format: it just continues the task text.
        text = messages[-1]["content"] + "\n"
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
    return response.split("</think>")[-1].strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="models/grpo_modal_v1/llm_best_31")
    args = parser.parse_args()

    scenes = [{"row": row, "out": {"full_prompt": row["prompt"][-1]["content"]}} for row in select_scenes()]

    for key, model_id in MODELS.items():
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).eval()
        for scene in scenes:
            messages = [
                {"role": "system", "content": SFT_SYSTEM_PROMPT},
                scene["row"]["prompt"][-1],
            ]
            scene["out"][key] = generate(model, tokenizer, messages)
            print(f"{key}: {scene['out'][key][:80]!r}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(RL_BASE)
    base = AutoModelForCausalLM.from_pretrained(RL_BASE, dtype=torch.float32)
    model = PeftModel.from_pretrained(base, args.adapter).eval()
    for scene in scenes:
        scene["out"]["rl"] = generate(model, tokenizer, scene["row"]["prompt"])
        print(f"rl: {scene['out']['rl'][:80]!r}", flush=True)

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump([scene["out"] for scene in scenes], f, indent=2, ensure_ascii=False)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
