"""
Merge a LoRA adapter into its base model and save the result.

Usage:
    python instruction_tuning/merge_adapter.py \
        --adapter_path models/cpt_arxiv_1495/final \
        --output_path models/cpt_arxiv_1495_merged
"""
import argparse
import json
import os
import torch


def merge_adapter(adapter_path: str, output_path: str):
    # Read base model from adapter config
    config_path = os.path.join(adapter_path, "adapter_config.json")
    with open(config_path) as f:
        adapter_config = json.load(f)
    base_model_id = adapter_config["base_model_name_or_path"]
    task_type = adapter_config.get("task_type")
    print(f"Base model: {base_model_id}")
    print(f"Task type:  {task_type}")
    print(f"Adapter:    {adapter_path}")
    print(f"Output:     {output_path}")

    # Strip the bnb-4bit suffix to get the fp16 base — merge requires full precision
    hf_base_id = base_model_id.replace("-bnb-4bit", "")
    if hf_base_id != base_model_id:
        print(f"Using non-quantized base for merge: {hf_base_id}")

    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    model_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "cpu"}
    if task_type == "SEQ_CLS":
        base = AutoModelForSequenceClassification.from_pretrained(
            hf_base_id,
            num_labels=1,
            **model_kwargs,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        base.config.pad_token_id = tokenizer.pad_token_id
    else:
        base = AutoModelForCausalLM.from_pretrained(
            hf_base_id,
            **model_kwargs,
        )
    model = PeftModel.from_pretrained(base, adapter_path)
    model = model.merge_and_unload()

    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"Saved merged model to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()
    merge_adapter(args.adapter_path, args.output_path)
