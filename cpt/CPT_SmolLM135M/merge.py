"""
TIES-style model merge: trims low-magnitude task-vector parameters, then
adds the sparse task vector back to the base model.

For a single fine-tuned model this reduces to task arithmetic with sparsification
(the Sign-Elect step of full TIES only matters when merging ≥2 fine-tuned models).

Usage:
    python merge.py \
        --finetuned models/exp01_full/final \
        --base HuggingFaceTB/SmolLM-135M \
        --output models/exp01_full_ties \
        --density 0.3 \
        --weight 1.0
"""
import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
    _peft_available = True
except ImportError:
    _peft_available = False


def ties_merge(base_sd, ft_sd, density: float, weight: float) -> dict:
    merged = {}
    for key in base_sd:
        base_param = base_sd[key].float()
        if key not in ft_sd:
            merged[key] = base_sd[key]
            continue

        task_vec = ft_sd[key].float() - base_param

        # Trim: zero out parameters below the (1-density) magnitude quantile.
        if task_vec.numel() > 1 and density < 1.0:
            threshold = torch.quantile(task_vec.abs().flatten(), 1.0 - density)
            task_vec = task_vec * (task_vec.abs() >= threshold)

        merged[key] = (base_param + weight * task_vec).to(base_sd[key].dtype)
    return merged


def main():
    parser = argparse.ArgumentParser(description="TIES merge of a fine-tuned model into its base.")
    parser.add_argument("--finetuned", required=True, help="Path to fine-tuned model (full or LoRA adapter).")
    parser.add_argument("--base", default="HuggingFaceTB/SmolLM-135M", help="Base model HF ID or path.")
    parser.add_argument("--output", required=True, help="Output directory for the merged model.")
    parser.add_argument("--density", type=float, default=0.3,
                        help="Fraction of task-vector params to keep by magnitude (default 0.3).")
    parser.add_argument("--weight", type=float, default=1.0,
                        help="Scaling factor applied to the task vector before merging (default 1.0).")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading base model: {args.base}")
    base_model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(args.base)

    is_adapter = os.path.exists(os.path.join(args.finetuned, "adapter_config.json"))
    if is_adapter:
        if not _peft_available:
            raise RuntimeError("peft is required to load LoRA adapters. Install it with: pip install peft")
        print(f"LoRA adapter detected — merging into base before TIES.")
        ft_model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.float32)
        ft_model = PeftModel.from_pretrained(ft_model, args.finetuned)
        ft_model = ft_model.merge_and_unload()
    else:
        print(f"Loading fine-tuned model: {args.finetuned}")
        ft_model = AutoModelForCausalLM.from_pretrained(args.finetuned, torch_dtype=torch.float32)

    print(f"Running TIES merge (density={args.density}, weight={args.weight})...")
    merged_sd = ties_merge(base_model.state_dict(), ft_model.state_dict(), args.density, args.weight)

    del ft_model
    base_model.load_state_dict(merged_sd)

    os.makedirs(args.output, exist_ok=True)
    base_model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"Merged model saved to: {args.output}")


if __name__ == "__main__":
    main()
