#!/bin/bash
set -e

echo "=== Starting exp04: Interleaved, no clean ==="
python train.py -d cpt_train_dataset_138_noclean.jsonl -td cpt_val_dataset_50.jsonl -o exp04_interleave_noclean --lora_r 16 --mix --mix_ratio 0.2 --wandb_project cpt-smollm135m

echo "=== Starting exp05: Interleaved, cleaned ==="
python train.py -d cpt_train_dataset_138.jsonl -td cpt_val_dataset_50.jsonl -o exp05_interleave_clean --lora_r 16 --mix --mix_ratio 0.2 --wandb_project cpt-smollm135m

echo "=== Starting exp06: Interleaved, cleaned, rsLoRA ==="
python train.py -d cpt_train_dataset_138.jsonl -td cpt_val_dataset_50.jsonl -o exp06_rslora_interleaved --lora_r 16 --mix --mix_ratio 0.2 --rslora --wandb_project cpt-smollm135m

echo "=== Starting exp05b: Interleaved, cleaned, r=32, max_steps=3000 ==="
python train.py -d cpt_train_dataset_138.jsonl -td cpt_val_dataset_50.jsonl -o exp05b_interleave_clean_r32 --lora_r 32 --mix --mix_ratio 0.2 --max_steps 3000 --wandb_project cpt-smollm135m

echo "=== All experiments done ==="
