## Config
lora_r: 32, lr: 2e-4, epochs: 10, mix: True, mix_ratio: 0.2, rslora: False
max_steps: 3000, stopping_strategy: all_exhausted

## Training Metrics
eval/loss: 3.116
train/loss (final step): 2.240
train/epoch: 0.19  (only 19% of one epoch — all_exhausted + 3000 steps barely touched custom data)
train/global_step: 3000
NOTE: all_exhausted strategy was wrong — model saw mostly HF data, lost domain signal entirely.
      Result worse than base model. Reverted to first_exhausted for future runs.
