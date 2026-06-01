"""Tiny config loader with single-level `extends` inheritance.

A config file may set `extends: base.yaml`; its keys override the parent's.
This keeps each experiment file down to the one knob it actually changes.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import yaml


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    parent_name = cfg.pop("extends", None)
    if parent_name:
        parent_path = os.path.join(os.path.dirname(path), parent_name)
        parent = load_config(parent_path)
        parent.update(cfg)  # child overrides parent
        cfg = parent
    return cfg


@dataclass
class RunConfig:
    name: str
    # model
    model_name: str
    max_seq_length: int
    load_in_4bit: bool
    # lora
    r: int
    lora_alpha: int
    lora_dropout: float
    use_rslora: bool
    use_gradient_checkpointing: Any
    random_state: int
    target_modules: list[str]
    # data
    dataset_name: str
    val_size: int
    seed: int
    packing: bool
    # optim
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    max_steps: int
    warmup_steps: int
    learning_rate: float
    embedding_learning_rate: float | None
    optim: str
    weight_decay: float
    lr_scheduler_type: str

    @classmethod
    def from_file(cls, path: str, **overrides: Any) -> "RunConfig":
        cfg = load_config(path)
        cfg.update({k: v for k, v in overrides.items() if v is not None})
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        unknown = set(cfg) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**cfg)

    @property
    def uses_decoupled_lr(self) -> bool:
        return self.embedding_learning_rate is not None
