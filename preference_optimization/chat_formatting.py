from __future__ import annotations

from typing import Any

from paper_researcher.tasks import SYSTEM_PROMPT

Message = dict[str, str]


def _as_messages(value: Any, default_role: str) -> list[Message]:
    if isinstance(value, str):
        return [{"role": default_role, "content": value}]
    return list(value)


def with_instruction_tuning_system_prompt(messages: Any) -> list[Message]:
    messages = _as_messages(messages, default_role="user")
    system_message = {"role": "system", "content": SYSTEM_PROMPT}
    if messages and messages[0].get("role") == "system":
        return [system_message, *messages[1:]]
    return [system_message, *messages]


def normalize_explicit_preference_example(example: dict[str, Any]) -> dict[str, Any]:
    chosen = _as_messages(example["chosen"], default_role="assistant")
    rejected = _as_messages(example["rejected"], default_role="assistant")
    if "prompt" in example and example["prompt"] is not None:
        prompt = with_instruction_tuning_system_prompt(example["prompt"])
        return {"prompt": prompt, "chosen": chosen, "rejected": rejected}
    return {
        "chosen": with_instruction_tuning_system_prompt(chosen),
        "rejected": with_instruction_tuning_system_prompt(rejected),
    }


def build_reward_conversations(example: dict[str, Any]) -> tuple[list[Message], list[Message]]:
    if "prompt" in example and example["prompt"] is not None:
        prompt = with_instruction_tuning_system_prompt(example["prompt"])
        chosen = _as_messages(example["chosen"], default_role="assistant")
        rejected = _as_messages(example["rejected"], default_role="assistant")
        return prompt + chosen, prompt + rejected

    chosen = with_instruction_tuning_system_prompt(example["chosen"])
    rejected = with_instruction_tuning_system_prompt(example["rejected"])
    return chosen, rejected
