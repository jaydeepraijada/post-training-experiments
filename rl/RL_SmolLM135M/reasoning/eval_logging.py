import json
import math
from pathlib import Path

from accelerate.utils import gather_object

try:
    from reasoning.env import _extract_think_content
except ModuleNotFoundError:
    from env import _extract_think_content


def _finite_number(value):
    value = float(value)
    return value if math.isfinite(value) else None


def append_eval_log(
    path: Path,
    step: int,
    logs,
    reward_weights: dict[str, float],
    references: list | None = None,
    generation_batch: int | None = None,
) -> int:
    prompts = list(logs["prompt"])
    completions = list(logs["completion"])
    advantages = list(logs["advantages"])
    rewards = {
        name: list(values)
        for name, values in logs["rewards"].items()
    }
    if references:
        sample_count = len(references)
        prompts = prompts[-sample_count:]
        completions = completions[-sample_count:]
        advantages = advantages[-sample_count:]
        rewards = {
            name: values[-sample_count:]
            for name, values in rewards.items()
        }

    row_count = min(
        len(prompts),
        len(completions),
        len(advantages),
        *(len(values) for values in rewards.values()),
    )
    if references is not None:
        row_count = min(row_count, len(references))
    if row_count == 0:
        return 0

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as log_file:
        for index in range(row_count):
            reasoning, response, _ = _extract_think_content(completions[index])
            reward_signals = {
                name: _finite_number(values[index])
                for name, values in rewards.items()
            }
            total_reward = sum(
                value * reward_weights.get(name, 1.0)
                for name, value in reward_signals.items()
                if value is not None
            )
            record = {
                "step": step,
                "generation_batch": generation_batch,
                "sample_index": index,
                "prompt": prompts[index],
                "reference": references[index] if references is not None else None,
                "completion": completions[index],
                "reasoning": reasoning,
                "response": response,
                "rewards": reward_signals,
                "reward_weights": reward_weights,
                "total_reward": total_reward,
                "advantage": _finite_number(advantages[index]),
            }
            log_file.write(json.dumps(record, ensure_ascii=True) + "\n")
    return row_count


def eval_log_path(log_dir: Path, step: int) -> Path:
    return log_dir / f"test_{step}.jsonl"


def build_eval_logging_trainer(base_trainer, log_dir: Path):
    class EvalLoggingTrainer(base_trainer):
        _eval_log_step = None
        _eval_log_batch_index = 0

        def _calculate_rewards(
            self,
            inputs,
            prompts,
            completions,
            completion_ids_list,
        ):
            rewards = super()._calculate_rewards(
                inputs,
                prompts,
                completions,
                completion_ids_list,
            )
            if not self.model.training:
                references = [example.get("answer") for example in inputs]
                self._eval_log_references = gather_object(references)
            return rewards

        def _generate_and_score_completions(self, inputs):
            result = super()._generate_and_score_completions(inputs)
            if not self.model.training and self.accelerator.is_main_process:
                path = eval_log_path(log_dir, self.state.global_step)
                if self._eval_log_step != self.state.global_step:
                    path.unlink(missing_ok=True)
                    self._eval_log_step = self.state.global_step
                    self._eval_log_batch_index = 0

                reward_weights = {
                    name: float(weight)
                    for name, weight in zip(
                        self.reward_func_names,
                        self.reward_weights.tolist(),
                    )
                }
                append_eval_log(
                    path,
                    self.state.global_step,
                    self._logs,
                    reward_weights,
                    getattr(self, "_eval_log_references", None),
                    self._eval_log_batch_index,
                )
            if not self.model.training:
                self._eval_log_batch_index += 1
                self._eval_log_references = []
            return result

    return EvalLoggingTrainer
