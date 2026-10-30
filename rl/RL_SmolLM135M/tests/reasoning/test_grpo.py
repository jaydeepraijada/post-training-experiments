import unittest
from collections import defaultdict, deque
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from reasoning import env
from reasoning import eval_logging


VALID_COMPLETION = "<think>Work through the problem.</think> The answer is 42."


class OutputFormatTests(unittest.TestCase):
    def test_expected_output_format_is_accepted(self):
        completions = [
            VALID_COMPLETION,
            "<think>\nWork through the problem.\n</think>\nThe answer is 42.",
        ]

        for completion in completions:
            with self.subTest(completion=completion):
                self.assertTrue(env._has_expected_format(completion))

    def test_malformed_output_format_is_rejected(self):
        completions = [
            "The answer is 42.",
            "<think></think> The answer is 42.",
            "<think>Work through it.</think>",
            "</think><think>Work through it.</think> The answer is 42.",
            "<think>One.</think><think>Two.</think> The answer is 42.",
            "<think>Work through it.</think> <answer>The answer is 42.</answer>",
            "<think>Work through it.</think> <ANSWER type='final'>42</ANSWER>",
        ]

        for completion in completions:
            with self.subTest(completion=completion):
                self.assertFalse(env._has_expected_format(completion))

    def test_grpo_reward_uses_untagged_answer(self):
        test_case = self

        class RewardModel:
            def score(self, response, reference):
                test_case.assertEqual(response, "The answer is 42.")
                test_case.assertEqual(reference, "The answer is 42.")
                return 0.75

        reward_model = RewardModel()
        completions = [[{"role": "assistant", "content": VALID_COMPLETION}]]
        rewards = {
            reward.__name__: reward for reward in env.trl_reward_functions()
        }

        with patch.object(env, "_reward_model", reward_model):
            self.assertEqual(
                rewards["think_format_reward"](
                    [],
                    completions,
                    ["The answer is 42."],
                ),
                [1.0],
            )
            self.assertEqual(
                rewards["neuraltxt_reward"](
                    [],
                    completions,
                    ["The answer is 42."],
                ),
                [0.75],
            )

    def test_output_schema_rewards_are_symmetric(self):
        completions = [
            [{"role": "assistant", "content": "<think>x</think> plain text"}],
            [{"role": "assistant", "content": '<think>x</think> {"ok": true}'}],
            [{"role": "assistant", "content": '<think>x</think> {"ok": true}'}],
            [{"role": "assistant", "content": "<think>x</think> plain text"}],
        ]
        references = [
            "plain reference",
            '{"ok": false}',
            "plain reference",
            '{"ok": false}',
        ]
        rewards = {
            reward.__name__: reward for reward in env.trl_reward_functions()
        }

        for reward_name in ("output_datatype_reward", "output_schema_reward"):
            with self.subTest(reward_name=reward_name):
                self.assertEqual(
                    rewards[reward_name]([], completions, references),
                    [0.5, 0.5, -0.5, -0.5],
                )

    def test_doom_loop_penalizes_four_consecutive_words(self):
        completions = [
            [{
                "role": "assistant",
                "content": (
                    "<think>Reasoning is where, WHERE where where.</think> answer"
                ),
            }],
            [{
                "role": "assistant",
                "content": "<think>where where where next where</think> answer",
            }],
        ]
        rewards = {
            reward.__name__: reward for reward in env.trl_reward_functions()
        }

        self.assertEqual(
            rewards["doom_loop_reward"](
                [],
                completions,
                ["answer", "answer"],
            ),
            [-1.0, 0.0],
        )
        self.assertTrue(
            env._has_doom_loop("Reasoning is where, WHERE where where.")
        )
        self.assertFalse(env._has_doom_loop("where where where next where"))

    def test_reward_signal_log_contains_granular_values(self):
        logs = {
            "prompt": deque([["prompt"]]),
            "completion": deque(["<think>x</think> answer"]),
            "rewards": defaultdict(
                deque,
                {
                    "think_format_reward": deque([1.0]),
                    "output_datatype_reward": deque([0.5]),
                    "output_schema_reward": deque([0.5]),
                    "doom_loop_reward": deque([0.0]),
                    "neuraltxt_reward": deque([0.75]),
                },
            ),
            "advantages": deque([1.25]),
        }

        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "reward_signals.jsonl"
            count = eval_logging.append_eval_log(
                path,
                step=3,
                logs=logs,
                reward_weights={
                    "think_format_reward": 1.0,
                    "output_datatype_reward": 1.0,
                    "output_schema_reward": 1.0,
                    "doom_loop_reward": 1.0,
                    "neuraltxt_reward": 1.0,
                },
                references=["expected answer"],
                generation_batch=2,
            )
            record = __import__("json").loads(path.read_text().strip())

        self.assertEqual(count, 1)
        self.assertEqual(record["step"], 3)
        self.assertEqual(record["generation_batch"], 2)
        self.assertEqual(record["reference"], "expected answer")
        self.assertEqual(record["reasoning"], "x")
        self.assertEqual(record["response"], "answer")
        self.assertEqual(record["total_reward"], 2.75)
        self.assertEqual(record["rewards"]["output_datatype_reward"], 0.5)
        self.assertEqual(record["rewards"]["output_schema_reward"], 0.5)

    def test_eval_log_path_uses_step(self):
        log_dir = Path("models/example/log")

        self.assertEqual(
            eval_logging.eval_log_path(log_dir, 25),
            Path("models/example/log/test_25.jsonl"),
        )


if __name__ == "__main__":
    unittest.main()
