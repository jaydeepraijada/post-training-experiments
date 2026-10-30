import unittest
from unittest.mock import patch

from reasoning import env
from reasoning.grpo.paper_dataset import PaperInstructionDataset


class PaperGRPOPipelineSmokeTests(unittest.TestCase):
    def test_paper_dataset_batches_prompts_and_rewards(self):
        class RewardModel:
            def __init__(self):
                self.responses = []
                self.references = []

            def score(self, response, reference):
                self.responses.append(response)
                self.references.append(reference)
                return 0.75

            def batch_score(self, responses, references):
                self.responses = responses
                self.references = references
                return [0.75 for _ in responses]

        reward_model = RewardModel()
        dataset = PaperInstructionDataset(
            "paperbd/paper_instructions_300K-v1",
            data_size=2,
            seed=env.SEED,
        )
        items = [dataset[i]["item"] for i in range(2)]
        completions = [
            f"<think>Use the passage.</think> {items[0]['answer']}",
            f"<think>Use the passage.</think> {items[1]['answer']}",
        ]

        with patch.object(env, "_reward_model", reward_model):
            reward_batch = env.score_completions(
                completions,
                [item["answer"] for item in items],
            )

        self.assertEqual(len(items), 2)
        self.assertEqual(len(reward_batch.total), 2)
        self.assertEqual(set(reward_batch.components), {
            "think_format_reward",
            "output_datatype_reward",
            "output_schema_reward",
            "doom_loop_reward",
            "neuraltxt_reward",
            "length_penalty_reward",
        })
        self.assertTrue(all(item["prompt"][0]["role"] == "system" for item in items))
        self.assertTrue(all(item["prompt"][1]["role"] == "user" for item in items))
        self.assertEqual(reward_model.references, [item["answer"] for item in items])
        self.assertEqual(
            reward_batch.components["think_format_reward"].tolist(),
            [1.0, 1.0],
        )
        self.assertEqual(
            reward_batch.components["neuraltxt_reward"].tolist(),
            [0.75, 0.75],
        )


if __name__ == "__main__":
    unittest.main()
