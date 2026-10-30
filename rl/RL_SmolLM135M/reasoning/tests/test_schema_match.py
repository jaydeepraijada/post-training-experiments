import unittest

from reasoning import env
from reasoning.schema_match import compare_output_schema, infer_output_schema


class SchemaInferenceTests(unittest.TestCase):
    def test_identifies_supported_output_types(self):
        cases = {
            '{"question": "What?", "answer": "This."}': "json",
            "['first point', 'second point']": "python_list",
            "- first point\n- second point": "markdown",
            "A direct answer in ordinary prose.": "prose",
        }

        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(infer_output_schema(text).kind, expected)

    def test_single_item_lists_are_structurally_invalid(self):
        self.assertEqual(infer_output_schema("- only one item").kind, "prose")
        python_list = infer_output_schema("['only one item']")
        json_list = infer_output_schema('["only one item"]')
        self.assertEqual(python_list.kind, "python_list")
        self.assertEqual(json_list.kind, "json")
        self.assertFalse(python_list.structurally_valid)
        self.assertFalse(json_list.structurally_valid)

    def test_json_schema_matches_keys_and_nested_types_without_item_count(self):
        reference = (
            '[{"question": "What is GRPO?", "answer": "A training method."}, '
            '{"question": "Why use it?", "answer": "For reasoning."}]'
        )
        matching = (
            '[{"question": "What is RL?", "answer": "Learning from rewards."}, '
            '{"question": "Why?", "answer": "To improve behavior."}, '
            '{"question": "Where?", "answer": "In an environment."}]'
        )
        wrong_keys = (
            '[{"prompt": "What is RL?", "response": "Learning from rewards."}, '
            '{"prompt": "Why?", "response": "To improve behavior."}]'
        )

        self.assertTrue(compare_output_schema(reference, matching).schema_matches)
        wrong = compare_output_schema(reference, wrong_keys)
        self.assertTrue(wrong.datatype_matches)
        self.assertFalse(wrong.schema_matches)

    def test_python_list_matches_item_shape_without_exact_length(self):
        reference = "['one', 'two', 'three']"
        response = "['alpha', 'beta']"
        wrong_items = "['alpha', 2]"

        self.assertTrue(compare_output_schema(reference, response).schema_matches)
        self.assertFalse(
            compare_output_schema(reference, wrong_items).schema_matches
        )

    def test_markdown_distinguishes_list_styles_and_question_answer_schema(self):
        bullets = "- first point\n- second point"
        other_bullets = "* alpha\n* beta\n* gamma"
        numbered = "1. first point\n2. second point"
        qa = "**Question:** What is GRPO?\n\n**Answer:** A training method."
        qa_with_heading = (
            "### Q1\n**Question:** What is GRPO?\n\n"
            "**Answer:** A training method."
        )

        self.assertTrue(compare_output_schema(bullets, other_bullets).schema_matches)
        list_mismatch = compare_output_schema(bullets, numbered)
        self.assertTrue(list_mismatch.datatype_matches)
        self.assertFalse(list_mismatch.schema_matches)
        self.assertTrue(compare_output_schema(qa, qa).schema_matches)
        self.assertFalse(compare_output_schema(qa, qa_with_heading).schema_matches)

    def test_json_scalars_are_json_not_prose(self):
        self.assertEqual(infer_output_schema("42").kind, "json")
        self.assertEqual(infer_output_schema('"answer"').kind, "json")
        self.assertEqual(infer_output_schema("null").kind, "json")

    def test_rejects_duplicate_or_single_item_list_hacks(self):
        json_reference = '["first", "second"]'
        python_reference = "['first', 'second']"
        markdown_reference = "- first\n- second"

        for reference, response in (
            (json_reference, '["same", "same"]'),
            (json_reference, '["only"]'),
            (python_reference, "['same', 'same']"),
            (python_reference, "['only']"),
            (markdown_reference, "- same\n- same"),
        ):
            with self.subTest(reference=reference, response=response):
                match = compare_output_schema(reference, response)
                self.assertFalse(match.schema_matches)

    def test_rejects_empty_markdown_qa_labels(self):
        reference = "**Question:** What?\n\n**Answer:** This."
        response = "**Question:**\n\n**Answer:**"

        match = compare_output_schema(reference, response)
        self.assertTrue(match.datatype_matches)
        self.assertFalse(match.schema_matches)

    def test_rejects_prose_appended_to_a_standalone_markdown_list(self):
        reference = "- first\n- second"
        response = "- first\n- second\n\nThe passage also discusses other material."

        match = compare_output_schema(reference, response)
        self.assertTrue(match.datatype_matches)
        self.assertFalse(match.schema_matches)


class SelectedReasoningModelFailureTests(unittest.TestCase):
    def test_run5_id_1_collapses_bullet_list_to_prose(self):
        reference = (
            "- GRPO-CARE encourages exploration of coherent reasoning paths.\n"
            "- SEED-Bench-R1 balances perception and reasoning.\n"
            "- GRPO-CARE outperforms standard GRPO."
        )
        response = (
            "The passage defines GRPO-CARE as an extension of GRPO, introduced "
            "to balance perception and reasoning in multimodal understanding."
        )

        match = compare_output_schema(reference, response)
        self.assertEqual(match.reference.kind, "markdown")
        self.assertEqual(match.response.kind, "prose")
        self.assertFalse(match.datatype_matches)
        self.assertFalse(match.schema_matches)

    def test_run5_id_78_collapses_numbered_questions_to_prose(self):
        reference = (
            "1. What is the primary goal of VeriTraCER?\n"
            "2. What training algorithm is developed?\n"
            "3. What are its key contributions?"
        )
        response = (
            "The passage defines Simul-CROWN as a new variation of verified "
            "training designed to obtain tighter bounds."
        )

        match = compare_output_schema(reference, response)
        self.assertEqual(match.reference.kind, "markdown")
        self.assertEqual(match.response.kind, "prose")
        self.assertFalse(match.schema_matches)

    def test_run5_id_12_collapses_markdown_qa_to_prose(self):
        reference = (
            "**Question:** What is the name of the novel model?\n\n"
            "**Answer:** The Stochastic Sparse Mixture of Experts (S2MoE)"
        )
        response = (
            "The passage defines the Stochastic Sparse Mixture of Experts "
            "(S2MoE) as a mixture of experts."
        )

        match = compare_output_schema(reference, response)
        self.assertEqual(match.reference.kind, "markdown")
        self.assertEqual(match.response.kind, "prose")
        self.assertFalse(match.schema_matches)

    def test_run5_id_6_collapses_python_list_to_prose(self):
        reference = (
            "['Models struggle with hard samples and text bias.', "
            "'SRPO addresses complex multimodal reasoning.', "
            "'DanceGRPO adapts GRPO to visual generation.']"
        )
        response = (
            "The passage defines two main challenges in visual reasoning: "
            "models struggle with hard samples and text bias."
        )

        match = compare_output_schema(reference, response)
        self.assertEqual(match.reference.kind, "python_list")
        self.assertEqual(match.response.kind, "prose")
        self.assertFalse(match.schema_matches)

    def test_run5_id_25_collapses_json_object_to_prose(self):
        reference = (
            '{"question": "What errors can LLM detection extend to?", '
            '"answer": "TypeError and ValueError."}'
        )
        response = (
            "The passage defines static analysis tools as a method for "
            "detecting runtime errors."
        )

        match = compare_output_schema(reference, response)
        self.assertEqual(match.reference.kind, "json")
        self.assertEqual(match.response.kind, "prose")
        self.assertFalse(match.schema_matches)

    def test_env_assigns_both_penalties_to_structured_output_collapse(self):
        reference = "- first point\n- second point"
        completion = "<think>Summarize the points.</think> The passage has two points."

        self.assertEqual(env.score_output_datatype(completion, reference), -0.5)
        self.assertEqual(env.score_output_schema(completion, reference), -0.5)

    def test_env_rewards_datatype_but_penalizes_wrong_markdown_schema(self):
        reference = "- first point\n- second point"
        completion = "<think>List the points.</think> 1. first point\n2. second point"

        self.assertEqual(env.score_output_datatype(completion, reference), 0.5)
        self.assertEqual(env.score_output_schema(completion, reference), -0.5)


if __name__ == "__main__":
    unittest.main()
