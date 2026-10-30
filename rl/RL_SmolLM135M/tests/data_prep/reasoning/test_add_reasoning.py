import unittest
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from text_albumentations import AlpacaDataset

from data_prep.reasoning import add_reasoning


class AddReasoningForRowTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.row = AlpacaDataset(
            instruction="Answer the question.",
            input="What is 2 + 2?",
            output="4",
        )

    async def test_retries_then_returns_generated_row(self):
        generated = self.row.model_copy(update={"reasoning": "Add the numbers."})

        with (
            patch.object(
                add_reasoning,
                "agenerate_reasoning",
                new=AsyncMock(side_effect=[ValueError("empty response"), generated]),
            ) as generate,
            patch.object(
                add_reasoning.asyncio,
                "sleep",
                new=AsyncMock(),
            ) as sleep,
        ):
            result = await add_reasoning.add_reasoning_for_row(
                4267,
                self.row,
                runtime=object(),
                max_attempts=3,
                retry_base_delay=0.5,
            )

        self.assertEqual(result, (4267, generated))
        self.assertEqual(generate.await_count, 2)
        sleep.assert_awaited_once_with(0.5)

    async def test_raises_after_attempts_are_exhausted(self):
        error = ValueError("empty response")

        with (
            patch.object(
                add_reasoning,
                "agenerate_reasoning",
                new=AsyncMock(side_effect=error),
            ) as generate,
            patch.object(
                add_reasoning.asyncio,
                "sleep",
                new=AsyncMock(),
            ) as sleep,
        ):
            with self.assertRaisesRegex(ValueError, "empty response"):
                await add_reasoning.add_reasoning_for_row(
                    4267,
                    self.row,
                    runtime=object(),
                    max_attempts=3,
                    retry_base_delay=0.5,
                )

        self.assertEqual(generate.await_count, 3)
        self.assertEqual(
            [call.args for call in sleep.await_args_list],
            [(0.5,), (1.0,)],
        )

    async def test_logs_complete_model_response_before_returning_it(self):
        response = SimpleNamespace(
            model_dump=lambda mode: {
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {
                            "content": None,
                            "reasoning": "all tokens were used thinking",
                        },
                    }
                ]
            }
        )
        create = AsyncMock(return_value=response)
        completions = SimpleNamespace(create=create)
        runtime = SimpleNamespace(
            model=SimpleNamespace(
                client=SimpleNamespace(
                    chat=SimpleNamespace(completions=completions)
                )
            )
        )

        with patch("builtins.print") as print_mock:
            add_reasoning.enable_model_response_logging(runtime)
            result = await completions.create(model="test-model")

        self.assertIs(result, response)
        create.assert_awaited_once_with(model="test-model")
        logged = print_mock.call_args.args[0]
        self.assertIn('"finish_reason": "length"', logged)
        self.assertIn('"content": null', logged)
        self.assertIn('"reasoning": "all tokens were used thinking"', logged)

    async def test_process_batch_writes_successes_and_skips_failures(self):
        generated = self.row.model_copy(update={"reasoning": "Add the numbers."})
        batch = [(10, self.row), (11, self.row)]

        with (
            patch.object(
                add_reasoning,
                "add_reasoning_for_row",
                new=AsyncMock(
                    side_effect=[
                        (10, generated),
                        ValueError("empty response"),
                    ]
                ),
            ),
            patch.object(add_reasoning, "write_reasoning_row") as write,
            patch("builtins.print") as print_mock,
        ):
            written = await add_reasoning.process_batch(
                batch,
                runtime=object(),
                out=StringIO(),
                max_attempts=1,
                retry_base_delay=0,
            )

        self.assertEqual(written, 1)
        write.assert_called_once_with(unittest.mock.ANY, 10, generated)
        self.assertTrue(
            any(
                "Skipped reasoning row 11" in call.args[0]
                for call in print_mock.call_args_list
            )
        )

    def test_output_progress_uses_source_indexes_after_skipped_rows(self):
        generated = self.row.model_copy(update={"reasoning": "Add the numbers."})

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.jsonl"
            output_path.write_text(
                add_reasoning.dump_reasoning_row(10, generated)
                + "\n"
                + add_reasoning.dump_reasoning_row(12, generated)
                + "\n"
            )

            progress = add_reasoning.get_output_progress(
                str(output_path),
                start_index=10,
            )

        self.assertEqual(progress, (2, 13))


if __name__ == "__main__":
    unittest.main()
