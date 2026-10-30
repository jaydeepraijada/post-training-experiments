import argparse
import asyncio
import json
import os

from text_albumentations import AlpacaDataset, OpenAIModel
from text_albumentations.reasoning import agenerate_reasoning


OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_jsonl", help="Path to Alpaca JSONL input")
    parser.add_argument("output_jsonl", help="Path to JSONL output with reasoning")
    parser.add_argument(
        "--model-name",
        type=str,
        default=os.environ.get("TEXT_ALBUMENTATIONS_MODEL", "gpt-5-mini"),
        help="OpenAI-compatible model name",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "openrouter"],
        default=os.environ.get("LLM_PROVIDER", "openai"),
        help="Which OpenAI-compatible provider client path to use",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Optional OpenAI-compatible base URL override",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI-compatible API key",
    )
    parser.add_argument(
        "--total-concurrent-calls",
        type=int,
        default=8,
        help="Maximum concurrent async model calls",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Maximum generation attempts per row",
    )
    parser.add_argument(
        "--retry-base-delay",
        type=float,
        default=1.0,
        help="Base delay in seconds for exponential retry backoff",
    )
    parser.add_argument(
        "--print-model-responses",
        action="store_true",
        help="Print complete OpenAI-compatible responses before parsing",
    )
    parser.add_argument(
        "--response-format",
        choices=["auto", "json_schema", "json_object"],
        default="auto",
        help="Structured-output mode passed to text-albumentations OpenAIModel",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh", "disabled"],
        default=os.environ.get("TEXT_ALBUMENTATIONS_REASONING_EFFORT", "low"),
        help="Reasoning effort for compatible endpoints; use disabled for local servers",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start processing from this 0-based row index",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Stop after this many input rows have been processed",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output_jsonl instead of resuming from it",
    )
    return parser.parse_args()


def resolve_client_config(args):
    if args.provider == "openrouter":
        return {
            "api_key": args.api_key or os.environ.get("OPENROUTER_API_KEY"),
            "base_url": args.base_url or OPENROUTER_BASE_URL,
        }

    return {
        "api_key": args.api_key or os.environ.get("OPENAI_API_KEY"),
        "base_url": (
            args.base_url
            or os.environ.get("OPENAI_BASE_URL")
            or OPENAI_BASE_URL
        ),
    }


def build_async_runtime(args):
    client_kwargs = resolve_client_config(args)
    reasoning_effort = (
        None if args.reasoning_effort == "disabled" else args.reasoning_effort
    )
    runtime = OpenAIModel(
        args.model_name,
        base_url=client_kwargs["base_url"],
        api_key=client_kwargs["api_key"],
        async_mode=True,
        total_concurrent_calls=args.total_concurrent_calls,
        response_format=args.response_format,
        reasoning_effort=reasoning_effort,
    )
    if args.print_model_responses:
        enable_model_response_logging(runtime)
    return runtime


def enable_model_response_logging(runtime):
    completions = runtime.model.client.chat.completions
    original_create = completions.create

    async def create_with_logging(*args, **kwargs):
        try:
            response = await original_create(*args, **kwargs)
        except Exception as exc:
            print(
                "MODEL_RESPONSE_ERROR "
                + json.dumps(
                    {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            raise

        print(
            "MODEL_RESPONSE "
            + json.dumps(
                response.model_dump(mode="json"),
                ensure_ascii=False,
            ),
            flush=True,
        )
        return response

    completions.create = create_with_logging


def iter_alpaca_rows(path: str, start_index: int, max_rows: int | None):
    emitted = 0
    with open(path) as f:
        for row_idx, line in enumerate(f):
            if row_idx < start_index:
                continue
            if max_rows is not None and emitted >= max_rows:
                break

            raw = line.strip()
            if not raw:
                continue

            data = json.loads(raw)
            yield row_idx, AlpacaDataset(
                instruction=data["instruction"],
                input=data["input"],
                output=data["output"],
            )
            emitted += 1


async def add_reasoning_for_row(
    row_idx: int,
    row: AlpacaDataset,
    runtime,
    max_attempts: int = 3,
    retry_base_delay: float = 1.0,
):
    for attempt in range(1, max_attempts + 1):
        try:
            row_with_reasoning = await agenerate_reasoning(row.input, row, runtime)
            return row_idx, row_with_reasoning
        except Exception as exc:
            if attempt == max_attempts:
                raise

            delay = retry_base_delay * (2 ** (attempt - 1))
            print(
                f"Retrying reasoning row {row_idx} after attempt "
                f"{attempt}/{max_attempts} failed: "
                f"{type(exc).__name__}: {exc}"
            )
            await asyncio.sleep(delay)


def dump_reasoning_row(row_idx: int, row: AlpacaDataset) -> str:
    return json.dumps(
        {
            "_source_row_idx": row_idx,
            "instruction": row.instruction,
            "input": row.input,
            "reasoning": row.reasoning,
            "output": row.output,
        },
        ensure_ascii=False,
    )


def _is_valid_reasoning_row(data) -> bool:
    return (
        isinstance(data, dict)
        and (
            "_source_row_idx" not in data
            or (
                isinstance(data["_source_row_idx"], int)
                and data["_source_row_idx"] >= 0
            )
        )
        and isinstance(data.get("instruction"), str)
        and isinstance(data.get("input"), str)
        and isinstance(data.get("reasoning"), str)
        and isinstance(data.get("output"), str)
    )


def get_output_progress(path: str, start_index: int) -> tuple[int, int]:
    if not os.path.exists(path):
        return 0, start_index

    valid_rows = 0
    next_input_index = start_index
    with open(path, "rb+") as f:
        while True:
            line_start = f.tell()
            line = f.readline()
            if not line:
                break

            try:
                if not line.endswith(b"\n"):
                    raise ValueError("trailing partial line")
                data = json.loads(line)
                if not _is_valid_reasoning_row(data):
                    raise ValueError("invalid reasoning row")
            except Exception:
                f.seek(line_start)
                f.truncate()
                print(
                    "Truncated output file to the last complete valid row "
                    f"at row_count={valid_rows}"
                )
                break

            valid_rows += 1
            source_row_idx = data.get("_source_row_idx")
            if source_row_idx is None:
                next_input_index = max(
                    next_input_index,
                    start_index + valid_rows,
                )
            else:
                next_input_index = max(next_input_index, source_row_idx + 1)

    return valid_rows, next_input_index


def write_reasoning_row(out, row_idx: int, row: AlpacaDataset):
    out.write(dump_reasoning_row(row_idx, row) + "\n")
    out.flush()
    os.fsync(out.fileno())


async def amain():
    args = parse_args()
    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be at least 1")
    if args.retry_base_delay < 0:
        raise ValueError("--retry-base-delay must be non-negative")

    runtime = build_async_runtime(args)
    if args.overwrite:
        completed_rows, effective_start_index = 0, args.start_index
    else:
        completed_rows, effective_start_index = get_output_progress(
            args.output_jsonl,
            args.start_index,
        )
    processed_rows = effective_start_index - args.start_index
    if args.max_rows is not None and processed_rows >= args.max_rows:
        print(
            f"Output already covers {processed_rows} input rows, "
            f"which satisfies max_rows={args.max_rows}."
        )
        return

    output_mode = "w" if args.overwrite else "a"
    remaining_rows = (
        None if args.max_rows is None else args.max_rows - processed_rows
    )
    total_written = 0

    print(f"input_jsonl={args.input_jsonl}")
    print(f"output_jsonl={args.output_jsonl}")
    print(f"provider={args.provider}")
    print(f"model_name={args.model_name}")
    print(f"base_url={resolve_client_config(args)['base_url']}")
    print(f"total_concurrent_calls={args.total_concurrent_calls}")
    print(f"max_attempts={args.max_attempts}")
    print(f"retry_base_delay={args.retry_base_delay}")
    print(f"print_model_responses={args.print_model_responses}")
    print(f"start_index={args.start_index}")
    print(f"max_rows={args.max_rows}")
    print(f"completed_output_rows={completed_rows}")
    print(f"processed_input_rows={processed_rows}")
    print(f"effective_start_index={effective_start_index}")
    print(f"remaining_rows={remaining_rows}")

    with open(args.output_jsonl, output_mode) as out:
        batch = []
        for row_idx, row in iter_alpaca_rows(
            args.input_jsonl,
            effective_start_index,
            remaining_rows,
        ):
            batch.append((row_idx, row))
            if len(batch) >= args.total_concurrent_calls:
                total_written += await process_batch(
                    batch,
                    runtime,
                    out,
                    args.max_attempts,
                    args.retry_base_delay,
                )
                batch = []

        if batch:
            total_written += await process_batch(
                batch,
                runtime,
                out,
                args.max_attempts,
                args.retry_base_delay,
            )

    print(f"Saved {total_written} rows with reasoning to {args.output_jsonl}")


async def process_batch(
    batch,
    runtime,
    out,
    max_attempts: int = 3,
    retry_base_delay: float = 1.0,
):
    tasks = [
        add_reasoning_for_row(
            row_idx,
            row,
            runtime,
            max_attempts,
            retry_base_delay,
        )
        for row_idx, row in batch
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    written = 0

    for (row_idx, _), result in zip(batch, results, strict=True):
        if isinstance(result, KeyboardInterrupt):
            raise result
        if isinstance(result, Exception):
            print(
                f"Skipped reasoning row {row_idx} after retries failed: "
                f"{type(result).__name__}: {result}"
            )

    for result in results:
        if isinstance(result, Exception):
            continue
        row_idx, row = result
        write_reasoning_row(out, row_idx, row)
        written += 1
        print(f"Saved reasoning row {row_idx}")

    return written


if __name__ == "__main__":
    asyncio.run(amain())
