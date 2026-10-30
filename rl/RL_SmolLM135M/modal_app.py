"""
Run GRPO on Modal. Checkpoints and logs persist on the `rl-grpo` Volume, and each
run resumes from the last saved eval. --budget-hours caps total GPU time for an
output id across restarts, so retries and preemptions can't overrun the budget.

Usage:
    modal run --detach modal_app.py --output-id grpo_modal_v1 --gpu L40S --budget-hours 46

    modal volume ls rl-grpo models/grpo_modal_v1
    modal volume get rl-grpo models/grpo_modal_v1/llm_best_0 ./models/grpo_modal_v1/
"""
import os
import subprocess
import threading
import time
from pathlib import Path, PurePosixPath

import modal

CODE_DIR = PurePosixPath("/root/rl")
VOLUME_DIR = PurePosixPath("/vol")
MAX_FUNCTION_TIMEOUT_SECONDS = 24 * 60 * 60
RUNTIME_SAVE_SECONDS = 60

app = modal.App("rl-grpo")
volume = modal.Volume.from_name("rl-grpo", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "accelerate>=1.0",
        "datasets>=3.6",
        "markdown-it-py>=3.0",
        "neural-txt[hf]>=0.1.6",
        "numpy>=1.26",
        "pandas>=2.2",
        "peft>=0.18.1",
        "pyyaml>=6.0",
        "rich>=14.3.2",
        "torch>=2.7",
        "tqdm>=4.66",
        "transformers>=4.57.6,<5",
    )
    .env({"HF_HOME": str(VOLUME_DIR / "hf"), "PYTHONUNBUFFERED": "1"})
    .add_local_dir("reasoning", str(CODE_DIR / "reasoning"))
)


@app.function(image=image, volumes={str(VOLUME_DIR): volume})
def train(config: str, output_id: str, budget_hours: float) -> None:
    # Relative models/ and logs/ paths in train.py land on the Volume.
    os.chdir(VOLUME_DIR)
    runtime_path = Path("models") / output_id / "runtime_seconds.txt"
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    previous_seconds = float(runtime_path.read_text()) if runtime_path.exists() else 0.0
    remaining_seconds = budget_hours * 3600 - previous_seconds
    print(f"Used {previous_seconds / 3600:.2f}h of {budget_hours}h budget")
    if remaining_seconds <= 0:
        print("Budget used up, not starting.")
        return

    start = time.time()
    finished = threading.Event()

    def save_runtime() -> None:
        while not finished.wait(RUNTIME_SAVE_SECONDS):
            runtime_path.write_text(str(previous_seconds + time.time() - start))
            volume.commit()

    threading.Thread(target=save_runtime, daemon=True).start()
    try:
        subprocess.run(
            [
                "python",
                str(CODE_DIR / "reasoning/grpo/train.py"),
                str(CODE_DIR / config),
                "-o",
                output_id,
                "--resume",
            ],
            check=True,
            timeout=remaining_seconds,
        )
    except subprocess.TimeoutExpired:
        print("Budget used up, stopping.")
    finally:
        finished.set()
        runtime_path.write_text(str(previous_seconds + time.time() - start))
        volume.commit()


@app.local_entrypoint()
def main(
    output_id: str,
    budget_hours: float,
    gpu: str = "L40S",
    config: str = "reasoning/grpo/modal_gpu.yaml",
) -> None:
    # spawn, not remote: a long-lived client's auth token expires, and that would stop the app.
    call = train.with_options(
        gpu=gpu,
        timeout=MAX_FUNCTION_TIMEOUT_SECONDS,
        retries=modal.Retries(max_retries=3, initial_delay=60.0, backoff_coefficient=2.0),
    ).spawn(config, output_id, budget_hours)
    print(f"Spawned {call.object_id}")
