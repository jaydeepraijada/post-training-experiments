"""Quick test: does outlines-constrained `<think>...</think>{json}` decoding recover
good structured outputs from a GRPO model that otherwise prose-collapses JSON tasks?

Compares, on JSON-output test examples, the SAME model under:
  - FREE   : ordinary greedy generation (the model rambles a prose answer)
  - FORCED : phase-A natural <think>...</think> (regex-closed) + phase-B JSON answer
             constrained to the reference's own key schema via outlines.

Both scored with the project's NeuralTxtReward (answer-only, same as training/eval).

Usage:
  .venv/bin/python reasoning/test_constrained_decode.py \
      --model models/mlx/run5_checkpoint-17500-v2 --n 12
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import env  # reuse SYSTEM_PROMPT / format_row / scoring  # noqa: E402


def flat_str_schema(ref_obj: dict) -> dict:
    """Object schema with the reference's keys, all strings (mirrors a known prod schema)."""
    keys = list(ref_obj.keys())
    return {
        "type": "object",
        "properties": {k: {"type": "string"} for k in keys},
        "required": keys,
        "additionalProperties": False,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/mlx/run5_checkpoint-17500-v2")
    ap.add_argument("--dataset", default="paperbd/paper_instructions_300K-v1")
    ap.add_argument("--n", type=int, default=12, help="# of JSON-output examples to test")
    ap.add_argument("--think_max_tokens", type=int, default=320)
    ap.add_argument("--json_max_tokens", type=int, default=256)
    ap.add_argument("--free_max_tokens", type=int, default=512)
    args = ap.parse_args()

    import outlines
    from outlines.types import Regex, JsonSchema
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler

    print(f"loading {args.model} ...", flush=True)
    mlx_model, tokenizer = load(args.model)
    om = outlines.from_mlxlm(mlx_model, tokenizer)
    greedy = make_sampler(temp=0.0)

    def take_think(text: str) -> str:
        """Natural reasoning up to and including the first </think> (unconstrained)."""
        if "</think>" in text:
            return text.split("</think>", 1)[0] + "</think>"
        return text.strip() + "</think>"

    from datasets import load_dataset
    ds = load_dataset(args.dataset, split="test")

    def build_prompt(example) -> str:
        row = env.format_row(example)
        return tokenizer.apply_chat_template(
            row["prompt"], tokenize=False, add_generation_prompt=True
        )

    # pick JSON-output examples (flat dict of scalars)
    picked = []
    for ex in ds:
        out = ex["output"]
        if not env._is_valid_json(out):
            continue
        obj = json.loads(out)
        if isinstance(obj, dict) and obj and all(not isinstance(v, (dict, list)) for v in obj.values()):
            picked.append(ex)
        if len(picked) >= args.n:
            break

    print(f"testing {len(picked)} JSON-output examples\n" + "=" * 90)
    agg = {"free_nt": [], "forced_nt": [], "free_json": [], "forced_json": []}

    for i, ex in enumerate(picked):
        prompt = build_prompt(ex)
        reference = ex["output"]
        schema = flat_str_schema(json.loads(reference))

        # --- FREE ---
        free_out = generate(mlx_model, tokenizer, prompt=prompt,
                            max_tokens=args.free_max_tokens, sampler=greedy, verbose=False)

        # --- FORCED: reuse the model's OWN free reasoning, constrain only the JSON ---
        think = take_think(free_out)
        ans = om(prompt + think, JsonSchema(schema), max_tokens=args.json_max_tokens)
        forced_out = think + ans

        free_nt = env.score_neuraltxt(free_out, reference)
        forced_nt = env.score_neuraltxt(forced_out, reference)
        _, free_resp, _ = env._extract_think_content(free_out)
        _, forced_resp, _ = env._extract_think_content(forced_out)
        free_json = env._is_valid_json(free_resp)
        forced_json = env._is_valid_json(forced_resp)

        agg["free_nt"].append(free_nt); agg["forced_nt"].append(forced_nt)
        agg["free_json"].append(free_json); agg["forced_json"].append(forced_json)

        print(f"\n[{i}] ref: {reference[:120]}")
        print(f"  FREE   nt={free_nt:.3f} json={free_json}  resp: {free_resp[:120]!r}")
        print(f"  FORCED nt={forced_nt:.3f} json={forced_json}  resp: {forced_resp[:120]!r}")

    n = len(picked)
    print("\n" + "=" * 90)
    print(f"mean NeuralTxt   FREE={sum(agg['free_nt'])/n:.3f}   FORCED={sum(agg['forced_nt'])/n:.3f}")
    print(f"valid-JSON rate  FREE={sum(agg['free_json'])/n:.0%}   FORCED={sum(agg['forced_json'])/n:.0%}")


if __name__ == "__main__":
    main()
