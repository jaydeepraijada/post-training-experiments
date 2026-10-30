import argparse, json, os
from datasets import load_dataset
from mlx_lm import load, batch_generate

SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.
You are an expert in AI, deep learning, and machine learning research and its applications.
Your answers are concise and helps directly solve any user query truthfully.
If you do not know the answer, you will inform the user that you do not know instead of making answers up.
    """

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", "-m", type=str, required=True)
parser.add_argument("--num_samples", "-n", type=int, default=1000)
parser.add_argument("--batch_size", "-bs", type=int, default=8)
parser.add_argument("--max_new_tokens", type=int, default=1024)
parser.add_argument("--seed", type=int, default=3407)
args = parser.parse_args()

model, tokenizer = load(args.model_path)

dataset = load_dataset("paperbd/paper_instructions_300K-v1", split="test")
dataset = dataset.shuffle(seed=args.seed).select(range(args.num_samples))

model_name = os.path.basename(os.path.normpath(args.model_path))
out_dir = "data/evals/rollouts"
os.makedirs(out_dir, exist_ok=True)
output_path = f"{out_dir}/{model_name}_n{args.num_samples}.jsonl"

# Skip if already partially done
done_ids = set()
if os.path.exists(output_path):
    with open(output_path) as f:
        for line in f:
            done_ids.add(json.loads(line)["id"])

examples = [(i, ex) for i, ex in enumerate(list(dataset)) if i not in done_ids]
total = len(examples)
print(f"{model_name}: {total} to generate (already done: {len(done_ids)})")
if total == 0:
    print(f"Already complete, exiting.")
    exit(0)

def build_prompt(example):
    instruction = example["instruction"]
    inp = example.get("input", "")
    question = instruction if not inp else f"{instruction}\n\n{inp}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    token_ids = tokenizer.encode(prompt_str)
    return question, token_ids

completed = 0
with open(output_path, "a") as f:
    for batch_start in range(0, total, args.batch_size):
        batch_items = examples[batch_start: batch_start + args.batch_size]
        batch = [ex for _, ex in batch_items]

        questions, prompts, ground_truths = [], [], []
        for ex in batch:
            q, token_ids = build_prompt(ex)
            questions.append(q)
            prompts.append(token_ids)
            ground_truths.append(ex["output"])

        result = batch_generate(model, tokenizer, prompts=prompts, max_tokens=args.max_new_tokens, verbose=False)

        for j, ((orig_id, ex), q, gt, response) in enumerate(zip(batch_items, questions, ground_truths, result.texts)):
            record = {
                "id": orig_id,
                "question": q,
                "response": response,
                "ground_truth": gt,
                "model": model_name,
            }
            f.write(json.dumps(record) + "\n")
            if (batch_start + j + 1) % 100 == 0:
                print(f"  [{batch_start + j + 1}/{total}] {response[:60].replace(chr(10), ' ')}...")

        completed += len(batch)
        print(f"  batch {batch_start // args.batch_size + 1} done — {completed}/{total}")

print(f"\nDone: {output_path} ({completed} records)")
