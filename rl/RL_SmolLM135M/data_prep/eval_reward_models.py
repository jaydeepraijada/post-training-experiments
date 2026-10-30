import json
import torch
import numpy as np
from scipy.stats import spearmanr, kendalltau

records = []
with open("instruction_tuning/evals/reward_model_dataset_clean.jsonl") as f:
    for line in f:
        records.append(json.loads(line))

rng = np.random.RandomState(42)
n = 30
indices = rng.choice(len(records), n, replace=False)
sample = [records[i] for i in indices]
print(f"Sample size: {n}")

# Compute all scores
print("Computing Skywork...")
from transformers import AutoModelForSequenceClassification, AutoTokenizer
rm = AutoModelForSequenceClassification.from_pretrained(
    "Skywork/Skywork-Reward-V2-Qwen3-0.6B",
    torch_dtype=torch.bfloat16, device_map="mps", num_labels=1,
)
tokenizer = AutoTokenizer.from_pretrained("Skywork/Skywork-Reward-V2-Qwen3-0.6B")
skywork_scores = []
for r in sample:
    conv = [{"role": "user", "content": r["instruction"]}, {"role": "assistant", "content": r["llm_response"]}]
    formatted = tokenizer.apply_chat_template(conv, tokenize=False)
    if tokenizer.bos_token and formatted.startswith(tokenizer.bos_token):
        formatted = formatted[len(tokenizer.bos_token):]
    inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=2048).to("mps")
    with torch.no_grad():
        skywork_scores.append(rm(**inputs).logits[0][0].item())

print("Computing Word F1...")
def word_f1(ref, hyp):
    rt = set(ref.lower().split()); ht = set(hyp.lower().split())
    if not ht: return 0.0
    p = len(rt & ht) / len(ht) if ht else 0
    rv = len(rt & ht) / len(rt) if rt else 0
    return 2*p*rv/(p+rv) if (p+rv)>0 else 0.0
f1_scores = [word_f1(r["reference"], r["llm_response"]) for r in sample]

print("Computing ROUGE-L...")
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
rouge_scores = [scorer.score(r["reference"], r["llm_response"])["rougeL"].fmeasure for r in sample]

print("Computing BERTScore...")
from bert_score import score as bertscore
refs = [r["reference"] for r in sample]
hyps = [r["llm_response"] for r in sample]
_, _, F1b = bertscore(hyps, refs, lang="en", verbose=False)
bert_scores = F1b.tolist()

print("Computing RewardBert...")
from qa_metrics.RewardBert import RewardBert
rb = RewardBert(device="mps")
rb_scores = []
for r in sample:
    norm, raw = rb.compute_score(r["reference"][:500], r["llm_response"][:500])
    rb_scores.append(norm)

gt = [r["score"] / 5.0 for r in sample]

all_methods = {
    "Skywork-V2 (0.6B)": skywork_scores,
    "Word F1": f1_scores,
    "ROUGE-L": rouge_scores,
    "BERTScore": bert_scores,
    "RewardBert": rb_scores,
}

print("\n" + "="*80)
print("CLEAN DATASET RESULTS (no think tags, n=30)")
print("="*80)
print(f"{'Method':<22} {'Spearman r':>12} {'Kendall tau':>12} {'Min':>10} {'Max':>10} {'Mean':>8}")
print("-"*80)
print(f"{'Judge':<22} {'-':>12} {'-':>12} {min(gt):>10.3f} {max(gt):>10.3f} {np.mean(gt):>8.3f}")
for name, scores in all_methods.items():
    sr, _ = spearmanr(gt, scores)
    kt, _ = kendalltau(gt, scores)
    s = np.array(scores)
    print(f"{name:<22} {sr:>12.4f} {kt:>12.4f} {s.min():>10.3f} {s.max():>10.3f} {s.mean():>8.3f}")

# Score buckets
print("\n" + "="*80)
print("BY SCORE BUCKET (Spearman r)")
print("="*80)
print(f"{'Method':<22}", end="")
for bname, fn in [("Low (<=0.4)", lambda x: x<=0.4), ("Mid (0.4-0.8)", lambda x: 0.4<x<0.8), ("High (>=0.8)", lambda x: x>=0.8)]:
    n_b = sum(1 for g in gt if fn(g))
    print(f"  {bname} (n={n_b})", end="")
print()
for name, scores in all_methods.items():
    print(f"{name:<22}", end="")
    for fn in [lambda x: x<=0.4, lambda x: 0.4<x<0.8, lambda x: x>=0.8]:
        idxs = [i for i,g in enumerate(gt) if fn(g)]
        if len(idxs) >= 3:
            sr, _ = spearmanr([gt[i] for i in idxs], [scores[i] for i in idxs])
            print(f"  r={sr:.3f}".ljust(20), end="")
        else:
            print(f"  n/a".ljust(20), end="")
    print()
