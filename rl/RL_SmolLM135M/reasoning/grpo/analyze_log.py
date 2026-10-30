import json
import os
files = os.listdir("results")

print(files)
for f in files:
    if "syl"  not in f:
        continue
    data = json.load(open(f"results/{f}"))
    correctness = [d["correctness"] for d in data]

    mean_ = sum(correctness)/len(correctness)

    print(f, mean_)
    
