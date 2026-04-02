"""
Recovery script: reconstruct results from v6 log, finish GPT-5, run debate.
"""
import json, random, time, os, re, threading
import numpy as np
from collections import defaultdict
from scipy import stats as sp_stats
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import everything from main script
exec(open("run_experiments.py").read().split("if __name__")[0])

# ============================================================
# 1. Regenerate problems (same seed = same problems)
# ============================================================
probs = generate_all()
prob_map = {p["id"]: p for p in probs}
print(f"Regenerated {len(probs)} problems")

# ============================================================
# 2. Parse log to recover results
# ============================================================
MODEL_LABEL_TO_KEY = {v["label"]: k for k, v in MODELS.items()}

results = []
with open("experiment_output_v6.log") as f:
    for line in f:
        m = re.match(r'\s+(\S.*?)\s+\[(\d+)/101\]\s+(\S+)\s+C=(\d)\s+R=(\d)', line)
        if m:
            label, idx, pid, c, r = m.groups()
            mk = MODEL_LABEL_TO_KEY.get(label.strip())
            if mk and pid in prob_map:
                p = prob_map[pid]
                results.append({
                    "problem_id": pid, "level": p["level"],
                    "mutation_type": p.get("mutation_type", "none"),
                    "model": mk, "response": "(recovered from log)",
                    "correctness": int(c), "reasoning": int(r),
                    "total": int(c) + int(r),
                })

# Count what we have
from collections import Counter
counts = Counter(r["model"] for r in results)
print("Recovered results:")
for mk, cnt in sorted(counts.items()):
    print(f"  {MODELS[mk]['label']}: {cnt}/101")

# ============================================================
# 3. Run missing GPT-5 problems (97-101)
# ============================================================
gpt5_done = {r["problem_id"] for r in results if r["model"] == "gpt-5"}
missing = [(i, p) for i, p in enumerate(probs) if p["id"] not in gpt5_done]
print(f"\nRunning {len(missing)} missing GPT-5 problems...")

for i, p in missing:
    print(f"  [{i+1}/101] {p['id']}", end=" ")
    try:
        resp = call_model(p["prompt"], "gpt-5")
        sc = auto_score(p, resp)
        results.append({
            "problem_id": p["id"], "level": p["level"],
            "mutation_type": p.get("mutation_type", "none"),
            "model": "gpt-5", "response": resp, **sc,
        })
        print(f"C={sc['correctness']} R={sc['reasoning']}")
    except Exception as e:
        print(f"ERROR: {e}")
        results.append({
            "problem_id": p["id"], "level": p["level"],
            "mutation_type": p.get("mutation_type", "none"),
            "model": "gpt-5", "response": str(e),
            "correctness": 0, "reasoning": 0, "total": 0,
        })

# Save all results
with open("all_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved {len(results)} results to all_results.json")

# Print stats
compute_and_print(results)

# ============================================================
# 4. Run debate (3 models only)
# ============================================================
DEBATE_MODELS = ["gpt-5", "llama-70b", "llama-8b"]
print(f"\n\nMULTI-AGENT DEBATE (20 L3 problems, {len(DEBATE_MODELS)} models)...")
l3_subset = [p for p in probs if p["level"] == 3][:20]
debate_results = []
debate_lock = threading.Lock()
print_lock = threading.Lock()

def run_debate_model(mk):
    model_debate = []
    with print_lock:
        print(f"\n  Debate {MODELS[mk]['label']}:")
    for p in l3_subset:
        try:
            r = run_debate(p, mk)
            model_debate.append(r)
            with print_lock:
                print(f"    Debate {MODELS[mk]['label']} {p['id']} C={r['correctness']} R={r['reasoning']}")
        except Exception as e:
            with print_lock:
                print(f"    Debate {MODELS[mk]['label']} {p['id']} ERROR: {e}")
    with debate_lock:
        debate_results.extend(model_debate)
    with print_lock:
        print(f"\n  >>> Debate {MODELS[mk]['label']} DONE")

with ThreadPoolExecutor(max_workers=3) as executor:
    futures = {executor.submit(run_debate_model, mk): mk for mk in DEBATE_MODELS}
    for f in as_completed(futures):
        f.result()

with open("debate_results.json", "w") as f:
    json.dump(debate_results, f, indent=2)

# Compare single vs debate
print("\nDEBATE TABLE (Table 3):")
for mk in DEBATE_MODELS:
    single = [r["correctness"] for r in results
              if r["model"] == mk and r["level"] == 3]
    debate = [r["correctness"] for r in debate_results
              if r["model"] == mk]
    s_pct = np.mean(single) * 100 if single else 0
    d_pct = np.mean(debate) * 100 if debate else 0
    print(f"  {MODELS[mk]['label']:18s}: single={s_pct:.0f}%, "
          f"debate={d_pct:.0f}%, delta={d_pct - s_pct:+.0f}%")

print("\n\n=== ALL DONE ===")
