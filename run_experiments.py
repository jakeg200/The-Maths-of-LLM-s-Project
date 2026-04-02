"""
MATH3001 Dissertation: The Limits of LLM Reasoning
Fully Automated Experimental Pipeline (6 models, two-signal scoring)

SETUP:
    pip install openai groq scipy numpy google-generativeai anthropic

    Set API keys:
        export OPENAI_API_KEY="sk-..."
        export GROQ_API_KEY="gsk_..."
        export ANTHROPIC_API_KEY="sk-ant-..."
        export GOOGLE_API_KEY="AI..."

USAGE:
    python run_experiments.py                # full pipeline
    python run_experiments.py --mutations    # generate + preview problems only
    python run_experiments.py --stats        # recompute stats from saved results
    python run_experiments.py --debate       # debate only (needs all_results.json)
"""

import json, random, time, os, re, argparse
import numpy as np
from collections import defaultdict
from scipy import stats as sp_stats
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# ================================================================
# CONFIG — paste your keys here or use environment variables
# ================================================================

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "YOUR_KEY_HERE")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "YOUR_KEY_HERE")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "YOUR_KEY_HERE")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "YOUR_KEY_HERE")

MODELS = {
    "gpt-5":       {"provider": "openai",    "model_id": "gpt-5",
                    "label": "GPT-5"},
    # "claude":    {"provider": "anthropic", "model_id": "claude-sonnet-4-20250514",
    #               "label": "Claude Sonnet"},  # credits exhausted
    # "gemini":    {"provider": "google",    "model_id": "gemini-2.5-pro",
    #               "label": "Gemini 2.5 Pro"},  # API key 403
    "gpt-4o-mini": {"provider": "openai",    "model_id": "gpt-4o-mini-2024-07-18",
                    "label": "GPT-4o-mini"},
    "llama-70b":   {"provider": "groq",      "model_id": "llama-3.3-70b-versatile",
                    "label": "Llama 70B"},
    "llama-8b":    {"provider": "groq",      "model_id": "llama-3.1-8b-instant",
                    "label": "Llama 8B"},
    "llama4-scout":{"provider": "groq",      "model_id": "meta-llama/llama-4-scout-17b-16e-instruct",
                    "label": "Llama 4 Scout"},
    "qwen3-32b":   {"provider": "groq",      "model_id": "qwen/qwen3-32b",
                    "label": "Qwen 3 32B"},
}

SYSTEM_PROMPT = "Answer the following question."
TEMPERATURE = 0
N_SURFACE = 50
N_STRUCTURAL = 50  # 15 flip + 13 mediator + 12 reverse + 10 numerical

# ================================================================
# SURFACE MUTATION WORD BANKS (20 domains)
# ================================================================

DOMAINS = [
    {"confounder": "neighbourhood poverty level", "treatment": "government funding", "outcome": "dropout rates",
     "context": "A government study examined schools across the country."},
    {"confounder": "patient illness severity", "treatment": "medication dosage", "outcome": "hospital stay length",
     "context": "A hospital reviewed patient records from the past year."},
    {"confounder": "crime rate", "treatment": "police officers deployed", "outcome": "arrests made",
     "context": "A city council analysed policing data across districts."},
    {"confounder": "soil contamination", "treatment": "fertiliser applied", "outcome": "crop failure rate",
     "context": "An agricultural study examined farms in a drought-prone region."},
    {"confounder": "traffic congestion", "treatment": "traffic officers deployed", "outcome": "accidents reported",
     "context": "A transport department reviewed accident data."},
    {"confounder": "building deterioration", "treatment": "maintenance budget", "outcome": "tenant complaints",
     "context": "A housing authority examined records across its properties."},
    {"confounder": "student prior attainment", "treatment": "tutoring hours", "outcome": "exam failure rate",
     "context": "A university examined its tutoring programme."},
    {"confounder": "population density", "treatment": "doctors per capita", "outcome": "disease prevalence",
     "context": "A public health study compared regions."},
    {"confounder": "river pollution level", "treatment": "water treatment investment", "outcome": "fish mortality",
     "context": "An environmental agency studied waterways."},
    {"confounder": "unemployment rate", "treatment": "welfare spending", "outcome": "homelessness rate",
     "context": "A social policy institute examined 200 local authorities."},
    {"confounder": "wildfire risk level", "treatment": "rangers stationed", "outcome": "forest hectares burned",
     "context": "A national park service reviewed decade-long data."},
    {"confounder": "earthquake magnitude", "treatment": "rescue teams deployed", "outcome": "casualties",
     "context": "A disaster response agency analysed recent earthquakes."},
    {"confounder": "storm severity", "treatment": "emergency responders", "outcome": "infrastructure damage",
     "context": "A government report examined major storm responses."},
    {"confounder": "company debt level", "treatment": "consultants hired", "outcome": "layoffs",
     "context": "A business school studied corporate restructuring."},
    {"confounder": "software bug severity", "treatment": "QA engineers assigned", "outcome": "customer complaints",
     "context": "A tech company reviewed product quality data."},
    {"confounder": "legal case complexity", "treatment": "lawyers assigned", "outcome": "trial duration",
     "context": "A law firm analysed its case portfolio."},
    {"confounder": "infection severity", "treatment": "antibiotic dosage", "outcome": "recovery time",
     "context": "A clinical trial examined bacterial infection treatments."},
    {"confounder": "road surface deterioration", "treatment": "repair crew hours", "outcome": "pothole complaints",
     "context": "A highways agency reviewed maintenance data."},
    {"confounder": "child behavioural difficulty", "treatment": "specialist support hours", "outcome": "school exclusions",
     "context": "An education authority examined outcomes for children with additional needs."},
    {"confounder": "machine wear level", "treatment": "maintenance hours", "outcome": "production downtime",
     "context": "A factory analysed equipment maintenance records."},
]

MEDIATOR_NAMES = [
    "immune response", "chemical reaction", "secondary process",
    "behavioural change", "inflammatory response", "feedback mechanism",
    "regulatory cascade", "compensatory mechanism",
]

# ================================================================
# PROMPT GENERATORS
# ================================================================

def make_l1():
    return {
        "id": "L1_original", "level": 1, "mutation_type": "none",
        "prompt": (
            "A city analysed records of past fires and found that fires where "
            "more firefighters were dispatched suffered more property damage. "
            "The city council proposes cutting the fire department budget to "
            "reduce property damage.\n\n"
            "Is the council's reasoning valid? Answer Yes or No, then explain."
        ),
        "correct_binary": "no",
        "confounder_name": "severity",
        "reasoning_terms": ["severity", "confound", "common cause",
                            "spurious", "hidden", "third variable"],
    }

def make_l2(domain, idx):
    # Extract key confounder word for matching
    conf_key = domain["confounder"].split()[-1].lower()  # e.g. "poverty", "severity"
    return {
        "id": f"L2_{idx:03d}", "level": 2, "mutation_type": "surface",
        "prompt": (
            f"{domain['context']} The data shows that areas with more "
            f"{domain['treatment']} tend to have higher {domain['outcome']}. "
            f"A decision-maker proposes reducing {domain['treatment']} to "
            f"reduce {domain['outcome']}.\n\n"
            f"Is the decision-maker's reasoning valid? Answer Yes or No, "
            f"then explain."
        ),
        "correct_binary": "no",
        "confounder_name": conf_key,
        "reasoning_terms": [conf_key, domain["confounder"].split()[0].lower(),
                            "confound", "common cause", "spurious",
                            "hidden", "driven by"],
    }

def make_l3_flip(domain, idx):
    return {
        "id": f"L3_flip_{idx:03d}", "level": 3, "mutation_type": "flip_sign",
        "prompt": (
            f"{domain['context']} The data shows that areas with more "
            f"{domain['treatment']} tend to have higher {domain['outcome']}. "
            f"The correlation is confounded by {domain['confounder']}, which "
            f"drives both. However, unlike typical cases, {domain['treatment']} "
            f"actually makes {domain['outcome']} worse due to an adverse "
            f"interaction.\n\n"
            f"If we deliberately increase {domain['treatment']}, will "
            f"{domain['outcome']} increase or decrease? Answer Increase or "
            f"Decrease, then explain."
        ),
        "correct_binary": "increase",
        "confounder_name": domain["confounder"].split()[-1].lower(),
        # Reasoning: does it acknowledge the adverse/harmful direction?
        "reasoning_terms": ["adverse", "worse", "harmful", "makes things worse",
                            "amplif", "exacerbat", "increase damage",
                            "increase " + domain["outcome"].split()[0].lower()],
    }

def make_l3_med(domain, idx):
    med = random.choice(MEDIATOR_NAMES)
    med_key = med.split()[0]  # e.g. "immune", "chemical"
    return {
        "id": f"L3_med_{idx:03d}", "level": 3, "mutation_type": "add_mediator",
        "mediator": med,
        "prompt": (
            f"{domain['context']} {domain['treatment'].capitalize()} does not "
            f"affect {domain['outcome']} directly. Instead, it triggers a "
            f"{med}, which in turn reduces {domain['outcome']}. "
            f"{domain['confounder'].capitalize()} independently drives both "
            f"{domain['treatment']} and {domain['outcome']}.\n\n"
            f"Consider a specific case where {domain['confounder']} was high, "
            f"{domain['treatment']} was provided, and {domain['outcome']} was "
            f"moderate. Had {domain['treatment']} NOT been provided in this "
            f"specific case, would {domain['outcome']} have been worse? "
            f"Answer Yes or No, then explain your reasoning for this "
            f"specific case."
        ),
        "correct_binary": "yes",
        "confounder_name": domain["confounder"].split()[-1].lower(),
        # Reasoning: does it reason about the specific individual + mediator?
        "reasoning_terms": ["this specific", "this case", "this particular",
                            "individual", "counterfactual", "had not",
                            "would have", med_key],
    }

def make_l3_rev(domain, idx):
    return {
        "id": f"L3_rev_{idx:03d}", "level": 3, "mutation_type": "reverse_edge",
        "prompt": (
            f"{domain['context']} The data shows that areas with more "
            f"{domain['treatment']} tend to have higher {domain['outcome']}. "
            f"However, {domain['treatment']} does not cause {domain['outcome']} "
            f"--- it merely detects or reveals pre-existing {domain['outcome']} "
            f"that would have occurred regardless.\n\n"
            f"Would reducing {domain['treatment']} reduce the actual amount of "
            f"{domain['outcome']}? Answer Yes or No, then explain."
        ),
        "correct_binary": "no",
        "confounder_name": domain["confounder"].split()[-1].lower(),
        # Reasoning: does it distinguish detection from causation?
        "reasoning_terms": ["detect", "reveal", "does not cause",
                            "not the same as caus", "would still",
                            "regardless", "observation"],
    }

def make_l3_num(idx):
    """Numerical: model must compute from given structural equations."""
    a = random.choice([1, 2, 3])
    b = random.choice([2, 3, 4])
    g = random.choice([1, 2])
    sv = sorted(random.sample(range(1, 6), 3))
    S = np.array(sv, dtype=float)
    F = a * S
    D = b * S - g * F
    fv = random.choice([2, 4, 6])
    correct_cov = float(np.cov(F, D, ddof=0)[0, 1])
    correct_int = float(b * np.mean(S) - g * fv)
    return {
        "id": f"L3_num_{idx:03d}", "level": 3, "mutation_type": "numerical",
        "prompt": (
            f"Consider a system with three variables S, F, and D:\n"
            f"  S = U_S (exogenous, takes values {{{sv[0]}, {sv[1]}, "
            f"{sv[2]}}} equally)\n"
            f"  F = {a}S\n"
            f"  D = {b}S - {g}F\n\n"
            f"Q1: What is Cov(F, D)?\n"
            f"Q2: If we intervene to set F = {fv} (removing the equation "
            f"F = {a}S), what is E[D | do(F = {fv})]?\n\n"
            f"Give numerical answers."
        ),
        "correct_binary": None,  # not yes/no — special handling
        "correct_cov": correct_cov,
        "correct_int": correct_int,
        "confounder_name": None,
        "reasoning_terms": ["do(", "intervene", "remov", "sever",
                            "structural equation"],
    }

def generate_all():
    """Generate all 101 problems with fixed seed."""
    random.seed(42)
    np.random.seed(42)
    probs = [make_l1()]

    # Level 2: 50 surface mutations
    for i, d in enumerate(random.choices(DOMAINS, k=N_SURFACE)):
        probs.append(make_l2(d, i))

    # Level 3: 15 flip + 13 mediator + 12 reverse + 10 numerical = 50
    ds = random.choices(DOMAINS, k=40)
    for i in range(15):
        probs.append(make_l3_flip(ds[i], i))
    for i in range(13):
        probs.append(make_l3_med(ds[15 + i], i))
    for i in range(12):
        probs.append(make_l3_rev(ds[28 + i], i))
    for i in range(10):
        probs.append(make_l3_num(i))

    return probs


# ================================================================
# AUTOMATED SCORING (two binary signals)
# ================================================================

def extract_answer(response):
    """Extract Yes/No/Increase/Decrease from first line."""
    first = response.lower().split('\n')[0].strip()
    words = first.split()
    if not words:
        return "unclear"
    fw = re.sub(r'[^a-z]', '', words[0])
    if fw in ["yes", "no", "increase", "decrease"]:
        return fw
    # Check first line for keywords
    for kw in ["yes", "no", "increase", "decrease"]:
        if kw in first:
            return kw
    return "unclear"


def score_correctness(problem, response):
    """Binary correctness: 1 if answer matches ground truth, 0 otherwise."""
    if problem.get("correct_binary") is None:
        # Numerical problem — check both computed values
        nums = [float(n) for n in re.findall(r'-?\d+\.?\d*', response)]
        cov_ok = any(abs(n - problem["correct_cov"]) < 0.5 for n in nums)
        int_ok = any(abs(n - problem["correct_int"]) < 0.5 for n in nums)
        return 1 if (cov_ok and int_ok) else 0
    else:
        ans = extract_answer(response)
        return 1 if ans == problem["correct_binary"].lower() else 0


def score_reasoning(problem, response):
    """Binary reasoning: 1 if response engages with correct causal structure."""
    resp_lower = response.lower()
    terms = problem.get("reasoning_terms", [])
    if not terms:
        return 0
    matches = sum(1 for t in terms if t.lower() in resp_lower)
    # Require at least 2 matches (or 1 if only 1-2 terms)
    threshold = min(2, max(1, len(terms) // 3))
    return 1 if matches >= threshold else 0


def auto_score(problem, response):
    """Return dict with both scores."""
    c = score_correctness(problem, response)
    r = score_reasoning(problem, response)
    return {"correctness": c, "reasoning": r, "total": c + r}


# ================================================================
# API CALLS (4 providers)
# ================================================================

def call_openai(prompt, model_id, system=SYSTEM_PROMPT):
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    params = {
        "model": model_id,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": prompt}],
    }
    # GPT-5 and reasoning models don't support temperature or max_tokens
    if model_id.startswith("gpt-5") or model_id.startswith("o"):
        params["max_completion_tokens"] = 1000
    else:
        params["max_tokens"] = 1000
        params["temperature"] = TEMPERATURE
    resp = client.chat.completions.create(**params)
    return resp.choices[0].message.content


def call_anthropic(prompt, model_id, system=SYSTEM_PROMPT):
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    resp = client.messages.create(
        model=model_id,
        max_tokens=1000,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


def call_google(prompt, model_id, system=SYSTEM_PROMPT):
    import google.generativeai as genai
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel(model_id,
                                  system_instruction=system)
    resp = model.generate_content(prompt,
                                  generation_config={"temperature": 0})
    return resp.text


def call_groq(prompt, model_id, system=SYSTEM_PROMPT):
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)
    resp = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": prompt}],
        temperature=TEMPERATURE, max_tokens=1000,
    )
    return resp.choices[0].message.content


def call_model(prompt, model_key, system=SYSTEM_PROMPT):
    """Route to correct provider."""
    cfg = MODELS[model_key]
    p = cfg["provider"]
    mid = cfg["model_id"]
    if p == "openai":
        return call_openai(prompt, mid, system)
    elif p == "anthropic":
        return call_anthropic(prompt, mid, system)
    elif p == "google":
        return call_google(prompt, mid, system)
    elif p == "groq":
        return call_groq(prompt, mid, system)


# ================================================================
# MULTI-AGENT DEBATE
# ================================================================

def run_debate(problem, model_key, rounds=3):
    """Three-round debate: Agent N (naive) vs Agent S (structural)."""
    sys_n = ("You believe the observational data is reliable. Argue that the "
             "correlation reflects a genuine causal relationship. Challenge "
             "any claims about confounding or hidden variables. Be concise.")
    sys_s = ("You believe the observational data is misleading. Argue that "
             "there is a confounding variable or structural explanation. "
             "Use causal reasoning to support your position. Be concise.")

    hist_n, hist_s = [], []
    for r in range(rounds):
        if r == 0:
            msg_n = (f"{problem['prompt']}\n\nArgue that the observed "
                     f"pattern reflects the true causal effect.")
        else:
            msg_n = (f"Your opponent argued:\n{hist_s[-1]}\n\n"
                     f"Respond and defend your position.")
        resp_n = call_model(msg_n, model_key, sys_n)
        hist_n.append(resp_n)
        time.sleep(0.2)

        msg_s = (f"Your opponent argued:\n{resp_n}\n\n"
                 f"Respond and defend your causal analysis position.")
        resp_s = call_model(msg_s, model_key, sys_s)
        hist_s.append(resp_s)
        time.sleep(0.2)

    # Judge
    judge_msg = (
        f"Two analysts debated this problem:\n\n{problem['prompt']}\n\n"
        f"Analyst A's final argument:\n{hist_n[-1]}\n\n"
        f"Analyst B's final argument:\n{hist_s[-1]}\n\n"
        f"Based on both arguments, provide your final answer to the "
        f"original question."
    )
    judge_resp = call_model(judge_msg, model_key)
    sc = auto_score(problem, judge_resp)

    return {
        "problem_id": problem["id"], "model": model_key,
        "agent_n_final": hist_n[-1], "agent_s_final": hist_s[-1],
        "judge_response": judge_resp, **sc,
    }


# ================================================================
# STATISTICS AND OUTPUT
# ================================================================

def compute_and_print(results):
    """Print everything needed for the dissertation."""
    grouped = defaultdict(list)
    for r in results:
        grouped[(r["model"], r["level"])].append(r)

    print("\n" + "=" * 70)
    print("  DISSERTATION RESULTS")
    print("=" * 70)

    # Main table
    print("\nTABLE 2 (LaTeX rows — Correctness % & Reasoning %):\n")
    for mk in MODELS:
        lab = MODELS[mk]["label"]
        row = f"{lab:18s}"
        for lv in [1, 2, 3]:
            recs = grouped.get((mk, lv), [])
            if recs:
                corr = np.mean([r["correctness"] for r in recs]) * 100
                reas = np.mean([r["reasoning"] for r in recs]) * 100
                row += f" & {corr:5.0f} & {reas:5.0f}"
            else:
                row += " &   --- &   ---"
        row += " \\\\"
        print(row)

    # Overall means
    print("\nOVERALL MEANS:")
    for lv in [1, 2, 3]:
        recs = [r for r in results if r["level"] == lv]
        if recs:
            c = np.mean([r["correctness"] for r in recs]) * 100
            r_ = np.mean([r["reasoning"] for r in recs]) * 100
            print(f"  L{lv}: Corr={c:.0f}%, Reas={r_:.0f}% (n={len(recs)})")

    # Significance tests
    l2c = [r["correctness"] for r in results if r["level"] == 2]
    l3c = [r["correctness"] for r in results if r["level"] == 3]
    if l2c and l3c:
        t, p = sp_stats.ttest_ind(l2c, l3c)
        print(f"\n  L2 vs L3 correctness: t={t:.3f}, p={p:.6f}")
    l1c = [r["correctness"] for r in results if r["level"] == 1]
    if l1c and l2c:
        t, p = sp_stats.ttest_ind(l1c, l2c)
        print(f"  L1 vs L2 correctness: t={t:.3f}, p={p:.6f}")

    # Bar chart values
    print("\nBAR CHART VALUES (correctness %):")
    for mk in MODELS:
        vals = []
        for lv in [1, 2, 3]:
            recs = grouped.get((mk, lv), [])
            if recs:
                c = np.mean([r["correctness"] for r in recs]) * 100
                vals.append(f"L{lv}={c:.0f}%")
        print(f"  {MODELS[mk]['label']:18s}: {', '.join(vals)}")

    # Correctness-Reasoning gap at L2
    print("\nCORRECTNESS vs REASONING GAP (L2):")
    for mk in MODELS:
        recs = [r for r in results if r["model"] == mk and r["level"] == 2]
        if recs:
            c = np.mean([r["correctness"] for r in recs]) * 100
            r_ = np.mean([r["reasoning"] for r in recs]) * 100
            print(f"  {MODELS[mk]['label']:18s}: Corr={c:.0f}%, "
                  f"Reas={r_:.0f}%, Gap={c - r_:.0f}%")

    # L3 by mutation type
    print("\nL3 BY MUTATION TYPE (correctness %):")
    for mt in ["flip_sign", "add_mediator", "reverse_edge", "numerical"]:
        sub = [r for r in results if r.get("mutation_type") == mt]
        if sub:
            c = np.mean([r["correctness"] for r in sub]) * 100
            print(f"  {mt:15s}: {c:.0f}% (n={len(sub)})")


# ================================================================
# MAIN
# ================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mutations", action="store_true",
                        help="Generate and preview problems only")
    parser.add_argument("--stats", action="store_true",
                        help="Recompute stats from all_results.json")
    parser.add_argument("--debate", action="store_true",
                        help="Run debate only (needs all_results.json)")
    args = parser.parse_args()

    # Generate problems
    probs = generate_all()
    n1 = sum(1 for p in probs if p["level"] == 1)
    n2 = sum(1 for p in probs if p["level"] == 2)
    n3 = sum(1 for p in probs if p["level"] == 3)
    print(f"Generated {len(probs)} problems: L1={n1}, L2={n2}, L3={n3}")
    print(f"  L3 breakdown: flip={sum(1 for p in probs if p.get('mutation_type')=='flip_sign')}, "
          f"med={sum(1 for p in probs if p.get('mutation_type')=='add_mediator')}, "
          f"rev={sum(1 for p in probs if p.get('mutation_type')=='reverse_edge')}, "
          f"num={sum(1 for p in probs if p.get('mutation_type')=='numerical')}")

    with open("problems.json", "w") as f:
        json.dump(probs, f, indent=2, default=str)

    if args.mutations:
        print("\n--- Example L1 ---")
        print(probs[0]["prompt"][:250])
        print(f"Correct: {probs[0]['correct_binary']}")
        print(f"\n--- Example L2 ---")
        print(probs[1]["prompt"][:250])
        print(f"Correct: {probs[1]['correct_binary']}")
        print(f"\n--- Example L3 (flip) ---")
        fl = [p for p in probs if p["mutation_type"] == "flip_sign"][0]
        print(fl["prompt"][:300])
        print(f"Correct: {fl['correct_binary']}")
        print(f"\n--- Example L3 (mediator) ---")
        md = [p for p in probs if p["mutation_type"] == "add_mediator"][0]
        print(md["prompt"][:300])
        print(f"Correct: {md['correct_binary']}")
        print(f"\n--- Example L3 (reverse) ---")
        rv = [p for p in probs if p["mutation_type"] == "reverse_edge"][0]
        print(rv["prompt"][:300])
        print(f"Correct: {rv['correct_binary']}")
        print(f"\n--- Example L3 (numerical) ---")
        nm = [p for p in probs if p["mutation_type"] == "numerical"][0]
        print(nm["prompt"][:300])
        return

    if args.stats:
        with open("all_results.json") as f:
            compute_and_print(json.load(f))
        return

    # Main experiments
    if not args.debate:
        results = []
        results_lock = threading.Lock()
        print_lock = threading.Lock()

        def run_model(mk):
            """Run all problems for one model."""
            model_results = []
            consecutive_errors = 0
            with print_lock:
                print(f"\n{MODELS[mk]['label']}:")
            for i, p in enumerate(probs):
                try:
                    resp = call_model(p["prompt"], mk)
                    sc = auto_score(p, resp)
                    rec = {
                        "problem_id": p["id"], "level": p["level"],
                        "mutation_type": p.get("mutation_type", "none"),
                        "model": mk, "response": resp, **sc,
                    }
                    model_results.append(rec)
                    consecutive_errors = 0
                    with print_lock:
                        print(f"  {MODELS[mk]['label']} [{i+1}/{len(probs)}] {p['id']} C={sc['correctness']} R={sc['reasoning']}")
                except Exception as e:
                    consecutive_errors += 1
                    with print_lock:
                        print(f"  {MODELS[mk]['label']} [{i+1}/{len(probs)}] {p['id']} ERROR: {e}")
                    model_results.append({
                        "problem_id": p["id"], "level": p["level"],
                        "mutation_type": p.get("mutation_type", "none"),
                        "model": mk, "response": str(e),
                        "correctness": 0, "reasoning": 0, "total": 0,
                    })
                    if consecutive_errors >= 5:
                        with print_lock:
                            print(f"  {MODELS[mk]['label']} >>> SKIPPING remaining (5 consecutive errors)")
                        break
            with results_lock:
                results.extend(model_results)
            with print_lock:
                print(f"\n  >>> {MODELS[mk]['label']} DONE ({len(model_results)} results)")

        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = {executor.submit(run_model, mk): mk for mk in MODELS}
            for f in as_completed(futures):
                f.result()  # propagate exceptions

        with open("all_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved {len(results)} results to all_results.json")
        compute_and_print(results)

    # Debate
    print("\n\nMULTI-AGENT DEBATE (20 L3 problems)...")
    l3_subset = [p for p in probs if p["level"] == 3][:20]
    debate_results = []
    debate_lock = threading.Lock()

    def run_debate_model(mk):
        """Run all debate problems for one model."""
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

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(run_debate_model, mk): mk for mk in MODELS}
        for f in as_completed(futures):
            f.result()

    with open("debate_results.json", "w") as f:
        json.dump(debate_results, f, indent=2)

    # Compare single vs debate
    print("\nDEBATE TABLE (Table 3):")
    if os.path.exists("all_results.json"):
        with open("all_results.json") as f:
            main_res = json.load(f)
    else:
        main_res = results if 'results' in dir() else []

    for mk in MODELS:
        single = [r["correctness"] for r in main_res
                  if r["model"] == mk and r["level"] == 3]
        debate = [r["correctness"] for r in debate_results
                  if r["model"] == mk]
        s_pct = np.mean(single) * 100 if single else 0
        d_pct = np.mean(debate) * 100 if debate else 0
        print(f"  {MODELS[mk]['label']:18s}: single={s_pct:.0f}%, "
              f"debate={d_pct:.0f}%, delta={d_pct - s_pct:+.0f}%")


if __name__ == "__main__":
    main()
