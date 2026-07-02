"""
cost_bench.py
-------------
The measured cost claim. Everything else in DAS is backed by a number; "reduces
cost" was, until now, only a demo estimate. This makes it a reproducible
benchmark on the one honest cost mechanism DAS actually has: **deflect traffic
from the frontier LLM, and send a thinner prompt on the traffic that remains.**

DAS does NOT claim to be a smarter or cheaper *model*. The frontier LLM stays in
the loop as the escalation fallback. The claim measured here is narrow and
defensible:

  1. DEFLECTION — queries a local specialist handles never hit the frontier API.
  2. THINNER PROMPTS — an escalated query carries only the routed specialist's
     policy, not every tenant's policy stuffed into one shared prompt.

Both save money; both are measured against a 100%-frontier baseline. And the
honest tension is measured too: deflection is governed by a confidence
threshold, and a linear router is over-confident on out-of-domain inputs (the
known routing bottleneck), so aggressive deflection MISROUTES genuinely-novel
queries. We sweep the threshold and report the trade-off — savings vs. answer
quality — rather than cherry-picking one flattering number.

What this does and doesn't show:
  ✓ the deflection + thin-prompt cost mechanic, measured end-to-end over the real
    das.platform deployment engine (router confidence drives every decision).
  ✗ it is NOT a real-traffic study: the query mix and token/price constants are
    illustrative (provider-neutral list prices, stated below and easily changed).
    Real savings depend on your domain mix, router quality, and provider price.

Pure NumPy + synthetic data, deterministic. Dogfoods das.platform.deploy.
"""
import numpy as np

from das.platform import deploy

# ── illustrative cost model (provider-neutral list prices; change freely) ──
# Dollars per token. Representative of a mid-tier frontier model's list price.
PRICE_IN = 3.0 / 1_000_000      # $/input token
PRICE_OUT = 15.0 / 1_000_000    # $/output token

# Token sizes for the two prompt shapes (illustrative).
TOK_SYS = 40          # system preamble
TOK_QUERY = 50        # the user's query
TOK_POLICY = 120      # one specialist's policy/context block
TOK_OUT = 150         # generated answer

# Fleet + query stream.
D_MODEL = 24
K_SPECIALISTS = 6         # domains that HAVE a local specialist
M_NOVEL = 3               # domains with NO specialist (must be escalated)
N_PER_DOMAIN = 60
THRESHOLDS = [0.0, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0]


def _query_cost(n_policies: int) -> float:
    """Cost of one frontier call whose prompt carries ``n_policies`` policy blocks."""
    tin = TOK_SYS + TOK_QUERY + n_policies * TOK_POLICY
    return tin * PRICE_IN + TOK_OUT * PRICE_OUT


def build_spec(k: int) -> dict:
    """A k-specialist single-client spec (one tenant per specialist keeps it simple)."""
    return {
        "client": "cost-bench",
        "d_model": D_MODEL,
        "escalation": {"frontier_llm": "frontier", "confidence_threshold": 0.0},
        "tenants": [{"name": f"t{i}", "experts": [{"name": f"spec-{i}"}]} for i in range(k)],
    }


def build_stream(trainer, k: int, m: int, n_per: int):
    """A labelled query stream: in-domain queries (a specialist exists, with a
    ground-truth label) and novel queries (no specialist — the correct action is
    to escalate). Returns list of (x, true_expert_or_None, label_or_None)."""
    rng = np.random.default_rng(7)
    stream = []
    for i in range(k):
        name = f"spec-{i}"
        X, y = trainer.sample(name, n_per, rng)
        for row, lab in zip(X, y):
            stream.append((row, name, int(lab)))
    for j in range(m):
        # novel domains the fleet was never trained on -> no correct local answer
        X, _ = trainer.sample(f"novel-{j}", n_per, rng)
        for row in X:
            stream.append((row, None, None))
    rng.shuffle(stream)
    return stream


def evaluate(dep, stream, threshold: float) -> dict:
    """Route every query at a given deflection threshold; tally cost + quality.

    Baseline = 100% frontier, fat prompt (all K policies). DAS = deflected
    queries cost $0 in API (served locally); escalated queries hit the frontier
    with a thin prompt (1 policy). Quality: a deflected query is correct only if
    it routed to the right specialist AND that specialist predicted the right
    label; an escalated query is assumed correct (the frontier has the context)."""
    root = dep.spec.root
    cp = dep.cp
    n = len(stream)
    deflected = correct = 0
    das_cost = 0.0
    baseline_cost = n * _query_cost(K_SPECIALISTS)   # fat prompt, every query
    for x, true_name, label in stream:
        rows = cp.route_explain(root, x[None, :])
        row = rows[0]
        if row["confidence"] >= threshold:
            # kept local (deflected from the frontier)
            deflected += 1
            hit = (true_name is not None
                   and row["name"] == true_name
                   and int(np.argmax(row["prediction"])) == label)
            if hit:
                correct += 1
            # else: a misroute — a wrong local answer (the cost of over-deflecting)
        else:
            # escalated to the frontier with a thin, single-policy prompt
            das_cost += _query_cost(1)
            correct += 1   # frontier answers correctly (it has the context)
    return {
        "threshold": threshold,
        "deflection_pct": 100.0 * deflected / n,
        "escalation_pct": 100.0 * (n - deflected) / n,
        "accuracy_pct": 100.0 * correct / n,
        "das_per_1k": 1000.0 * das_cost / n,
        "baseline_per_1k": 1000.0 * baseline_cost / n,
        "savings_pct": 100.0 * (baseline_cost - das_cost) / baseline_cost,
    }


def run(k: int = K_SPECIALISTS, m: int = M_NOVEL, n_per: int = N_PER_DOMAIN):
    """Stand up the fleet, build the stream, sweep the threshold. Returns a dict
    with the parameters, the baseline cost, and one row per threshold."""
    dep = deploy(build_spec(k), secret="bench-secret")
    stream = build_stream(dep.trainer, k, m, n_per)
    rows = [evaluate(dep, stream, t) for t in THRESHOLDS]
    return {
        "k_specialists": k,
        "m_novel": m,
        "n_queries": len(stream),
        "baseline_per_1k": rows[0]["baseline_per_1k"],
        "rows": rows,
    }


def main():
    res = run()
    print("=" * 78)
    print(" DAS cost benchmark — deflection + thin prompts vs a 100%-frontier baseline")
    print("=" * 78)
    print(f"\n  fleet: {res['k_specialists']} specialists + {res['m_novel']} novel "
          f"(unserved) domains | {res['n_queries']} queries")
    print(f"  baseline (all-frontier, fat prompt): ${res['baseline_per_1k']:.2f} / 1k queries")
    print(f"  price model: ${PRICE_IN*1e6:.0f}/M in, ${PRICE_OUT*1e6:.0f}/M out "
          f"(illustrative list price)\n")
    print(f"  {'thresh':>7}{'deflect%':>10}{'escal%':>9}{'quality%':>10}"
          f"{'DAS $/1k':>11}{'savings%':>10}")
    print("  " + "-" * 70)
    for r in res["rows"]:
        print(f"  {r['threshold']:>7.2f}{r['deflection_pct']:>10.1f}{r['escalation_pct']:>9.1f}"
              f"{r['accuracy_pct']:>10.1f}{r['das_per_1k']:>11.2f}{r['savings_pct']:>10.1f}")
    print("\n  How to read it honestly:")
    print("  - threshold 1.00 = escalate everything: $0 deflection, but STILL cheaper")
    print("    than baseline because escalated prompts are thin (1 policy, not K).")
    print("  - lowering the threshold deflects more (bigger savings) until the")
    print("    over-confident router starts keeping NOVEL queries local -> quality")
    print("    drops. The right operating point is the knee, not the cheapest row.")
    print("  - constants at the top are illustrative; real savings depend on your")
    print("    domain mix, router quality, and provider price.")


if __name__ == "__main__":
    main()
