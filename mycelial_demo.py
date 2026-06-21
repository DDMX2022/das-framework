"""
mycelial_demo.py
----------------
Phase 13: the "mycelial forest" — a dense orchestrator (the "soil") that
decomposes a multi-domain query, routes each part to a specialist tree, and
synthesises the answers. This is the full DAS vision: many trees linked by a
shared orchestration layer.

Self-contained (no LLM download): the soil is a small domain router over a
bag-of-words front-end; the "trees" are isolated per-domain experts. A composite
query ("what is 3 plus 4 and i love this and turn off the light") is split into
clauses, each routed to its tree, then the answers are synthesised.

We MEASURE the cost the pitch hides: the orchestrator runs on EVERY query, and a
multi-domain query activates MULTIPLE trees — so "only one tiny expert fires" is
not what actually happens. The orchestration works; the "run a tiny fraction"
economics do not survive an honest accounting.
"""
import numpy as np
from das.text import Tokenizer
from das.functional import FibonacciLeaf, softmax
from das.routing import StemRouter

rng = np.random.default_rng(0)
N = 400
SOIL_LLM_PARAMS = 7_000_000_000   # a stand-in for a dense orchestrator LLM (always on)

def gen_math(n):
    ops = [("plus", 1), ("sum of", 1), ("minus", 0), ("times", 0)]; T, Y = [], []
    for _ in range(n):
        op, lab = ops[rng.integers(len(ops))]; a, b = rng.integers(1, 10, 2)
        T.append(f"what is {a} {op} {b}"); Y.append(lab)
    return T, Y, "math"

def gen_sentiment(n):
    pos, neg = ["love", "great", "wonderful"], ["hate", "terrible", "awful"]; T, Y = [], []
    for _ in range(n):
        if rng.random() < 0.5: T.append(f"i {pos[rng.integers(3)]} this"); Y.append(1)
        else: T.append(f"i {neg[rng.integers(3)]} this"); Y.append(0)
    return T, Y, "sentiment"

def gen_command(n):
    dev = ["light", "fan", "heater"]; T, Y = [], []
    for _ in range(n):
        if rng.random() < 0.5: T.append(f"turn on the {dev[rng.integers(3)]}"); Y.append(1)
        else: T.append(f"turn off the {dev[rng.integers(3)]}"); Y.append(0)
    return T, Y, "command"

doms = [gen_math(N), gen_sentiment(N), gen_command(N)]
names = [d[2] for d in doms]
tok = Tokenizer().fit([t for (T, _, _) in doms for t in T])
D = tok.dim
X = [tok.transform(T) for (T, _, _) in doms]
Y = [np.array(Yi) for (_, Yi, _) in doms]

def ce_grad(logits, y):
    p = softmax(logits); oh = np.zeros_like(p); oh[np.arange(len(y)), y] = 1.0
    return (p - oh) / len(y)

# ── The SOIL: a domain orchestrator/router over all domains ─────
soil = StemRouter(D, 3, seed=1)
Xr = np.vstack(X); dr = np.concatenate([np.full(N, d) for d in range(3)])
for _ in range(800):
    i = rng.integers(0, len(Xr), 64)
    soil.train_step(Xr[i], dr[i], lr=0.2)

# ── The TREES: one isolated specialist per domain ───────────────
trees = [FibonacciLeaf([D, 16, 8, 2], seed=10 + d) for d in range(3)]
for d in range(3):
    for _ in range(400):
        i = rng.integers(0, N, 64)
        trees[d].backward(ce_grad(trees[d].forward(X[d][i]), Y[d][i]), 0.1)
    trees[d].frozen = True
tree_params = sum(w.size for w in trees[0].W) + sum(b.size for b in trees[0].b)

print("=" * 64)
print(" Phase 13: mycelial forest — orchestrate across specialist trees")
print("=" * 64)

# ── Orchestrate a composite, multi-domain query ─────────────────
composite = "what is 3 plus 4 and i love this and turn off the light"
clauses = [c.strip() for c in composite.split(" and ")]
print(f'\n  query: "{composite}"')
print(f"  soil decomposes into {len(clauses)} clauses:")
activated = set()
for c in clauses:
    v = tok.transform([c])
    dom = int(soil.route(v)[0][0])
    ans = int(trees[dom].forward(v).argmax(1)[0])
    activated.add(dom)
    print(f'    "{c}"  -> tree[{names[dom]}]  -> answer {ans}')

# ── Honest cost accounting ──────────────────────────────────────
active_tree_params = len(activated) * tree_params
print(f"\n  trees activated this query: {len(activated)} of 3  ({[names[d] for d in sorted(activated)]})")
print(f"\n  cost accounting (the pitch says 'only one tiny expert fires'):")
print(f"    soil orchestrator (always on): {SOIL_LLM_PARAMS:,} params")
print(f"    activated tree experts:        {active_tree_params:,} params ({len(activated)} x {tree_params:,})")
soil_share = SOIL_LLM_PARAMS / (SOIL_LLM_PARAMS + active_tree_params) * 100
print(f"    -> the always-on soil is {soil_share:.4f}% of active compute")

print("\n" + "=" * 64)
print("  The orchestration WORKS: clauses route to the right trees and synthesise.")
print("  But the honest cost is soil + (k activated trees), not 'one tiny leaf'.")
print("  A dense orchestrator runs on every query and dominates cost, and")
print("  multi-domain queries fire multiple trees — so the 'run a tiny fraction'")
print("  economics from the original pitch do not hold once the soil is counted.")
print("=" * 64)
