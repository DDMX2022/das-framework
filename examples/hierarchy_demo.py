"""
hierarchy_demo.py
-----------------
The specialty TREE made tangible (das/hierarchy.py) — the answer to "can I have
a React tree with leaves for hooks, rehydration, etc.":

    Specialty router -> react  -> hooks | rehydration
                     -> math   -> algebra | calculus

Shows: two-level provenance (specialty -> leaf, both confidences), the ROUTING
ISOLATION guarantee (grafting inside react leaves math's router byte-identical
— the flat forest cannot make that promise), and branch-level right-to-be-
forgotten (prune the whole react specialty, math proven untouched).

Pure NumPy, synthetic clusters, deterministic. Run from the repo root.
"""
import numpy as np

from das.functional import softmax
from das.hierarchy import HierarchicalDASForest

D = 18
rng = np.random.default_rng(0)

TREE_SPEC = {
    "react": ["hooks", "rehydration"],
    "math": ["algebra", "calculus"],
}


def cluster(seed, scale):
    r = np.random.default_rng(seed)
    c = r.normal(0, 1, D)
    return c / np.linalg.norm(c) * scale


def ce_grad(logits, y):
    p = softmax(logits)
    oh = np.zeros_like(p)
    oh[np.arange(len(y)), y] = 1.0
    return (p - oh) / len(y)


def train_leaf(leaf, X, y, steps=250):
    leaf.frozen = False
    r = np.random.default_rng(1)
    for _ in range(steps):
        i = r.integers(0, len(X), 32)
        leaf.backward(ce_grad(leaf.forward(X[i]), y[i]), 0.05)
    leaf.frozen = True


print("=" * 68)
print(" HierarchicalDASForest — a real specialty tree, two-level provenance")
print("=" * 68)

tree = HierarchicalDASForest(D, seed=7)
DATA = {}
for s, (branch, leaves) in enumerate(TREE_SPEC.items()):
    tree.add_branch(branch, num_leaves=len(leaves), seed=10 + s)
    base = cluster(100 + s, 5.0)
    for l, leaf_name in enumerate(leaves):
        c = base + cluster(200 + s * 10 + l, 1.5)
        X = c + rng.normal(0, 0.4, (150, D))
        y = (X @ cluster(300 + s * 10 + l, 1.0) > 0).astype(int)
        DATA[(branch, l)] = (X, leaf_name)
        train_leaf(tree.branches[branch].leaves[l], X, y)
    Xs = np.vstack([DATA[(branch, l)][0] for l in range(len(leaves))])
    ys = np.concatenate([np.full(150, l) for l in range(len(leaves))])
    r = np.random.default_rng(2)
    for _ in range(350):
        i = r.integers(0, len(Xs), 32)
        tree.branches[branch].router.train_step(Xs[i], ys[i], lr=0.2)

Xt = np.vstack([DATA[(b, l)][0] for b in TREE_SPEC for l in range(len(TREE_SPEC[b]))])
yt = np.concatenate([np.full(300, s) for s in range(len(TREE_SPEC))])
tree.train_router(Xt, yt, steps=500, seed=4)

print(f"\n[tree] {tree.branch_names} — {tree.leaf_count()} leaves total\n")

# ── two-level provenance ─────────────────────────────────────────────
print("[provenance] one query per leaf, routed through both levels:")
for (branch, l), (X, leaf_name) in DATA.items():
    row = tree.route_explain(X[:1])[0]
    got = TREE_SPEC[row["specialty"]][row["leaf"]]
    print(f"   {branch}/{leaf_name:<12} -> {row['specialty']}/{got:<12} "
          f"(specialty {row['specialty_confidence']:.2f}, leaf {row['leaf_confidence']:.2f})")

# ── routing isolation on graft ───────────────────────────────────────
X0, _ = DATA[("react", 0)][0], None
def train_fn(forest, idx):
    train_leaf(forest.leaves[idx], X0, (X0 @ cluster(999, 1.0) > 0).astype(int), steps=120)
    r = np.random.default_rng(5)
    for _ in range(120):
        i = r.integers(0, len(X0), 32)
        forest.router.train_step(X0[i], np.full(32, idx), lr=0.1)

idx, leaves_ok, routers_ok = tree.graft_leaf("react", train_fn, seed=42)
print(f"\n[graft] 'react-suspense' grafted into react (leaf {idx})")
print(f"   math's leaves byte-identical:              {leaves_ok}")
print(f"   math's router AND top router untouched:    {routers_ok}   <- the tree-only guarantee")

# ── branch-level right-to-be-forgotten ───────────────────────────────
result = tree.prune_branch("react")
print(f"\n[prune] deleted the entire react specialty "
      f"({result['removed_leaves']} leaves structurally gone)")
print(f"   survivors byte-identical: {result['survivors_byte_identical']}")
print(f"   remaining tree: {tree.branch_names}")
