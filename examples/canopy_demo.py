"""
canopy_demo.py
--------------
Phase 10: the canopy. Hard top-1 routing commits 100% to one leaf, so a misroute
is fatal. The canopy blends the top-k leaves by routing weight, so the right leaf
usually still contributes even when it wasn't the #1 pick — graceful degradation.

This ONLY makes sense when leaves share an output space (here: one global 2-class
rule, leaves specialised by input region). We deliberately UNDER-train the router
so it misroutes near region boundaries, then compare top-1 vs top-2 (canopy).
Expectation: with a shaky router, top-2 >= top-1. With a strong router they
converge — the canopy is insurance for routing uncertainty, not free accuracy.
"""
import numpy as np
from das.model import DASForest
from das.functional import softmax

rng = np.random.default_rng(0)
D, LEAF, N = 8, [8, 13, 8, 2], 600
w_global = rng.normal(0, 1, D)   # ONE shared rule -> leaves share output meaning

def region(did, n):
    centers = {0: np.eye(D)[0]*3, 1: np.eye(D)[3]*3, 2: np.eye(D)[6]*3}
    X = centers[did] + rng.normal(0, 1.4, (n, D))      # wide spread -> overlap
    return X, (X @ w_global > 0).astype(int), np.full(n, did)

def ce_grad(logits, y):
    p = softmax(logits); oh = np.zeros_like(p); oh[np.arange(len(y)), y] = 1.0
    return (p - oh) / len(y)

def acc(logits, y):
    return (logits.argmax(1) == y).mean()

doms = {d: region(d, N) for d in range(3)}
Xall = np.vstack([doms[d][0] for d in range(3)])
yall = np.concatenate([doms[d][1] for d in range(3)])
dall = np.concatenate([doms[d][2] for d in range(3)])

print("=" * 60)
print(" DAS canopy (Phase 10): top-1 vs top-2 under a shaky router")
print("=" * 60)

def build(router_steps):
    f = DASForest(D, LEAF, num_leaves=3, seed=7)
    for _ in range(router_steps):
        idx = rng.integers(0, len(Xall), 128)
        f.router.train_step(Xall[idx], dall[idx], lr=0.1)
    for d in range(3):
        leaf = f.leaves[d]; leaf.frozen = False
        Xd, yd, _ = doms[d]
        for _ in range(400):
            idx = rng.integers(0, len(Xd), 128)
            leaf.backward(ce_grad(leaf.forward(Xd[idx]), yd[idx]), 0.05)
        leaf.frozen = True
    return f

for label, steps in [("shaky router (40 steps)", 40), ("strong router (600 steps)", 600)]:
    f = build(steps)
    racc = (f.router.route(Xall)[0] == dall).mean()
    a1 = acc(f.predict(Xall)[0], yall)
    a2 = acc(f.predict_canopy(Xall, k=2)[0], yall)
    print(f"\n  {label}: routing acc {racc:.3f}")
    print(f"    top-1 (hard):   {a1:.3f}")
    print(f"    top-2 (canopy): {a2:.3f}   ({'canopy helps' if a2 > a1 + 1e-6 else 'no gain'})")

print("\n" + "=" * 60)
print("  Canopy is insurance against routing uncertainty, not free accuracy.")
print("=" * 60)
