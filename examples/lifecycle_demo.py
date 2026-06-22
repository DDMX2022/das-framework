"""
lifecycle_demo.py
-----------------
Proves the full forest lifecycle on synthetic multi-domain data:
  GROW   — train 4 leaves in isolation.
  ROUTE  — send a production workload where domain 3 never appears.
  MONITOR— observe that leaf 3 is dormant (0 traffic).
  PRUNE  — evict the dormant leaf + its router column.
  REGROW — graft a fresh leaf for a brand-new domain and train it.

The point: pruning a leaf and grafting a new one both leave every OTHER leaf
byte-identical (SHA-256), and routing for the surviving domains keeps working
with no retraining. That's the "works like a forest" guarantee.
"""
import numpy as np
from das.model import DASForest
from das.functional import softmax
from das.lifecycle import ForestLifecycle

rng = np.random.default_rng(0)
D, LEAF, N = 21, [21, 13, 8, 2], 500

def make_domain(did, n, rng):
    centers = {0: np.eye(D)[0]*4, 1: np.eye(D)[5]*4, 2: np.eye(D)[10]*4,
               3: np.eye(D)[15]*4, 4: np.eye(D)[18]*4}
    rule = np.random.default_rng(100 + did).normal(0, 1, D)
    X = centers[did] + rng.normal(0, 1.0, (n, D))
    return X, (X @ rule > 0).astype(int), np.full(n, did)

def ce_grad(logits, y):
    p = softmax(logits); oh = np.zeros_like(p); oh[np.arange(len(y)), y] = 1.0
    return (p - oh) / len(y)

def acc(logits, y):
    return (logits.argmax(1) == y).mean()

def grow_leaf(forest, lid, X, y, steps=400, lr=0.05):
    leaf = forest.leaves[lid]; leaf.frozen = False
    for _ in range(steps):
        idx = rng.integers(0, len(X), 128)
        leaf.backward(ce_grad(leaf.forward(X[idx]), y[idx]), lr)
    leaf.frozen = True
    return acc(leaf.forward(X), y)

print("=" * 64)
print(" DAS FOREST — lifecycle: grow -> graft -> prune -> regrow")
print("=" * 64)

# ── GROW: 4 domains, 4 leaves ───────────────────────────────────
doms = {d: make_domain(d, N, rng) for d in range(4)}
Xr = np.vstack([doms[d][0] for d in range(4)])
dr = np.concatenate([doms[d][2] for d in range(4)])
forest = DASForest(D, LEAF, num_leaves=4, seed=7)
for _ in range(800):
    idx = rng.integers(0, len(Xr), 128)
    forest.router.train_step(Xr[idx], dr[idx], lr=0.1)
for d in range(4):
    a = grow_leaf(forest, d, doms[d][0], doms[d][1])
    print(f"  [grow]  leaf {d} trained — acc {a:.3f}")

life = ForestLifecycle(forest)

# ── ROUTE: production workload from domains 0,1,2 only (3 never appears) ──
print("\n  [route] serving a workload from domains 0,1,2 (domain 3 absent) ...")
for d in [0, 1, 2]:
    life.route(doms[d][0])
print("  [monitor] usage:", life.monitor())

# ── PRUNE the dormant leaf (<1% of traffic counts as dormant) ───
DORMANT_SHARE = 0.01
dormant = life.dormant_leaves(max_share=DORMANT_SHARE)
print(f"\n  [monitor] dormant leaves (<{DORMANT_SHARE:.0%} traffic): {dormant}")
before = {i: forest.leaves[i].weight_hash() for i in [0, 1, 2]}
pruned = life.prune_dormant(max_share=DORMANT_SHARE)
print(f"  [prune]  evicted {pruned} -> forest now has {len(forest.leaves)} leaves")
after = {i: forest.leaves[i].weight_hash() for i in [0, 1, 2]}
prune_ok = all(before[i] == after[i] for i in [0, 1, 2])
print(f"  [proof]  survivors byte-identical after prune: {'PASS' if prune_ok else 'FAIL'}")

# routing still works for the survivors, no retraining
route_ok = True
for d in [0, 1, 2]:
    out, _ = forest.predict(doms[d][0])
    if acc(out, doms[d][1]) < 0.85:
        route_ok = False
print(f"  [check]  survivors still classify their domains: {'PASS' if route_ok else 'FAIL'}")

# ── REGROW: graft a fresh leaf for a NEW domain 4 ───────────────
X4, y4, _ = make_domain(4, N, rng)
before2 = {i: forest.leaves[i].weight_hash() for i in [0, 1, 2]}
new_id = life.graft(seed=321)
# router must learn the new route (honest: experts isolated, router is not)
Xr2 = np.vstack([doms[0][0], doms[1][0], doms[2][0], X4])
dr2 = np.concatenate([np.full(N, 0), np.full(N, 1), np.full(N, 2), np.full(N, new_id)])
for _ in range(400):
    idx = rng.integers(0, len(Xr2), 128)
    forest.router.train_step(Xr2[idx], dr2[idx], lr=0.1)
a4 = grow_leaf(forest, new_id, X4, y4)
print(f"\n  [regrow] grafted leaf {new_id} for new domain 4 — acc {a4:.3f}")
after2 = {i: forest.leaves[i].weight_hash() for i in [0, 1, 2]}
regrow_ok = all(before2[i] == after2[i] for i in [0, 1, 2])
print(f"  [proof]  old leaves byte-identical after regrow: {'PASS' if regrow_ok else 'FAIL'}")

# ── REDUNDANCY: graft a duplicate of domain 0, detect + prune it ─
print("\n  [redundancy] grafting a second leaf trained on domain 0 (a duplicate) ...")
dup_id = life.graft(seed=777)
grow_leaf(forest, dup_id, doms[0][0], doms[0][1])
# probe on domain-0 data — where both the original and the duplicate operate
probe, THRESH = doms[0][0], 0.90
red = life.find_redundant(probe, threshold=THRESH)
print(f"  [redundancy] detected redundant pairs (i,j,agreement): {red}")
n_before = len(forest.leaves)
dropped = life.prune_redundant(probe, threshold=THRESH)
print(f"  [redundancy] pruned duplicates {dropped} -> {n_before} leaves became {len(forest.leaves)}")
redundancy_ok = len(forest.leaves) < n_before

# ── PERSISTENCE: usage counters survive a save/load round trip ───
life.reset_usage(); life.route(doms[0][0]); life.route(doms[1][0])
import os, tempfile
p = os.path.join(tempfile.gettempdir(), 'das_usage.json')
life.save_usage(p)
usage_before = life.usage.copy(); total_before = life.total_routed
life.usage[:] = 0; life.total_routed = 0
life.load_usage(p)
persist_ok = bool((life.usage == usage_before).all() and life.total_routed == total_before)
print(f"\n  [persist] usage restored after save/load: {'PASS' if persist_ok else 'FAIL'}")

print("\n" + "=" * 64)
overall = prune_ok and route_ok and regrow_ok and redundancy_ok and persist_ok
print(f"  Overall lifecycle proof: {'PASS' if overall else 'FAIL'}")
print("=" * 64)
import sys; sys.exit(0 if overall else 1)
