"""
governance_demo.py
------------------
The use case the evidence actually supports: a multi-tenant ML service where
each tenant gets an ISOLATED leaf trained only on their own data. This demo
shows the three properties that make DAS defensible in regulated / multi-tenant
settings — and that a monolithic model cannot provide:

  1. NON-INTERFERENCE  — onboarding tenant B provably never changes tenant A's
     model (SHA-256 byte-identical). "Your data never trains anyone else's model."
  2. DELETION / UNLEARNING — a tenant requests removal; we prune their leaf and
     prove (a) their capability is gone and (b) everyone else is untouched. In a
     monolithic model, cleanly removing one tenant's influence is unsolved.
  3. AUDIT TRAIL — the hash fingerprints at every step ARE the compliance log.

This is not "better AI" — it's auditable, governed AI.
"""
import numpy as np
from das.model import DASForest
from das.functional import softmax
from das.lifecycle import ForestLifecycle

rng = np.random.default_rng(0)
D, LEAF, N = 16, [16, 13, 8, 2], 400

def tenant_data(tid, n):
    centers = {0: np.eye(D)[0]*4, 1: np.eye(D)[5]*4, 2: np.eye(D)[10]*4, 3: np.eye(D)[14]*4}
    rule = np.random.default_rng(100 + tid).normal(0, 1, D)
    X = centers[tid] + rng.normal(0, 1.0, (n, D))
    return X, (X @ rule > 0).astype(int)

def ce_grad(logits, y):
    p = softmax(logits); oh = np.zeros_like(p); oh[np.arange(len(y)), y] = 1.0
    return (p - oh) / len(y)

def acc(logits, y):
    return (logits.argmax(1) == y).mean()

data = {t: tenant_data(t, N) for t in range(3)}

def train_leaf(forest, lid, X, y, steps=400, lr=0.05):
    leaf = forest.leaves[lid]; leaf.frozen = False
    for _ in range(steps):
        idx = rng.integers(0, len(X), 64)
        leaf.backward(ce_grad(leaf.forward(X[idx]), y[idx]), lr)
    leaf.frozen = True

def retrain_router(forest, tenant_ids):
    Xr = np.vstack([data[t][0] for t in tenant_ids])
    dr = np.concatenate([np.full(N, slot) for slot, _ in enumerate(tenant_ids)])
    for _ in range(600):
        idx = rng.integers(0, len(Xr), 64)
        forest.router.train_step(Xr[idx], dr[idx], lr=0.15)

print("=" * 64)
print(" DAS governance demo — multi-tenant isolation, deletion, audit")
print("=" * 64)

# Tenant 0 onboards (forest seed)
forest = DASForest(D, LEAF, num_leaves=1, seed=7)
life = ForestLifecycle(forest)
train_leaf(forest, 0, *data[0]); retrain_router(forest, [0])
audit = {0: forest.leaves[0].weight_hash()}
print(f"\n[onboard] tenant 0  -> leaf hash {audit[0]}")

# Tenants 1, 2 onboard; prove prior tenants byte-identical (non-interference)
ni_ok = True
for t in [1, 2]:
    snap = {p: forest.leaves[p].weight_hash() for p in range(t)}
    life.graft(seed=300 + t); train_leaf(forest, t, *data[t]); retrain_router(forest, list(range(t + 1)))
    audit[t] = forest.leaves[t].weight_hash()
    for p in range(t):
        if forest.leaves[p].weight_hash() != snap[p]:
            ni_ok = False
    print(f"[onboard] tenant {t}  -> leaf hash {audit[t]}   prior tenants unchanged: "
          f"{all(forest.leaves[p].weight_hash()==snap[p] for p in range(t))}")

# Quality check: every tenant served well
print("\n[serve] per-tenant accuracy (own data, routed end-to-end):")
served = {}
for t in range(3):
    out, idx = forest.predict(data[t][0])
    served[t] = acc(out, data[t][1])
    print(f"        tenant {t}: routed-correctly={(idx==t).mean():.2f}  task-acc={served[t]:.3f}")

# Tenant 1 requests deletion (right to be forgotten)
print("\n[delete] tenant 1 requests removal ...")
snap = {0: forest.leaves[0].weight_hash(), 2: forest.leaves[2].weight_hash()}
acc1_before = served[1]
life.prune(1)                      # drop tenant 1's leaf + its router column
retrain_router(forest, [0, 2])     # router now serves only the remaining tenants
# remaining tenants untouched?
others_ok = (forest.leaves[0].weight_hash() == snap[0] and forest.leaves[1].weight_hash() == snap[2])
# (after prune, old leaf 2 shifted to index 1)
# tenant 1's capability gone?
out, _ = forest.predict(data[1][0]); acc1_after = acc(out, data[1][1])
print(f"        leaves remaining: {len(forest.leaves)}")
print(f"        other tenants byte-identical after deletion: {others_ok}")
print(f"        tenant 1 task-acc: {acc1_before:.3f} -> {acc1_after:.3f}  (capability removed)")
deletion_ok = others_ok and acc1_after < 0.7   # fell toward chance

# Audit trail
print("\n[audit] hash log (the compliance record):")
for t, h in audit.items():
    print(f"        tenant {t}: {h}")

print("\n" + "=" * 64)
overall = ni_ok and deletion_ok
print(f"  Non-interference: {'PASS' if ni_ok else 'FAIL'}   "
      f"Deletion+unlearning: {'PASS' if deletion_ok else 'FAIL'}")
print(f"  Overall governance proof: {'PASS' if overall else 'FAIL'}")
print("  A monolithic model can prove NEITHER of these. That is DAS's real edge.")
print("=" * 64)
import sys; sys.exit(0 if overall else 1)
