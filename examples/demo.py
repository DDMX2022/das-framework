"""
demo.py
-------
Full DAS lifecycle on a synthetic multi-domain task, proving the ONE claim
that actually holds: training a new expert leaf cannot disturb the others.
Synthetic setup:
  - 3 "domains", each a distinct cluster of inputs in R^d.
  - Each domain has its OWN binary classification rule.
  - The router must learn which domain an input belongs to.
  - Each leaf must learn its own domain's rule, in isolation.
"""
import numpy as np
from das.model import DASForest

rng = np.random.default_rng(42)

D_MODEL = 21          # Fibonacci :)  input dimension
LEAF_DIMS = [21, 13, 8, 2]   # Fibonacci descent, output = 2 classes
N_PER_DOMAIN = 600

def make_domain(domain_id, n, rng):
    """Distinct input cluster + a domain-specific labelling rule."""
    center = rng.normal(0, 3, D_MODEL) * 0  # placeholder, set below
    centers = {
        0: np.eye(D_MODEL)[0] * 4,
        1: np.eye(D_MODEL)[7] * 4,
        2: np.eye(D_MODEL)[14] * 4,
    }
    rules = {  # each domain classifies along a different direction
        0: rng_rule(0),
        1: rng_rule(1),
        2: rng_rule(2),
    }
    X = centers[domain_id] + rng.normal(0, 1.0, (n, D_MODEL))
    y = (X @ rules[domain_id] > 0).astype(int)
    dom = np.full(n, domain_id)
    return X, y, dom

def rng_rule(seed):
    return np.random.default_rng(100 + seed).normal(0, 1, D_MODEL)

def cross_entropy_grad(logits, y):
    from das.functional import softmax
    p = softmax(logits)
    N = logits.shape[0]
    onehot = np.zeros_like(p); onehot[np.arange(N), y] = 1.0
    return (p - onehot) / N

def accuracy(logits, y):
    return (logits.argmax(axis=1) == y).mean()

def train_leaf_in_isolation(forest, leaf_id, X, y, steps=400, lr=0.05, batch=128):
    """Train ONE leaf. Router is frozen; all other leaves are frozen."""
    leaf = forest.leaves[leaf_id]
    leaf.frozen = False
    n = X.shape[0]
    for s in range(steps):
        idx = rng.integers(0, n, batch)
        xb, yb = X[idx], y[idx]
        logits = leaf.forward(xb)
        leaf.backward(cross_entropy_grad(logits, yb), lr)
    leaf.frozen = True
    return accuracy(leaf.forward(X), y)

print("=" * 64)
print(" DAS FRAMEWORK — minimal working prototype (NumPy, CPU)")
print("=" * 64)

# ---- Build data for the first two domains ----
X0, y0, d0 = make_domain(0, N_PER_DOMAIN, rng)
X1, y1, d1 = make_domain(1, N_PER_DOMAIN, rng)
Xr = np.vstack([X0, X1])
dr = np.concatenate([d0, d1])

# ---- Build the forest: 2 leaves to start ----
forest = DASForest(D_MODEL, LEAF_DIMS, num_leaves=2, seed=7)

# ---- PHASE 1: train the Stem Router (domains 0 & 1) ----
print("\n[Phase 1] Training Stem Router on domains 0 & 1 ...")
for s in range(600):
    idx = rng.integers(0, Xr.shape[0], 128)
    loss, acc = forest.router.train_step(Xr[idx], dr[idx], lr=0.1)
ridx, _ = forest.router.route(Xr)
print(f"          router routing accuracy: {(ridx == dr).mean():.3f}")

# ---- PHASE 2: train each leaf IN ISOLATION ----
print("\n[Phase 2] Training Leaf 0 (domain 0) in isolation ...")
a0 = train_leaf_in_isolation(forest, 0, X0, y0)
print(f"          leaf 0 accuracy on its domain: {a0:.3f}")
print("[Phase 2] Training Leaf 1 (domain 1) in isolation ...")
a1 = train_leaf_in_isolation(forest, 1, X1, y1)
print(f"          leaf 1 accuracy on its domain: {a1:.3f}")

# ---- Snapshot weight fingerprints BEFORE grafting ----
before = forest.leaf_hashes()
print("\n[Snapshot] Leaf fingerprints before grafting a new domain:")
for k, v in before.items():
    print(f"           leaf {k}: {v}")

# ---- PHASE 3: GRAFT a third leaf for a brand-new domain ----
print("\n[Phase 3] Grafting Leaf 2 for a NEW domain 2 ...")
X2, y2, d2 = make_domain(2, N_PER_DOMAIN, rng)
new_id = forest.graft(seed=321)

# router must learn the new route (honest: router is NOT frozen forever)
Xr_all = np.vstack([X0, X1, X2])
dr_all = np.concatenate([d0, d1, d2])
for s in range(400):
    idx = rng.integers(0, Xr_all.shape[0], 128)
    forest.router.train_step(Xr_all[idx], dr_all[idx], lr=0.1)

# freeze the two old leaves explicitly, then train ONLY the new leaf
forest.leaves[0].frozen = True
forest.leaves[1].frozen = True
a2 = train_leaf_in_isolation(forest, new_id, X2, y2)
print(f"          leaf 2 accuracy on its domain: {a2:.3f}")

# ---- Snapshot AFTER grafting ----
after = forest.leaf_hashes()
print("\n[Snapshot] Leaf fingerprints AFTER grafting + training leaf 2:")
for k in before:
    print(f"           leaf {k}: {after[k]}")

# ---- THE PROOF ----
print("\n" + "=" * 64)
print(" PROOF — zero catastrophic forgetting")
print("=" * 64)
unchanged = all(before[k] == after[k] for k in before)
for k in before:
    same = "UNCHANGED" if before[k] == after[k] else "*** CHANGED ***"
    print(f"   leaf {k}: {same}")
print(f"\n   Result: {'PASS — old leaves are byte-identical.' if unchanged else 'FAIL'}")

# ---- Final whole-system accuracy across all 3 domains ----
print("\n[Final] Whole-forest accuracy across all 3 domains:")
for X, y, name, did in [(X0, y0, 'domain 0', 0), (X1, y1, 'domain 1', 1), (X2, y2, 'domain 2', 2)]:
    out, leaf_idx = forest.predict(X)
    routed = (leaf_idx == did).mean()
    print(f"          {name}: routed-correctly={routed:.2f}  task-acc={accuracy(out, y):.3f}")
print("=" * 64)
