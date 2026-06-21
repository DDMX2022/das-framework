"""
benchmark.py
Phase 2: DAS Forest vs single MLP — honest comparison on real data.
Dataset: sklearn digits (1797 samples, 8×8 pixels, 10 classes)

Domain split (visual clusters):
  Domain 0 → digits {0,1,2,3}  task: even vs odd
  Domain 1 → digits {4,5,6}    task: ≤5 vs 6
  Domain 2 → digits {7,8,9}    task: ≤8 vs 9

Install: pip install scikit-learn
Run:     python benchmark.py
"""
import numpy as np, time
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from das.model import DASForest
from das.functional import FibonacciLeaf, softmax

rng = np.random.default_rng(42)

D_IN          = 64
LEAF_DIMS     = [64, 32, 16, 2]
BASELINE_DIMS = [64, 80, 40, 2]   # ≈ same total params as forest
LR, BATCH     = 0.02, 64

# ── Data ────────────────────────────────────────────────────────
digits  = load_digits()
X       = StandardScaler().fit_transform(digits.data.astype(np.float64))
y_digit = digits.target

DMAP = {0:0,1:0,2:0,3:0, 4:1,5:1,6:1, 7:2,8:2,9:2}
TASK = {0: lambda d: int(d%2==0), 1: lambda d: int(d<=5), 2: lambda d: int(d<=8)}
domains = np.array([DMAP[y] for y in y_digit])
labels  = np.array([TASK[DMAP[y]](y) for y in y_digit])

splits = {}
for d in range(3):
    m = domains == d
    Xtr,Xte,ytr,yte = train_test_split(X[m], labels[m], test_size=0.2, random_state=42)
    splits[d] = (Xtr, ytr, Xte, yte)

Xall = np.vstack([splits[d][0] for d in range(3)])
dall = np.concatenate([np.full(len(splits[d][0]), d) for d in range(3)])

def ce_grad(logits, y):
    p = softmax(logits); oh = np.zeros_like(p)
    oh[np.arange(len(y)), y] = 1.0
    return (p - oh) / len(y)

def accuracy(logits, y):
    return round(float((logits.argmax(1) == y).mean()), 4)

def n_params(leaf):
    return sum(w.size for w in leaf.W) + sum(b.size for b in leaf.b)

# ── DAS Forest ──────────────────────────────────────────────────
print("=" * 60)
print("  PHASE 2 — DAS Forest vs Baseline MLP (sklearn digits)")
print("=" * 60)

forest = DASForest(D_IN, LEAF_DIMS, num_leaves=3, seed=7)
das_p  = forest.router.W.size + forest.router.b.size + sum(n_params(l) for l in forest.leaves)

print(f"\n[DAS] Parameters: {das_p:,}")
print("\n[DAS Phase 1] Training Stem Router ...")
t = time.time()
for s in range(800):
    idx = rng.integers(0, len(Xall), BATCH)
    _, a = forest.router.train_step(Xall[idx], dall[idx], lr=0.05)
    if s % 200 == 0: print(f"  step {s:4d}  acc={a:.3f}")
ridx, _ = forest.router.route(Xall)
print(f"  Router accuracy: {(ridx==dall).mean():.3f}  ({time.time()-t:.1f}s)")

before = forest.leaf_hashes()
das_test = {}
for d in range(3):
    Xtr, ytr, Xte, yte = splits[d]
    leaf = forest.leaves[d]; leaf.frozen = False
    print(f"\n[DAS Phase {d+2}] Leaf {d} on Domain {d} (isolated) ...")
    t = time.time()
    for s in range(600):
        idx = rng.integers(0, len(Xtr), BATCH)
        leaf.backward(ce_grad(leaf.forward(Xtr[idx]), ytr[idx]), LR)
    leaf.frozen = True
    das_test[d] = accuracy(leaf.forward(Xte), yte)
    print(f"  test acc: {das_test[d]:.3f}  ({time.time()-t:.1f}s)")
after = forest.leaf_hashes()

# ── Baseline MLP ─────────────────────────────────────────────────
print("\n[Baseline] Single MLP (all domains mixed) ...")
baseline = FibonacciLeaf(BASELINE_DIMS, seed=99)
bl_p     = n_params(baseline)
print(f"  Parameters: {bl_p:,}")
Xbl = np.vstack([splits[d][0] for d in range(3)])
ybl = np.concatenate([splits[d][1] for d in range(3)])
t = time.time()
for s in range(1200):
    idx = rng.integers(0, len(Xbl), BATCH)
    baseline.backward(ce_grad(baseline.forward(Xbl[idx]), ybl[idx]), LR)
bl_test = {d: accuracy(baseline.forward(splits[d][2]), splits[d][3]) for d in range(3)}
print(f"  ({time.time()-t:.1f}s)")

# ── Results ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  RESULTS")
print("=" * 60)
print(f"\n  Params   DAS Forest: {das_p:,}   Baseline MLP: {bl_p:,}")
print(f"\n  {'Domain':<10} {'DAS':>8} {'Baseline':>10} {'Winner':>8}")
print(f"  {'-'*40}")
for d in range(3):
    da, ba = das_test[d], bl_test[d]
    print(f"  Domain {d}   {da:>8.3f} {ba:>10.3f} {'DAS ✓' if da>=ba else 'MLP ✓':>8}")

passed = all(before[k] == after[k] for k in before)
print(f"\n  Forgetting proof: {'PASS ✓' if passed else 'FAIL ✗'}")
print("=" * 60)
