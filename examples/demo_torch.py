"""
demo_torch.py
-------------
PyTorch analog of demo.py: the full DAS lifecycle (route, train leaves in
isolation, graft a new leaf, train it, prove the old leaves never moved) —
but on real MNIST instead of synthetic clusters, and on the autograd path
instead of manual backprop.

Setup, Split-MNIST style: 3 binary "domains", each a pair of digits.
  domain 0: 0 vs 1
  domain 1: 2 vs 3
  domain 2: 4 vs 5
The router's job is to look at a raw 784-pixel image and say which domain
it came from. Each leaf's job is the binary classification within its own
domain. Leaves 0 and 1 are trained first; leaf 2 is grafted in afterwards
to prove that learning a new domain cannot disturb the old ones.

Runs on CPU with a fixed seed so the SHA-256 weight hashes are byte-exact
and reproducible (MPS float ops are not bit-reproducible across runs/devices,
which would make a "byte-identical" proof meaningless).
"""
import gzip
import json
import os
import struct
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '.')
from das_torch import (
    DASForest, leaf_hash, train_router, train_leaf_isolated, save_forest,
)

DEVICE = "cpu"
torch.manual_seed(0)
np.random.seed(0)

D_MODEL = 784
LEAF_DIMS = [784, 256, 64, 2]   # Fibonacci-ish descent, binary output
DOMAIN_PAIRS = {0: (0, 1), 1: (2, 3), 2: (4, 5)}  # domain id -> (neg digit, pos digit)
N_PER_DOMAIN = 800  # train examples per domain (balanced over the 2 digits)

# ── Load real MNIST from the IDX files already on disk ─────────────────
def _read_mnist_idx(path):
    """Parse a gzipped MNIST IDX file using only stdlib + numpy. (Same
    recipe as app.py's _read_mnist_idx — duplicated here so demo_torch.py
    has no import-time dependency on app.py or torchvision.)"""
    with gzip.open(path, 'rb') as f:
        magic = struct.unpack('>I', f.read(4))[0]
        ndims = magic & 0xFF
        shape = tuple(struct.unpack('>I', f.read(4))[0] for _ in range(ndims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

def load_mnist():
    base = './data/MNIST/raw'
    X_tr = _read_mnist_idx(f'{base}/train-images-idx3-ubyte.gz').reshape(-1, 784).astype(np.float32) / 255.0
    y_tr = _read_mnist_idx(f'{base}/train-labels-idx1-ubyte.gz').astype(np.int64)
    X_te = _read_mnist_idx(f'{base}/t10k-images-idx3-ubyte.gz').reshape(-1, 784).astype(np.float32) / 255.0
    y_te = _read_mnist_idx(f'{base}/t10k-labels-idx1-ubyte.gz').astype(np.int64)
    return X_tr, y_tr, X_te, y_te

def make_domain_split(X, y, domain_id, n_per_domain, rng):
    """Build a balanced binary split for one domain: (neg_digit, pos_digit)."""
    neg_digit, pos_digit = DOMAIN_PAIRS[domain_id]
    pos_idx = np.where(y == pos_digit)[0]
    neg_idx = np.where(y == neg_digit)[0]
    n = min(n_per_domain, len(pos_idx), len(neg_idx))
    pos_pick = rng.choice(pos_idx, n, replace=False)
    neg_pick = rng.choice(neg_idx, n, replace=False)
    idx = np.concatenate([pos_pick, neg_pick])
    Xd = X[idx]
    yd = np.concatenate([np.ones(n, dtype=np.int64), np.zeros(n, dtype=np.int64)])
    dom = np.full(2 * n, domain_id, dtype=np.int64)
    perm = rng.permutation(len(Xd))
    return Xd[perm], yd[perm], dom[perm]

print("=" * 64)
print(" DAS FRAMEWORK (PyTorch) — full lifecycle on real MNIST")
print("=" * 64)
print(f"\n  Device: {DEVICE}  (forced CPU for byte-exact hash proofs)")

rng = np.random.default_rng(42)
X_tr, y_tr, X_te, y_te = load_mnist()
print(f"  MNIST: {len(X_tr):,} train  {len(X_te):,} test")

# ---- Build domains 0 & 1 (train + test) ----
X0_tr, y0_tr, d0_tr = make_domain_split(X_tr, y_tr, 0, N_PER_DOMAIN, rng)
X1_tr, y1_tr, d1_tr = make_domain_split(X_tr, y_tr, 1, N_PER_DOMAIN, rng)
X0_te, y0_te, _ = make_domain_split(X_te, y_te, 0, 400, rng)
X1_te, y1_te, _ = make_domain_split(X_te, y_te, 1, 400, rng)

Xr01 = np.vstack([X0_tr, X1_tr])
dr01 = np.concatenate([d0_tr, d1_tr])

t_X0_tr, t_y0_tr = torch.from_numpy(X0_tr), torch.from_numpy(y0_tr)
t_X1_tr, t_y1_tr = torch.from_numpy(X1_tr), torch.from_numpy(y1_tr)

# ---- Build the forest: 2 leaves to start ----
forest = DASForest(D_MODEL, LEAF_DIMS, num_leaves=2).to(DEVICE)
n_params = sum(p.numel() for p in forest.parameters())
print(f"  Forest params (2 leaves): {n_params:,}")

# ---- PHASE 1: train the Stem Router on domains 0 & 1 ----
print("\n[Phase 1] Training Stem Router on domains 0 & 1 (0v1, 2v3) ...")
t0 = time.time()
router_acc = train_router(
    forest,
    torch.from_numpy(Xr01), torch.from_numpy(dr01),
    steps=600, lr=1e-3, batch=128, device=DEVICE,
)
print(f"          router routing accuracy (train): {router_acc:.3f}  ({time.time()-t0:.1f}s)")

# ---- PHASE 2: train each leaf IN ISOLATION ----
print("\n[Phase 2] Training Leaf 0 (digits 0v1) in isolation ...")
t0 = time.time()
a0 = train_leaf_isolated(forest, 0, t_X0_tr, t_y0_tr, steps=600, lr=1e-3, batch=128, device=DEVICE)
print(f"          leaf 0 train accuracy: {a0:.3f}  ({time.time()-t0:.1f}s)")

print("[Phase 2] Training Leaf 1 (digits 2v3) in isolation ...")
t0 = time.time()
a1 = train_leaf_isolated(forest, 1, t_X1_tr, t_y1_tr, steps=600, lr=1e-3, batch=128, device=DEVICE)
print(f"          leaf 1 train accuracy: {a1:.3f}  ({time.time()-t0:.1f}s)")

# ---- Snapshot weight fingerprints BEFORE grafting ----
before = {i: leaf_hash(forest.leaves[i]) for i in range(2)}
print("\n[Snapshot] Leaf fingerprints before grafting a new domain:")
for k, v in before.items():
    print(f"           leaf {k}: {v}")

# ---- PHASE 3: GRAFT a third leaf for a brand-new domain (4v5) ----
print("\n[Phase 3] Grafting Leaf 2 for a NEW domain 2 (digits 4v5) ...")
X2_tr, y2_tr, d2_tr = make_domain_split(X_tr, y_tr, 2, N_PER_DOMAIN, rng)
X2_te, y2_te, _ = make_domain_split(X_te, y_te, 2, 400, rng)
t_X2_tr, t_y2_tr = torch.from_numpy(X2_tr), torch.from_numpy(y2_tr)

new_id = forest.graft()
print(f"          grafted leaf id: {new_id}  (total leaves: {len(forest.leaves)})")

# Router must learn the new route — honest: the router is NOT frozen forever,
# only the leaves are. Train it on all 3 domains now that domain 2 exists.
Xr_all = np.vstack([X0_tr, X1_tr, X2_tr])
dr_all = np.concatenate([d0_tr, d1_tr, d2_tr])
t0 = time.time()
router_acc_3 = train_router(
    forest,
    torch.from_numpy(Xr_all), torch.from_numpy(dr_all),
    steps=600, lr=1e-3, batch=128, device=DEVICE,
)
print(f"          router routing accuracy (train, 3-way): {router_acc_3:.3f}  ({time.time()-t0:.1f}s)")

# Train ONLY leaf 2 in isolation (train_leaf_isolated re-freezes 0 & 1 itself).
print("[Phase 3] Training Leaf 2 (digits 4v5) in isolation ...")
t0 = time.time()
a2 = train_leaf_isolated(forest, new_id, t_X2_tr, t_y2_tr, steps=600, lr=1e-3, batch=128, device=DEVICE)
print(f"          leaf 2 train accuracy: {a2:.3f}  ({time.time()-t0:.1f}s)")

# ---- Snapshot AFTER grafting + training leaf 2 ----
after = {i: leaf_hash(forest.leaves[i]) for i in before}
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
print(f"\n   Result: {'PASS — old leaves are byte-identical.' if unchanged else 'FAIL — isolation leaked.'}")

# ---- Checkpoint the finished forest so serve.py has something to load ----
# (serve.py is the REST front-end added in Stage 4; it loads whatever forest
# was last saved here. Saved unconditionally — even a FAILing proof run still
# trains something real, and "save what you trained" matches how the rest of
# this codebase treats checkpoints.)
FOREST_DIR = './checkpoints/forest'
print(f"\n[Checkpoint] Saving trained forest to {FOREST_DIR}/ ...")
save_forest(forest, FOREST_DIR)
# Sidecar file: which (neg_digit, pos_digit) each leaf's binary output maps
# to. save_forest()'s manifest is generic (dims/types only); this is specific
# to this demo's Split-MNIST domain setup, so it lives next to it, not inside it.
with open(os.path.join(FOREST_DIR, 'domain_labels.json'), 'w') as f:
    json.dump({str(k): list(v) for k, v in DOMAIN_PAIRS.items()}, f, indent=2)
print(f"          saved router.pt, leaf_0.pt, leaf_1.pt, leaf_2.pt, manifest.json, domain_labels.json")

# ---- Held-out test accuracy + routing, all 3 domains ----
print("\n[Final] Held-out test accuracy across all 3 domains:")

def eval_domain(X_np, y_np, did, leaf_dims_out=2):
    Xd = torch.from_numpy(X_np)
    yd = torch.from_numpy(y_np)
    with torch.no_grad():
        out, leaf_idx = forest.predict(Xd)
    routed_correct = (leaf_idx == did).float().mean().item()
    task_acc = (out.argmax(1) == yd).float().mean().item()
    return routed_correct, task_acc

for X_np, y_np, name, did in [
    (X0_te, y0_te, 'domain 0 (0v1)', 0),
    (X1_te, y1_te, 'domain 1 (2v3)', 1),
    (X2_te, y2_te, 'domain 2 (4v5)', 2),
]:
    routed, acc = eval_domain(X_np, y_np, did)
    print(f"          {name}: routed-correctly={routed:.3f}  task-acc={acc:.3f}")

print("\n" + "=" * 64)
print(" SUMMARY")
print("=" * 64)
print(f"  Router accuracy (final, 3-way, train set): {router_acc_3:.3f}")
print(f"  Leaf 0 (0v1) train acc: {a0:.3f}   Leaf 1 (2v3) train acc: {a1:.3f}   Leaf 2 (4v5) train acc: {a2:.3f}")
print(f"  Forgetting proof: {'PASS' if unchanged else 'FAIL'}")
print("=" * 64)

if not unchanged:
    sys.exit(1)
