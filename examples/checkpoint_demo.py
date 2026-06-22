"""
checkpoint_demo.py
-------------------
Proves that checkpointing is real: a leaf (or a whole forest) saved to disk
and loaded back is BYTE-IDENTICAL to the original, not just "close" or
"same accuracy". This is what makes grafting operationally useful — you can
train a leaf once, freeze it to disk, and attach it to a different forest
later (graft_leaf) without retraining.

Three proofs, in order:
  1. save_leaf / load_leaf  — single leaf round-trip, hash + accuracy match.
  2. save_forest / load_forest — whole-forest round-trip, every leaf hash
     matches and predict() gives identical routing on a test batch.
  3. graft_leaf — load a previously-saved leaf into a brand-new forest and
     classify correctly with ZERO retraining.

Runs on CPU with a fixed seed (same reasoning as demo_torch.py: byte-exact
hash proofs require deterministic float ops, which MPS does not guarantee).
"""
import gzip
import os
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, '.')
from das_torch import (
    DASForest, FibonacciLeaf, leaf_hash, train_leaf_isolated, train_router,
    save_leaf, load_leaf, save_forest, load_forest,
)

DEVICE = "cpu"
torch.manual_seed(0)
np.random.seed(0)

D_MODEL = 784
LEAF_DIMS = [784, 256, 64, 2]
CKPT_DIR = './checkpoints'

def _read_mnist_idx(path):
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

def make_binary_split(X, y, neg_digit, pos_digit, n_per_class, rng):
    pos_idx = np.where(y == pos_digit)[0]
    neg_idx = np.where(y == neg_digit)[0]
    n = min(n_per_class, len(pos_idx), len(neg_idx))
    pos_pick = rng.choice(pos_idx, n, replace=False)
    neg_pick = rng.choice(neg_idx, n, replace=False)
    idx = np.concatenate([pos_pick, neg_pick])
    Xd = X[idx]
    yd = np.concatenate([np.ones(n, dtype=np.int64), np.zeros(n, dtype=np.int64)])
    perm = rng.permutation(len(Xd))
    return Xd[perm], yd[perm]

print("=" * 64)
print(" DAS FRAMEWORK (PyTorch) — checkpoint / restore proofs")
print("=" * 64)
print(f"\n  Device: {DEVICE}  (forced CPU for byte-exact hash proofs)")

rng = np.random.default_rng(7)
X_tr, y_tr, X_te, y_te = load_mnist()

# digits used for this demo's single-leaf task: 0 vs 1
X_tr01, y_tr01 = make_binary_split(X_tr, y_tr, 0, 1, 800, rng)
X_te01, y_te01 = make_binary_split(X_te, y_te, 0, 1, 400, rng)
t_X_tr01, t_y_tr01 = torch.from_numpy(X_tr01), torch.from_numpy(y_tr01)
t_X_te01, t_y_te01 = torch.from_numpy(X_te01), torch.from_numpy(y_te01)

# ── PROOF 1: single-leaf save/load round-trip ──────────────────────────
print("\n[Proof 1] Single leaf: train -> save_leaf -> load_leaf -> compare")

forest_a = DASForest(D_MODEL, LEAF_DIMS, num_leaves=1).to(DEVICE)
acc_before = train_leaf_isolated(forest_a, 0, t_X_tr01, t_y_tr01, steps=600, lr=1e-3, batch=128, device=DEVICE)
original_leaf = forest_a.leaves[0]
hash_original = leaf_hash(original_leaf)
with torch.no_grad():
    acc_original_test = (original_leaf(t_X_te01).argmax(1) == t_y_te01).float().mean().item()
print(f"          trained leaf: train acc={acc_before:.3f}  test acc={acc_original_test:.3f}  hash={hash_original}")

leaf_path = os.path.join(CKPT_DIR, 'proof1_leaf.pt')
save_leaf(original_leaf, leaf_path)
print(f"          saved to {leaf_path}")

loaded_leaf = load_leaf(leaf_path, device=DEVICE)
hash_loaded = leaf_hash(loaded_leaf)
with torch.no_grad():
    acc_loaded_test = (loaded_leaf(t_X_te01).argmax(1) == t_y_te01).float().mean().item()
print(f"          loaded leaf:  test acc={acc_loaded_test:.3f}  hash={hash_loaded}")

proof1_hash_match = hash_original == hash_loaded
proof1_acc_match = acc_original_test == acc_loaded_test
print(f"          hash match: {proof1_hash_match}   accuracy match: {proof1_acc_match}")
print(f"          PROOF 1: {'PASS' if proof1_hash_match and proof1_acc_match else 'FAIL'}")

# ── PROOF 2: whole-forest save/load round-trip ─────────────────────────
print("\n[Proof 2] Whole forest: train 2 leaves + router -> save_forest -> load_forest -> compare")

X_tr23, y_tr23 = make_binary_split(X_tr, y_tr, 2, 3, 800, rng)
t_X_tr23, t_y_tr23 = torch.from_numpy(X_tr23), torch.from_numpy(y_tr23)

forest_b = DASForest(D_MODEL, LEAF_DIMS, num_leaves=2).to(DEVICE)
dom01 = np.concatenate([np.zeros(len(X_tr01), dtype=np.int64), np.ones(len(X_tr23), dtype=np.int64)])
Xr = np.vstack([X_tr01, X_tr23])
train_router(forest_b, torch.from_numpy(Xr), torch.from_numpy(dom01), steps=400, lr=1e-3, batch=128, device=DEVICE)
train_leaf_isolated(forest_b, 0, t_X_tr01, t_y_tr01, steps=600, lr=1e-3, batch=128, device=DEVICE)
train_leaf_isolated(forest_b, 1, t_X_tr23, t_y_tr23, steps=600, lr=1e-3, batch=128, device=DEVICE)

hashes_before = [leaf_hash(l) for l in forest_b.leaves]
test_batch = torch.from_numpy(np.vstack([X_te01[:50], X_te[y_te == 2][:50]]))
with torch.no_grad():
    out_before, idx_before = forest_b.predict(test_batch)

forest_dir = os.path.join(CKPT_DIR, 'proof2_forest')
save_forest(forest_b, forest_dir)
print(f"          saved forest to {forest_dir}/  (router.pt, leaf_0.pt, leaf_1.pt, manifest.json)")

forest_b_loaded = load_forest(forest_dir, device=DEVICE)
hashes_after = [leaf_hash(l) for l in forest_b_loaded.leaves]
with torch.no_grad():
    out_after, idx_after = forest_b_loaded.predict(test_batch)

proof2_hashes_match = hashes_before == hashes_after
proof2_routing_match = torch.equal(idx_before, idx_after)
proof2_output_match = torch.equal(out_before, out_after)
for i, (hb, ha) in enumerate(zip(hashes_before, hashes_after)):
    print(f"          leaf {i}: before={hb}  after={ha}  match={hb == ha}")
print(f"          routing identical: {proof2_routing_match}   raw output identical: {proof2_output_match}")
print(f"          PROOF 2: {'PASS' if proof2_hashes_match and proof2_routing_match and proof2_output_match else 'FAIL'}")

# ── PROOF 3: graft_leaf — attach a saved leaf to a fresh forest, no retrain ──
print("\n[Proof 3] graft_leaf: load proof1's saved leaf into a brand-new forest, zero retraining")

fresh_forest = DASForest(D_MODEL, LEAF_DIMS, num_leaves=1).to(DEVICE)  # 1 placeholder leaf, untrained
grafted_leaf = load_leaf(leaf_path, device=DEVICE)  # the 0-vs-1 leaf saved in Proof 1
new_id = fresh_forest.graft_leaf(grafted_leaf)
print(f"          grafted leaf id: {new_id}  (forest now has {len(fresh_forest.leaves)} leaves)")

with torch.no_grad():
    grafted_out = fresh_forest.leaves[new_id](t_X_te01)
    grafted_acc = (grafted_out.argmax(1) == t_y_te01).float().mean().item()
hash_grafted = leaf_hash(fresh_forest.leaves[new_id])
print(f"          grafted leaf test acc on 0v1 (no retraining): {grafted_acc:.3f}")
print(f"          grafted leaf hash matches proof1 original: {hash_grafted == hash_original}")
proof3_pass = grafted_acc > 0.9 and hash_grafted == hash_original
print(f"          PROOF 3: {'PASS' if proof3_pass else 'FAIL'}")

# ── Summary ──────────────────────────────────────────────────────────
print("\n" + "=" * 64)
print(" SUMMARY")
print("=" * 64)
all_pass = (proof1_hash_match and proof1_acc_match and
            proof2_hashes_match and proof2_routing_match and proof2_output_match and
            proof3_pass)
print(f"  Proof 1 (save_leaf/load_leaf byte-exact):       {'PASS' if proof1_hash_match and proof1_acc_match else 'FAIL'}")
print(f"  Proof 2 (save_forest/load_forest byte-exact):   {'PASS' if proof2_hashes_match and proof2_routing_match and proof2_output_match else 'FAIL'}")
print(f"  Proof 3 (graft_leaf works with zero retraining): {'PASS' if proof3_pass else 'FAIL'}")
print(f"\n  Overall: {'PASS' if all_pass else 'FAIL'}")
print("=" * 64)

if not all_pass:
    sys.exit(1)
