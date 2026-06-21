"""
conv_demo.py
------------
Proves a CNN leaf works through the same machinery as the MLP leaves: train
in isolation, freeze, checkpoint to disk, restore byte-exact. The point is
NOT that a CNN beats an MLP on MNIST 1-vs-rest (on a task this easy, both
get >99%, so don't read too much into the head-to-head) — it's that
DASForest's leaf API (flat-in, flat-out, freeze/unfreeze, save/load) does
not force every expert to be an MLP. A leaf can be architecturally whatever
it needs to be.

Task: 1-vs-rest on digit 7 (arbitrary choice), trained on real MNIST.

Runs on CPU with a fixed seed (same reasoning as demo_torch.py: byte-exact
hash proofs require deterministic float ops).
"""
import gzip
import os
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, '.')
from das_torch import ConvLeaf, leaf_hash, save_leaf, load_leaf

DEVICE = "cpu"
torch.manual_seed(0)
np.random.seed(0)

DIGIT = 7
CKPT_DIR = './checkpoints'
LEAF_PATH = os.path.join(CKPT_DIR, 'conv_leaf_7.pt')

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

def make_1vrest_split(X, y, digit, n_pos, rng):
    """Balanced 1-vs-rest split: n_pos positives + n_pos negatives sampled
    uniformly from the other 9 digits."""
    pos_idx = np.where(y == digit)[0]
    neg_idx = np.where(y != digit)[0]
    n = min(n_pos, len(pos_idx))
    pos_pick = rng.choice(pos_idx, n, replace=False)
    neg_pick = rng.choice(neg_idx, n, replace=False)
    idx = np.concatenate([pos_pick, neg_pick])
    Xd = X[idx]
    yd = np.concatenate([np.ones(n, dtype=np.int64), np.zeros(n, dtype=np.int64)])
    perm = rng.permutation(len(Xd))
    return Xd[perm], yd[perm]

print("=" * 64)
print(f" DAS FRAMEWORK (PyTorch) — ConvLeaf, heterogeneous expert (digit {DIGIT} 1-vs-rest)")
print("=" * 64)
print(f"\n  Device: {DEVICE}  (forced CPU for byte-exact hash proofs)")

rng = np.random.default_rng(3)
X_tr, y_tr, X_te, y_te = load_mnist()
X_tr7, y_tr7 = make_1vrest_split(X_tr, y_tr, DIGIT, 1000, rng)
X_te7, y_te7 = make_1vrest_split(X_te, y_te, DIGIT, 500, rng)
t_X_tr7, t_y_tr7 = torch.from_numpy(X_tr7), torch.from_numpy(y_tr7)
t_X_te7, t_y_te7 = torch.from_numpy(X_te7), torch.from_numpy(y_te7)
print(f"  train: {len(X_tr7)} examples (balanced)   test: {len(X_te7)} examples (balanced)")

# ── Train a lone ConvLeaf in isolation (same discipline as train_leaf_isolated,
#    just inlined here since this leaf isn't sitting inside a DASForest) ──
print("\n[Train] ConvLeaf on digit-7 1-vs-rest ...")
leaf = ConvLeaf([784, 2]).to(DEVICE)
opt = torch.optim.Adam(leaf.parameters(), lr=1e-3)
n = len(t_X_tr7)
leaf.train()
for step in range(500):
    idx = torch.randint(0, n, (128,))
    xb, yb = t_X_tr7[idx], t_y_tr7[idx]
    opt.zero_grad()
    loss = torch.nn.functional.cross_entropy(leaf(xb), yb)
    loss.backward()
    opt.step()
leaf.freeze()
leaf.eval()

with torch.no_grad():
    train_acc = (leaf(t_X_tr7).argmax(1) == t_y_tr7).float().mean().item()
    test_acc_before = (leaf(t_X_te7).argmax(1) == t_y_te7).float().mean().item()
hash_before = leaf_hash(leaf)
n_params = sum(p.numel() for p in leaf.parameters())
print(f"          params: {n_params:,}   train acc: {train_acc:.3f}   test acc: {test_acc_before:.3f}   hash: {hash_before}")

# ── Checkpoint + restore ────────────────────────────────────────────────
print(f"\n[Checkpoint] save_leaf -> {LEAF_PATH}")
save_leaf(leaf, LEAF_PATH)

restored = load_leaf(LEAF_PATH, device=DEVICE)
hash_after = leaf_hash(restored)
with torch.no_grad():
    test_acc_after = (restored(t_X_te7).argmax(1) == t_y_te7).float().mean().item()
print(f"[Restore]    loaded leaf type: {type(restored).__name__}   test acc: {test_acc_after:.3f}   hash: {hash_after}")

hash_match = hash_before == hash_after
acc_match = test_acc_before == test_acc_after
acc_good = test_acc_before > 0.9

print("\n" + "=" * 64)
print(" RESULT")
print("=" * 64)
print(f"  ConvLeaf test accuracy (1-vs-rest, digit {DIGIT}): {test_acc_before:.3f}  (>0.9 target: {'met' if acc_good else 'NOT met'})")
print(f"  Hash byte-match after restore: {hash_match}")
print(f"  Accuracy preserved after restore: {acc_match}")
overall = hash_match and acc_match and acc_good
print(f"\n  Overall: {'PASS' if overall else 'FAIL'}")
print("=" * 64)

if not overall:
    sys.exit(1)
