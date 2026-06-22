"""
backbone_demo.py
----------------
Phase 9: shared frozen backbone + isolated heads, on real MNIST.

Pipeline:
  1. Train a SHARED backbone (784 -> 256 -> 64 features) with a throwaway
     10-class head on all of MNIST, then FREEZE it.
  2. Train the router on backbone FEATURES (not raw pixels) to pick the task.
  3. Train one tiny isolated HEAD (64 -> 2) per task on those features.
  4. Graft a new head for a new task; prove the old heads stay byte-identical.

Why this matters: the heads now SHARE features (no re-learning vision 5x) and
the router routes on LEARNED features (not raw pixels). The cost: the backbone
is a shared trainable component — fine as long as it stays frozen.

Forced CPU for byte-exact hash proofs.
"""
import gzip
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from das_torch import BackboneForest, leaf_hash, train_head_isolated

DEVICE = "cpu"
torch.manual_seed(0)
np.random.seed(0)

def read_idx(path):
    with gzip.open(path, 'rb') as f:
        magic = struct.unpack('>I', f.read(4))[0]
        ndims = magic & 0xFF
        shape = tuple(struct.unpack('>I', f.read(4))[0] for _ in range(ndims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

base = './data/MNIST/raw'
X_tr = read_idx(f'{base}/train-images-idx3-ubyte.gz').reshape(-1, 784).astype(np.float32) / 255.0
y_tr = read_idx(f'{base}/train-labels-idx1-ubyte.gz').astype(np.int64)
X_te = read_idx(f'{base}/t10k-images-idx3-ubyte.gz').reshape(-1, 784).astype(np.float32) / 255.0
y_te = read_idx(f'{base}/t10k-labels-idx1-ubyte.gz').astype(np.int64)

TASKS = [(0, 1), (2, 3), (4, 5)]
NEW_TASK = (6, 7)

def binary_split(X, y, d0, d1):
    m = (y == d0) | (y == d1)
    return torch.from_numpy(X[m]), torch.from_numpy((y[m] == d1).astype(np.int64))

print("=" * 64)
print(" DAS — Phase 9: shared backbone + isolated heads (MNIST)")
print("=" * 64)

forest = BackboneForest(in_dim=784, feat_dim=64, head_dims=[64, 2], num_leaves=len(TASKS)).to(DEVICE)

# ── 1. Train + freeze the shared backbone ───────────────────────
print("\n[1] Training shared backbone (10-class pretext) then freezing ...")
sub = np.random.choice(len(X_tr), 12000, replace=False)
Xb = torch.from_numpy(X_tr[sub]).to(DEVICE)
yb = torch.from_numpy(y_tr[sub]).to(DEVICE)
temp_head = nn.Linear(64, 10).to(DEVICE)
opt = torch.optim.Adam(list(forest.backbone.parameters()) + list(temp_head.parameters()), lr=1e-3)
for _ in range(1500):
    idx = torch.randint(0, len(Xb), (128,), device=DEVICE)
    opt.zero_grad()
    loss = F.cross_entropy(temp_head(forest.backbone(Xb[idx])), yb[idx])
    loss.backward(); opt.step()
forest.freeze_backbone()
n_bb = sum(p.numel() for p in forest.backbone.parameters())
print(f"    backbone frozen ({n_bb:,} params)")

# ── 2. Train the router ON FEATURES ─────────────────────────────
print("\n[2] Training router on backbone FEATURES (3-way task id) ...")
Xr_parts, dr_parts = [], []
for t, (d0, d1) in enumerate(TASKS):
    Xt, _ = binary_split(X_tr, y_tr, d0, d1)
    Xr_parts.append(Xt); dr_parts.append(torch.full((len(Xt),), t, dtype=torch.int64))
Xr = torch.cat(Xr_parts); dr = torch.cat(dr_parts)
with torch.no_grad():
    Fr = forest.backbone(Xr.to(DEVICE)).cpu()       # features for the router
# train_router expects a thing with .router; reuse it by routing on features
router_opt = torch.optim.Adam(forest.router.parameters(), lr=1e-3)
Frd, drd = Fr.to(DEVICE), dr.to(DEVICE)
for _ in range(800):
    idx = torch.randint(0, len(Frd), (128,), device=DEVICE)
    router_opt.zero_grad()
    _, _, logits = forest.router(Frd[idx])
    loss = F.cross_entropy(logits, drd[idx]); loss.backward(); router_opt.step()
with torch.no_grad():
    _, _, logits = forest.router(Frd)
    router_acc = (logits.argmax(1) == drd).float().mean().item()
print(f"    router accuracy on features: {router_acc:.4f}")

# ── 3. Train one isolated head per task ─────────────────────────
print("\n[3] Training isolated heads on shared features ...")
test_feats, test_y = {}, {}
for t, (d0, d1) in enumerate(TASKS):
    Xt, yt = binary_split(X_tr, y_tr, d0, d1)
    Xe, ye = binary_split(X_te, y_te, d0, d1)
    with torch.no_grad():
        ft = forest.backbone(Xt.to(DEVICE)).cpu()
        fe = forest.backbone(Xe.to(DEVICE)).cpu()
    test_feats[t], test_y[t] = fe, ye
    a = train_head_isolated(forest, t, ft, yt, steps=400, device=DEVICE)
    te = (forest.heads[t](fe.to(DEVICE)).argmax(1) == ye.to(DEVICE)).float().mean().item()
    print(f"    head {t} ({d0}v{d1}): train {a:.4f}  test {te:.4f}")
n_head = sum(p.numel() for p in forest.heads[0].parameters())
print(f"    head size: {n_head:,} params  (vs backbone {n_bb:,}) — heads are ~{n_bb//n_head}x smaller")

# ── 4. Graft a new head; prove old heads byte-identical ─────────
print(f"\n[4] Grafting a head for new task {NEW_TASK} ...")
before = {t: leaf_hash(forest.heads[t]) for t in range(len(TASKS))}
nid = forest.graft_head()
Xn, yn = binary_split(X_tr, y_tr, *NEW_TASK)
with torch.no_grad():
    fn = forest.backbone(Xn.to(DEVICE)).cpu()
an = train_head_isolated(forest, nid, fn, yn, steps=400, device=DEVICE)
after = {t: leaf_hash(forest.heads[t]) for t in range(len(TASKS))}
unchanged = all(before[t] == after[t] for t in range(len(TASKS)))
print(f"    new head {nid} trained — train acc {an:.4f}")
for t in range(len(TASKS)):
    print(f"    head {t}: {'UNCHANGED' if before[t] == after[t] else '*** CHANGED ***'}")

print("\n" + "=" * 64)
print(f"  Router on features: {router_acc:.3f}   Forgetting proof: {'PASS' if unchanged else 'FAIL'}")
print("=" * 64)
import sys; sys.exit(0 if unchanged else 1)
