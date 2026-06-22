"""
backbone_cifar_bench.py
-----------------------
Closes the Phase 8 / Phase 9 loop on REAL images. Phase 8 showed the linear
router on raw CIFAR pixels collapses to ~0.42. Phase 9 (shared frozen backbone,
route on learned features) recovered the router to ~0.98 — but only on MNIST.
This runs Phase 9 on CIFAR to see if the fix holds on hard images.

Pipeline:
  1. Train a shared CONV backbone (10-class pretext on CIFAR), then freeze.
  2. Train the router on backbone FEATURES (5-way task id) -> the key number.
  3. Train one isolated head per task on those features.
  4. Graft a head; prove the others stay byte-identical.

Compare the feature-router accuracy here against the 0.42 raw-pixel baseline.
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets

from das_torch import HeadLeaf, StemRouter, leaf_hash

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)
TASKS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
FEAT = 128

class ConvBackbone(nn.Module):
    def __init__(self, feat=FEAT):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 32->16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 16->8
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 8->4
        )
        self.fc = nn.Linear(128 * 4 * 4, feat)

    def forward(self, x):
        x = self.conv(x.view(-1, 3, 32, 32)).flatten(1)
        return F.relu(self.fc(x))

    def freeze(self):
        for p in self.parameters():
            p.requires_grad_(False)

def load_cifar():
    tr = datasets.CIFAR10(root='./data', train=True, download=True)
    te = datasets.CIFAR10(root='./data', train=False, download=True)
    Xtr = tr.data.astype(np.float32).transpose(0, 3, 1, 2).reshape(-1, 3072) / 255.0
    Xte = te.data.astype(np.float32).transpose(0, 3, 1, 2).reshape(-1, 3072) / 255.0
    return Xtr, np.array(tr.targets, dtype=np.int64), Xte, np.array(te.targets, dtype=np.int64)

def binary_split(X, y, d0, d1, n_per, rng):
    m = (y == d0) | (y == d1)
    Xs, ys = X[m], y[m]
    pos, neg = np.where(ys == d1)[0], np.where(ys == d0)[0]
    n = min(n_per, len(pos), len(neg))
    idx = np.concatenate([rng.choice(pos, n, replace=False), rng.choice(neg, n, replace=False)])
    yb = np.concatenate([np.ones(n, dtype=np.int64), np.zeros(n, dtype=np.int64)])
    return torch.from_numpy(Xs[idx]), torch.from_numpy(yb)

print("=" * 64)
print(" Phase 9 on CIFAR — does a shared backbone fix the router?")
print("=" * 64)
rng = np.random.default_rng(42)
Xtr, ytr, Xte, yte = load_cifar()
t0 = time.time()

# 1. shared conv backbone, 10-class pretext, then freeze
bb = ConvBackbone().to(DEVICE)
sub = np.random.choice(len(Xtr), 15000, replace=False)
Xb = torch.from_numpy(Xtr[sub]).to(DEVICE); yb = torch.from_numpy(ytr[sub]).to(DEVICE)
temp = nn.Linear(FEAT, 10).to(DEVICE)
opt = torch.optim.Adam(list(bb.parameters()) + list(temp.parameters()), lr=1e-3)
bb.train()
for _ in range(2000):
    i = torch.randint(0, len(Xb), (128,), device=DEVICE)
    opt.zero_grad(); F.cross_entropy(temp(bb(Xb[i])), yb[i]).backward(); opt.step()
bb.freeze(); bb.eval()
print(f"\n[1] conv backbone frozen ({sum(p.numel() for p in bb.parameters()):,} params)  ({time.time()-t0:.0f}s)")

# task splits + cached features
tasks = []
for d0, d1 in TASKS:
    Xt, yt = binary_split(Xtr, ytr, d0, d1, 2500, rng)
    Xe, ye = binary_split(Xte, yte, d0, d1, 800, rng)
    with torch.no_grad():
        ft = bb(Xt.to(DEVICE)); fe = bb(Xe.to(DEVICE))
    tasks.append((ft, yt.to(DEVICE), fe, ye.to(DEVICE)))

# 2. router on FEATURES
router = StemRouter(FEAT, len(TASKS)).to(DEVICE)
Fr = torch.cat([tasks[t][0] for t in range(len(TASKS))])
dr = torch.cat([torch.full((len(tasks[t][0]),), t, device=DEVICE) for t in range(len(TASKS))])
Fe = torch.cat([tasks[t][2] for t in range(len(TASKS))])
de = torch.cat([torch.full((len(tasks[t][2]),), t, device=DEVICE) for t in range(len(TASKS))])
ro = torch.optim.Adam(router.parameters(), lr=1e-3)
for _ in range(1500):
    i = torch.randint(0, len(Fr), (128,), device=DEVICE)
    ro.zero_grad(); _, _, lg = router(Fr[i]); F.cross_entropy(lg, dr[i]).backward(); ro.step()
with torch.no_grad():
    _, _, lg = router(Fe); router_acc = (lg.argmax(1) == de).float().mean().item()
print(f"[2] router accuracy ON FEATURES: {router_acc:.4f}   (raw-pixel baseline was 0.42)")

# 3. isolated heads on features
print("[3] isolated heads on features:")
head_accs = []
heads = []
for t in range(len(TASKS)):
    ft, yt, fe, ye = tasks[t]
    head = HeadLeaf([FEAT, 2]).to(DEVICE); heads.append(head)
    o = torch.optim.Adam(head.parameters(), lr=1e-3); head.train()
    for _ in range(500):
        i = torch.randint(0, len(ft), (128,), device=DEVICE)
        o.zero_grad(); F.cross_entropy(head(ft[i]), yt[i]).backward(); o.step()
    head.eval()
    with torch.no_grad():
        a = (head(fe).argmax(1) == ye).float().mean().item()
    head_accs.append(a)
    print(f"    head {t} {TASKS[t]}: {a:.4f}")

# 4. forgetting proof on graft
before = [leaf_hash(h) for h in heads]
extra = HeadLeaf([FEAT, 2]).to(DEVICE)
ft, yt, _, _ = tasks[0]
o = torch.optim.Adam(extra.parameters(), lr=1e-3)
for _ in range(300):
    i = torch.randint(0, len(ft), (128,), device=DEVICE)
    o.zero_grad(); F.cross_entropy(extra(ft[i]), yt[i]).backward(); o.step()
after = [leaf_hash(h) for h in heads]
unchanged = before == after

print("\n" + "=" * 64)
print(f"  Router on features:  {router_acc:.3f}   (vs 0.42 on raw pixels — recovered: {router_acc>0.7})")
print(f"  Mean head accuracy:  {np.mean(head_accs):.3f}")
print(f"  Forgetting proof:    {'PASS' if unchanged else 'FAIL'}")
print(f"  Runtime: {time.time()-t0:.0f}s")
print("=" * 64)
import sys; sys.exit(0 if unchanged else 1)
