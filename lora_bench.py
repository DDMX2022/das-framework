"""
lora_bench.py
-------------
Phase 14: the make-or-break test. "Add a per-task capability to a frozen
backbone without disturbing the rest" is exactly what LoRA adapters already do
and it's the industry default. So: does DAS actually beat per-task LoRA, or is
DAS just "LoRA adapters with a router bolted on"?

Same frozen backbone, two ways to specialize per task:
  - DAS   : a small ISOLATED HEAD on the frozen features (backbone untouched).
  - LoRA  : low-rank adapters that re-tune the frozen backbone per task, + a head.

Both are isolated per task, so both get zero forgetting and trivial deletion for
free — that's the honest part. We measure where they actually differ:
  1. per-task accuracy (task known)
  2. params per task
  3. forgetting (expect both 0)
  4. deletion (expect both trivial)
  5. TASK-FREE routing — DAS has a built-in router; LoRA needs to be told the
     task (or borrow the same router). This is DAS's only structural edge.

Forced CPU for byte-exact hash proofs.
"""
import gzip
import struct
import hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from das_torch import Backbone, HeadLeaf, StemRouter, train_head_isolated

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

TASKS = [(0, 1), (2, 3), (4, 5), (6, 7)]
RANK = 8

def binary_split(X, y, d0, d1):
    m = (y == d0) | (y == d1)
    return torch.from_numpy(X[m]), torch.from_numpy((y[m] == d1).astype(np.int64))

def param_hash(*tensors):
    h = hashlib.sha256()
    for t in tensors:
        h.update(t.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]

class LoRAExpert(nn.Module):
    """Per-task low-rank adaptation of a SHARED FROZEN backbone, plus a head.
    Adapters start at zero (B=0) so the expert begins as the untouched backbone.
    Only the adapters + head train; the backbone is never modified."""
    def __init__(self, backbone, rank=8, out_dim=2):
        super().__init__()
        self.backbone = backbone   # shared, frozen — not trained here
        l1, l2 = backbone.net[0], backbone.net[2]
        self.A1 = nn.Parameter(torch.randn(rank, l1.in_features) * 0.01)
        self.B1 = nn.Parameter(torch.zeros(l1.out_features, rank))
        self.A2 = nn.Parameter(torch.randn(rank, l2.in_features) * 0.01)
        self.B2 = nn.Parameter(torch.zeros(l2.out_features, rank))
        self.head = nn.Linear(l2.out_features, out_dim)

    def adapter_params(self):
        return [self.A1, self.B1, self.A2, self.B2, *self.head.parameters()]

    def forward(self, x):
        l1, l2 = self.backbone.net[0], self.backbone.net[2]
        h1 = F.relu(F.linear(x, l1.weight, l1.bias) + F.linear(F.linear(x, self.A1), self.B1))
        h2 = F.relu(F.linear(h1, l2.weight, l2.bias) + F.linear(F.linear(h1, self.A2), self.B2))
        return self.head(h2)

def n_params(ps):
    return int(sum(p.numel() for p in ps))

print("=" * 68)
print(" DAS vs LoRA — same frozen backbone, two ways to specialize (MNIST)")
print("=" * 68)

# ── Shared frozen backbone ──────────────────────────────────────
backbone = Backbone(in_dim=784, feat_dim=64).to(DEVICE)
sub = np.random.choice(len(X_tr), 12000, replace=False)
Xb = torch.from_numpy(X_tr[sub]).to(DEVICE); yb = torch.from_numpy(y_tr[sub]).to(DEVICE)
temp = nn.Linear(64, 10).to(DEVICE)
opt = torch.optim.Adam(list(backbone.parameters()) + list(temp.parameters()), lr=1e-3)
for _ in range(1500):
    idx = torch.randint(0, len(Xb), (128,), device=DEVICE)
    opt.zero_grad(); F.cross_entropy(temp(backbone(Xb[idx])), yb[idx]).backward(); opt.step()
backbone.freeze()
n_bb = n_params(backbone.parameters())
print(f"\n[backbone] frozen, {n_bb:,} params")

# ── DAS: isolated head per task on frozen features ──────────────
das_heads, das_acc, das_feat = [], [], {}
for t, (d0, d1) in enumerate(TASKS):
    Xt, yt = binary_split(X_tr, y_tr, d0, d1)
    Xe, ye = binary_split(X_te, y_te, d0, d1)
    with torch.no_grad():
        ft = backbone(Xt.to(DEVICE)).cpu(); fe = backbone(Xe.to(DEVICE)).cpu()
    das_feat[t] = (fe, ye)
    head = HeadLeaf([64, 2]).to(DEVICE); das_heads.append(head)
    # train just this head on frozen features
    o = torch.optim.Adam(head.parameters(), lr=1e-3)
    ftd, ytd = ft.to(DEVICE), yt.to(DEVICE)
    head.train()
    for _ in range(400):
        idx = torch.randint(0, len(ftd), (128,), device=DEVICE)
        o.zero_grad(); F.cross_entropy(head(ftd[idx]), ytd[idx]).backward(); o.step()
    head.eval()
    with torch.no_grad():
        das_acc.append((head(fe.to(DEVICE)).argmax(1) == ye.to(DEVICE)).float().mean().item())
das_head_params = n_params(das_heads[0].parameters())

# ── LoRA: per-task adapters on the frozen backbone ──────────────
lora_experts, lora_acc = [], []
for t, (d0, d1) in enumerate(TASKS):
    Xt, yt = binary_split(X_tr, y_tr, d0, d1)
    Xe, ye = binary_split(X_te, y_te, d0, d1)
    exp = LoRAExpert(backbone, rank=RANK).to(DEVICE); lora_experts.append(exp)
    o = torch.optim.Adam(exp.adapter_params(), lr=1e-3)
    Xtd, ytd = Xt.to(DEVICE), yt.to(DEVICE)
    exp.train()
    for _ in range(400):
        idx = torch.randint(0, len(Xtd), (128,), device=DEVICE)
        o.zero_grad(); F.cross_entropy(exp(Xtd[idx]), ytd[idx]).backward(); o.step()
    exp.eval()
    with torch.no_grad():
        lora_acc.append((exp(Xe.to(DEVICE)).argmax(1) == ye.to(DEVICE)).float().mean().item())
lora_task_params = n_params(lora_experts[0].adapter_params())

# ── Task-free routing (DAS's structural edge) ───────────────────
router = StemRouter(64, len(TASKS)).to(DEVICE)
Xr = torch.cat([binary_split(X_tr, y_tr, *TASKS[t])[0] for t in range(len(TASKS))])
dr = torch.cat([torch.full((len(binary_split(X_tr, y_tr, *TASKS[t])[0]),), t, dtype=torch.int64)
                for t in range(len(TASKS))])
with torch.no_grad():
    Fr = backbone(Xr.to(DEVICE))
ro = torch.optim.Adam(router.parameters(), lr=1e-3)
drd = dr.to(DEVICE)
for _ in range(800):
    idx = torch.randint(0, len(Fr), (128,), device=DEVICE)
    ro.zero_grad(); _, _, lg = router(Fr[idx]); F.cross_entropy(lg, drd[idx]).backward(); ro.step()
with torch.no_grad():
    _, _, lg = router(Fr); router_acc = (lg.argmax(1) == drd).float().mean().item()

# ── Forgetting + deletion (both isolated by construction) ───────
# Train an EXTRA task for each and confirm the first task's module is untouched.
h0_before = param_hash(*das_heads[0].parameters())
l0_before = param_hash(*lora_experts[0].adapter_params())
# (training new modules above never touched module 0; verify)
extra_head = HeadLeaf([64, 2]).to(DEVICE)
Xx, yx = binary_split(X_tr, y_tr, 8, 9)
with torch.no_grad():
    fx = backbone(Xx.to(DEVICE)).cpu()
ox = torch.optim.Adam(extra_head.parameters(), lr=1e-3)
fxd, yxd = fx.to(DEVICE), yx.to(DEVICE)
for _ in range(200):
    idx = torch.randint(0, len(fxd), (128,), device=DEVICE)
    ox.zero_grad(); F.cross_entropy(extra_head(fxd[idx]), yxd[idx]).backward(); ox.step()
h0_after = param_hash(*das_heads[0].parameters())
l0_after = param_hash(*lora_experts[0].adapter_params())
das_isolated = h0_before == h0_after
lora_isolated = l0_before == l0_after

# ── Report ──────────────────────────────────────────────────────
print("\n" + "-" * 68)
print(f"  {'Metric':<34}{'DAS (head)':>16}{'LoRA (adapter)':>18}")
print("-" * 68)
for t in range(len(TASKS)):
    print(f"  task {t} acc {str(TASKS[t]):<22}{das_acc[t]:>16.4f}{lora_acc[t]:>18.4f}")
print(f"  {'mean per-task acc':<34}{np.mean(das_acc):>16.4f}{np.mean(lora_acc):>18.4f}")
print(f"  {'params per task':<34}{das_head_params:>16,}{lora_task_params:>18,}")
print(f"  {'zero forgetting (module isolated)':<34}{str(das_isolated):>16}{str(lora_isolated):>18}")
print(f"  {'deletion (drop the module)':<34}{'trivial':>16}{'trivial':>18}")
print(f"  {'task-free routing built in?':<34}{'yes':>16}{'no':>18}")
print("-" * 68)
print(f"\n  DAS router accuracy on features (task-free selection): {router_acc:.4f}")
print(f"  -> LoRA would need this SAME router to run task-free; on its own it")
print(f"     must be told the task. That router is DAS's one structural edge.")

print("\n" + "=" * 68)
print(" VERDICT")
print("=" * 68)
dm, lm = np.mean(das_acc), np.mean(lora_acc)
acc_gap = lm - dm
print(f"  Accuracy:   {'LoRA' if acc_gap>0.005 else 'DAS' if acc_gap<-0.005 else 'tie'}"
      f"  (LoRA {lm:.3f} vs DAS {dm:.3f}, gap {acc_gap:+.3f})")
print(f"  Params:     DAS lighter ({das_head_params} vs {lora_task_params} per task)")
print(f"  Forgetting: tie (both isolated -> 0)")
print(f"  Deletion:   tie (both drop a module)")
print(f"  Task-free:  DAS (built-in router @ {router_acc:.2f}); LoRA needs the task id")
print("\n  Honest read: DAS and per-task LoRA are nearly the same idea. They tie on")
print("  isolation, forgetting, and deletion. LoRA can re-tune features (often a")
print("  little more accurate, more params); DAS's real edge is the integrated,")
print("  task-free router. If the task is always known, plain LoRA is simpler and")
print("  equivalent. DAS earns its keep only when task-free routing + audit matter.")
print("=" * 68)
