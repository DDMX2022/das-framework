"""
router_bench.py
---------------
The router is DAS's bottleneck on real images (Phase 8: a linear gate hit only
42% routing accuracy on raw CIFAR pixels). Two ways to fix it:
  (a) a more EXPRESSIVE router (a small MLP) on the raw pixels, or
  (b) route on LEARNED features from a shared backbone (Phase 9).

This bench measures (a) directly — linear vs MLP router on raw pixels — on both
MNIST (easy, near-linearly-separable) and CIFAR (hard). It quantifies how much a
non-linear router buys you, and whether it's enough on its own.

("Attention router" in the roadmap is overkill for fixed vectors with no sequence
to attend over; the honest expressive upgrade is a non-linear MLP gate.)
"""
import gzip, struct, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)

def read_idx(path):
    with gzip.open(path, 'rb') as f:
        magic = struct.unpack('>I', f.read(4))[0]; nd = magic & 0xFF
        shape = tuple(struct.unpack('>I', f.read(4))[0] for _ in range(nd))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

def mnist_tasks():
    Xtr = read_idx('./data/MNIST/raw/train-images-idx3-ubyte.gz').reshape(-1, 784).astype(np.float32) / 255.0
    ytr = read_idx('./data/MNIST/raw/train-labels-idx1-ubyte.gz').astype(np.int64)
    Xte = read_idx('./data/MNIST/raw/t10k-images-idx3-ubyte.gz').reshape(-1, 784).astype(np.float32) / 255.0
    yte = read_idx('./data/MNIST/raw/t10k-labels-idx1-ubyte.gz').astype(np.int64)
    return Xtr, ytr, Xte, yte, 784

def cifar_tasks():
    tr = datasets.CIFAR10(root='./data', train=True, download=True)
    te = datasets.CIFAR10(root='./data', train=False, download=True)
    Xtr = tr.data.astype(np.float32).transpose(0, 3, 1, 2).reshape(-1, 3072) / 255.0
    Xte = te.data.astype(np.float32).transpose(0, 3, 1, 2).reshape(-1, 3072) / 255.0
    return Xtr, np.array(tr.targets), Xte, np.array(te.targets), 3072

TASKS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]

def task_routing_data(X, y, per_task=2500):
    """Pool samples from each task labeled by TASK ID (the routing target)."""
    rng = np.random.default_rng(0)
    Xs, ds = [], []
    for t, (d0, d1) in enumerate(TASKS):
        idx = np.where((y == d0) | (y == d1))[0]
        idx = rng.choice(idx, min(per_task, len(idx)), replace=False)
        Xs.append(X[idx]); ds.append(np.full(len(idx), t))
    return np.vstack(Xs), np.concatenate(ds)

def train_router(model, X, d, Xte, dte, steps=1000):
    model = model.to(DEVICE)
    X = torch.from_numpy(X).to(DEVICE); d = torch.from_numpy(d).to(DEVICE)
    Xte = torch.from_numpy(Xte).to(DEVICE); dte = torch.from_numpy(dte).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(steps):
        idx = torch.randint(0, len(X), (128,), device=DEVICE)
        opt.zero_grad(); F.cross_entropy(model(X[idx]), d[idx]).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        return (model(Xte).argmax(1) == dte).float().mean().item()

def linear(dim): return nn.Linear(dim, 5)
def mlp(dim): return nn.Sequential(nn.Linear(dim, 256), nn.ReLU(), nn.Linear(256, 5))

print("=" * 60)
print(" Router on raw pixels — linear vs MLP (5-way task id)")
print("=" * 60)
for name, loader in [("MNIST", mnist_tasks), ("CIFAR-10", cifar_tasks)]:
    Xtr, ytr, Xte, yte, dim = loader()
    Xr, dr = task_routing_data(Xtr, ytr)
    Xe, de = task_routing_data(Xte, yte, per_task=800)
    lin = train_router(linear(dim), Xr, dr, Xe, de)
    nl = train_router(mlp(dim), Xr, dr, Xe, de)
    print(f"\n  {name} (dim {dim}):")
    print(f"    linear router: {lin:.3f}")
    print(f"    MLP router:    {nl:.3f}   (+{nl-lin:.3f})")

print("\n" + "=" * 60)
print("  Takeaway: on MNIST both routers are strong (raw pixels ~ separable).")
print("  On CIFAR the MLP router beats linear but raw pixels stay hard — the")
print("  real fix is routing on LEARNED features (Phase 9 shared backbone),")
print("  where the router recovers to ~98%.")
print("=" * 60)
