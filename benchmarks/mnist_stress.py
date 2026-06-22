"""
mnist_stress.py
---------------
MNIST stress test: 10 specialist leaves, one per digit class.
Each leaf is a deep MLP trained IN ISOLATION on 1-vs-rest binary task.
Router does 10-way routing on full 784-dim pixel vectors.

This tests:
  1. Does routing scale to 10 classes on real image data?
  2. Do deeper Fibonacci nets (784→512→256→2) stay isolated?
  3. Can DAS leaves specialise better than a shared 10-class MLP?
  4. Is the forgetting proof still true across 10 leaves?

Device: auto-selects MPS (Apple Silicon) → CUDA → CPU

Install:  pip install torchvision
Run:      python mnist_stress.py
"""
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np, sys, time, hashlib

sys.path.insert(0, '.')
from das_torch import DASForest, FibonacciLeaf

device = ("mps"  if torch.backends.mps.is_available() else
          "cuda" if torch.cuda.is_available()          else "cpu")
print(f"\n  Device: {device}")

# ── Config ────────────────────────────────────────────────────────
LEAF_DIMS  = [784, 512, 256, 2]    # deeper Fibonacci descent
BASE_DIMS  = [784, 1024, 512, 10]  # 10-class baseline, ~same total params
LR         = 1e-3
LEAF_EP    = 8     # epochs per leaf (isolated)
ROUTER_EP  = 5
BASE_EP    = 10
BATCH      = 256
torch.manual_seed(0)
np.random.seed(0)

# ── Load MNIST ────────────────────────────────────────────────────
flat = transforms.Compose([transforms.ToTensor(),
                            transforms.Lambda(lambda x: x.view(-1))])
train_ds = datasets.MNIST('./data', train=True,  download=True, transform=flat)
test_ds  = datasets.MNIST('./data', train=False, download=True, transform=flat)
print(f"  MNIST: {len(train_ds):,} train  {len(test_ds):,} test")

# ── Dataset helpers ───────────────────────────────────────────────
class BinaryDataset(Dataset):
    """Wraps MNIST for 1-vs-rest binary classification on one digit."""
    def __init__(self, ds, digit):
        self.data, self.labels = [], []
        targets = ds.targets.numpy()
        pos = np.where(targets == digit)[0]
        neg = np.random.choice(np.where(targets != digit)[0], len(pos), replace=False)
        for i in np.concatenate([pos, neg]):
            x, _ = ds[i]
            self.data.append(x)
            self.labels.append(1 if int(ds.targets[i]) == digit else 0)
    def __len__(self):  return len(self.data)
    def __getitem__(self, i): return self.data[i], self.labels[i]

def leaf_accuracy(leaf, ds, digit, device):
    bd = BinaryDataset(ds, digit)
    loader = DataLoader(bd, batch_size=512)
    leaf.eval(); correct = 0; n = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), torch.tensor(y).to(device)
            correct += (leaf(X).argmax(1) == y).sum().item(); n += len(y)
    leaf.train()
    return correct / n

def leaf_hash(leaf):
    h = hashlib.sha256()
    for p in leaf.parameters():
        h.update(p.data.cpu().numpy().tobytes())
    return h.hexdigest()[:14]

# ── Build DAS Forest: 10 leaves ───────────────────────────────────
print(f"\n  Building DAS Forest: 10 leaves × {LEAF_DIMS} ...")
forest = DASForest(d_model=784, leaf_dims=LEAF_DIMS, num_leaves=10).to(device)
das_p  = sum(p.numel() for p in forest.parameters())
print(f"  DAS total params: {das_p:,}")

# ── Phase 1: Router — 10-way routing ─────────────────────────────
print("\n[Phase 1] Stem Router — 10-way routing on 784-dim pixels ...")
full_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
router_opt  = torch.optim.Adam(forest.router.parameters(), lr=LR)

t0 = time.time()
for epoch in range(ROUTER_EP):
    correct = 0; n_total = 0
    for X, y in full_loader:
        X, y = X.to(device), y.to(device)
        router_opt.zero_grad()
        _, tau, logits = forest.router(X)
        loss = F.cross_entropy(logits, y)
        loss.backward(); router_opt.step()
        correct += (logits.argmax(1) == y).sum().item(); n_total += len(y)
    print(f"  epoch {epoch+1}/{ROUTER_EP}  routing acc: {correct/n_total:.3f}")
print(f"  Router done  ({time.time()-t0:.0f}s)")

# ── Phases 2-11: Each leaf in isolation ───────────────────────────
print("\n[Phases 2-11] Training 10 leaves in isolation ...")
for leaf in forest.leaves: leaf.freeze()

leaf_test_accs = {}
# Proof: for each leaf d, record hashes of all ALREADY-TRAINED (frozen) leaves
# before training leaf d, then verify they are unchanged after.
proof_violations = {}   # {(trained_leaf, checked_leaf): True/False}
frozen_hashes = {}      # hash of leaf d right after it finished training
t_total = time.time()

for digit in range(10):
    # Snapshot all previously-trained leaves before we touch leaf `digit`
    snap_before = {d: frozen_hashes[d] for d in range(digit)}

    leaf = forest.leaves[digit]; leaf.unfreeze()

    train_bd = BinaryDataset(train_ds, digit)
    loader   = DataLoader(train_bd, batch_size=BATCH, shuffle=True)
    opt      = torch.optim.Adam(leaf.parameters(), lr=LR)

    t0 = time.time()
    for epoch in range(LEAF_EP):
        correct = 0; n_total = 0
        for X, y in loader:
            X = X.to(device); y = torch.tensor(y).long().to(device)
            opt.zero_grad()
            loss = F.cross_entropy(leaf(X), y)
            loss.backward(); opt.step()
            correct += (leaf(X).argmax(1) == y).sum().item(); n_total += len(y)
    leaf.freeze()

    # Record this leaf's final hash
    frozen_hashes[digit] = leaf_hash(leaf)

    # Verify ALL previously-frozen leaves are still byte-identical
    for d in range(digit):
        changed = leaf_hash(forest.leaves[d]) != snap_before[d]
        proof_violations[(digit, d)] = changed

    test_acc = leaf_accuracy(leaf, test_ds, digit, device)
    leaf_test_accs[digit] = test_acc
    print(f"  Leaf {digit:2d} (digit {digit})  train ep{LEAF_EP} acc={correct/n_total:.3f}"
          f"  test acc={test_acc:.3f}  ({time.time()-t0:.0f}s)")

print(f"\n  All leaves done  (total {time.time()-t_total:.0f}s)")

# ── Phase 12: Baseline 10-class MLP ──────────────────────────────
print("\n[Phase 12] Baseline 10-class MLP ...")

class BaselineMLP(nn.Module):
    def __init__(self, dims):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2: layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

baseline = BaselineMLP(BASE_DIMS).to(device)
bl_p     = sum(p.numel() for p in baseline.parameters())
print(f"  Baseline params: {bl_p:,}")
bl_opt   = torch.optim.Adam(baseline.parameters(), lr=LR)

t0 = time.time()
for epoch in range(BASE_EP):
    correct = 0; n_total = 0
    for X, y in DataLoader(train_ds, batch_size=BATCH, shuffle=True):
        X, y = X.to(device), y.to(device)
        bl_opt.zero_grad()
        out = baseline(X)
        loss = F.cross_entropy(out, y); loss.backward(); bl_opt.step()
        correct += (out.argmax(1)==y).sum().item(); n_total += len(y)
    if (epoch+1) % 2 == 0:
        print(f"  epoch {epoch+1}/{BASE_EP}  acc={correct/n_total:.3f}")

# Baseline test accuracy (full 10-class)
correct = 0; n_total = 0
with torch.no_grad():
    for X, y in DataLoader(test_ds, batch_size=512):
        X, y = X.to(device), y.to(device)
        correct += (baseline(X).argmax(1)==y).sum().item(); n_total += len(y)
bl_test_acc = correct / n_total
print(f"  Baseline 10-class test acc: {bl_test_acc:.3f}  ({time.time()-t0:.0f}s)")

# ── Results ────────────────────────────────────────────────────────
print("\n" + "="*64)
print("  STRESS TEST RESULTS — DAS vs Baseline on MNIST")
print("="*64)
print(f"\n  Architecture      Params       Task")
print(f"  DAS Forest        {das_p:>10,}   10 × 1-vs-rest binary")
print(f"  Baseline MLP      {bl_p:>10,}   10-class softmax")

print(f"\n  {'Leaf/Digit':<12} {'DAS 1-vs-rest':>14}")
print(f"  {'-'*30}")
for d in range(10):
    print(f"  Leaf {d:2d} / {d}    {leaf_test_accs[d]:>14.3f}")

avg_leaf = sum(leaf_test_accs.values()) / 10
print(f"\n  DAS avg leaf acc (1-vs-rest):  {avg_leaf:.3f}")
print(f"  Baseline 10-class test acc:    {bl_test_acc:.3f}")

# Forgetting proof: no violations means all frozen leaves were byte-stable
# while subsequent leaves were being trained.
all_stable = not any(proof_violations.values())
if proof_violations:
    print(f"\n  Forgetting proof details:")
    for (trained, checked), changed in sorted(proof_violations.items()):
        status = "✗ CHANGED" if changed else "✓ stable"
        print(f"    Leaf {checked} while training leaf {trained}: {status}")
print(f"\n  Forgetting proof (10 leaves):  {'PASS ✓ — all previously-frozen leaves byte-identical across subsequent training' if all_stable else 'FAIL ✗ — some frozen leaf changed'}")

print(f"\n  Note: DAS leaf acc measures 1-vs-rest (binary, ~balanced).")
print(f"  Baseline measures 10-class accuracy (harder task, more comparable to real use).")
print("="*64)
