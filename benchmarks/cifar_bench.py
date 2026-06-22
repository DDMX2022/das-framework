"""
cifar_bench.py
---------------
Phase 8: does hard top-1 routing survive REAL images?

Everything up to this point (demo_torch.py, conv_demo.py, app.py's Split-MNIST
continual bench) used MNIST — 28x28 grayscale digits, a benchmark so easy a
linear classifier gets ~92% on raw pixels. That makes MNIST a bad test of
whether DAS's router (a single nn.Linear + softmax + argmax) can actually
read an image well enough to route it. CIFAR-10 (32x32x3 color photos of
real objects) is a much harder, more realistic stress test for exactly the
part of DAS that MNIST was too easy to expose: the router.

This script builds Split-CIFAR (5 binary tasks: 0v1, 2v3, 4v5, 6v7, 8v9 —
same task shape as the MNIST continual bench in app.py) and compares:
  - DAS:        a linear router (raw 3072-dim pixels -> 5-way) + 5 CNN
                leaves, each trained in ISOLATION and frozen.
  - Fine-tuned: one shared CNN, trained sequentially task by task (no
                protection at all — expected to forget).
  - Multi-task: one shared CNN, trained on all 5 tasks jointly (upper bound,
                cheats by seeing everything at once).

The headline number is the DAS ROUTER's accuracy on raw CIFAR pixels — if a
linear gate can't tell "cat or dog" from "0 or 1" reliably from raw pixels,
that's a real finding about where this architecture's ceiling is, not a bug
to be tuned away. The per-task leaf accuracy is graded separately (with the
task id KNOWN, same as the MNIST bench) precisely so a weak router doesn't
contaminate the leaf-quality measurement, and vice versa.

Runs on MPS for speed (training only — see the note below on why this does
not weaken the forgetting proof). Aim: ~3-8 minutes end to end.

NOTE on determinism: the zero-forgetting proof is a hash equality check on
FROZEN parameters. A frozen parameter's bytes do not change regardless of
what device produced them or whether MPS float ops are bit-reproducible
across runs — there is no gradient flowing into it, full stop. So unlike
demo_torch.py / conv_demo.py (which force CPU because they want the printed
accuracy numbers to be exactly reproducible run-to-run), this script trains
on MPS for speed and the BWT=0 proof is just as valid.

Run:      conda run -n das python cifar_bench.py
First run downloads CIFAR-10 (~170MB) into ./data via torchvision.
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets

sys.path.insert(0, '.')
from das_torch import (
    ConvLeaf, DASForest, leaf_hash, train_leaf_isolated, train_router,
)

DEVICE = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu"
)
torch.manual_seed(0)
np.random.seed(0)

D_MODEL = 3072            # 32*32*3, flattened
CHANNELS = 3
LEAF_DIMS = [3072, 2]     # ConvLeaf only needs [in_dim, out_dim]; internal shape is fixed by channels=3
TASKS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]   # CIFAR-10 class indices, same task shape as Split-MNIST
N_TASKS = len(TASKS)

# CIFAR-10 class names, for readability in printouts only.
CIFAR_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                  'dog', 'frog', 'horse', 'ship', 'truck']

# Keep runtime tractable: subset per task rather than the full 5000/class.
N_TRAIN_PER_TASK = 1500   # per task, balanced across its 2 classes -> 3000 images/task
N_TEST_PER_TASK = 500     # per task, balanced
LEAF_STEPS = 500
ROUTER_STEPS = 800
FT_STEPS = 500
MT_STEPS = 1200
LR = 1e-3
BATCH = 128

CKPT_DIR = './checkpoints/cifar_forest'


# ── Data ─────────────────────────────────────────────────────────────────
def load_cifar():
    """Load CIFAR-10 via torchvision (downloads to ./data on first run).
    Flatten to (N, 3072) float32 in [0, 1] — channel-first order (3,32,32)
    flattened, matching what ConvLeaf's view(-1, channels, side, side) expects
    on reshape. [0,1] scaling (not standardized) to keep this script's data
    pipeline simple and match conv_demo.py's /255.0 convention for MNIST."""
    train = datasets.CIFAR10(root='./data', train=True, download=True)
    test = datasets.CIFAR10(root='./data', train=False, download=True)
    # train.data / test.data are uint8 arrays shaped (N, 32, 32, 3) (HWC).
    X_tr = train.data.astype(np.float32).transpose(0, 3, 1, 2).reshape(-1, 3072) / 255.0
    y_tr = np.array(train.targets, dtype=np.int64)
    X_te = test.data.astype(np.float32).transpose(0, 3, 1, 2).reshape(-1, 3072) / 255.0
    y_te = np.array(test.targets, dtype=np.int64)
    return X_tr, y_tr, X_te, y_te


def binary_split(X, y, d0, d1, n_per_class, rng):
    """Balanced binary train/test split: label=1 if class==d1, else 0.
    Same recipe as app.py's _cl_binary_split, just NumPy-side here since
    CIFAR doesn't need the standardization MNIST's version does."""
    mask = (y == d0) | (y == d1)
    Xs, ys = X[mask], y[mask]
    pos_idx = np.where(ys == d1)[0]
    neg_idx = np.where(ys == d0)[0]
    n = min(n_per_class, len(pos_idx), len(neg_idx))
    pos_pick = rng.choice(pos_idx, n, replace=False)
    neg_pick = rng.choice(neg_idx, n, replace=False)
    idx = np.concatenate([pos_pick, neg_pick])
    Xd = Xs[idx]
    yd = np.concatenate([np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.int64)])
    perm = rng.permutation(len(Xd))
    return Xd[perm], yd[perm]


def acc_of(model, X, y, device, batch=512):
    """Batched accuracy eval (CIFAR test sets are big enough that a single
    forward pass through a CNN on MPS benefits from chunking)."""
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = X[i:i + batch].to(device)
            yb = y[i:i + batch].to(device)
            pred = model(xb).argmax(1)
            correct += (pred == yb).sum().item()
    return correct / len(X)


def main():
    t_start = time.time()
    print("=" * 72)
    print(" DAS FRAMEWORK — Phase 8: Split-CIFAR-10 stress test")
    print(" Hard top-1 routing on REAL images. Does it survive?")
    print("=" * 72)
    print(f"\n  Device: {DEVICE}  (training on accelerator for speed; frozen-weight")
    print(f"           hashes are byte-exact regardless of device/nondeterminism)")

    rng = np.random.default_rng(42)

    # ── Load + split data ──────────────────────────────────────────────
    print("\n[Data] Loading CIFAR-10 (downloads on first run) ...")
    t0 = time.time()
    X_tr, y_tr, X_te, y_te = load_cifar()
    print(f"        train: {len(X_tr):,}  test: {len(X_te):,}  ({time.time()-t0:.1f}s)")

    task_splits = []
    for d0, d1 in TASKS:
        Xtr, ytr = binary_split(X_tr, y_tr, d0, d1, N_TRAIN_PER_TASK, rng)
        Xte, yte = binary_split(X_te, y_te, d0, d1, N_TEST_PER_TASK, rng)
        task_splits.append((
            torch.from_numpy(Xtr), torch.from_numpy(ytr),
            torch.from_numpy(Xte), torch.from_numpy(yte),
        ))
    task_labels = [f'{CIFAR_CLASSES[a]}({a}) vs {CIFAR_CLASSES[b]}({b})' for a, b in TASKS]
    for t, lbl in enumerate(task_labels):
        print(f"        task {t}: {lbl}  "
              f"train={len(task_splits[t][0])}  test={len(task_splits[t][2])}")

    # Router training set: pool a slice of every task's train data, labeled by task id.
    Xr = torch.cat([task_splits[t][0] for t in range(N_TASKS)])
    dr = torch.cat([torch.full((len(task_splits[t][0]),), t, dtype=torch.int64) for t in range(N_TASKS)])

    # ════════════════════════════════════════════════════════════════════
    # STAGE 2 — DAS: linear router + 5 isolated CNN leaves
    # ════════════════════════════════════════════════════════════════════
    print("\n" + "-" * 72)
    print(" DAS — linear router (raw pixels) + 5 isolated CNN leaves")
    print("-" * 72)

    forest = DASForest(D_MODEL, LEAF_DIMS, num_leaves=N_TASKS).to(DEVICE)
    # DASForest defaults to FibonacciLeaf (MLP); swap in CNN leaves for this bench.
    forest.leaves = torch.nn.ModuleList([
        ConvLeaf(LEAF_DIMS, channels=CHANNELS) for _ in range(N_TASKS)
    ]).to(DEVICE)

    print(f"\n[Router] Training linear gate on raw 3072-dim pixels, 5-way task id ...")
    t0 = time.time()
    router_train_acc = train_router(forest, Xr, dr, steps=ROUTER_STEPS, lr=LR, batch=BATCH, device=DEVICE)
    t_router = time.time() - t0
    # Also measure router accuracy on HELD-OUT data (train acc alone overstates it).
    Xr_te = torch.cat([task_splits[t][2] for t in range(N_TASKS)])
    dr_te = torch.cat([torch.full((len(task_splits[t][2]),), t, dtype=torch.int64) for t in range(N_TASKS)])
    with torch.no_grad():
        _, _, logits = forest.router(Xr_te.to(DEVICE))
        router_test_acc = (logits.argmax(1) == dr_te.to(DEVICE)).float().mean().item()
    print(f"          router train acc: {router_train_acc:.4f}   "
          f"router TEST acc: {router_test_acc:.4f}   ({t_router:.1f}s)")

    das_matrix = [[None] * N_TASKS for _ in range(N_TASKS)]
    t_leaves = []
    leaf_train_accs = []
    # Incremental forgetting proof: capture each leaf's hash the instant it
    # freezes, and BEFORE training each new leaf re-check that every
    # previously-frozen leaf is still byte-identical. This actually exercises
    # the isolation guarantee (training leaf t must not perturb leaves 0..t-1),
    # unlike a single snapshot taken after everything has already trained.
    frozen_hashes = {}            # leaf id -> hash captured right after it froze
    proof_violations = []         # (trained_leaf, perturbed_earlier_leaf) pairs
    for t in range(N_TASKS):
        Xtr, ytr, Xte, yte = task_splits[t]
        snap_before = {i: frozen_hashes[i] for i in range(t)}   # earlier leaves, pre-training-t
        print(f"\n[Leaf {t}] Training in isolation on task {t} ({task_labels[t]}) ...")
        t0 = time.time()
        leaf_acc = train_leaf_isolated(forest, t, Xtr, ytr, steps=LEAF_STEPS, lr=LR, batch=BATCH, device=DEVICE)
        dt = time.time() - t0
        t_leaves.append(dt)
        leaf_train_accs.append(leaf_acc)
        print(f"          leaf {t} train acc: {leaf_acc:.4f}   ({dt:.1f}s)")
        # The actual proof: did training leaf t disturb any already-frozen leaf?
        for i in range(t):
            if leaf_hash(forest.leaves[i]) != snap_before[i]:
                proof_violations.append((t, i))
        frozen_hashes[t] = leaf_hash(forest.leaves[t])
        for ev in range(t + 1):
            _, _, Xe, ye = task_splits[ev]
            das_matrix[t][ev] = round(acc_of(forest.leaves[ev], Xe, ye, DEVICE), 4)
        print(f"          eval so far: {das_matrix[t][:t+1]}")

    print("\n[Proof] Incremental forgetting check (training leaf t vs leaves 0..t-1):")
    n_checks = N_TASKS * (N_TASKS - 1) // 2
    unchanged = len(proof_violations) == 0
    if unchanged:
        print(f"          {n_checks}/{n_checks} pairwise checks PASS — every frozen leaf stayed byte-identical")
    else:
        for (trained, perturbed) in proof_violations:
            print(f"          *** CHANGED *** leaf {perturbed} moved while training leaf {trained}")
    for i in range(N_TASKS):
        print(f"          leaf {i} final hash: {frozen_hashes[i]}")

    # ── Checkpoint + round-trip integrity: save the forest, reload it, and
    #    confirm every leaf comes back byte-for-byte (the Phase 7 machinery
    #    working on CIFAR CNNs, not just MNIST MLPs). ──
    print("\n[Checkpoint] Saving DAS forest + reloading to verify byte-exact restore ...")
    os.makedirs(CKPT_DIR, exist_ok=True)
    from das_torch import save_forest, load_forest
    save_forest(forest, CKPT_DIR)
    with open(os.path.join(CKPT_DIR, 'task_labels.json'), 'w') as f:
        json.dump({'tasks': TASKS, 'labels': task_labels}, f, indent=2)
    reloaded = load_forest(CKPT_DIR, device=DEVICE)
    restore_ok = all(leaf_hash(reloaded.leaves[i]) == frozen_hashes[i] for i in range(N_TASKS))
    print(f"          saved to {CKPT_DIR}/   restore byte-exact: {'PASS' if restore_ok else 'FAIL'}")

    das_bwt = round(sum(das_matrix[N_TASKS - 1][i] - das_matrix[i][i] for i in range(N_TASKS - 1))
                     / (N_TASKS - 1), 4)

    # ── Cross-domain contamination: every leaf on every task's test set ──
    print("\n[Contamination] Each leaf evaluated on EVERY task's test set ...")
    contam = [[round(acc_of(forest.leaves[i], task_splits[j][2], task_splits[j][3], DEVICE), 4)
               for j in range(N_TASKS)] for i in range(N_TASKS)]
    diag = [contam[i][i] for i in range(N_TASKS)]
    off = [contam[i][j] for i in range(N_TASKS) for j in range(N_TASKS) if i != j]
    diag_mean = round(sum(diag) / len(diag), 4)
    off_mean = round(sum(off) / len(off), 4)
    print(f"          diagonal mean (own task): {diag_mean:.4f}")
    print(f"          off-diagonal mean (wrong task, ~chance=0.5): {off_mean:.4f}")

    t_das_total = t_router + sum(t_leaves)

    # ════════════════════════════════════════════════════════════════════
    # STAGE 3a — Fine-tuned CNN baseline (sequential, no protection)
    # ════════════════════════════════════════════════════════════════════
    print("\n" + "-" * 72)
    print(" Fine-tuned CNN — ONE shared net, trained sequentially (forgetting expected)")
    print("-" * 72)

    ft_net = ConvLeaf(LEAF_DIMS, channels=CHANNELS).to(DEVICE)
    ft_net.unfreeze()
    ft_matrix = [[None] * N_TASKS for _ in range(N_TASKS)]
    t_ft = []
    for t in range(N_TASKS):
        Xtr, ytr, Xte, yte = task_splits[t]
        print(f"\n[FT task {t}] Overwriting shared CNN on task {t} ({task_labels[t]}) ...")
        Xtr_d, ytr_d = Xtr.to(DEVICE), ytr.to(DEVICE)
        opt = torch.optim.Adam(ft_net.parameters(), lr=LR)
        t0 = time.time()
        ft_net.train()
        n = len(Xtr_d)
        for s in range(FT_STEPS):
            idx = torch.randint(0, n, (min(BATCH, n),), device=DEVICE)
            xb, yb = Xtr_d[idx], ytr_d[idx]
            opt.zero_grad()
            loss = F.cross_entropy(ft_net(xb), yb)
            loss.backward()
            opt.step()
        ft_net.eval()
        dt = time.time() - t0
        t_ft.append(dt)
        train_acc = acc_of(ft_net, Xtr, ytr, DEVICE)
        print(f"          train acc: {train_acc:.4f}   ({dt:.1f}s)")
        for ev in range(t + 1):
            _, _, Xe, ye = task_splits[ev]
            ft_matrix[t][ev] = round(acc_of(ft_net, Xe, ye, DEVICE), 4)
        print(f"          eval so far: {ft_matrix[t][:t+1]}")

    ft_bwt = round(sum(ft_matrix[N_TASKS - 1][i] - ft_matrix[i][i] for i in range(N_TASKS - 1))
                    / (N_TASKS - 1), 4)
    t_ft_total = sum(t_ft)

    # ════════════════════════════════════════════════════════════════════
    # STAGE 3b — Multi-task CNN baseline (joint training, upper bound)
    # ════════════════════════════════════════════════════════════════════
    print("\n" + "-" * 72)
    print(" Multi-task CNN — ONE shared net trained on ALL tasks jointly (upper bound)")
    print("-" * 72)

    # 10-way softmax over the original CIFAR class ids actually present
    # (0..9) keeps this simple: same net shape (ConvLeaf), just out_dim=10.
    mt_net = ConvLeaf([3072, 10], channels=CHANNELS).to(DEVICE)
    mt_net.unfreeze()
    Xmt = torch.cat([task_splits[t][0] for t in range(N_TASKS)]).to(DEVICE)
    # labels: map back to the ORIGINAL CIFAR class id (not the per-task binary label)
    ymt_parts = []
    for t, (d0, d1) in enumerate(TASKS):
        _, ytr, _, _ = task_splits[t]
        ymt_parts.append(torch.where(ytr == 0, torch.tensor(d0), torch.tensor(d1)))
    ymt = torch.cat(ymt_parts).to(DEVICE)

    print(f"\n[Multi-task] Training on {len(Xmt)} pooled images across all 5 tasks, 10-way ...")
    opt = torch.optim.Adam(mt_net.parameters(), lr=LR)
    t0 = time.time()
    mt_net.train()
    n = len(Xmt)
    for s in range(MT_STEPS):
        idx = torch.randint(0, n, (BATCH,), device=DEVICE)
        xb, yb = Xmt[idx], ymt[idx]
        opt.zero_grad()
        loss = F.cross_entropy(mt_net(xb), yb)
        loss.backward()
        opt.step()
    mt_net.eval()
    t_mt = time.time() - t0
    print(f"          ({t_mt:.1f}s)")

    mt_accs = []
    for t, (d0, d1) in enumerate(TASKS):
        _, _, Xte, yte = task_splits[t]
        Xte_d = Xte.to(DEVICE)
        with torch.no_grad():
            logits = mt_net(Xte_d)
        # binary decision restricted to the task's own two classes, same trick
        # app.py's multi-task baseline uses: compare logit(d1) vs logit(d0).
        pred = (logits[:, d1] > logits[:, d0]).long().cpu()
        mt_accs.append(round((pred == yte).float().mean().item(), 4))
    print(f"          per-task acc (binary decision within task's 2 classes): {mt_accs}")

    # ════════════════════════════════════════════════════════════════════
    # STAGE 4 — Final report
    # ════════════════════════════════════════════════════════════════════
    t_total = time.time() - t_start

    print("\n" + "=" * 72)
    print(" FINAL REPORT — Split-CIFAR-10 (Phase 8)")
    print("=" * 72)

    print("\n  Backward Transfer (BWT) — negative means forgetting; 0 means none:")
    print(f"    DAS:         {das_bwt:+.4f}")
    print(f"    Fine-tuned:  {ft_bwt:+.4f}")
    print(f"    Multi-task:  n/a (not sequential, no forgetting to measure)")

    print(f"\n  *** HEADLINE: DAS router accuracy on raw CIFAR pixels ***")
    print(f"    train: {router_train_acc:.4f}   held-out test: {router_test_acc:.4f}")
    bottleneck = router_test_acc < 0.85
    if bottleneck:
        interpretation = (
            "routing is the bottleneck here — a single linear gate on raw pixels "
            "cannot cleanly separate 5 visually-overlapping object categories the "
            "way it could separate MNIST digit pairs."
        )
    else:
        interpretation = "the linear router holds up surprisingly well even on raw CIFAR pixels."
    print(f"    Interpretation: {interpretation}")

    print(f"\n  Per-task DAS leaf accuracy (task id KNOWN — isolates leaf quality from router quality):")
    for t in range(N_TASKS):
        print(f"    leaf {t} ({task_labels[t]}): train={leaf_train_accs[t]:.4f}  "
              f"final test={das_matrix[N_TASKS-1][t]:.4f}")

    print(f"\n  Forgetting proof (frozen leaf hashes unchanged): {'PASS' if unchanged else 'FAIL'}")
    print(f"  Checkpoint restore byte-exact (save -> reload): {'PASS' if restore_ok else 'FAIL'}")

    print(f"\n  Cross-domain contamination:")
    print(f"    diagonal mean (own task):        {diag_mean:.4f}")
    print(f"    off-diagonal mean (wrong task):   {off_mean:.4f}  (chance = 0.5)")

    print(f"\n  Runtime:")
    print(f"    DAS total:        {t_das_total:.1f}s  (router {t_router:.1f}s + leaves {sum(t_leaves):.1f}s)")
    print(f"    Fine-tuned total:  {t_ft_total:.1f}s")
    print(f"    Multi-task total:  {t_mt:.1f}s")
    print(f"    Full script:       {t_total:.1f}s")

    print("\n" + "=" * 72)
    overall_pass = unchanged and restore_ok
    print(f"  Overall (forgetting proof + restore): {'PASS' if overall_pass else 'FAIL'}")
    print("=" * 72)

    if not overall_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
