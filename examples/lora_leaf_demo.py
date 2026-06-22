"""
lora_leaf_demo.py
-----------------
Proves the LoRALeaf expert type: experts are now standard LoRA adapters on a
shared frozen backbone (the format the project decided to standardize on —
PRODUCT_PLAN.md Phase 1). Demonstrates the guarantees still hold for LoRA experts:

  - ISOLATION / zero-forgetting: graft + train a new LoRA expert; existing
    experts stay byte-identical (SHA-256).
  - per-expert accuracy on its task.
  - CHECKPOINT byte-exact: LoRAForest.save -> load reproduces every expert's hash.

Forced CPU for bit-reproducible hashes. Synthetic (no downloads).

Production note: swap the local Backbone for a frozen HuggingFace encoder + peft
LoRA — the forest API (route → one adapter → logits) is unchanged. Not done here
(no transformer libs / slow net); see the hook comment in das_torch.LoRALeaf.
"""
import tempfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from das_torch import LoRAForest, train_leaf_isolated_lora, leaf_hash

DEVICE = "cpu"
torch.manual_seed(0); np.random.seed(0)
D, FEAT, N = 20, 32, 400

def domain(did, n):
    c = np.zeros(D); c[did * 4] = 4.0
    rule = np.random.default_rng(100 + did).normal(0, 1, D)
    X = c + np.random.default_rng(did).normal(0, 1.0, (n, D))
    return torch.tensor(X, dtype=torch.float32), torch.tensor((X @ rule > 0).astype(np.int64))

doms = [domain(d, N) for d in range(3)]

print("=" * 60)
print(" LoRALeaf — experts as LoRA adapters on a shared frozen backbone")
print("=" * 60)

forest = LoRAForest(D, FEAT, out_dim=2, num_leaves=2, rank=8).to(DEVICE)

# pretrain + freeze the shared backbone (a stand-in for a frozen LM encoder)
Xp = torch.cat([doms[d][0] for d in range(2)]); yp = torch.cat([doms[d][1] for d in range(2)])
tmp = nn.Linear(FEAT, 2)
opt = torch.optim.Adam(list(forest.backbone.parameters()) + list(tmp.parameters()), lr=1e-3)
for _ in range(500):
    i = torch.randint(0, len(Xp), (64,))
    opt.zero_grad(); F.cross_entropy(tmp(forest.backbone(Xp[i])), yp[i]).backward(); opt.step()
forest.freeze_backbone()
print(f"\n  shared backbone frozen ({sum(p.numel() for p in forest.backbone.parameters()):,} params)")

# train two LoRA experts in isolation
for t in range(2):
    a = train_leaf_isolated_lora(forest, t, doms[t][0], doms[t][1], steps=400, device=DEVICE)
    n_ad = sum(p.numel() for p in forest.leaves[t].adapter_params())
    print(f"  LoRA expert {t}: acc {a:.3f}  ({n_ad:,} trainable adapter params)")

# ── isolation / forgetting proof ────────────────────────────────
before = [leaf_hash(l) for l in forest.leaves]
nid = forest.graft_leaf()
a2 = train_leaf_isolated_lora(forest, nid, doms[2][0], doms[2][1], steps=400, device=DEVICE)
after = [leaf_hash(forest.leaves[i]) for i in range(len(before))]
forget_ok = before == after
print(f"\n  grafted LoRA expert {nid}: acc {a2:.3f}")
print(f"  existing experts byte-identical after graft+train: {'PASS' if forget_ok else 'FAIL'}")

# ── checkpoint byte-exact round-trip ────────────────────────────
d = tempfile.mkdtemp()
forest.save(d)
loaded = LoRAForest.load(d, device=DEVICE)
restore_ok = all(leaf_hash(loaded.leaves[i]) == leaf_hash(forest.leaves[i]) for i in range(len(forest.leaves)))
print(f"  checkpoint save→load byte-exact ({len(forest.leaves)} experts): {'PASS' if restore_ok else 'FAIL'}")

print("\n" + "=" * 60)
ok = forget_ok and restore_ok
print(f"  Overall: {'PASS' if ok else 'FAIL'} — experts are LoRA adapters, isolation + checkpointing hold")
print("=" * 60)
import sys; sys.exit(0 if ok else 1)
