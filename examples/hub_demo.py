"""
hub_demo.py
-----------
The leaf marketplace in action: train an expert, PUBLISH it to a shared hub, then
have a BRAND-NEW forest PULL and graft it with zero retraining — verified
byte-identical by its fingerprint. This is "share/graft community leaves".
"""
import os
import tempfile
import numpy as np
from das.model import DASForest
from das.functional import FibonacciLeaf, softmax
from das import hub

rng = np.random.default_rng(0)
D, LEAF, N = 16, [16, 13, 8, 2], 400

def domain(did, n):
    center = np.eye(D)[did * 5] * 4
    rule = np.random.default_rng(100 + did).normal(0, 1, D)
    X = center + rng.normal(0, 1.0, (n, D))
    return X.astype(np.float64), (X @ rule > 0).astype(int)

def ce_grad(logits, y):
    p = softmax(logits); oh = np.zeros_like(p); oh[np.arange(len(y)), y] = 1.0
    return (p - oh) / len(y)

def acc(logits, y):
    return (logits.argmax(1) == y).mean()

HUB = os.path.join(tempfile.gettempdir(), "das_hub")

print("=" * 60)
print(" DAS leaf marketplace — publish, list, pull, graft")
print("=" * 60)

# Author A trains two experts and publishes them
for name, did, dom in [("math-v1", 0, "math"), ("vision-v1", 1, "vision")]:
    X, y = domain(did, N)
    leaf = FibonacciLeaf(LEAF, seed=did)
    for _ in range(400):
        i = rng.integers(0, N, 64); leaf.backward(ce_grad(leaf.forward(X[i]), y[i]), 0.05)
    leaf.frozen = True
    meta = hub.publish(leaf, name, HUB, domain=dom, author="alice")
    print(f"\n  published '{name}'  domain={dom}  hash={meta['hash']}  acc={acc(leaf.forward(X), y):.3f}")

# Author B browses the hub
print("\n  hub catalog:")
for name, m in hub.list_leaves(HUB).items():
    print(f"    {name:<12} domain={m['domain']:<8} by {m['author']}  {m['hash']}")

# Author B builds a fresh forest and grafts a community leaf — no retraining
print("\n  Author B: fresh forest, graft 'math-v1' from the hub (no retraining) ...")
forest = DASForest(D, LEAF, num_leaves=1, seed=99)   # B's own leaf 0 (untrained here)
gid = hub.graft_from_hub(forest, "math-v1", HUB)
Xtest, ytest = domain(0, 200)
pulled = forest.leaves[gid]
print(f"  grafted as leaf {gid}  | hash {pulled.weight_hash()}  | "
      f"matches published: {pulled.weight_hash() == hub.list_leaves(HUB)['math-v1']['hash']}")
print(f"  pulled leaf accuracy on math domain (zero retraining): {acc(pulled.forward(Xtest), ytest):.3f}")

# Integrity check: a tampered catalog hash is rejected on pull
print("\n  integrity: pulling with a corrupted fingerprint must fail ...")
idx = hub.list_leaves(HUB); idx["vision-v1"]["hash"] = "deadbeefdeadbeef"
hub._save_index(HUB, idx)
try:
    hub.pull("vision-v1", HUB); ok = False
except ValueError as e:
    ok = True; print(f"    rejected: {e}")

print("\n" + "=" * 60)
ok_all = ok and acc(pulled.forward(Xtest), ytest) > 0.8
print(f"  Marketplace: publish/list/pull/graft + hash verification: {'PASS' if ok_all else 'FAIL'}")
print("=" * 60)
import sys; sys.exit(0 if ok_all else 1)
