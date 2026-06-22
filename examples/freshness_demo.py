"""
freshness_demo.py — refuse a rolled-back snapshot (closes SECURITY_REVIEW F1)
----------------------------------------------------------------------------
The signed audit chain proves a snapshot is *internally* consistent, not that it's
the *latest*. So an attacker who can write `DAS_STATE` can restore an older, valid
snapshot — silently undoing a deletion — and `verify()` + `state_matches_audit()`
both still pass. This demo shows the gap, then closes it with a FreshnessAnchor.

  1. save state, then delete a tenant (right-to-be-forgotten) and save again,
  2. attacker restores the OLD snapshot over DAS_STATE,
  3. WITHOUT an anchor: it loads clean — the deleted tenant is silently back,
  4. WITH an anchor (kept outside DAS_STATE): the load is REFUSED (RollbackDetected).

Pure NumPy, no downloads.
"""
import shutil
import tempfile
import os

import numpy as np
from das.model import DASForest
from das.functional import softmax
from das.governance import ControlPlane
from das.freshness import FreshnessAnchor, RollbackDetected

rng = np.random.default_rng(0)
D, LEAF, N = 16, [16, 13, 8, 2], 300
DATA = {}


def expert_data(key, n):
    c = np.zeros(D); c[(abs(hash(key)) % (D // 2)) * 2] = 4.0
    rule = np.random.default_rng(abs(hash(key)) % (2**31)).normal(0, 1, D)
    X = c + rng.normal(0, 1.0, (n, D))
    return X, (X @ rule > 0).astype(int)


def ce_grad(logits, y):
    p = softmax(logits); oh = np.zeros_like(p); oh[np.arange(len(y)), y] = 1.0
    return (p - oh) / len(y)


def train_fn(key, cp):
    DATA[key] = expert_data(key, N)
    def fn(forest, idx):
        leaf = forest.leaves[idx]; leaf.frozen = False
        X, y = DATA[key]
        for _ in range(250):
            i = rng.integers(0, len(X), 64); leaf.backward(ce_grad(leaf.forward(X[i]), y[i]), 0.05)
        leaf.frozen = True
        keys = [r["name"] for r in cp.experts] + [key]
        Xr = np.vstack([DATA[k][0] for k in keys])
        dr = np.concatenate([np.full(N, s) for s, _ in enumerate(keys)])
        for _ in range(350):
            i = rng.integers(0, len(Xr), 64); forest.router.train_step(Xr[i], dr[i], lr=0.15)
    return fn


def rule(s):
    print("\n" + "─" * 72 + f"\n{s}\n" + "─" * 72)


def main():
    work = tempfile.mkdtemp()
    STATE = os.path.join(work, "state")          # untrusted: the attacker can write here
    ANCHORP = os.path.join(work, "anchor.log")   # trusted: a SEPARATE store
    anchor = FreshnessAnchor(ANCHORP)

    rule("1 · Build a 2-tenant fleet, save (anchor records the chain tip)")
    DATA["acme-tax"] = expert_data("acme-tax", N)
    forest = DASForest(D, LEAF, num_leaves=1, seed=7)
    leaf = forest.leaves[0]; leaf.frozen = False
    X, y = DATA["acme-tax"]
    for _ in range(250):
        i = rng.integers(0, N, 64); leaf.backward(ce_grad(leaf.forward(X[i]), y[i]), 0.05)
    leaf.frozen = True
    cp = ControlPlane(forest, seed_tenant="acme", seed_name="acme-tax", secret="prod")
    cp.register_tenant("root", "globex")
    cp.graft("root", "globex", "globex-vision", train_fn("globex-vision", cp))
    cp.save(STATE, anchor=anchor)
    print(f"  tenants: {sorted(cp.tenants)}   audit entries: {len(cp.audit.entries)}")
    print(f"  anchor latest: {anchor.latest()}")

    rule("2 · Attacker backs up this snapshot, then we DELETE tenant 'globex'")
    backup = os.path.join(work, "backup")
    shutil.copytree(STATE, backup)               # attacker keeps the old snapshot
    cp.delete_tenant("root", "globex")           # right-to-be-forgotten
    cp.save(STATE, anchor=anchor)
    print(f"  tenants now: {sorted(cp.tenants)}   audit entries: {len(cp.audit.entries)}")
    print(f"  anchor latest: {anchor.latest()}  (advanced)")

    rule("3 · Attacker restores the OLD snapshot over DAS_STATE")
    shutil.rmtree(STATE); shutil.copytree(backup, STATE)
    print("  DAS_STATE rolled back to before the deletion.")

    rule("4 · Load WITHOUT an anchor — the rollback is invisible (the F1 gap)")
    rolled = ControlPlane.load(STATE, secret="prod")
    chain_ok = rolled.verify_audit("root")["ok"]
    bound_ok = rolled.state_matches_audit()
    print(f"  loaded tenants: {sorted(rolled.tenants)}  ← 'globex' is silently BACK")
    print(f"  audit chain ok: {chain_ok}   state↔audit bound: {bound_ok}  (both pass!)")

    rule("5 · Load WITH the anchor — the rollback is REFUSED")
    try:
        ControlPlane.load(STATE, secret="prod", anchor=anchor)
        print("  ✗ load succeeded — anchor failed to catch the rollback")
        caught = False
    except RollbackDetected as e:
        print(f"  ✓ RollbackDetected: {e}")
        caught = True

    rule("Done — freshness, not just consistency")
    print("  The anchor turns 'this snapshot is internally valid' into 'this snapshot")
    print("  is the latest committed one'. Keep it on a store the DAS_STATE writer")
    print("  cannot roll back. Pairs with Ed25519 (F7): authorship + recency.")

    ok = (chain_ok and bound_ok            # the gap really is invisible without the anchor
          and "globex" in rolled.tenants
          and caught)                       # and the anchor really catches it
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
