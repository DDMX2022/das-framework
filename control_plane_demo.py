"""
control_plane_demo.py
---------------------
The governance CONTROL PLANE made tangible (das/governance.py). Shows the three
things that make DAS a *governed* fleet rather than just a sparse model:

  1. RBAC          — roles + tenant scoping decide who may graft/prune/delete;
                     denied attempts are themselves recorded.
  2. MULTI-TENANCY — experts belong to tenants; one tenant's operator cannot
                     touch another's experts.
  3. DELETE TENANT — right-to-be-forgotten removes only that tenant's experts
                     and proves every other tenant is byte-identical.

Plus the tamper-evident audit log catches any after-the-fact edit.
Pure NumPy, synthetic data, no downloads.
"""
import numpy as np
from das.model import DASForest
from das.functional import softmax
from das.governance import ControlPlane, AccessDenied

rng = np.random.default_rng(0)
D, LEAF, N = 16, [16, 13, 8, 2], 300

# each expert is a distinct cluster + a random linear rule -> binary label
def expert_data(key, n):
    c = np.zeros(D); c[(abs(hash(key)) % (D // 2)) * 2] = 4.0
    rule = np.random.default_rng(abs(hash(key)) % (2**31)).normal(0, 1, D)
    X = c + rng.normal(0, 1.0, (n, D))
    return X, (X @ rule > 0).astype(int)

def ce_grad(logits, y):
    p = softmax(logits); oh = np.zeros_like(p); oh[np.arange(len(y)), y] = 1.0
    return (p - oh) / len(y)

# registry of every expert's data, so the router can be retrained on graft
DATA = {}
def make_train_fn(key, cp):
    """Returns train_fn(forest, leaf_index): train that leaf in isolation, then
    retrain the (non-isolated) router over all currently-registered experts."""
    DATA[key] = expert_data(key, N)
    def train_fn(forest, idx):
        leaf = forest.leaves[idx]; leaf.frozen = False
        X, y = DATA[key]
        for _ in range(400):
            i = rng.integers(0, len(X), 64); leaf.backward(ce_grad(leaf.forward(X[i]), y[i]), 0.05)
        leaf.frozen = True
        keys = [r["name"] for r in cp.experts]            # router slots, in leaf order
        Xr = np.vstack([DATA[k][0] for k in keys])
        dr = np.concatenate([np.full(N, s) for s, _ in enumerate(keys)])
        for _ in range(600):
            i = rng.integers(0, len(Xr), 64); forest.router.train_step(Xr[i], dr[i], lr=0.15)
    return train_fn

print("=" * 66)
print(" DAS governance control plane — RBAC · multi-tenancy · audit")
print("=" * 66)

# ── seed: tenant 'acme' onboards with its first expert ──────────────
forest = DASForest(D, LEAF, num_leaves=1, seed=7)
DATA["acme-math"] = expert_data("acme-math", N)
leaf = forest.leaves[0]; leaf.frozen = False
for _ in range(400):
    i = rng.integers(0, N, 64); leaf.backward(ce_grad(leaf.forward(DATA["acme-math"][0][i]), DATA["acme-math"][1][i]), 0.05)
leaf.frozen = True

cp = ControlPlane(forest, seed_tenant="acme", seed_name="acme-math")
print("\n[admin] root admin bootstrapped; seed expert 'acme-math' (tenant acme)")

# ── admin sets up tenants + scoped users ────────────────────────────
cp.register_tenant("root", "globex")
cp.add_user("root", "alice", role="operator", tenant="acme")     # scoped to acme
cp.add_user("root", "bob",   role="operator", tenant="globex")   # scoped to globex
cp.add_user("root", "carol", role="auditor")                      # read-only
cp.add_user("root", "dave",  role="viewer")                       # predict + read
print("[admin] tenants: acme, globex; users: alice(op/acme) bob(op/globex) carol(auditor) dave(viewer)")

# ── RBAC in action ──────────────────────────────────────────────────
eid_lang = cp.graft("alice", "acme", "acme-language", make_train_fn("acme-language", cp))
print(f"\n[alice] grafted 'acme-language' for acme  -> eid {eid_lang}  ✓ allowed")

eid_vis = cp.graft("bob", "globex", "globex-vision", make_train_fn("globex-vision", cp))
print(f"[bob]   grafted 'globex-vision' for globex -> eid {eid_vis}  ✓ allowed")

rbac = {}
try:
    cp.graft("alice", "globex", "x", make_train_fn("x", cp))           # cross-tenant
except AccessDenied as e:
    rbac["cross_tenant"] = True; print(f"[alice] graft into globex          -> ✗ DENIED ({e})")
try:
    cp.prune("dave", eid_lang)                                          # viewer can't prune
except AccessDenied as e:
    rbac["viewer_prune"] = True; print(f"[dave]  prune acme-language        -> ✗ DENIED ({e})")
try:
    cp.add_user("carol", "mallory", "admin")                            # auditor can't manage
except AccessDenied as e:
    rbac["auditor_manage"] = True; print(f"[carol] add admin user            -> ✗ DENIED ({e})")

# carol (auditor) verifies the log — including the denied attempts
v = cp.verify_audit("carol")
print(f"\n[carol] audit verify: ok={v['ok']}  entries={v['entries']} (denials are logged too)")

# ── right-to-be-forgotten at tenant granularity ─────────────────────
globex_before = [r for r in cp.experts if r["tenant"] == "globex"]
globex_hashes = {r["eid"]: forest.leaves[i].weight_hash()
                 for i, r in enumerate(cp.experts) if r["tenant"] == "globex"}
res = cp.delete_tenant("root", "acme")
globex_after = {r["eid"]: forest.leaves[i].weight_hash()
                for i, r in enumerate(cp.experts) if r["tenant"] == "globex"}
forgotten = all(globex_after.get(eid) == h for eid, h in globex_hashes.items())
print(f"\n[admin] delete tenant 'acme': removed {res['removed']} experts; "
      f"globex byte-identical: {res['non_interference']}")

# ── tamper detection ────────────────────────────────────────────────
cp.audit.entries[1]["detail"] += " (tampered)"
ok, idx, reason = cp.audit.verify()
print(f"[tamper] edited audit entry 1 -> verify now ok={ok} at index {idx}: {reason}")

print("\n" + "=" * 66)
passed = (all(rbac.get(k) for k in ("cross_tenant", "viewer_prune", "auditor_manage"))
          and res["non_interference"] and forgotten and v["ok"] and not ok)
print(f"  RBAC denials enforced:        {'PASS' if all(rbac.get(k) for k in ('cross_tenant','viewer_prune','auditor_manage')) else 'FAIL'}")
print(f"  Tenant deletion + isolation:  {'PASS' if res['non_interference'] and forgotten else 'FAIL'}")
print(f"  Audit verify then tamper:     {'PASS' if v['ok'] and not ok else 'FAIL'}")
print(f"  Overall control-plane proof:  {'PASS' if passed else 'FAIL'}")
print("=" * 66)
import sys; sys.exit(0 if passed else 1)
