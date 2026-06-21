"""
Governance control-plane tests (das/governance.py). Pure NumPy — runs in CI.
"""
import numpy as np
import pytest

from das.model import DASForest
from das.functional import softmax
from das.governance import ControlPlane, AccessDenied

D, LEAF, N = 16, [16, 13, 8, 2], 120
rng = np.random.default_rng(0)


def _data(key):
    c = np.zeros(D); c[(abs(hash(key)) % (D // 2)) * 2] = 4.0
    rule = np.random.default_rng(abs(hash(key)) % (2**31)).normal(0, 1, D)
    X = c + rng.normal(0, 1.0, (N, D))
    return X, (X @ rule > 0).astype(int)


def _ce(logits, y):
    p = softmax(logits); oh = np.zeros_like(p); oh[np.arange(len(y)), y] = 1.0
    return (p - oh) / len(y)


def _train_fn(key):
    X, y = _data(key)
    def fn(forest, idx):
        leaf = forest.leaves[idx]; leaf.frozen = False
        for _ in range(80):
            i = rng.integers(0, N, 32); leaf.backward(_ce(leaf.forward(X[i]), y[i]), 0.05)
        leaf.frozen = True
    return fn


def _seed_cp():
    forest = DASForest(D, LEAF, num_leaves=1, seed=7)
    _train_fn("acme-0")(forest, 0)
    cp = ControlPlane(forest, seed_tenant="acme", seed_name="acme-0")
    cp.register_tenant("root", "globex")
    cp.add_user("root", "alice", "operator", tenant="acme")
    cp.add_user("root", "bob", "operator", tenant="globex")
    cp.add_user("root", "carol", "auditor")
    cp.add_user("root", "dave", "viewer")
    return cp


def test_rbac_role_denials():
    cp = _seed_cp()
    eid = cp.graft("alice", "acme", "acme-1", _train_fn("acme-1"))
    with pytest.raises(AccessDenied):
        cp.prune("dave", eid)            # viewer lacks prune
    with pytest.raises(AccessDenied):
        cp.add_user("carol", "x", "admin")  # auditor lacks manage
    with pytest.raises(AccessDenied):
        cp.graft("zzz", "acme", "y", _train_fn("y"))  # unknown user


def test_rbac_tenant_scope():
    cp = _seed_cp()
    with pytest.raises(AccessDenied):
        cp.graft("alice", "globex", "x", _train_fn("x"))  # acme operator can't touch globex
    # but the matching operator can
    assert isinstance(cp.graft("bob", "globex", "globex-0", _train_fn("globex-0")), int)


def test_graft_non_interference():
    cp = _seed_cp()
    before = [cp.forest.leaves[i].weight_hash() for i in range(len(cp.experts))]
    cp.graft("alice", "acme", "acme-1", _train_fn("acme-1"))
    after = [cp.forest.leaves[i].weight_hash() for i in range(len(before))]
    assert after == before          # existing experts byte-identical


def test_delete_tenant_isolation():
    cp = _seed_cp()
    cp.graft("alice", "acme", "acme-1", _train_fn("acme-1"))
    cp.graft("bob", "globex", "globex-0", _train_fn("globex-0"))
    globex = {r["eid"]: cp.forest.leaves[i].weight_hash()
              for i, r in enumerate(cp.experts) if r["tenant"] == "globex"}
    res = cp.delete_tenant("root", "acme")
    assert res["removed"] == 2 and res["non_interference"] is True
    after = {r["eid"]: cp.forest.leaves[i].weight_hash()
             for i, r in enumerate(cp.experts) if r["tenant"] == "globex"}
    assert after == globex                       # globex untouched
    assert all(r["tenant"] == "globex" for r in cp.experts)  # acme fully gone


def test_denials_are_audited_and_chain_holds():
    cp = _seed_cp()
    n0 = len(cp.audit.entries)
    with pytest.raises(AccessDenied):
        cp.prune("dave", 0)
    assert len(cp.audit.entries) == n0 + 1
    assert cp.audit.entries[-1]["event"] == "denied"
    ok, _, _ = cp.audit.verify()
    assert ok                                    # chain still valid after denial


def test_audit_tamper_detected():
    cp = _seed_cp()
    cp.graft("alice", "acme", "acme-1", _train_fn("acme-1"))
    assert cp.verify_audit("carol")["ok"] is True
    cp.audit.entries[1]["detail"] += " (tampered)"
    assert cp.audit.verify()[0] is False


def test_list_experts_tenant_scoped():
    cp = _seed_cp()
    cp.graft("bob", "globex", "globex-0", _train_fn("globex-0"))
    assert {r["tenant"] for r in cp.list_experts("root")} == {"acme", "globex"}  # global
    assert {r["tenant"] for r in cp.list_experts("alice")} == {"acme"}            # scoped
