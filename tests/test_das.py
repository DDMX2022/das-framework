"""
Core guarantees, as tests. Pure-NumPy (no torch) so CI is fast.
Run: pytest -q
"""
import os
import tempfile
import numpy as np
import pytest

from das.model import DASForest
from das.functional import FibonacciLeaf, softmax
from das.lifecycle import ForestLifecycle
from das.audit import AuditLog
from das import hub

def ce_grad(logits, y):
    p = softmax(logits); oh = np.zeros_like(p); oh[np.arange(len(y)), y] = 1.0
    return (p - oh) / len(y)

def _train(leaf, X, y, rng, steps=120):
    leaf.frozen = False
    for _ in range(steps):
        i = rng.integers(0, len(X), 32)
        leaf.backward(ce_grad(leaf.forward(X[i]), y[i]), 0.05)
    leaf.frozen = True

def test_zero_forgetting_on_graft():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (120, 8)); y = (X[:, 0] > 0).astype(int)
    f = DASForest(8, [8, 6, 2], num_leaves=2, seed=1)
    _train(f.leaves[0], X, y, rng)
    h0, h1 = f.leaves[0].weight_hash(), f.leaves[1].weight_hash()
    nid = f.graft(seed=2)
    _train(f.leaves[nid], X, y, rng)
    assert f.leaves[0].weight_hash() == h0   # byte-identical after grafting+training a new leaf
    assert f.leaves[1].weight_hash() == h1

def test_prune_preserves_survivors():
    f = DASForest(8, [8, 6, 2], num_leaves=3, seed=1)
    life = ForestLifecycle(f)
    before = [l.weight_hash() for l in f.leaves]
    life.prune(2)
    assert len(f.leaves) == 2
    assert [l.weight_hash() for l in f.leaves] == before[:2]

def test_canopy_shapes():
    f = DASForest(8, [8, 6, 2], num_leaves=3, seed=1)
    X = np.random.default_rng(0).normal(0, 1, (10, 8))
    out, idx = f.predict_canopy(X, k=2)
    assert out.shape == (10, 2) and idx.shape == (10, 2)

def test_audit_detects_tampering():
    log = AuditLog("secret")
    log.append("graft", "added math")
    log.append("graft", "added vision")
    log.append("prune", "removed math")
    ok, idx, _ = log.verify()
    assert ok and idx == -1
    log.entries[1]["detail"] = "added EVIL"      # tamper
    ok, idx, _ = log.verify()
    assert (not ok) and idx == 1

def test_audit_detects_deletion():
    log = AuditLog("secret")
    for k in range(4):
        log.append("e", f"d{k}")
    del log.entries[2]                            # remove an entry
    ok, idx, _ = log.verify()
    assert not ok

def test_hub_roundtrip_and_integrity():
    leaf = FibonacciLeaf([8, 6, 2], seed=3); leaf.frozen = True
    d = tempfile.mkdtemp()
    hub.publish(leaf, "x", d)
    pulled = hub.pull("x", d)
    assert pulled.weight_hash() == leaf.weight_hash()    # byte-exact restore
    idx = hub.list_leaves(d); idx["x"]["hash"] = "deadbeefdeadbeef"; hub._save_index(d, idx)
    with pytest.raises(ValueError):                       # tampered fingerprint rejected
        hub.pull("x", d)
