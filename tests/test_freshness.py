"""
Tests for the freshness / rollback anchor (SECURITY_REVIEW F1).
Pure stdlib + numpy (base dep) — runs in CI.
"""
import os
import shutil
import tempfile

import numpy as np
import pytest

from das.model import DASForest
from das.functional import softmax
from das.governance import ControlPlane
from das.freshness import FreshnessAnchor, RollbackDetected


def _trained_forest():
    rng = np.random.default_rng(0)
    D, LEAF, N = 12, [12, 8, 2], 200
    c = np.zeros(D); c[0] = 4.0
    X = c + rng.normal(0, 1.0, (N, D))
    rule = rng.normal(0, 1, D)
    y = (X @ rule > 0).astype(int)
    forest = DASForest(D, LEAF, num_leaves=1, seed=7)
    leaf = forest.leaves[0]; leaf.frozen = False

    def ce(logits, yy):
        p = softmax(logits); oh = np.zeros_like(p); oh[np.arange(len(yy)), yy] = 1.0
        return (p - oh) / len(yy)
    for _ in range(120):
        i = rng.integers(0, N, 32); leaf.backward(ce(leaf.forward(X[i]), y[i]), 0.05)
    leaf.frozen = True
    return forest


# ── the anchor primitive ───────────────────────────────────────────────
def test_anchor_record_latest_and_check():
    d = tempfile.mkdtemp()
    a = FreshnessAnchor(os.path.join(d, "anchor.log"))
    assert a.latest() is None
    assert a.check([])[0] is True              # no anchor yet → ok
    a.record(0, "sigA")
    a.record(1, "sigB")
    assert a.latest() == (1, "sigB")
    # a chain that contains sigB at index 1 and is long enough → fresh
    entries = [{"sig": "sigA"}, {"sig": "sigB"}, {"sig": "sigC"}]
    assert a.check(entries)[0] is True


def test_anchor_detects_rollback_and_fork():
    d = tempfile.mkdtemp()
    a = FreshnessAnchor(os.path.join(d, "anchor.log"))
    a.record(2, "head2")
    assert a.check([{"sig": "x"}, {"sig": "y"}])[0] is False        # too short → rollback
    ok, reason = a.check([{"sig": "x"}, {"sig": "y"}, {"sig": "DIFFERENT"}])
    assert ok is False and "fork" in reason                         # wrong head at index 2
    with pytest.raises(RollbackDetected):
        a.enforce([{"sig": "x"}])


# ── end-to-end through the control plane ────────────────────────────────
def _graft_fn(forest, idx):
    """Minimal isolated training so a grafted leaf is real (enough to delete)."""
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (60, 12)); y = (X[:, 0] > 0).astype(int)
    leaf = forest.leaves[idx]; leaf.frozen = False

    def ce(logits, yy):
        p = softmax(logits); oh = np.zeros_like(p); oh[np.arange(len(yy)), yy] = 1.0
        return (p - oh) / len(yy)
    for _ in range(40):
        i = rng.integers(0, 60, 16); leaf.backward(ce(leaf.forward(X[i]), y[i]), 0.05)
    leaf.frozen = True


def test_load_refuses_rolled_back_state():
    work = tempfile.mkdtemp()
    state = os.path.join(work, "state")
    anchor = FreshnessAnchor(os.path.join(work, "anchor.log"))

    cp = ControlPlane(_trained_forest(), seed_tenant="acme", seed_name="acme-tax", secret="k")
    cp.register_tenant("root", "globex")
    cp.graft("root", "globex", "globex-vision", _graft_fn)
    cp.save(state, anchor=anchor)
    backup = os.path.join(work, "backup")
    shutil.copytree(state, backup)             # old snapshot

    cp.delete_tenant("root", "globex")
    cp.save(state, anchor=anchor)              # anchor advances

    shutil.rmtree(state); shutil.copytree(backup, state)   # roll back DAS_STATE

    # without the anchor the stale snapshot loads clean (documents the F1 gap)
    rolled = ControlPlane.load(state, secret="k")
    assert rolled.verify_audit("root")["ok"] is True
    assert rolled.state_matches_audit() is True

    # with the anchor it is refused
    with pytest.raises(RollbackDetected):
        ControlPlane.load(state, secret="k", anchor=anchor)


def test_load_current_state_with_anchor_ok():
    work = tempfile.mkdtemp()
    state = os.path.join(work, "state")
    anchor = FreshnessAnchor(os.path.join(work, "anchor.log"))
    cp = ControlPlane(_trained_forest(), seed_tenant="acme", seed_name="acme-tax", secret="k")
    cp.save(state, anchor=anchor)
    reloaded = ControlPlane.load(state, secret="k", anchor=anchor)   # must not raise
    assert reloaded.verify_audit("root")["ok"] is True


def test_anchor_is_append_only_across_saves():
    work = tempfile.mkdtemp()
    state = os.path.join(work, "state")
    path = os.path.join(work, "anchor.log")
    anchor = FreshnessAnchor(path)
    cp = ControlPlane(_trained_forest(), seed_tenant="acme", seed_name="acme-tax", secret="k")
    cp.save(state, anchor=anchor)
    cp.register_tenant("root", "globex")
    cp.save(state, anchor=anchor)
    lines = [l for l in open(path).read().splitlines() if l.strip()]
    assert len(lines) == 2                      # one appended per save, never rewritten
