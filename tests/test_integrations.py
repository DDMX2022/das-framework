"""
Integration-adapter tests (das/integrations). The DAS node itself is pure NumPy
and runs in CI; the langgraph-compiled path is skipped unless langgraph is
installed.
"""
import numpy as np
import pytest

from das.model import DASForest
from das.functional import softmax
from das.governance import ControlPlane
from das.integrations import DASExpertNode, build_graph

D, LEAF, N = 16, [16, 13, 8, 2], 120
rng = np.random.default_rng(0)

# each expert gets its own well-separated cluster center (distinct dim), so the
# router can actually discriminate them — we're testing provenance, not routing.
_CENTERS = {"acme-0": 0, "globex-0": 6}


def _data(key):
    c = np.zeros(D); c[_CENTERS[key]] = 6.0
    rule = np.random.default_rng(abs(hash(key)) % (2**31)).normal(0, 1, D)
    X = c + rng.normal(0, 0.6, (N, D))
    return X, (X @ rule > 0).astype(int)


def _ce(logits, y):
    p = softmax(logits); oh = np.zeros_like(p); oh[np.arange(len(y)), y] = 1.0
    return (p - oh) / len(y)


def _train_fn(key, cp):
    """Train the new leaf in isolation, then retrain the router over every
    registered expert so routing actually discriminates them."""
    X, y = _data(key)
    DATA[key] = (X, y)

    def fn(forest, idx):
        leaf = forest.leaves[idx]; leaf.frozen = False
        for _ in range(120):
            i = rng.integers(0, N, 32); leaf.backward(_ce(leaf.forward(X[i]), y[i]), 0.05)
        leaf.frozen = True
        # graft() appends the new expert record AFTER this callback, so include the
        # expert being grafted (it lands at leaf index `idx`) when retraining router.
        keys = [r["name"] for r in cp.experts] + [key]
        Xr = np.vstack([DATA[k][0] for k in keys])
        dr = np.concatenate([np.full(N, s) for s, _ in enumerate(keys)])
        for _ in range(800):
            i = rng.integers(0, len(Xr), 64); forest.router.train_step(Xr[i], dr[i], lr=0.25)
    return fn


DATA = {}


def _seed_cp():
    forest = DASForest(D, LEAF, num_leaves=1, seed=7)
    DATA["acme-0"] = _data("acme-0")
    leaf = forest.leaves[0]; leaf.frozen = False
    X, y = DATA["acme-0"]
    for _ in range(200):
        i = rng.integers(0, N, 32); leaf.backward(_ce(leaf.forward(X[i]), y[i]), 0.05)
    leaf.frozen = True
    cp = ControlPlane(forest, seed_tenant="acme", seed_name="acme-0")
    cp.register_tenant("root", "globex")
    cp.add_user("root", "alice", "operator", tenant="acme")
    cp.add_user("root", "carol", "auditor")        # no 'predict' permission
    cp.graft("root", "globex", "globex-0", _train_fn("globex-0", cp))
    return cp


def test_node_writes_provenance():
    cp = _seed_cp()
    node = DASExpertNode(cp)
    # a query drawn from globex's cluster should be served by a globex expert
    X, _ = DATA["globex-0"]
    upd = node({"embedding": X[0], "actor": "root"})
    assert upd["das_denied"] is False
    assert upd["das_tenant"] == "globex"
    assert upd["das_expert"] == "globex-0"
    assert 0.0 <= upd["das_confidence"] <= 1.0
    assert len(upd["das_prediction"]) == LEAF[-1]
    assert upd["das_actor"] == "root"


def test_node_surfaces_denial_as_state():
    cp = _seed_cp()
    node = DASExpertNode(cp)
    n0 = len(cp.audit.entries)
    upd = node({"embedding": DATA["acme-0"][0][0], "actor": "carol"})  # auditor: no predict
    assert upd["das_denied"] is True and "predict" in upd["das_denied_reason"]
    assert "das_prediction" not in upd
    # the denial was recorded tamper-evidently, and the chain still verifies
    assert len(cp.audit.entries) == n0 + 1
    assert cp.audit.entries[-1]["event"] == "denied"
    assert cp.audit.verify()[0] is True


def test_node_can_raise_on_denied():
    from das.governance import AccessDenied
    cp = _seed_cp()
    node = DASExpertNode(cp, raise_on_denied=True)
    with pytest.raises(AccessDenied):
        node({"embedding": DATA["acme-0"][0][0], "actor": "carol"})


def test_default_actor_used_when_absent():
    cp = _seed_cp()
    node = DASExpertNode(cp, default_actor="root")
    upd = node({"embedding": DATA["acme-0"][0][0]})   # no 'actor' in state
    assert upd["das_actor"] == "root" and upd["das_denied"] is False


def test_build_graph_when_langgraph_present():
    pytest.importorskip("langgraph")
    cp = _seed_cp()
    graph = build_graph(cp)
    out = graph.invoke({"embedding": list(map(float, DATA["globex-0"][0][0])), "actor": "root"})
    assert out["das_tenant"] in ("acme", "globex")
    assert out["das_denied"] is False
