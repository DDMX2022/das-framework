"""HierarchicalDASNode — the specialty tree as an agent node: two-level
provenance in graph state + per-branch escalation policy."""
import numpy as np
import pytest

from das.functional import softmax
from das.hierarchy import HierarchicalDASForest
from das.integrations import HierarchicalDASNode

D = 16
rng = np.random.default_rng(0)


def _cluster(seed, scale):
    r = np.random.default_rng(seed)
    c = r.normal(0, 1, D)
    return c / np.linalg.norm(c) * scale


def _ce_grad(logits, y):
    p = softmax(logits)
    oh = np.zeros_like(p)
    oh[np.arange(len(y)), y] = 1.0
    return (p - oh) / len(y)


@pytest.fixture(scope="module")
def tree_and_probe():
    """react/math tree, trained; returns (tree, {(branch, leaf): probe_vector})."""
    t = HierarchicalDASForest(D, seed=3)
    probes = {}
    for s, branch in enumerate(["react", "math"]):
        t.add_branch(branch, num_leaves=2, seed=10 + s)
        base = _cluster(100 + s, 5.0)
        Xs, ys = [], []
        for l in range(2):
            c = base + _cluster(200 + s * 10 + l, 1.5)
            X = c + rng.normal(0, 0.4, (100, D))
            y = (X @ _cluster(300 + s * 10 + l, 1.0) > 0).astype(int)
            probes[(branch, l)] = c
            leaf = t.branches[branch].leaves[l]
            leaf.frozen = False
            r = np.random.default_rng(1)
            for _ in range(150):
                i = r.integers(0, len(X), 32)
                leaf.backward(_ce_grad(leaf.forward(X[i]), y[i]), 0.05)
            leaf.frozen = True
            Xs.append(X)
            ys.append(np.full(100, l))
        Xb, yb = np.vstack(Xs), np.concatenate(ys)
        r = np.random.default_rng(2)
        for _ in range(250):
            i = r.integers(0, len(Xb), 32)
            t.branches[branch].router.train_step(Xb[i], yb[i], lr=0.2)
    Xt = np.vstack([probes[(b, l)] + rng.normal(0, 0.4, (100, D))
                    for b in ["react", "math"] for l in range(2)])
    yt = np.concatenate([np.full(200, s) for s in range(2)])
    t.train_router(Xt, yt, steps=400, seed=4)
    return t, probes


def test_node_writes_two_level_provenance(tree_and_probe):
    tree, probes = tree_and_probe
    node = HierarchicalDASNode(tree, frontier="claude-sonnet-5")
    out = node({"embedding": probes[("react", 1)].tolist()})
    assert out["das_specialty"] == "react" and out["das_leaf"] == 1
    assert 0 < out["das_specialty_confidence"] <= 1
    assert 0 < out["das_leaf_confidence"] <= 1
    assert out["das_decision"] == "local" and out["das_frontier"] is None
    assert isinstance(out["das_prediction"], list)


def test_per_branch_escalation_policy(tree_and_probe):
    tree, probes = tree_and_probe
    # react is a regulated domain here: it demands impossible confidence
    node = HierarchicalDASNode(tree, confidence_threshold=0.0,
                               branch_thresholds={"react": 1.01},
                               frontier="claude-sonnet-5")
    react = node({"embedding": probes[("react", 0)]})
    math_ = node({"embedding": probes[("math", 0)]})
    assert react["das_decision"] == "escalate"
    assert react["das_frontier"] == "claude-sonnet-5"
    assert math_["das_decision"] == "local"           # default threshold applies


def test_escalates_when_either_level_is_unsure(tree_and_probe):
    tree, probes = tree_and_probe
    node = HierarchicalDASNode(tree, confidence_threshold=1.01)
    out = node({"embedding": probes[("math", 1)]})
    assert out["das_decision"] == "escalate"          # softmax < 1.01 always


def test_langgraph_compilation_if_installed(tree_and_probe):
    pytest.importorskip("langgraph")
    from das.integrations import build_hierarchical_graph
    tree, probes = tree_and_probe
    graph = build_hierarchical_graph(tree, frontier="f")
    out = graph.invoke({"embedding": probes[("react", 0)].tolist()})
    assert out["das_specialty"] == "react"