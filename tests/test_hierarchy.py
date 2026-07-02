"""HierarchicalDASForest — the specialty tree: two-level routing/provenance,
branch lifecycle, and the guarantees (leaf AND routing isolation)."""
import numpy as np
import pytest

from das.functional import softmax
from das.hierarchy import HierarchicalDASForest

D = 16
rng = np.random.default_rng(0)


def _cluster(seed, scale=4.0):
    r = np.random.default_rng(seed)
    c = r.normal(0, 1, D)
    return c / np.linalg.norm(c) * scale


def _ce_grad(logits, y):
    p = softmax(logits)
    oh = np.zeros_like(p)
    oh[np.arange(len(y)), y] = 1.0
    return (p - oh) / len(y)


def _train_leaf(leaf, X, y, steps=200):
    leaf.frozen = False
    r = np.random.default_rng(1)
    for _ in range(steps):
        i = r.integers(0, len(X), 32)
        leaf.backward(_ce_grad(leaf.forward(X[i]), y[i]), 0.05)
    leaf.frozen = True


@pytest.fixture
def tree():
    """react (2 leaves: hooks, state) + math (2 leaves: algebra, calculus),
    all leaves and both router levels trained on separable clusters."""
    t = HierarchicalDASForest(D, seed=3)
    data = {}
    for s, branch in enumerate(["react", "math"]):
        t.add_branch(branch, num_leaves=2, seed=10 + s)
        base = _cluster(100 + s, scale=5.0)
        for l in range(2):
            c = base + _cluster(200 + s * 10 + l, scale=1.5)
            X = c + rng.normal(0, 0.4, (120, D))
            y = (X @ _cluster(300 + s * 10 + l, scale=1.0) > 0).astype(int)
            data[(branch, l)] = (X, y)
            _train_leaf(t.branches[branch].leaves[l], X, y)
        # sub-router for this branch
        Xs = np.vstack([data[(branch, l)][0] for l in range(2)])
        ys = np.concatenate([np.full(120, l) for l in range(2)])
        r = np.random.default_rng(2)
        for _ in range(300):
            i = r.integers(0, len(Xs), 32)
            t.branches[branch].router.train_step(Xs[i], ys[i], lr=0.2)
    Xt = np.vstack([data[(b, l)][0] for b in ["react", "math"] for l in range(2)])
    yt = np.concatenate([np.full(240, s) for s in range(2)])
    t.train_router(Xt, yt, steps=400, seed=4)
    t._test_data = data
    return t


def test_add_branch_proves_others_intact():
    t = HierarchicalDASForest(D, seed=1)
    t.add_branch("react", num_leaves=2)
    _forest, intact = t.add_branch("math", num_leaves=3)
    assert intact is True
    assert t.branch_names == ["react", "math"]
    assert t.leaf_count() == 5
    with pytest.raises(ValueError, match="already exists"):
        t.add_branch("react")


def test_two_level_provenance(tree):
    X, _ = tree._test_data[("react", 0)]
    rows = tree.route_explain(X[:20])
    hits = sum(1 for r in rows if r["specialty"] == "react" and r["leaf"] == 0)
    assert hits >= 18                                # routed through both levels
    r = rows[0]
    assert set(r) == {"specialty", "specialty_confidence", "leaf",
                      "leaf_confidence", "prediction"}
    assert 0 < r["specialty_confidence"] <= 1 and 0 < r["leaf_confidence"] <= 1


def test_end_to_end_two_level_routing(tree):
    for (branch, leaf), (X, _y) in tree._test_data.items():
        rows = tree.route_explain(X[:30])
        acc = np.mean([r["specialty"] == branch and r["leaf"] == leaf for r in rows])
        assert acc > 0.8, f"{branch}/{leaf} routed at {acc}"


def test_routing_isolation_on_graft(tree):
    """THE tree guarantee the flat forest cannot make: grafting+training inside
    react leaves math's leaves AND math's router AND the top router untouched."""
    def train_fn(forest, idx):
        X, y = tree._test_data[("react", 0)]
        _train_leaf(forest.leaves[idx], X, y, steps=100)
        # even retraining react's own sub-router stays inside the branch
        r = np.random.default_rng(5)
        for _ in range(100):
            i = r.integers(0, len(X), 32)
            forest.router.train_step(X[i], np.full(32, idx), lr=0.1)

    idx, leaves_intact, routers_intact = tree.graft_leaf("react", train_fn, seed=42)
    assert idx == 2
    assert leaves_intact is True                     # math's leaves byte-identical
    assert routers_intact is True                    # math's router AND top router too


def test_prune_branch_survivors_byte_identical(tree):
    math_hashes = tree.hashes()["math"]
    result = tree.prune_branch("react")
    assert result["removed_leaves"] == 2
    assert result["survivors_byte_identical"] is True
    assert tree.branch_names == ["math"]
    assert tree.hashes()["math"] == math_hashes
    # routing falls through: everything now lands in math
    X, _ = tree._test_data[("math", 1)]
    rows = tree.route_explain(X[:10])
    assert all(r["specialty"] == "math" for r in rows)
    with pytest.raises(KeyError):
        tree.prune_branch("react")


def test_router_hashes_cover_top_and_branches(tree):
    h = tree.router_hashes()
    assert set(h) == {"react", "math", "__top__"}
    assert all(len(v) == 64 for v in h.values())


def test_d_model_mismatch_rejected():
    from das.model import DASForest
    t = HierarchicalDASForest(D)
    t.add_branch("a")
    with pytest.raises(ValueError, match="d_model"):
        t.add_branch("b", forest=DASForest(D + 2, [D + 2, 8, 2], num_leaves=1))
