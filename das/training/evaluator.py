"""Evaluation and isolated candidate-training helpers."""

from dataclasses import dataclass

import numpy as np

from das.functional import FibonacciLeaf, softmax


@dataclass
class ExpertEvalSet:
    """Frozen evaluation set for one expert id."""

    eid: int
    name: str
    X: np.ndarray
    y: np.ndarray


def clone_leaf(leaf):
    """Deep-copy a NumPy FibonacciLeaf so live weights are not trained directly."""
    clone = FibonacciLeaf(leaf.dims)
    clone.W = [w.copy() for w in leaf.W]
    clone.b = [b.copy() for b in leaf.b]
    clone.frozen = bool(leaf.frozen)
    return clone


def accuracy(logits, y):
    return float((np.argmax(logits, axis=1) == y).mean())


def _ce_grad(logits, y):
    p = softmax(logits)
    oh = np.zeros_like(p)
    oh[np.arange(len(y)), y] = 1.0
    return (p - oh) / len(y)


def evaluate_leaf(leaf, X, y):
    if len(X) == 0:
        return 0.0
    return accuracy(leaf.forward(X), y)


def train_leaf(leaf, X, y, steps=120, lr=0.05, batch=32, seed=0, momentum=0.0):
    """Train one candidate leaf in isolation and return it frozen.

    ``momentum`` > 0 switches to SGD-with-momentum (velocity on every W/b) —
    plain SGD on tiny nets over nonconvex curricula is init-sensitive, and
    momentum tightens the spread. Default 0 keeps the original behaviour."""
    rng = np.random.default_rng(seed)
    leaf.frozen = False
    n = len(X)
    if momentum > 0:
        vW = [np.zeros_like(w) for w in leaf.W]
        vb = [np.zeros_like(b) for b in leaf.b]
        for _ in range(int(steps)):
            idx = rng.integers(0, n, min(int(batch), n))
            gW, gb = leaf.grads(_ce_grad(leaf.forward(X[idx]), y[idx]))
            for i in range(len(leaf.W)):
                vW[i] = momentum * vW[i] - lr * gW[i]
                vb[i] = momentum * vb[i] - lr * gb[i]
                leaf.W[i] += vW[i]
                leaf.b[i] += vb[i]
    else:
        for _ in range(int(steps)):
            idx = rng.integers(0, n, min(int(batch), n))
            leaf.backward(_ce_grad(leaf.forward(X[idx]), y[idx]), lr)
    leaf.frozen = True
    return leaf


def _eid_to_index(control_plane):
    return {r["eid"]: i for i, r in enumerate(control_plane.experts)}


def evaluate_suite(control_plane, eval_sets):
    """Directly score each expert on its own frozen eval set."""
    eid_to_index = _eid_to_index(control_plane)
    scores = {}
    for ev in eval_sets or []:
        if ev.eid not in eid_to_index:
            continue
        leaf = control_plane.forest.leaves[eid_to_index[ev.eid]]
        scores[str(ev.eid)] = evaluate_leaf(leaf, ev.X, ev.y)
    return scores


def evaluate_candidate_suite(control_plane, target_eid, candidate_leaf, eval_sets):
    """Score a hypothetical replacement while leaving the live forest untouched."""
    eid_to_index = _eid_to_index(control_plane)
    scores = {}
    for ev in eval_sets or []:
        if ev.eid not in eid_to_index:
            continue
        leaf = candidate_leaf if ev.eid == target_eid else control_plane.forest.leaves[eid_to_index[ev.eid]]
        scores[str(ev.eid)] = evaluate_leaf(leaf, ev.X, ev.y)
    return scores


def router_accuracy(control_plane, eval_sets):
    """How often the current router selects the expert owning each eval set."""
    eval_sets = eval_sets or []
    eid_to_index = _eid_to_index(control_plane)
    Xs, labels = [], []
    for ev in eval_sets:
        if ev.eid not in eid_to_index:
            continue
        Xs.append(ev.X)
        labels.append(np.full(len(ev.X), eid_to_index[ev.eid], dtype=int))
    if not Xs:
        return None
    X = np.vstack(Xs)
    y = np.concatenate(labels)
    routed, _ = control_plane.forest.router.route(X)
    return float((routed == y).mean())
