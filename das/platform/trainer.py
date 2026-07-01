"""
das/platform/trainer.py
-----------------------
The default expert trainer for the deployment engine.

``ControlPlane.graft`` is backend-agnostic: it takes a ``train_fn(forest, idx)``
callback and never trains anything itself. This module supplies the *default*
callback so ``das deploy`` runs end-to-end with only NumPy — every expert gets a
real, isolated, deterministically-trained leaf, and the router is retrained over
all registered experts on each graft.

It is deliberately synthetic: each expert owns a distinct, separable data cluster
derived deterministically from its name (via SHA-256, NOT Python's salted
``hash``), so deployments are reproducible run-to-run. In production an FDE swaps
this for a teacher-backed trainer (``das.training``) behind the same seam — the
governance guarantees are identical either way.
"""
from __future__ import annotations

import hashlib

import numpy as np

from das.functional import softmax
from das.model import DASForest


def _stable_seed(text: str) -> int:
    """Deterministic 31-bit seed from a string (reproducible across processes)."""
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) % (2 ** 31)


def _ce_grad(logits, y):
    p = softmax(logits)
    oh = np.zeros_like(p)
    oh[np.arange(len(y)), y] = 1.0
    return (p - oh) / len(y)


class SyntheticTrainer:
    """Deterministic, dependency-free expert trainer.

    Owns the mapping from an expert *name* to its (reproducible) training data and
    cluster center, so it can (a) seed the first expert, (b) provide a graft
    ``train_fn`` for each subsequent expert, and (c) tell a connector where an
    expert's center is (for sensible demo routing).
    """

    def __init__(self, d_model: int, leaf_dims, n: int = 300,
                 leaf_steps: int = 400, router_steps: int = 600,
                 leaf_lr: float = 0.05, router_lr: float = 0.15):
        self.d_model = d_model
        self.leaf_dims = list(leaf_dims)
        self.n = n
        self.leaf_steps = leaf_steps
        self.router_steps = router_steps
        self.leaf_lr = leaf_lr
        self.router_lr = router_lr

    # ── deterministic per-expert data ────────────────────────────────
    def center(self, name: str) -> np.ndarray:
        """This expert's cluster center — a deterministic, near-orthogonal dense
        vector derived from the name. Random directions in d_model space are
        well-separated (unlike one-hot axes, which collide once you have more
        experts than dimensions), so the router can route many experts apart."""
        rng = np.random.default_rng(_stable_seed("center:" + name))
        v = rng.normal(0, 1.0, self.d_model)
        v /= np.linalg.norm(v) + 1e-9
        return v * 4.0

    def rule(self, name: str) -> np.ndarray:
        """This expert's fixed linear decision rule (maps a point to a binary
        label). Deterministic per name."""
        return np.random.default_rng(_stable_seed("rule:" + name)).normal(0, 1, self.d_model)

    def data(self, name: str):
        """Reproducible (X, y) for an expert: points around its center, labelled
        by a fixed random linear rule. Same name -> same data, every process."""
        rng = np.random.default_rng(_stable_seed("data:" + name))
        X = self.center(name) + rng.normal(0, 1.0, (self.n, self.d_model))
        y = (X @ self.rule(name) > 0).astype(int)
        return X, y

    def sample(self, name: str, n: int, rng=None):
        """Draw fresh held-out points from an expert's distribution (for eval /
        benchmarking) — same cluster + rule as ``data`` but independent draws."""
        rng = rng or np.random.default_rng(_stable_seed("sample:" + name))
        X = self.center(name) + rng.normal(0, 1.0, (n, self.d_model))
        y = (X @ self.rule(name) > 0).astype(int)
        return X, y

    def _train_leaf(self, leaf, X, y):
        rng = np.random.default_rng(_stable_seed("fit"))
        leaf.frozen = False
        for _ in range(self.leaf_steps):
            i = rng.integers(0, len(X), 64)
            leaf.backward(_ce_grad(leaf.forward(X[i]), y[i]), self.leaf_lr)
        leaf.frozen = True

    def _train_router(self, forest, names):
        rng = np.random.default_rng(_stable_seed("router"))
        Xr = np.vstack([self.data(n)[0] for n in names])
        dr = np.concatenate([np.full(self.n, s) for s in range(len(names))])
        for _ in range(self.router_steps):
            i = rng.integers(0, len(Xr), 64)
            forest.router.train_step(Xr[i], dr[i], lr=self.router_lr)

    # ── seed + graft hooks ───────────────────────────────────────────
    def seed_forest(self, seed_name: str) -> DASForest:
        """Build a one-leaf forest with the seed expert trained. This is what the
        ControlPlane is constructed over."""
        forest = DASForest(self.d_model, self.leaf_dims, num_leaves=1,
                           seed=_stable_seed("forest:" + seed_name))
        X, y = self.data(seed_name)
        self._train_leaf(forest.leaves[0], X, y)
        return forest

    def train_fn(self, name: str, cp):
        """Return a ``train_fn(forest, idx)`` for grafting expert ``name``: train
        its leaf in isolation, then retrain the router over every registered
        expert plus this one (graft appends the record *after* this callback, so
        we include ``name`` explicitly)."""
        def _fn(forest, idx):
            X, y = self.data(name)
            self._train_leaf(forest.leaves[idx], X, y)
            names = [r["name"] for r in cp.experts] + [name]
            self._train_router(forest, names)
        return _fn

    def seed_for(self, name: str):
        """A stable graft seed for an expert, so leaf init is reproducible."""
        return _stable_seed("leaf:" + name)
