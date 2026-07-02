"""
das/hierarchy.py
----------------
The "Forest of Forests" — the specialty TREE that SPECIALTY_FORESTS.md sketched.

    Specialty router          chooses: react | math | physics
      -> branch (DASForest)   chooses the leaf inside that specialty
           -> expert leaf     answers

Two-level hard routing: a top-level StemRouter commits each input to exactly one
named BRANCH (a whole DASForest), and that branch's own router commits it to
exactly one leaf. Provenance is two-level — (specialty, specialty_confidence,
leaf, leaf_confidence) — which is what an audit record needs to say "this answer
came from react → hooks".

Why hierarchy instead of one flat router (measured, not asserted — see
benchmarks/hierarchy_bench.py): a flat softmax must separate ALL leaves of ALL
specialties at once, and the known routing bottleneck grows with total leaf
count. The tree divides the problem: the top router only separates K
specialties; each sub-router only separates its own handful. Active router
compute per query drops from d·N to d·(K + n_branch).

Every existing guarantee is inherited, not re-implemented: each branch IS a
DASForest, so byte-identical grafts, provable prunes, and weight hashing hold
inside every branch exactly as before. This wrapper only adds routing structure
on top — and one new, cleaner operation: prune an entire branch (delete a whole
specialty) with the survivors proven byte-identical.

HONEST NOTE: like the flat router, the specialty router is trained supervision-
style (you tell it which specialty each input belongs to) and must be given a
short update when a branch is added. The leaves never move; the routers learn.
Only depends on numpy.
"""
import hashlib

import numpy as np

from .model import DASForest
from .routing import StemRouter


class HierarchicalDASForest:
    """A specialty router over named DASForest branches (two-level top-1)."""

    def __init__(self, d_model, seed=0):
        self.d_model = d_model
        self._seed = seed
        self.branch_names = []     # slot order — aligns with the specialty router's columns
        self.branches = {}         # name -> DASForest
        self.router = None         # created when the first branch is added

    # ── structure ────────────────────────────────────────────────────
    def add_branch(self, name, forest=None, leaf_dims=None, num_leaves=1, seed=None):
        """Graft a whole specialty. Pass an existing DASForest (adopting it as a
        branch) or let one be created. Proves every OTHER branch's leaves are
        byte-identical across the operation and returns (forest, intact)."""
        if name in self.branches:
            raise ValueError(f"branch '{name}' already exists")
        before = self.hashes()
        if forest is None:
            dims = leaf_dims or [self.d_model, 13, 8, 2]
            forest = DASForest(self.d_model, dims, num_leaves=num_leaves,
                               seed=self._seed if seed is None else seed)
        if forest.d_model != self.d_model:
            raise ValueError(f"branch d_model {forest.d_model} != {self.d_model}")
        self.branches[name] = forest
        self.branch_names.append(name)
        if self.router is None:
            self.router = StemRouter(self.d_model, 1, seed=self._seed)
        else:
            self.router.expand(seed=abs(hash(name)) % 10000)
        after = self.hashes()
        intact = all(after[b] == h for b, h in before.items())
        return forest, intact

    def prune_branch(self, name):
        """Delete an entire specialty — its leaves are structurally gone, its
        router column removed — and prove every surviving branch byte-identical.
        The flat forest can only prune leaf-by-leaf; this is the tree's cleaner
        right-to-be-forgotten at specialty granularity."""
        if name not in self.branches:
            raise KeyError(f"no branch '{name}'")
        survivors_before = {b: h for b, h in self.hashes().items() if b != name}
        slot = self.branch_names.index(name)
        removed_leaves = len(self.branches[name].leaves)
        del self.branches[name]
        self.branch_names.pop(slot)
        self.router.W = np.delete(self.router.W, slot, axis=1)
        self.router.b = np.delete(self.router.b, slot)
        self.router.num_leaves -= 1
        intact = self.hashes() == survivors_before
        return {"branch": name, "removed_leaves": removed_leaves,
                "survivors_byte_identical": intact}

    def graft_leaf(self, branch, train_fn=None, new_leaf_dims=None, seed=99):
        """Graft one leaf INSIDE a branch (the flat graft, scoped to a specialty).

        Proves the tree's strongest guarantee — ROUTING ISOLATION, which the flat
        forest structurally cannot offer: in a flat forest the single router is
        shared mutable state, retrained over everyone's data on every graft, so
        adding one tenant's expert changes every other tenant's routing behaviour.
        Here the operation touches only this branch's sub-router; every other
        branch's LEAVES **and ROUTER** — and the top specialty router — are
        byte-identical afterwards. Returns (idx, leaves_intact, routing_intact)."""
        leaves_before = {b: h for b, h in self.hashes().items() if b != branch}
        routers_before = {b: h for b, h in self.router_hashes().items()
                          if b not in (branch,)}
        forest = self.branches[branch]
        idx = forest.graft(new_leaf_dims=new_leaf_dims, seed=seed)
        if train_fn is not None:
            train_fn(forest, idx)
        leaves_intact = {b: h for b, h in self.hashes().items() if b != branch} == leaves_before
        routers_intact = {b: h for b, h in self.router_hashes().items()
                          if b not in (branch,)} == routers_before
        return idx, leaves_intact, routers_intact

    # ── two-level routing ────────────────────────────────────────────
    def route(self, h):
        """Top-level only: which BRANCH each input goes to. Returns (idx, tau)."""
        if self.router is None:
            raise RuntimeError("no branches yet")
        return self.router.route(h)

    def route_explain(self, h):
        """Two-level provenance per input:
        {specialty, specialty_confidence, leaf, leaf_confidence, prediction}."""
        h = np.asarray(h, dtype=float)
        if h.ndim == 1:
            h = h[None, :]
        top_idx, top_tau = self.route(h)
        rows = [None] * h.shape[0]
        for slot, name in enumerate(self.branch_names):
            mask = (top_idx == slot)
            if not mask.any():
                continue
            sub = self.branches[name]
            leaf_idx, leaf_tau = sub.router.route(h[mask])
            out, _ = sub.predict(h[mask])
            for row_pos, n in enumerate(np.where(mask)[0]):
                li = int(leaf_idx[row_pos])
                rows[int(n)] = {
                    "specialty": name,
                    "specialty_confidence": float(top_tau[n, slot]),
                    "leaf": li,
                    "leaf_confidence": float(leaf_tau[row_pos, li]),
                    "prediction": out[row_pos].tolist(),
                }
        return rows

    def predict(self, h):
        """Route through both levels; returns (out, branch_idx, leaf_idx)."""
        h = np.asarray(h, dtype=float)
        if h.ndim == 1:
            h = h[None, :]
        top_idx, _ = self.route(h)
        d_out = None
        outs, leaf_ids = None, np.zeros(h.shape[0], dtype=int)
        for slot, name in enumerate(self.branch_names):
            mask = (top_idx == slot)
            if not mask.any():
                continue
            out, li = self.branches[name].predict(h[mask])
            if outs is None:
                d_out = out.shape[1]
                outs = np.zeros((h.shape[0], d_out))
            outs[mask] = out
            leaf_ids[mask] = li
        return outs, top_idx, leaf_ids

    # ── training the specialty router ────────────────────────────────
    def train_router(self, X, branch_labels, steps=400, lr=0.15, batch=64, seed=0):
        """Supervised top-level routing: labels are branch SLOT indices (use
        `slot_of` to map names). The branches' own routers are trained per-branch
        by whoever trains their leaves — same contract as the flat forest."""
        rng = np.random.default_rng(seed)
        X = np.asarray(X, dtype=float)
        y = np.asarray(branch_labels, dtype=int)
        for _ in range(int(steps)):
            i = rng.integers(0, len(X), min(int(batch), len(X)))
            self.router.train_step(X[i], y[i], lr=lr)

    def slot_of(self, name):
        return self.branch_names.index(name)

    # ── proofs ───────────────────────────────────────────────────────
    def hashes(self):
        """Two-level fingerprint: {branch: {leaf_idx: sha256}} — the audit payload."""
        return {name: self.branches[name].leaf_hashes() for name in self.branch_names}

    def router_hashes(self):
        """Fingerprint of every ROUTER: each branch's sub-router plus the top
        specialty router ('__top__'). This is what makes routing isolation
        provable, not just claimed."""
        def _h(router):
            m = hashlib.sha256()
            m.update(np.ascontiguousarray(router.W).tobytes())
            m.update(np.ascontiguousarray(router.b).tobytes())
            return m.hexdigest()

        out = {name: _h(self.branches[name].router) for name in self.branch_names}
        if self.router is not None:
            out["__top__"] = _h(self.router)
        return out

    def leaf_count(self):
        return sum(len(f.leaves) for f in self.branches.values())
