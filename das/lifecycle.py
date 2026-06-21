"""
das/lifecycle.py
----------------
The Forest Lifecycle Manager — the missing third of the "works like a forest"
behaviour. The base DASForest already supports GROW (train a leaf in isolation)
and GRAFT (add a new leaf). This adds the rest of the organic loop:

  MONITOR  — count how often each leaf is actually routed to in production.
  PRUNE    — remove a leaf that's dormant / stale / redundant, and shrink the
             router gate to match.
  REGROW   — graft a fresh leaf into a freed slot.

HONEST NOTE on pruning and the router: removing leaf k means deleting column k
from the router's gate (W, b). This is cleaner than it sounds. The OTHER leaves'
gate columns are left exactly as they were, so:
  - their weights are byte-identical (the forgetting proof still holds), and
  - argmax over the *remaining* columns is unchanged for any input that wasn't
    already going to the pruned leaf.
So pruning needs NO retraining for the survivors to keep routing correctly.
Inputs that *used* to route to the pruned leaf simply fall through to their
next-best leaf — which is exactly the desired "that capability is gone" behaviour.
"""
import numpy as np


class ForestLifecycle:
    """Wraps a DASForest and tracks usage so leaves can be pruned and regrown.
    GROW is just `train` your leaves as usual (isolated); this class owns the
    monitor / prune / regraft half of the lifecycle."""

    def __init__(self, forest):
        self.forest = forest
        self.usage = np.zeros(len(forest.leaves), dtype=np.int64)
        self.total_routed = 0

    # ── MONITOR ──────────────────────────────────────────────────────────
    def _log(self, leaf_idx):
        for i in range(len(self.forest.leaves)):
            self.usage[i] += int((leaf_idx == i).sum())
        self.total_routed += len(leaf_idx)

    def route(self, h):
        """Route a batch and record which leaf each input went to."""
        leaf_idx, tau = self.forest.router.route(h)
        self._log(leaf_idx)
        return leaf_idx, tau

    def predict(self, h):
        """Full predict (route + run leaves), logging usage along the way."""
        out, leaf_idx = self.forest.predict(h)
        self._log(leaf_idx)
        return out, leaf_idx

    def monitor(self):
        """Per-leaf usage report: raw route count and share of total traffic."""
        n = len(self.forest.leaves)
        shares = (self.usage / self.total_routed) if self.total_routed else np.zeros(n)
        return {i: {'routes': int(self.usage[i]), 'share': round(float(shares[i]), 4)}
                for i in range(n)}

    def dormant_leaves(self, max_share=0.0):
        """Leaf ids whose traffic share is <= max_share (default: exactly zero —
        leaves that never fired since the last usage reset)."""
        if self.total_routed == 0:
            return []
        shares = self.usage / self.total_routed
        return [i for i, s in enumerate(shares) if s <= max_share]

    def reset_usage(self):
        """Start a fresh monitoring window (e.g. per day/week before deciding
        what to prune)."""
        self.usage[:] = 0
        self.total_routed = 0

    # ── PRUNE ────────────────────────────────────────────────────────────
    def prune(self, leaf_id):
        """Remove one leaf and its router-gate column. Survivors are untouched."""
        if not (0 <= leaf_id < len(self.forest.leaves)):
            raise IndexError(f"no leaf {leaf_id} (forest has {len(self.forest.leaves)})")
        del self.forest.leaves[leaf_id]
        r = self.forest.router
        r.W = np.delete(r.W, leaf_id, axis=1)     # drop that leaf's gate column
        r.b = np.delete(r.b, leaf_id)
        r.num_leaves -= 1
        self.usage = np.delete(self.usage, leaf_id)
        return leaf_id

    def prune_dormant(self, max_share=0.0):
        """Evict every dormant leaf in one pass. Returns the original ids pruned.
        Prunes from the highest index down so earlier indices stay valid."""
        ids = sorted(self.dormant_leaves(max_share), reverse=True)
        for i in ids:
            self.prune(i)
        return ids

    # ── REGROW ───────────────────────────────────────────────────────────
    def graft(self, new_leaf_dims=None, seed=99):
        """Graft a fresh leaf (then GROW it by training in isolation as usual)."""
        nid = self.forest.graft(new_leaf_dims, seed=seed)
        self.usage = np.append(self.usage, 0)
        return nid
