"""
das/model.py
------------
The "Forest" — assembles the Stem Router and the leaves into one system.
The "Canopy" in the marketing is the layer that would combine multiple leaf
outputs; in this minimal prototype each input goes to exactly one leaf, so the
canopy is just "return the chosen leaf's output".
"""
import numpy as np
from .functional import FibonacciLeaf
from .routing import StemRouter

class DASForest:
    def __init__(self, d_model, leaf_dims, num_leaves, seed=0):
        self.d_model = d_model
        self.leaf_dims = list(leaf_dims)
        self.router = StemRouter(d_model, num_leaves, seed=seed)
        self.leaves = [FibonacciLeaf(leaf_dims, seed=seed + 1 + i)
                       for i in range(num_leaves)]

    def predict(self, h):
        """Route each input to its leaf, collect outputs."""
        leaf_idx, _ = self.router.route(h)
        out = np.zeros((h.shape[0], self.leaf_dims[-1]))
        for i, leaf in enumerate(self.leaves):
            mask = (leaf_idx == i)
            if mask.any():
                out[mask] = leaf.forward(h[mask])
        return out, leaf_idx

    def graft(self, new_leaf_dims=None, seed=99):
        """Add a brand new expert leaf and a router slot for it."""
        dims = new_leaf_dims or self.leaf_dims
        self.leaves.append(FibonacciLeaf(dims, seed=seed))
        self.router.expand(seed=seed)
        return len(self.leaves) - 1

    def freeze_all_leaves(self):
        for leaf in self.leaves:
            leaf.frozen = True

    def leaf_hashes(self):
        return {i: leaf.weight_hash() for i, leaf in enumerate(self.leaves)}
