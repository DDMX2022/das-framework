"""
das/hub.py
----------
A minimal "leaf marketplace": publish a trained leaf to a shared hub (a folder),
list what's available, and pull + graft a leaf into any forest with ZERO
retraining. Each entry carries a SHA-256 fingerprint, so a pulled leaf is verified
on load — the same audit-trail property that makes DAS defensible, now extended to
sharing experts between people/projects.

Format: each leaf is an .npz (weights + dims); `index.json` is the catalog
(name, dims, domain, author, hash).
"""
import os
import json
import numpy as np
from .functional import FibonacciLeaf

def _index_path(hub_dir):
    return os.path.join(hub_dir, "index.json")

def _load_index(hub_dir):
    p = _index_path(hub_dir)
    return json.load(open(p)) if os.path.exists(p) else {}

def _save_index(hub_dir, idx):
    json.dump(idx, open(_index_path(hub_dir), "w"), indent=2)

def publish(leaf, name, hub_dir, domain="", author="", metadata=None):
    """Save a (frozen) leaf to the hub with its fingerprint in the catalog."""
    os.makedirs(hub_dir, exist_ok=True)
    blob = {f"W{i}": w for i, w in enumerate(leaf.W)}
    blob.update({f"b{i}": b for i, b in enumerate(leaf.b)})
    np.savez(os.path.join(hub_dir, f"{name}.npz"), dims=np.array(leaf.dims), **blob)
    idx = _load_index(hub_dir)
    idx[name] = {"name": name, "dims": list(leaf.dims), "domain": domain,
                 "author": author, "hash": leaf.weight_hash(), "file": f"{name}.npz"}
    if metadata:
        idx[name].update(metadata)
    _save_index(hub_dir, idx)
    return idx[name]

def list_leaves(hub_dir):
    """Catalog of available leaves."""
    return _load_index(hub_dir)

def pull(name, hub_dir):
    """Load a leaf from the hub, frozen, and VERIFY its fingerprint."""
    idx = _load_index(hub_dir)
    if name not in idx:
        raise KeyError(f"no leaf '{name}' in hub {hub_dir}")
    d = np.load(os.path.join(hub_dir, idx[name]["file"]))
    dims = [int(x) for x in d["dims"]]
    leaf = FibonacciLeaf(dims)
    leaf.W = [d[f"W{i}"] for i in range(len(dims) - 1)]
    leaf.b = [d[f"b{i}"] for i in range(len(dims) - 1)]
    leaf.frozen = True
    if leaf.weight_hash() != idx[name]["hash"]:
        raise ValueError(f"hash mismatch on '{name}' — corrupted or tampered")
    return leaf

def graft_from_hub(forest, name, hub_dir):
    """Pull a leaf from the hub and graft it into `forest` (no retraining)."""
    leaf = pull(name, hub_dir)
    forest.leaves.append(leaf)
    forest.router.expand()
    return len(forest.leaves) - 1
