"""
LoRALeaf tests — torch-guarded so the NumPy-only CI skips them gracefully.
Run locally (torch present): pytest -q tests/test_lora.py
"""
import tempfile
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from das_torch import LoRAForest, train_leaf_isolated_lora, leaf_hash

D, FEAT = 20, 32

def _data(did, n=200):
    c = np.zeros(D); c[did * 4] = 4.0
    rule = np.random.default_rng(100 + did).normal(0, 1, D)
    X = c + np.random.default_rng(did).normal(0, 1.0, (n, D))
    return torch.tensor(X, dtype=torch.float32), torch.tensor((X @ rule > 0).astype(np.int64))

def _forest():
    torch.manual_seed(0)
    f = LoRAForest(D, FEAT, out_dim=2, num_leaves=2, rank=8)
    f.freeze_backbone()
    for t in range(2):
        train_leaf_isolated_lora(f, t, *_data(t), steps=120)
    return f

def test_lora_isolation_on_graft():
    f = _forest()
    before = [leaf_hash(l) for l in f.leaves]
    nid = f.graft_leaf()
    train_leaf_isolated_lora(f, nid, *_data(2), steps=120)
    assert [leaf_hash(f.leaves[i]) for i in range(2)] == before   # existing experts unchanged

def test_lora_checkpoint_byte_exact():
    f = _forest()
    d = tempfile.mkdtemp()
    f.save(d)
    g = LoRAForest.load(d)
    assert [leaf_hash(g.leaves[i]) for i in range(len(f.leaves))] == \
           [leaf_hash(f.leaves[i]) for i in range(len(f.leaves))]

def test_lora_backbone_not_trained():
    """Training an adapter must never move the shared backbone."""
    f = _forest()
    bb = [p.detach().clone() for p in f.backbone.parameters()]
    train_leaf_isolated_lora(f, 0, *_data(0), steps=120)
    assert all(torch.equal(a, b) for a, b in zip(f.backbone.parameters(), bb))
