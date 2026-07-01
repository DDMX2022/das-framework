"""ContextSource contract + reference connectors."""
import numpy as np
import pytest

from das.platform import (
    ClientSpec, StaticContextSource, CallableContextSource, SpecKeywordConnector,
)
from das.platform.trainer import SyntheticTrainer


def test_embed_enforces_shape():
    src = CallableContextSource(4, lambda q: np.zeros(4))
    assert src.embed("x").shape == (1, 4)
    bad = CallableContextSource(4, lambda q: np.zeros(3))
    with pytest.raises(ValueError, match="expected \\(1, 4\\)"):
        bad.embed("x")


def test_static_source_lookup():
    src = StaticContextSource(3, {"a": [1, 2, 3]})
    assert np.allclose(src.embed("a"), [[1, 2, 3]])
    with pytest.raises(KeyError):
        src.embed("missing")


def test_callable_source_accepts_row_or_flat():
    flat = CallableContextSource(2, lambda q: np.array([1.0, 2.0]))
    row = CallableContextSource(2, lambda q: np.array([[1.0, 2.0]]))
    assert flat.embed("q").shape == row.embed("q").shape == (1, 2)


def _spec():
    return ClientSpec.from_dict({
        "client": "c", "d_model": 16,
        "tenants": [
            {"name": "t1", "experts": [{"name": "card", "keywords": ["charge", "card"]}]},
            {"name": "t2", "experts": [{"name": "claim", "keywords": ["mri", "denied"]}]},
        ],
    })


def test_keyword_connector_matches_expert():
    spec = _spec()
    trainer = SyntheticTrainer(spec.d_model, spec.resolved_leaf_dims())
    conn = SpecKeywordConnector(spec, trainer)
    assert conn.match("my card charge is wrong") == "card"
    assert conn.match("mri was denied") == "claim"
    assert conn.match("totally unrelated text") is None


def test_keyword_connector_embeds_near_center():
    spec = _spec()
    trainer = SyntheticTrainer(spec.d_model, spec.resolved_leaf_dims())
    conn = SpecKeywordConnector(spec, trainer, noise=0.0)
    vec = conn.embed("charge on my card")
    assert np.allclose(vec[0], trainer.center("card"))
