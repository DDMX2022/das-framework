"""
Tests for the real-encoder text path (das_text).

The corpus check is torch-free and always runs (keeps CI green without torch).
The encoder + isolation test downloads a ~90 MB model and needs torch +
sentence-transformers, so it is opt-in: set DAS_HF_TEST=1 to run it.

    pip install -e ".[hf]"
    DAS_HF_TEST=1 pytest tests/test_hf_text.py -q
"""
import os
import pytest

import das_text  # torch-free import (lazy heavy deps)


def test_demo_corpus_wellformed():
    corpus = das_text.DEMO_CORPUS
    assert corpus, "DEMO_CORPUS is empty"
    for domain, rows in corpus.items():
        assert len(rows) >= 4, f"{domain} has too few examples"
        labels = {lab for _, lab in rows}
        assert labels == {0, 1}, f"{domain} must have both classes, got {labels}"
        for text, lab in rows:
            assert isinstance(text, str) and text.strip(), f"empty text in {domain}"
            assert lab in (0, 1)


def test_embed_domains_rejects_unknown_domain():
    with pytest.raises(KeyError):
        das_text.embed_domains(encoder=None, domains=["not_a_domain"])


@pytest.mark.skipif(not os.environ.get("DAS_HF_TEST"),
                    reason="set DAS_HF_TEST=1 to run the model-download encoder test")
def test_real_encoder_routing_and_isolation():
    torch = pytest.importorskip("torch")
    pytest.importorskip("sentence_transformers")
    from das_torch import LoRAForest, train_leaf_isolated_lora, train_router, leaf_hash

    enc = das_text.TextEncoder(device="cpu")
    assert enc.dim and enc.dim > 0

    domains = ["legal", "medical"]
    cache = das_text.embed_domains(enc, domains)
    emb0, _ = cache["legal"]
    assert emb0.shape[1] == enc.dim

    torch.manual_seed(0)
    forest = LoRAForest(enc.dim, 64, out_dim=2, num_leaves=len(domains), rank=8)
    # freeze a (randomly-initialised) backbone — isolation must hold regardless
    forest.freeze_backbone()
    for i, name in enumerate(domains):
        e, y = cache[name]
        train_leaf_isolated_lora(forest, i, e, y, steps=60, device="cpu")

    before = [leaf_hash(forest.leaves[i]) for i in range(len(domains))]
    nid = forest.graft_leaf()
    e, y = das_text.embed_domains(enc, ["finance"])["finance"]
    train_leaf_isolated_lora(forest, nid, e, y, steps=60, device="cpu")
    after = [leaf_hash(forest.leaves[i]) for i in range(len(domains))]
    assert before == after, "grafting a new expert changed an existing one"
