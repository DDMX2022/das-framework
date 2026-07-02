"""Real-semantics connectors — the Phase-1 substance seam: queries and teacher
lessons embedded by a REAL frozen encoder (MiniLM), not keywords or hashes.
Needs the [hf] extra; skipped without it. End-to-end story lives in
examples/hf_governance_demo.py."""
import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("sentence_transformers")

from das.platform import MiniLMContextSource, RealTextLessonEncoder
from das.training.teachers import EndpointLLMTeacher, HashingTextEncoder


@pytest.fixture(scope="module")
def source():
    return MiniLMContextSource()          # cached MiniLM, CPU


def test_embeds_real_text_at_the_connector_contract(source):
    v = source.embed("unknown recurring card charge from a merchant")
    assert v.shape == (1, 384)
    assert np.isclose(np.linalg.norm(v), 1.0, atol=1e-3)   # L2-normalised


def test_embeddings_are_semantic_not_lexical(source):
    """The point of the real encoder: MEANING drives distance. Two differently-
    worded finance sentences must sit closer than finance vs. medical."""
    card = source.embed("I want to dispute a charge on my credit card")[0]
    fraud = source.embed("someone billed my account for a purchase I never made")[0]
    mri = source.embed("my MRI scan appointment was cancelled by the clinic")[0]
    assert card @ fraud > card @ mri


def test_lesson_encoder_matches_query_geometry(source):
    """Teacher lessons and routed queries must share one geometry — encode the
    same sentence both ways and get the same vector."""
    enc = RealTextLessonEncoder()
    text = "useState stores component state between renders"
    lesson_vec = enc.encode([{"input": text, "label": 1}])
    query_vec = source.embed(text)
    assert lesson_vec.shape == (1, 384)
    assert np.allclose(lesson_vec, query_vec, atol=1e-5)


def test_llm_teacher_accepts_real_encoder():
    enc = RealTextLessonEncoder()
    teacher = EndpointLLMTeacher("real-teacher", enc.d_model, provider="ollama",
                                 endpoint="http://example.invalid", model="x",
                                 encoder=enc)
    assert teacher.encoder is enc                          # real semantics wired
    default = EndpointLLMTeacher("hash-teacher", 384)
    assert isinstance(default.encoder, HashingTextEncoder)  # fallback unchanged