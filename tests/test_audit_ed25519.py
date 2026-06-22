"""
Tests for Ed25519 (asymmetric) audit signing — public-key-verifiable docs (F7).
Guarded: skipped unless the optional `cryptography` dependency is installed, so
CI stays torch-/crypto-free and green.

    pip install -e ".[crypto]"
    pytest tests/test_audit_ed25519.py -q
"""
import copy

import pytest

pytest.importorskip("cryptography")

from das.audit import AuditLog, generate_keypair, verify_document


def _ed_log():
    priv_pem, pub_hex = generate_keypair()
    log = AuditLog(private_key=priv_pem)
    log.append("init", "seed", payload={"acme": "aaaa111122223333"})
    log.append("graft", "added globex", payload={"acme": "aaaa111122223333",
                                                 "globex": "bbbb444455556666"})
    log.append("prune", "removed globex", payload={"acme": "aaaa111122223333"})
    return log, pub_hex


def test_scheme_and_embedded_public_key():
    log, pub_hex = _ed_log()
    assert "ed25519" in log.scheme
    assert log.public_key_hex == pub_hex
    doc = log.to_document()
    assert doc["public_key"] == pub_hex
    # the private key must never leak into the document
    assert "private" not in __import__("json").dumps(doc).lower()


def test_verify_with_public_key_only():
    log, pub_hex = _ed_log()
    doc = log.to_document()
    assert verify_document(doc, public_key=pub_hex) == (True, [])
    # no key passed → falls back to the embedded public key (self-consistent)
    assert verify_document(doc)[0] is True


def test_wrong_public_key_rejected():
    log, _ = _ed_log()
    doc = log.to_document()
    _, other_pub = generate_keypair()
    ok, issues = verify_document(doc, public_key=other_pub)
    assert ok is False and any("signature" in r for _, r in issues)


def test_content_edit_rejected_under_pubkey():
    log, pub_hex = _ed_log()
    doc = log.to_document()
    doc["entries"][1]["detail"] += " (altered)"
    assert verify_document(doc, public_key=pub_hex)[0] is False


def test_structural_checks_still_keyless():
    log, _ = _ed_log()
    doc = log.to_document()
    del doc["entries"][1]                       # delete an entry, no key at all
    ok, issues = verify_document(doc)
    assert ok is False and any("chain broken" in r for _, r in issues)


def test_pem_string_and_object_accepted():
    priv_pem, _ = generate_keypair()
    # bytes PEM
    AuditLog(private_key=priv_pem).append("x", "y")
    # str PEM
    AuditLog(private_key=priv_pem.decode()).append("x", "y")
    # an Ed25519 object
    from cryptography.hazmat.primitives.serialization import load_pem_private_key
    obj = load_pem_private_key(priv_pem, password=None)
    log = AuditLog(private_key=obj)
    log.append("x", "y")
    assert log.verify()[0] is True
