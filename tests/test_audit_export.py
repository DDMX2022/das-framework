"""
Tests for the exportable, independently-verifiable audit document
(das.audit.to_document / verify_document and the export/load round-trip).
Pure stdlib + numpy (base dep) — runs in CI.
"""
import copy
import json
import os
import tempfile

from das.audit import AuditLog, verify_document

SECRET = "compliance-key"


def _log():
    log = AuditLog(SECRET)
    log.append("init", "seed expert", payload={"acme-tax": "aaaa111122223333"})
    log.append("graft", "added globex-vision", payload={"acme-tax": "aaaa111122223333",
                                                        "globex-vision": "bbbb444455556666"})
    log.append("prune", "removed globex-vision", payload={"acme-tax": "aaaa111122223333"})
    return log


def test_document_is_self_contained():
    doc = _log().to_document(meta={"note": "x"})
    assert doc["count"] == 3
    assert doc["head"] == doc["entries"][-1]["sig"]
    assert "hmac" in doc["scheme"]
    # the actual fingerprints travel in the document (not just their hash)
    assert doc["entries"][1]["payload"]["globex-vision"] == "bbbb444455556666"


def test_verify_keyless_and_authenticated():
    doc = _log().to_document()
    assert verify_document(doc) == (True, [])               # keyless / structural
    assert verify_document(doc, secret=SECRET)[0] is True   # authenticated
    assert verify_document(doc, secret="wrong-key")[0] is False


def test_export_load_roundtrip(tmp_path=None):
    d = tempfile.mkdtemp()
    path = os.path.join(d, "audit.json")
    log = _log()
    log.export(path)
    doc = json.load(open(path))
    assert verify_document(doc, secret=SECRET)[0] is True
    reloaded = AuditLog.load(path, secret=SECRET)
    assert reloaded.verify()[0] is True


def test_keyless_catches_fingerprint_edit():
    doc = _log().to_document()
    doc["entries"][0]["payload"]["acme-tax"] = "deadbeefdeadbeef"
    ok, issues = verify_document(doc)                        # no secret
    assert ok is False and any("fingerprint" in r for _, r in issues)


def test_keyless_catches_reorder_and_delete():
    doc = _log().to_document()
    del doc["entries"][1]
    ok, issues = verify_document(doc)
    assert ok is False and any("chain broken" in r for _, r in issues)


def test_content_edit_needs_the_key():
    doc = _log().to_document()
    doc["entries"][1]["detail"] += " (altered)"
    # a pure text edit is invisible to keyless structural checks ...
    assert verify_document(doc)[0] is True
    # ... but the HMAC signature catches it
    assert verify_document(doc, secret=SECRET)[0] is False


def test_live_log_verify_catches_payload_tamper():
    log = _log()
    assert log.verify()[0] is True
    log.entries[0]["payload"]["acme-tax"] = "0000000000000000"
    assert log.verify()[0] is False
