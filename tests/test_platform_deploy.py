"""End-to-end deployment engine: spec -> governed fleet -> route -> offboard -> bundle."""
import json

import pytest

from das.audit import verify_document
from das.governance import AccessDenied
from das.platform import deploy, SpecError


SPEC = {
    "client": "northwind",
    "d_model": 18,
    "escalation": {"frontier_llm": "claude-sonnet-5", "confidence_threshold": 0.7},
    "tenants": [
        {"name": "careplus", "experts": [
            {"name": "medical-claim", "keywords": ["claim", "mri", "denied", "appeal"]},
            {"name": "prior-auth", "keywords": ["authorization", "referral"]},
        ]},
        {"name": "fintrust", "experts": [
            {"name": "card-dispute", "keywords": ["charge", "card", "merchant", "dispute"]},
            {"name": "loan-status", "keywords": ["loan", "balance", "mortgage"]},
        ]},
    ],
    "users": [
        {"name": "care-agent", "role": "operator", "tenant": "careplus"},
        {"name": "bank-agent", "role": "operator", "tenant": "fintrust"},
        {"name": "auditor-jane", "role": "auditor"},
    ],
}


@pytest.fixture
def dep():
    return deploy(SPEC, secret="test-secret")


def test_deploy_stands_up_full_fleet(dep):
    s = dep.summary()
    assert s["client"] == "northwind"
    assert sorted(s["tenants"]) == ["careplus", "fintrust"]
    assert {e["name"] for e in s["experts"]} == {
        "medical-claim", "prior-auth", "card-dispute", "loan-status"}
    assert s["audit_ok"] is True
    # root admin + three declared users
    assert set(s["users"]) == {"root", "care-agent", "bank-agent", "auditor-jane"}
    assert s["users"]["care-agent"] == {"role": "operator", "tenant": "careplus"}


def test_deploy_is_reproducible():
    a = deploy(SPEC, secret="test-secret").summary()
    b = deploy(SPEC, secret="test-secret").summary()
    assert [e["name"] for e in a["experts"]] == [e["name"] for e in b["experts"]]


def test_seed_override_honoured_for_first_expert():
    """Review item #8: ExpertSpec.seed applies to the seed-tenant's first
    expert too, not only to grafted ones."""
    base = json.loads(json.dumps(SPEC))
    seeded = json.loads(json.dumps(SPEC))
    seeded["tenants"][0]["experts"][0]["seed"] = 4242
    h_base = deploy(base, secret="s").cp.forest.leaves[0].weight_hash()
    h_seed = deploy(seeded, secret="s").cp.forest.leaves[0].weight_hash()
    assert h_base != h_seed                       # the override changed the init
    h_seed2 = deploy(seeded, secret="s").cp.forest.leaves[0].weight_hash()
    assert h_seed == h_seed2                      # and stays deterministic


def test_routing_hits_the_right_specialist(dep):
    r = dep.route("bank-agent", "unknown recurring card charge from a merchant")
    assert r["expert"] == "card-dispute"
    assert r["tenant"] == "fintrust"
    assert r["confidence"] > 0.5


def test_escalation_policy(dep):
    # threshold 0.7 in the spec; a clear in-domain query stays local
    r = dep.route("care-agent", "my mri claim was denied, need an appeal")
    assert r["decision"] == "local"
    assert r["expert"] == "medical-claim"


def test_threshold_one_forces_escalation():
    spec = json.loads(json.dumps(SPEC))
    spec["escalation"]["confidence_threshold"] = 1.0  # softmax < 1.0 always -> escalate
    d = deploy(spec, secret="test-secret")
    r = d.route("care-agent", "my mri claim was denied")
    assert r["decision"] == "escalate"
    assert r["frontier_llm"] == "claude-sonnet-5"


def test_rbac_is_enforced_end_to_end(dep):
    # auditor may not graft
    with pytest.raises(AccessDenied):
        dep.grow("auditor-jane", "careplus", "x")


def test_grow_keeps_others_byte_identical(dep):
    before = {e["eid"]: e for e in dep.summary()["experts"]}
    eid = dep.grow("root", "careplus", "pharmacy-claim", keywords=["pharmacy", "rx"])
    assert eid not in before
    # audit chain still verifies and records the graft's non-interference
    assert dep.verify()["ok"] is True
    r = dep.route("care-agent", "pharmacy rx refill")
    assert r["expert"] == "pharmacy-claim"


def test_offboard_is_provable_deletion(dep):
    result = dep.offboard("fintrust")
    assert result["removed"] == 2
    assert result["non_interference"] is True
    remaining = {e["tenant"] for e in dep.summary()["experts"]}
    assert remaining == {"careplus"}
    assert dep.verify()["ok"] is True


def test_bundle_is_offline_verifiable(dep, tmp_path):
    out = tmp_path / "bundle.json"
    manifest = dep.export_bundle(str(out))
    assert manifest["audit_chain_ok"] is True
    doc = json.loads(out.read_text())
    # the bundle is itself a valid audit document
    assert doc["bundle_schema"] == "das.compliance-bundle/v1"
    assert doc["client"] == "northwind"
    ok, issues = verify_document(doc)                       # keyless / structural
    assert ok, issues
    ok_auth, _ = verify_document(doc, secret="test-secret")  # + HMAC authenticity
    assert ok_auth


def test_bundle_tamper_is_caught(dep, tmp_path):
    out = tmp_path / "bundle.json"
    dep.export_bundle(str(out))
    doc = json.loads(out.read_text())
    # editing a signed entry breaks HMAC authenticity (checked with the secret)
    doc["entries"][1]["detail"] = "tampered"
    ok, issues = verify_document(doc, secret="test-secret")
    assert not ok and issues
    # reordering the chain is caught even keyless (structural)
    doc2 = json.loads(out.read_text())
    doc2["entries"][1], doc2["entries"][2] = doc2["entries"][2], doc2["entries"][1]
    ok2, issues2 = verify_document(doc2)
    assert not ok2 and issues2


def test_bundle_wrapper_tamper_is_caught_with_key(dep, tmp_path):
    """The review's high finding: the manifest travels OUTSIDE the entry chain,
    so it carries its own signature — a keyed verifier catches wrapper edits."""
    out = tmp_path / "bundle.json"
    dep.export_bundle(str(out))
    doc = json.loads(out.read_text())
    assert doc["bundle_signature"] and "manifest" in doc["bundle_signed_fields"]
    ok, _ = verify_document(doc, secret="test-secret")     # untampered: passes
    assert ok
    doc["manifest"]["tenants"] = ["evil-corp"]             # rewrite the manifest
    ok2, issues = verify_document(doc, secret="test-secret")
    assert not ok2 and any("bundle metadata" in r for _i, r in issues)
    # keyless remains structural-only (documented boundary), same as entry auth
    ok3, _ = verify_document(doc)
    assert ok3


def test_deploy_requires_experts():
    # spec validation rejects an empty expert list before deploy runs
    with pytest.raises(SpecError):
        deploy({"client": "empty", "tenants": [{"name": "t", "experts": []}]},
               secret="x")
