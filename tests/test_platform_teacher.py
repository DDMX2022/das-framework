"""The teacher bridge — Growing-Child teachers behind the platform's train_fn
seam: teacher-driven grow, policy-gated improve, quarantine, audit, RBAC."""
import pytest

from das.governance import AccessDenied
from das.platform import deploy
from das.training.growth import GrowthPolicy

import platform_console as pc


SPEC = {
    "client": "northwind",
    "d_model": 18,
    "escalation": {"frontier_llm": "claude-sonnet-5", "confidence_threshold": 0.7},
    "tenants": [
        {"name": "careplus", "experts": [
            {"name": "medical-claim", "keywords": ["claim", "mri", "denied"]}]},
        {"name": "fintrust", "experts": [
            {"name": "card-dispute", "keywords": ["charge", "card", "dispute"]}]},
    ],
    "users": [{"name": "auditor-jane", "role": "auditor"}],
}


@pytest.fixture
def dep():
    return deploy(SPEC, secret="s")


def _events(dep):
    return [e["event"] for e in dep.cp.audit.entries]


# ── teacher-driven grow ──────────────────────────────────────────────
def test_grow_with_teacher_reports_and_isolates(dep):
    before = {r["eid"]: dep.cp.forest.leaves[i].weight_hash()
              for i, r in enumerate(dep.cp.experts)}
    eid = dep.grow("root", "careplus", "pharmacy-claim",
                   keywords=["pharmacy", "rx"], teacher="local-teacher")
    # prior experts byte-identical (position by eid, indexes shifted by graft)
    for i, r in enumerate(dep.cp.experts):
        if r["eid"] in before:
            assert dep.cp.forest.leaves[i].weight_hash() == before[r["eid"]]
    report = dep.teacher_trainer.reports["pharmacy-claim"]
    assert report["teacher"] == "local-teacher"
    assert report["eval_accuracy"] > 0.8            # learned the teacher's lessons
    assert "pharmacy-claim" in dep.teacher_trainer.lessons
    assert dep.verify()["ok"] and eid not in before


def test_teacher_grown_expert_routes_coherently(dep):
    dep.grow("root", "careplus", "pharmacy-claim",
             keywords=["pharmacy", "rx"], teacher="local-teacher")
    r = dep.route("root", "pharmacy rx refill request")
    assert r["expert"] == "pharmacy-claim"
    assert r["confidence"] > 0.5


def test_grow_with_unknown_teacher_rejected(dep):
    with pytest.raises(KeyError, match="unknown teacher"):
        dep.grow("root", "careplus", "x", teacher="ghost-teacher")


def test_teacher_grow_is_reproducible():
    a = deploy(SPEC, secret="s")
    b = deploy(SPEC, secret="s")
    for d in (a, b):
        d.grow("root", "careplus", "pharmacy-claim", teacher="local-teacher")
    ha = a.cp.forest.leaves[-1].weight_hash()
    hb = b.cp.forest.leaves[-1].weight_hash()
    assert ha == hb


# ── policy-gated improve (the Growing-Child loop) ────────────────────
def test_improve_accepted_and_audited(dep):
    eid = dep.cp.experts[0]["eid"]
    result = dep.improve("root", eid)
    assert result["accepted"] is True
    assert result["target_accuracy_after"] >= result["target_accuracy_before"]
    assert result["old_experts_unchanged"] is True
    assert "growth_update" in _events(dep)
    assert dep.verify()["ok"] and dep.cp.state_matches_audit()


def test_improve_rejected_leaves_live_expert_untouched(dep):
    eid = dep.cp.experts[0]["eid"]
    live_hash = dep.cp.forest.leaves[0].weight_hash()
    result = dep.improve("root", eid, policy=GrowthPolicy(min_accuracy=1.01))
    assert result["accepted"] is False
    assert "below minimum" in result["reason"]
    assert dep.cp.forest.leaves[0].weight_hash() == live_hash   # quarantine held
    assert "growth_rejected" in _events(dep)
    assert dep.verify()["ok"] and dep.cp.state_matches_audit()


def test_improve_checks_regression_across_all_experts(dep):
    result = dep.improve("root", dep.cp.experts[0]["eid"])
    # a frozen eval set existed for every expert in the fleet
    assert set(result["previous_accuracy_before"]) == {
        f"{r['eid']}" for r in dep.cp.experts}


def test_improve_rbac_denied_for_auditor(dep):
    with pytest.raises(AccessDenied):
        dep.improve("auditor-jane", dep.cp.experts[0]["eid"])
    assert "denied" in _events(dep)


def test_register_llm_teacher_config(dep):
    meta = dep.register_teacher({
        "id": "phone-ollama", "provider": "ollama",
        "endpoint": "http://192.168.1.50:11434",
        "model": "qwen2.5:7b-instruct", "api_key": "secret-key"})
    assert meta["provider"] == "ollama" and meta["dynamic"] is True
    assert "api_key" not in meta                     # sanitized metadata only
    assert "phone-ollama" in dep.teachers


# ── console surface ──────────────────────────────────────────────────
@pytest.fixture
def client():
    pc.app.config["TESTING"] = True
    pc.DEPLOYMENTS.clear()
    pc.STATS.clear()
    pc._bootstrap()
    return pc.app.test_client()


def test_console_grow_with_teacher(client):
    r = client.post("/api/deployments/northwind/grow",
                    json={"tenant": "careplus", "name": "pharmacy-claim",
                          "teacher": "local-teacher"}).get_json()
    assert r["others_byte_identical"] is True
    assert r["teacher_report"]["teacher"] == "local-teacher"


def test_console_improve_endpoint(client):
    r = client.post("/api/deployments/northwind/improve", json={"eid": 0})
    j = r.get_json()
    assert r.status_code == 200 and j["accepted"] in (True, False)
    assert j["audit_ok"] is True
    denied = client.post("/api/deployments/northwind/improve",
                         json={"eid": 0, "actor": "auditor-jane"})
    assert denied.status_code == 403


def test_console_teacher_registration(client):
    r = client.post("/api/deployments/northwind/teachers",
                    json={"id": "lab-llm", "provider": "openai-compatible",
                          "endpoint": "http://localhost:8000/v1", "model": "qwen"})
    assert r.status_code == 201
    detail = client.get("/api/deployments/northwind").get_json()
    assert {t["id"] for t in detail["teachers"]} >= {"local-teacher", "lab-llm"}
    bad = client.post("/api/deployments/northwind/teachers", json={"provider": "ollama"})
    assert bad.status_code == 400                    # teacher id required