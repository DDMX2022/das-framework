"""The platform console — the Rancher-style multi-deployment API surface."""
import json

import pytest

import platform_console as pc


@pytest.fixture
def client():
    pc.app.config["TESTING"] = True
    # reset console state so tests are order-independent
    pc.DEPLOYMENTS.clear()
    pc.STATS.clear()
    pc._bootstrap()
    return pc.app.test_client()


def test_lists_bootstrapped_deployments(client):
    j = client.get("/api/deployments").get_json()
    names = {d["client"] for d in j["deployments"]}
    assert names == {"northwind", "meridian"}
    assert all(d["audit_ok"] for d in j["deployments"])


def test_detail_includes_live_weight_hashes(client):
    j = client.get("/api/deployments/northwind").get_json()
    assert len(j["experts"]) == 4
    assert all(len(e["hash"]) == 16 for e in j["experts"])
    assert client.get("/api/deployments/ghost").status_code == 404


def test_route_tallies_deflection(client):
    r = client.post("/api/deployments/northwind/route",
                    json={"actor": "bank-agent",
                          "query": "unknown card charge from a merchant"}).get_json()
    assert r["expert"] == "card-dispute"
    assert r["decision"] in ("local", "escalate")
    assert r["stats"]["local"] + r["stats"]["escalate"] == 1


def test_route_requires_query(client):
    assert client.post("/api/deployments/northwind/route", json={}).status_code == 400


def test_rbac_denial_surfaces_as_403(client):
    r = client.post("/api/deployments/northwind/grow",
                    json={"actor": "auditor-jane", "tenant": "careplus", "name": "x"})
    assert r.status_code == 403
    assert r.get_json()["denied"] is True


def test_grow_proves_others_intact(client):
    r = client.post("/api/deployments/northwind/grow",
                    json={"tenant": "careplus", "name": "pharmacy-claim",
                          "keywords": ["pharmacy", "rx"]}).get_json()
    assert r["others_byte_identical"] is True and r["audit_ok"] is True
    # duplicate name is rejected
    dup = client.post("/api/deployments/northwind/grow",
                      json={"tenant": "careplus", "name": "pharmacy-claim"})
    assert dup.status_code == 409


def test_offboard_is_provable(client):
    r = client.post("/api/deployments/northwind/offboard",
                    json={"tenant": "fintrust"}).get_json()
    assert r["removed"] == 2 and r["non_interference"] is True and r["audit_ok"] is True


def test_deploy_new_client_via_api(client):
    spec = {"client": "acme", "d_model": 16,
            "tenants": [{"name": "t", "experts": [{"name": "e1"}]}]}
    r = client.post("/api/deployments", json={"spec": spec})
    assert r.status_code == 201
    assert client.post("/api/deployments", json={"spec": spec}).status_code == 409  # duplicate
    bad = client.post("/api/deployments", json={"spec": {"client": "x"}})
    assert bad.status_code == 400  # SpecError surfaces cleanly


def test_bundle_downloads_verifiable_document(client):
    from das.audit import verify_document
    r = client.get("/api/deployments/meridian/bundle")
    assert r.status_code == 200
    doc = json.loads(r.data)
    assert doc["client"] == "meridian"
    ok, issues = verify_document(doc)
    assert ok, issues


def test_audit_endpoint_verifies(client):
    j = client.get("/api/deployments/northwind/audit").get_json()
    assert j["verify"]["ok"] is True
    assert j["entries"]  # init + grafts + user adds
