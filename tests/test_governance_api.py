"""
Governance REST API tests (governance_api.py). Uses Flask's test client against
the bootstrapped in-memory fleet. Skipped if flask isn't installed.
"""
import numpy as np
import pytest

pytest.importorskip("flask")

import governance_api as api


@pytest.fixture()
def client():
    return api.app.test_client()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    j = r.get_json()
    assert j["status"] == "ok" and j["backend"].startswith("numpy")
    assert j["experts"] >= 1 and j["audit_chain_ok"] is True
    assert j["state_matches_audit"] is True


def test_predict_returns_provenance(client):
    dim = api.DIM
    # a query at globex-vision's cluster (center dim 8) should be served by globex
    emb = [0.0] * dim; emb[8] = 6.0
    r = client.post("/predict", json={"embedding": emb}, headers={"X-DAS-Actor": "root"})
    assert r.status_code == 200
    j = r.get_json()
    assert j["tenant"] in api.cp.tenants
    assert any(rec["name"] == j["name"] and rec["tenant"] == j["tenant"] for rec in api.cp.experts)
    assert 0.0 <= j["confidence"] <= 1.0 and j["actor"] == "root"


def test_predict_bad_dim(client):
    r = client.post("/predict", json={"embedding": [1.0, 2.0]})
    assert r.status_code == 400


def test_predict_denied_for_auditor(client):
    dim = api.DIM
    r = client.post("/predict", json={"embedding": [0.0] * dim},
                    headers={"X-DAS-Actor": "carol"})   # auditor lacks predict
    assert r.status_code == 403
    assert "denied" in r.get_json()["error"]


def test_experts_tenant_scoped(client):
    glob = client.get("/experts", headers={"X-DAS-Actor": "root"}).get_json()
    scoped = client.get("/experts", headers={"X-DAS-Actor": "alice"}).get_json()  # acme-bound
    assert {e["tenant"] for e in glob["experts"]} >= {"acme", "globex"}
    assert {e["tenant"] for e in scoped["experts"]} == {"acme"}


def test_audit_verify(client):
    r = client.get("/audit/verify", headers={"X-DAS-Actor": "carol"})
    assert r.status_code == 200 and r.get_json()["ok"] is True


def test_prune_denied_for_auditor(client):
    eid = api.cp.experts[-1]["eid"]
    r = client.post("/prune", json={"eid": eid}, headers={"X-DAS-Actor": "carol"})
    assert r.status_code == 403
