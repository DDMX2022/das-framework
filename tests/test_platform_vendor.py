"""Vendor-side license administration — the superadmin store + console API."""
import datetime
import json

import pytest

pytest.importorskip("cryptography")
pytest.importorskip("flask")

from das.platform import LicenseError, deploy, verify_license
from das.platform.license import load_license
from das.platform.vendor import VendorStore

import vendor_console as vc


UTC = datetime.timezone.utc


@pytest.fixture
def store(tmp_path):
    return VendorStore(str(tmp_path / "vendor"))


@pytest.fixture
def client(store, monkeypatch):
    monkeypatch.setattr(vc, "store", store)
    monkeypatch.setattr(vc, "TOKEN", None)          # dev mode unless a test gates it
    vc.app.config["TESTING"] = True
    return vc.app.test_client()


# ── the store ────────────────────────────────────────────────────────
def test_keypair_created_once(store):
    pub1 = store.ensure_keypair()
    pub2 = store.ensure_keypair()
    assert pub1 == pub2 and len(pub1) == 64          # 32-byte ed25519 key, hex


def test_issue_records_and_verifies(store):
    rec = store.issue(customer="Acme", days=90, max_tenants=3)
    doc = rec["license"]
    ok, issues = verify_license(doc, store.public_key_hex)
    assert ok, issues
    assert store.get(doc["license_id"])["status"] == "active"


def test_effective_statuses(store):
    t0 = datetime.datetime(2026, 1, 1, tzinfo=UTC)
    store.issue(customer="Fresh", days=365, now=t0)
    store.issue(customer="Closing", days=20, now=t0)     # <=30d -> expiring
    store.issue(customer="Gone", days=5, now=t0)
    by = {r["customer"]: r["status"]
          for r in store.list(now=t0 + datetime.timedelta(days=10))}
    assert by == {"Fresh": "active", "Closing": "expiring", "Gone": "expired"}


def test_revoke_and_renew_lifecycle(store):
    rec = store.issue(customer="Acme", days=90, max_experts=5)
    lid = rec["license"]["license_id"]
    new = store.renew(lid, days=365)
    assert new["license"]["entitlements"]["max_experts"] == 5   # claims carried over
    assert store.get(lid)["status"] == "superseded"
    with pytest.raises(LicenseError, match="superseded"):
        store.revoke(lid)                                        # revoke the renewal instead
    new_id = new["license"]["license_id"]
    store.revoke(new_id, reason="non-payment")
    assert store.get(new_id)["revoked_reason"] == "non-payment"
    with pytest.raises(LicenseError, match="revoked"):
        store.renew(new_id)


def test_license_id_path_traversal_rejected(store):
    with pytest.raises(LicenseError, match="bad license id"):
        store.get("../../etc/passwd")


# ── the console API ──────────────────────────────────────────────────
def test_status_and_counts(client):
    client.post("/api/vendor/licenses", json={"customer": "Acme", "days": 30})
    s = client.get("/api/vendor/status").get_json()
    assert len(s["public_key_hex"]) == 64
    assert s["counts"]["total"] == 1


def test_issue_validation(client):
    assert client.post("/api/vendor/licenses", json={"customer": ""}).status_code == 400
    assert client.post("/api/vendor/licenses",
                       json={"customer": "A", "days": 0}).status_code == 400


def test_download_ships_only_the_signed_doc(client, store):
    lid = client.post("/api/vendor/licenses",
                      json={"customer": "Acme"}).get_json()["license"]["license_id"]
    r = client.get(f"/api/vendor/licenses/{lid}/download")
    doc = json.loads(r.data)
    assert "status" not in doc and "record_version" not in doc   # registry stays home
    ok, issues = verify_license(doc, store.public_key_hex)
    assert ok, issues


def test_verify_endpoint_catches_tamper(client, store):
    doc = store.issue(customer="Acme")["license"]
    assert client.post("/api/vendor/verify",
                       json={"license": doc}).get_json()["result"] == "VALID"
    doc["customer"] = "Evil Corp"
    assert client.post("/api/vendor/verify",
                       json={"license": doc}).get_json()["result"] == "INVALID"


def test_token_gate(client, monkeypatch):
    monkeypatch.setattr(vc, "TOKEN", "s3cret")
    assert client.get("/api/vendor/status").status_code == 401
    ok = client.get("/api/vendor/status", headers={"X-DAS-Vendor-Token": "s3cret"})
    assert ok.status_code == 200
    assert client.get("/").status_code == 200        # the page itself may load


# ── the full vendor -> customer loop ─────────────────────────────────
def test_issued_license_drives_a_customer_deploy(client, store, tmp_path, monkeypatch):
    lid = client.post("/api/vendor/licenses", json={
        "customer": "Northwind", "days": 365,
        "max_tenants": 1, "max_experts": 2}).get_json()["license"]["license_id"]
    lic_file = tmp_path / "license.json"
    lic_file.write_bytes(client.get(f"/api/vendor/licenses/{lid}/download").data)

    monkeypatch.setenv("DAS_LICENSE", str(lic_file))
    monkeypatch.setenv("DAS_LICENSE_PUBKEY", store.public_key_hex)
    lic = load_license()
    within = {"client": "c", "d_model": 16,
              "tenants": [{"name": "t", "experts": [{"name": "e1"}, {"name": "e2"}]}]}
    assert len(deploy(within, secret="s", license=lic).summary()["experts"]) == 2
    beyond = {"client": "c", "d_model": 16,
              "tenants": [{"name": "t", "experts": [{"name": "e1"}]},
                          {"name": "t2", "experts": [{"name": "e2"}]}]}
    with pytest.raises(LicenseError, match="2 tenants but the license allows 1"):
        deploy(beyond, secret="s", license=lic)
