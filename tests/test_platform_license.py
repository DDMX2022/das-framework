"""Offline subscription licensing — signing, pinning, expiry, entitlements,
and fail-closed enforcement in the deployment engine."""
import datetime
import json

import pytest

pytest.importorskip("cryptography")

from das.audit import generate_keypair
from das.platform import ClientSpec, License, LicenseError, deploy, issue_license, verify_license
from das.platform.license import load_license


@pytest.fixture(scope="module")
def keys():
    pem, pub = generate_keypair()
    return pem, pub


def _issue(pem, **kw):
    kw.setdefault("customer", "Northwind Health")
    return issue_license(pem, **kw)


SPEC = {"client": "northwind", "d_model": 16,
        "tenants": [{"name": "t1", "experts": [{"name": "e1"}, {"name": "e2"}]},
                    {"name": "t2", "experts": [{"name": "e3"}]}]}


# ── signing + verification ──────────────────────────────────────────
def test_issue_verify_roundtrip(keys):
    pem, pub = keys
    doc = _issue(pem, days=365, tier="platform", max_tenants=10)
    ok, issues = verify_license(doc, pub)
    assert ok, issues


def test_tampered_claims_fail_closed(keys):
    pem, pub = keys
    doc = _issue(pem, days=30, max_tenants=1)
    doc["entitlements"]["max_tenants"] = 999          # upgrade yourself? no.
    ok, issues = verify_license(doc, pub)
    assert not ok and any("signature invalid" in i for i in issues)


def test_wrong_vendor_key_rejected(keys):
    pem, _ = keys
    _, other_pub = generate_keypair()                  # attacker's key
    doc = _issue(pem)
    ok, issues = verify_license(doc, other_pub)
    assert not ok and any("signature invalid" in i for i in issues)


def test_embedded_key_is_never_trusted(keys):
    pem, _ = keys
    doc = _issue(pem)
    # no pinned key -> refuse, even though doc carries signer_public_key
    ok, issues = verify_license(doc, "")
    assert not ok and any("pinned" in i for i in issues)


def test_expiry(keys):
    pem, pub = keys
    t0 = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
    doc = _issue(pem, days=30, now=t0)
    ok, _ = verify_license(doc, pub, now=t0 + datetime.timedelta(days=29))
    assert ok
    ok, issues = verify_license(doc, pub, now=t0 + datetime.timedelta(days=31))
    assert not ok and any("expired" in i for i in issues)


def test_from_doc_and_status(keys):
    pem, pub = keys
    t0 = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
    lic = License.from_doc(_issue(pem, days=20, now=t0), pub, now=t0)
    s = lic.status(now=t0)
    assert s["licensed"] and s["customer"] == "Northwind Health"
    assert s["days_remaining"] == 20 and s["warning"]          # under 30d -> warn


# ── entitlement enforcement ─────────────────────────────────────────
def test_check_spec_limits(keys):
    pem, pub = keys
    spec = ClientSpec.from_dict(SPEC)                  # 2 tenants, 3 experts
    ok_lic = License.from_doc(_issue(pem, max_tenants=2, max_experts=3), pub)
    ok_lic.check_spec(spec)                            # within limits
    tight = License.from_doc(_issue(pem, max_tenants=1), pub)
    with pytest.raises(LicenseError, match="2 tenants but the license allows 1"):
        tight.check_spec(spec)
    tiny = License.from_doc(_issue(pem, max_experts=2), pub)
    with pytest.raises(LicenseError, match="3 experts but the license allows 2"):
        tiny.check_spec(spec)


def test_check_fleet(keys):
    pem, pub = keys
    lic = License.from_doc(_issue(pem, max_deployments=2), pub)
    lic.check_fleet(2)
    with pytest.raises(LicenseError, match="allows 2 deployment"):
        lic.check_fleet(3)


def test_deploy_enforces_license(keys):
    pem, pub = keys
    good = License.from_doc(_issue(pem, max_tenants=5, max_experts=10), pub)
    dep = deploy(SPEC, secret="s", license=good)
    assert len(dep.summary()["experts"]) == 3
    bad = License.from_doc(_issue(pem, max_experts=1), pub)
    with pytest.raises(LicenseError):
        deploy(SPEC, secret="s", license=bad)


# ── load_license policy (env-driven) ────────────────────────────────
def test_no_license_is_evaluation_mode(monkeypatch):
    monkeypatch.delenv("DAS_LICENSE", raising=False)
    monkeypatch.delenv("DAS_LICENSE_REQUIRED", raising=False)
    assert load_license() is None                      # eval mode, permitted


def test_required_without_license_fails(monkeypatch):
    monkeypatch.delenv("DAS_LICENSE", raising=False)
    monkeypatch.setenv("DAS_LICENSE_REQUIRED", "1")
    with pytest.raises(LicenseError, match="DAS_LICENSE_REQUIRED"):
        load_license()


def test_configured_license_loads_and_fails_closed(keys, tmp_path, monkeypatch):
    pem, pub = keys
    path = tmp_path / "license.json"
    path.write_text(json.dumps(_issue(pem, days=365)))
    monkeypatch.setenv("DAS_LICENSE", str(path))
    monkeypatch.setenv("DAS_LICENSE_PUBKEY", pub)
    lic = load_license()
    assert lic.customer == "Northwind Health"
    # tamper the file -> configured license must fail CLOSED, not fall back
    doc = json.loads(path.read_text()); doc["customer"] = "Evil Corp"
    path.write_text(json.dumps(doc))
    with pytest.raises(LicenseError, match="signature invalid"):
        load_license()


def test_pubkey_from_file(keys, tmp_path, monkeypatch):
    pem, pub = keys
    keyfile = tmp_path / "vendor.pub"; keyfile.write_text(pub + "\n")
    lic_path = tmp_path / "license.json"
    lic_path.write_text(json.dumps(_issue(pem)))
    monkeypatch.setenv("DAS_LICENSE", str(lic_path))
    monkeypatch.setenv("DAS_LICENSE_PUBKEY", str(keyfile))
    assert load_license().tier == "platform"


# ── CLI ─────────────────────────────────────────────────────────────
def test_cli_keygen_issue_verify(tmp_path, monkeypatch):
    from das.platform.cli import main
    monkeypatch.delenv("DAS_LICENSE_PUBKEY", raising=False)
    key = tmp_path / "vendor.pem"; lic = tmp_path / "license.json"
    assert main(["license", "keygen", "--out", str(key)]) == 0
    assert main(["license", "issue", "--key", str(key), "--customer", "Acme",
                 "--days", "90", "--max-tenants", "4", "--out", str(lic)]) == 0
    # verify needs the pinned pubkey; recover it from the PEM
    from cryptography.hazmat.primitives.serialization import load_pem_private_key
    pub = load_pem_private_key(key.read_bytes(), None).public_key().public_bytes_raw().hex()
    assert main(["license", "verify", str(lic), "--pubkey", pub]) == 0
    assert main(["license", "verify", str(lic)]) == 1          # no pinned key -> fail
    assert main(["license", "show", str(lic)]) == 0
