"""
Tests for API deployment hardening — F3 (default-secret guard) and F2 (authn-proxy
contract). Needs flask, so skipped in the torch-/flask-free CI core (like the other
API tests); runs wherever the [web] extra is installed.
"""
import pytest

pytest.importorskip("flask")

import apps.governance_api as api


# ── F3: production startup guard ────────────────────────────────────────
def test_dev_config_is_never_fatal():
    assert api._startup_errors("development", "das-dev-key", False, None) == []


def test_prod_refuses_default_secret_and_missing_proxy():
    errs = api._startup_errors("production", "das-dev-key", False, None)
    assert len(errs) == 2                                    # both F3 and F2 fire


def test_prod_ok_with_privkey_or_strong_secret_plus_proxy():
    # Ed25519 private key satisfies F3; proxy secret satisfies F2
    assert api._startup_errors("production", "das-dev-key", True, "tok") == []
    # a strong HMAC secret + proxy secret also passes
    assert api._startup_errors("production", "strong-random-secret", False, "tok") == []


def test_prod_strong_secret_but_no_proxy_still_fails_f2():
    errs = api._startup_errors("production", "strong-random-secret", False, None)
    assert len(errs) == 1 and "F2" in errs[0]


# ── F2: trusted-proxy enforcement at request time ───────────────────────
def test_proxy_auth_enforced_when_configured(monkeypatch):
    monkeypatch.setattr(api, "PROXY_SECRET", "tok")
    c = api.app.test_client()
    assert c.get("/health").status_code == 200               # liveness probe exempt
    assert c.get("/experts", headers={"X-DAS-Actor": "root"}).status_code == 401
    assert c.get("/experts", headers={"X-DAS-Actor": "root",
                                      "X-DAS-Proxy-Auth": "tok"}).status_code == 200
    assert c.get("/experts", headers={"X-DAS-Actor": "root",
                                      "X-DAS-Proxy-Auth": "wrong"}).status_code == 401


def test_no_enforcement_when_unset(monkeypatch):
    monkeypatch.setattr(api, "PROXY_SECRET", None)
    c = api.app.test_client()
    assert c.get("/experts", headers={"X-DAS-Actor": "root"}).status_code == 200
