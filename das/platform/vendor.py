"""
das/platform/vendor.py
----------------------
VENDOR-SIDE license administration — the superadmin's store. This module (and
the console over it) is never shipped to customers: it holds the Ed25519
SIGNING key that mints licenses. Customers only ever hold license files and the
pinned PUBLIC key.

File layout under ``root`` (``DAS_VENDOR_DIR``):

    signing_key.pem          the private key (0600; created on first use)
    public_key.hex           what commercial builds bake in / customers pin
    licenses/<id>.json       one record per issued license:
                             {"record_version":1, "status": ..., "license": {signed doc}}

Statuses: ``active`` (stored) -> effectively ``expiring`` (<= 30 days left) or
``expired`` (past); ``revoked`` and ``superseded`` (renewed) are terminal.

Revocation honesty: licenses verify OFFLINE, so revoking here is registry
bookkeeping + refusing renewal — it cannot remotely kill a file a customer
already holds. That is the deliberate price of no-phone-home; the mitigation is
short subscription windows (the default 365 days, or less).
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from .license import (
    GRACE_WARNING_DAYS,
    LicenseError,
    _fmt_ts,
    _parse_ts,
    _utcnow,
    issue_license,
    verify_license,
)

RECORD_VERSION = 1


class VendorStore:
    """The superadmin's registry of issued licenses + the signing keypair."""

    def __init__(self, root: str):
        self.root = root
        self.key_path = os.path.join(root, "signing_key.pem")
        self.pub_path = os.path.join(root, "public_key.hex")
        self.lic_dir = os.path.join(root, "licenses")
        os.makedirs(self.lic_dir, exist_ok=True)

    # ── keypair ──────────────────────────────────────────────────────
    def ensure_keypair(self) -> str:
        """Create the signing keypair on first use; return the public hex."""
        if not os.path.exists(self.key_path):
            from das.audit import generate_keypair
            pem, pub_hex = generate_keypair()
            # created 0600 from the first byte — no default-perms window
            fd = os.open(self.key_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "wb") as fh:
                fh.write(pem)
            with open(self.pub_path, "w", encoding="utf-8") as fh:
                fh.write(pub_hex + "\n")
        return self.public_key_hex

    @property
    def public_key_hex(self) -> str:
        with open(self.pub_path, "r", encoding="utf-8") as fh:
            return fh.read().strip()

    def _private_pem(self) -> bytes:
        with open(self.key_path, "rb") as fh:
            return fh.read()

    # ── records ──────────────────────────────────────────────────────
    def _path(self, license_id: str) -> str:
        safe = "".join(c for c in license_id if c.isalnum())
        if not safe or safe != license_id:
            raise LicenseError(f"bad license id {license_id!r}")
        return os.path.join(self.lic_dir, f"{safe}.json")

    def _write(self, record: Dict[str, Any]) -> None:
        with open(self._path(record["license"]["license_id"]), "w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2, sort_keys=True)

    def get(self, license_id: str) -> Dict[str, Any]:
        path = self._path(license_id)
        if not os.path.exists(path):
            raise LicenseError(f"no license with id '{license_id}'")
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def _effective_status(self, record: Dict[str, Any], now=None) -> str:
        status = record.get("status", "active")
        if status != "active":
            return status
        now = now or _utcnow()
        expires = _parse_ts(record["license"]["expires_at"], "expires_at")
        if now >= expires:
            return "expired"
        if (expires - now).days <= GRACE_WARNING_DAYS:
            return "expiring"
        return "active"

    def list(self, now=None) -> List[Dict[str, Any]]:
        """Registry rows, newest first, with the effective status derived."""
        rows = []
        for name in os.listdir(self.lic_dir):
            if not name.endswith(".json"):
                continue
            with open(os.path.join(self.lic_dir, name), "r", encoding="utf-8") as fh:
                record = json.load(fh)
            lic = record["license"]
            expires = _parse_ts(lic["expires_at"], "expires_at")
            rows.append({
                "license_id": lic["license_id"],
                "customer": lic["customer"],
                "tier": lic.get("tier", "platform"),
                "issued_at": lic["issued_at"],
                "expires_at": lic["expires_at"],
                "days_remaining": max(0, (expires - (now or _utcnow())).days),
                "entitlements": lic.get("entitlements", {}),
                "status": self._effective_status(record, now=now),
                "revoked_reason": record.get("revoked_reason"),
                "renewed_from": record.get("renewed_from"),
                "superseded_by": record.get("superseded_by"),
            })
        rows.sort(key=lambda r: r["issued_at"], reverse=True)
        return rows

    # ── lifecycle ────────────────────────────────────────────────────
    def issue(self, *, customer: str, days: int = 365, tier: str = "platform",
              max_deployments: Optional[int] = None,
              max_tenants: Optional[int] = None,
              max_experts: Optional[int] = None,
              features: Optional[List[str]] = None,
              renewed_from: Optional[str] = None,
              now=None) -> Dict[str, Any]:
        self.ensure_keypair()
        doc = issue_license(
            self._private_pem(), customer=customer, days=days, tier=tier,
            max_deployments=max_deployments, max_tenants=max_tenants,
            max_experts=max_experts, features=features, now=now,
        )
        record = {"record_version": RECORD_VERSION, "status": "active",
                  "license": doc}
        if renewed_from:
            record["renewed_from"] = renewed_from
        self._write(record)
        return record

    def revoke(self, license_id: str, reason: str = "") -> Dict[str, Any]:
        """Registry-side revocation (see module docstring for what this can and
        cannot do). Refuses to revoke a superseded record (revoke the renewal)."""
        record = self.get(license_id)
        if record.get("status") == "superseded":
            raise LicenseError(f"license '{license_id}' was superseded by "
                               f"'{record.get('superseded_by')}' — revoke that one")
        record["status"] = "revoked"
        record["revoked_at"] = _fmt_ts(_utcnow())
        record["revoked_reason"] = reason or "unspecified"
        self._write(record)
        return record

    def renew(self, license_id: str, days: int = 365, now=None) -> Dict[str, Any]:
        """Issue a NEW license with the same claims and a fresh expiry; the old
        record is marked superseded. Revoked licenses are not renewable."""
        old = self.get(license_id)
        if old.get("status") == "revoked":
            raise LicenseError(f"license '{license_id}' is revoked — issue a new "
                               f"license instead of renewing")
        lic = old["license"]
        ent = lic.get("entitlements", {})
        new = self.issue(
            customer=lic["customer"], days=days, tier=lic.get("tier", "platform"),
            max_deployments=ent.get("max_deployments"),
            max_tenants=ent.get("max_tenants"),
            max_experts=ent.get("max_experts"),
            features=ent.get("features"),
            renewed_from=license_id, now=now,
        )
        old["status"] = "superseded"
        old["superseded_by"] = new["license"]["license_id"]
        self._write(old)
        return new

    def verify_doc(self, doc: Dict[str, Any]):
        """Check any pasted license document against OUR public key."""
        self.ensure_keypair()
        return verify_license(doc, self.public_key_hex)
