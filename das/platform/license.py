"""
das/platform/license.py
-----------------------
Offline subscription licensing for the commercial platform tier.

A license is an **Ed25519-signed JSON file** of entitlement claims (customer,
expiry, limits). The vendor signs it once with a private key; the shipped code
verifies it **offline** against a pinned public key — no license server, no
phone-home. That matters because DAS deploys inside the customer's trust
boundary (VPC / on-prem / air-gapped), where a call-home check would contradict
the product's own privacy pitch. A subscription is simply a short-dated license,
renewed by issuing a new signed file.

Trust model (deliberately strict):
  * Verification REQUIRES an explicitly pinned public key — from the
    ``public_key`` argument, the ``DAS_LICENSE_PUBKEY`` env (hex or a file path),
    or the ``VENDOR_PUBLIC_KEY_HEX`` constant a commercial build bakes in. The
    key embedded in the license document is informational only and is NEVER
    trusted for verification (an attacker who can re-sign with their own key can
    also embed it).
  * No license configured  -> evaluation mode (None; callers may proceed with a
    notice) — unless ``DAS_LICENSE_REQUIRED=1``, the commercial build's switch.
  * License configured but invalid / expired / tampered -> fail CLOSED
    (``LicenseError``), never a silent downgrade to evaluation mode.

Honesty note: a key check in source-available Python is a compliance mechanism,
not DRM. The EULA does the legal work; this makes entitlements auditable and
renewal operable. That is the industry-standard posture (GitLab EE, Rancher).

Vendor side:  das license keygen / das license issue
Customer side: das license verify / show; `das deploy` enforces automatically
when ``DAS_LICENSE`` points at the file.

Requires the ``[crypto]`` extra (cryptography) — same dependency as Ed25519
audit signing, reusing ``das.audit.generate_keypair``.
"""
from __future__ import annotations

import datetime
import json
import os
import secrets as _secrets
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

LICENSE_SCHEMA = "das.license/v1"

# A commercial build bakes the vendor's public key here (hex). The OSS tree
# ships None: licensing is then active only when DAS_LICENSE_PUBKEY is provided.
VENDOR_PUBLIC_KEY_HEX: Optional[str] = None

TIERS = ("core", "control-plane", "platform")
GRACE_WARNING_DAYS = 30


class LicenseError(Exception):
    """A configured license is missing, malformed, tampered, expired, or an
    entitlement was exceeded. Deployments fail closed on this."""


# ── time helpers ─────────────────────────────────────────────────────
def _utcnow() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _parse_ts(value: str, where: str) -> datetime.datetime:
    try:
        return datetime.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise LicenseError(f"{where}: bad timestamp {value!r}") from exc


def _fmt_ts(dt: datetime.datetime) -> str:
    return dt.astimezone(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── signing (vendor side) ────────────────────────────────────────────
def _canonical(payload: Dict[str, Any]) -> bytes:
    """The signed bytes: canonical JSON of the document minus its signature."""
    body = {k: v for k, v in payload.items() if k != "signature"}
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")


def issue_license(private_key, *, customer: str, days: int = 365,
                  tier: str = "platform",
                  max_deployments: Optional[int] = None,
                  max_tenants: Optional[int] = None,
                  max_experts: Optional[int] = None,
                  features: Optional[List[str]] = None,
                  now: Optional[datetime.datetime] = None) -> Dict[str, Any]:
    """Vendor-side: create and sign a license document.

    ``private_key`` is an Ed25519 PEM (bytes/str) or key object — the same
    format ``das.audit.generate_keypair`` produces. Limits of ``None`` mean
    unlimited. Returns the signed document (a plain dict, ready to serialize).
    """
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    if isinstance(private_key, (bytes, str)):
        pem = private_key.encode() if isinstance(private_key, str) else private_key
        private_key = serialization.load_pem_private_key(pem, password=None)
    if not isinstance(private_key, Ed25519PrivateKey):
        raise TypeError("private_key must be an Ed25519 private key (PEM bytes/str or object)")
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r} (choose from {TIERS})")
    if not customer or not str(customer).strip():
        raise ValueError("customer must be a non-empty string")
    if days < 1:
        raise ValueError("days must be >= 1")

    now = now or _utcnow()
    doc: Dict[str, Any] = {
        "schema": LICENSE_SCHEMA,
        "license_id": _secrets.token_hex(8),
        "customer": str(customer).strip(),
        "tier": tier,
        "issued_at": _fmt_ts(now),
        "expires_at": _fmt_ts(now + datetime.timedelta(days=days)),
        "entitlements": {
            "max_deployments": max_deployments,
            "max_tenants": max_tenants,
            "max_experts": max_experts,
            "features": sorted(features or []),
        },
        # informational only — verification always uses a PINNED key, never this
        "signer_public_key": private_key.public_key().public_bytes_raw().hex(),
    }
    doc["signature"] = private_key.sign(_canonical(doc)).hex()
    return doc


# ── verification (customer side) ─────────────────────────────────────
def verify_license(doc: Dict[str, Any], public_key: str,
                   now: Optional[datetime.datetime] = None) -> Tuple[bool, List[str]]:
    """Verify a license document against a PINNED vendor public key (hex).

    Checks: schema, required fields, Ed25519 signature over the canonical
    payload, and expiry. Returns (ok, issues). The embedded
    ``signer_public_key`` is deliberately ignored.
    """
    issues: List[str] = []
    if not isinstance(doc, dict):
        return False, ["document is not a JSON object"]
    if doc.get("schema") != LICENSE_SCHEMA:
        issues.append(f"schema is {doc.get('schema')!r}, expected {LICENSE_SCHEMA!r}")
    for key in ("customer", "issued_at", "expires_at", "entitlements", "signature"):
        if key not in doc:
            issues.append(f"missing field '{key}'")
    if issues:
        return False, issues

    if not public_key:
        return False, ["no trusted public key pinned — set DAS_LICENSE_PUBKEY "
                       "(hex or a file path) or pass --pubkey; the key embedded "
                       "in the license is never trusted"]
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        pub = Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key.strip()))
    except (ValueError, TypeError) as exc:
        return False, [f"bad public key: {exc}"]
    try:
        pub.verify(bytes.fromhex(doc["signature"]), _canonical(doc))
    except (InvalidSignature, ValueError):
        issues.append("signature invalid — the license was tampered with or was "
                      "signed by a different key")
        return False, issues

    now = now or _utcnow()
    issued = _parse_ts(doc["issued_at"], "issued_at")
    if now < issued:
        issues.append(f"license not yet valid — issued_at {doc['issued_at']} "
                      f"is in the future (clock skew or a forged date)")
        return False, issues
    expires = _parse_ts(doc["expires_at"], "expires_at")
    if now >= expires:
        issues.append(f"license expired at {doc['expires_at']}")
        return False, issues
    return True, []


# ── the runtime object ───────────────────────────────────────────────
@dataclass(frozen=True)
class License:
    """A VERIFIED license. Construct via ``License.from_doc`` /
    ``load_license`` only — holding one implies the signature already checked."""
    customer: str
    tier: str
    expires_at: datetime.datetime
    entitlements: Dict[str, Any] = field(default_factory=dict)
    license_id: str = ""

    @classmethod
    def from_doc(cls, doc: Dict[str, Any], public_key: str,
                 now: Optional[datetime.datetime] = None) -> "License":
        ok, issues = verify_license(doc, public_key, now=now)
        if not ok:
            raise LicenseError("invalid license: " + "; ".join(issues))
        return cls(
            customer=doc["customer"],
            tier=doc.get("tier", "platform"),
            expires_at=_parse_ts(doc["expires_at"], "expires_at"),
            entitlements=dict(doc.get("entitlements") or {}),
            license_id=doc.get("license_id", ""),
        )

    def days_remaining(self, now: Optional[datetime.datetime] = None) -> int:
        delta = self.expires_at - (now or _utcnow())
        return max(0, delta.days)

    def status(self, now: Optional[datetime.datetime] = None) -> Dict[str, Any]:
        now = now or _utcnow()
        days = self.days_remaining(now)
        expired = now >= self.expires_at
        if expired:
            warning = "license EXPIRED — deploys and grows fail closed; renew"
        elif days <= GRACE_WARNING_DAYS:
            warning = f"license expires in {days} days — renew soon"
        else:
            warning = None
        return {
            "licensed": True,
            "expired": expired,          # a long-running console must not lie
            "customer": self.customer,
            "tier": self.tier,
            "expires_at": _fmt_ts(self.expires_at),
            "days_remaining": days,
            "warning": warning,
            "entitlements": self.entitlements,
        }

    # ── enforcement ──────────────────────────────────────────────────
    def _limit(self, name: str) -> Optional[int]:
        v = self.entitlements.get(name)
        return None if v is None else int(v)

    def check_spec(self, spec, now: Optional[datetime.datetime] = None) -> None:
        """Enforce per-deployment entitlements against a ClientSpec. Raises
        ``LicenseError`` on any exceeded limit; expiry is re-checked so a
        long-running process can't outlive its license at the next deploy."""
        if (now or _utcnow()) >= self.expires_at:
            raise LicenseError(f"license for '{self.customer}' expired at "
                               f"{_fmt_ts(self.expires_at)} — renew to deploy")
        max_t = self._limit("max_tenants")
        if max_t is not None and len(spec.tenant_names) > max_t:
            raise LicenseError(
                f"spec declares {len(spec.tenant_names)} tenants but the license "
                f"allows {max_t} per deployment")
        max_e = self._limit("max_experts")
        if max_e is not None and len(spec.experts) > max_e:
            raise LicenseError(
                f"spec declares {len(spec.experts)} experts but the license "
                f"allows {max_e} per deployment")

    def check_fleet(self, n_deployments_after: int) -> None:
        """Fleet-level entitlement (used by the console / multi-client hosts)."""
        max_d = self._limit("max_deployments")
        if max_d is not None and n_deployments_after > max_d:
            raise LicenseError(
                f"license allows {max_d} deployment(s); this would be "
                f"#{n_deployments_after}")

    def check_expert_count(self, n_after: int,
                           now: Optional[datetime.datetime] = None) -> None:
        """Post-deploy entitlement: growing an expert must re-check the limit
        AND expiry — without this, a fleet licensed for N experts could grow
        past N through the very console it ships with."""
        if (now or _utcnow()) >= self.expires_at:
            raise LicenseError(f"license for '{self.customer}' expired at "
                               f"{_fmt_ts(self.expires_at)} — renew to grow")
        max_e = self._limit("max_experts")
        if max_e is not None and n_after > max_e:
            raise LicenseError(
                f"growing to {n_after} experts exceeds the license limit of "
                f"{max_e} per deployment")


# ── loading / policy ─────────────────────────────────────────────────
def resolve_public_key(explicit: Optional[str] = None) -> Optional[str]:
    """The pinned vendor key: explicit arg > DAS_LICENSE_PUBKEY (hex or a path
    to a file containing hex) > the baked-in VENDOR_PUBLIC_KEY_HEX."""
    cand = explicit or os.environ.get("DAS_LICENSE_PUBKEY") or VENDOR_PUBLIC_KEY_HEX
    if cand and os.path.exists(cand):
        with open(cand, "r", encoding="utf-8") as fh:
            cand = fh.read().strip()
    return cand


def load_license(path: Optional[str] = None, public_key: Optional[str] = None,
                 now: Optional[datetime.datetime] = None) -> Optional[License]:
    """Load-and-verify the license per the trust model.

    Returns a verified ``License``, or ``None`` for evaluation mode (nothing
    configured and not required). Raises ``LicenseError`` (fail closed) when a
    license IS configured but is unreadable/invalid/expired, or when
    ``DAS_LICENSE_REQUIRED=1`` and none is configured.
    """
    path = path or os.environ.get("DAS_LICENSE")
    required = os.environ.get("DAS_LICENSE_REQUIRED", "").lower() in ("1", "true", "yes")
    if not path:
        if required:
            raise LicenseError("DAS_LICENSE_REQUIRED is set but no license is "
                               "configured — set DAS_LICENSE to the license file")
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise LicenseError(f"cannot read license '{path}': {exc}") from exc
    return License.from_doc(doc, resolve_public_key(public_key) or "", now=now)


def evaluation_notice() -> str:
    return ("evaluation mode — no license configured (set DAS_LICENSE to a "
            "signed license file); not for production use")
