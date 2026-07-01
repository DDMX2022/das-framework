"""
das/platform/bundle.py
----------------------
The compliance "leave-behind". An FDE deploys, then rolls to the next client; the
client's security/compliance team stays. What survives the FDE's departure is the
*proof*, and this packages it: a dated, self-contained document wrapping the
signed audit export plus a manifest of what was deployed and how to verify it
offline — no system access, and (keyless) no secret.

    write_bundle(deployment, "northwind_2026-07-02.json")
    # recipient, later, on any machine:
    #   das-verify northwind_2026-07-02.json

The audit document inside is exactly what ``ControlPlane.export_audit`` produces
(chain + real weight fingerprints), so ``das-verify`` / ``das.audit.verify_document``
accept it directly.
"""
from __future__ import annotations

import datetime
import json
from typing import Optional

BUNDLE_SCHEMA = "das.compliance-bundle/v1"


def build_bundle(deployment, actor: Optional[str] = None) -> dict:
    """Assemble (but don't write) the compliance bundle for a deployment.

    The bundle *is* the exportable audit document (its keys sit at the top level,
    so ``das-verify bundle.json`` verifies it directly) with the platform's
    compliance metadata added under distinct, non-colliding keys.
    """
    actor = actor or deployment.spec.root
    audit_doc = deployment.cp.export_audit(actor)
    verification = deployment.cp.verify_audit(actor)
    generated = datetime.datetime.now(datetime.timezone.utc).isoformat()
    signed = "ed25519" if deployment.spec.audit.private_key_file else "hmac"

    bundle = dict(audit_doc)  # top-level audit keys: scheme/entries/head/…
    bundle["bundle_schema"] = BUNDLE_SCHEMA
    bundle["client"] = deployment.spec.client
    bundle["generated_at"] = generated
    bundle["generated_by"] = actor
    bundle["manifest"] = {
        "tenants": sorted(deployment.cp.tenants),
        "experts": [{"eid": r["eid"], "tenant": r["tenant"], "name": r["name"]}
                    for r in deployment.cp.experts],
        "audit_signing": signed,
        "audit_chain_ok": verification["ok"],
        "audit_entries": verification["entries"],
    }
    bundle["verify_instructions"] = {
        "tool": "das-verify",
        "keyless": "das-verify <bundle.json>",
        "hmac": "das-verify <bundle.json> --secret $SECRET",
        "ed25519": "das-verify <bundle.json> --pubkey <public-hex>",
        "note": ("This file is itself the verifiable audit document. Keyless "
                 "verification proves the chain and weight fingerprints are "
                 "internally consistent; a key proves authorship."),
    }
    return bundle


def write_bundle(deployment, path: str, actor: Optional[str] = None) -> dict:
    """Write the compliance bundle to ``path`` and return the manifest section."""
    bundle = build_bundle(deployment, actor=actor)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(bundle, fh, indent=2, sort_keys=True)
    return bundle["manifest"]
