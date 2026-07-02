"""
das — the FDE deployment-engine CLI.

    das deploy   client.yaml [--save DIR] [--bundle OUT.json] [--license LIC.json]
    das route    client.yaml --actor NAME --query "text"
    das offboard client.yaml --tenant NAME [--bundle OUT.json]
    das bundle   client.yaml --out OUT.json
    das verify   BUNDLE.json [--secret KEY | --pubkey HEX]

    das license keygen --out vendor_key.pem          # vendor: create signing keypair
    das license issue  --key vendor_key.pem --customer NAME [--days 365] [...limits]
    das license verify LIC.json --pubkey HEX_OR_FILE # customer: offline check
    das license show   LIC.json                      # claims without verification

``deploy`` stands up a governed, multi-tenant expert fleet from a single spec.
The other verbs drive the lifecycle: route a query (with escalation), offboard a
tenant (provable deletion), emit the leave-behind compliance bundle, or verify a
bundle offline (delegates to the same engine as ``das-verify``).

Licensing is offline: a license is an Ed25519-signed claims file verified
against a pinned vendor public key (DAS_LICENSE_PUBKEY) — no phone-home, so it
works air-gapped. No license configured = evaluation mode (noticed, permitted);
a configured-but-invalid license fails closed. See das/platform/license.py.

The audit secret is read at runtime from the file/env the spec names, or from
``--secret``; it is never written to disk.
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def _print(obj):
    print(json.dumps(obj, indent=2, sort_keys=True))


def _deploy(args):
    from das.platform import deploy
    from das.platform.license import evaluation_notice, load_license
    lic = load_license(path=args.license)          # fail-closed on invalid; None = eval
    dep = deploy(args.spec, secret=args.secret, license=lic)
    summary = dep.summary()
    summary["license"] = lic.status() if lic else {"licensed": False,
                                                   "notice": evaluation_notice()}
    if args.save:
        dep.save(args.save)
        summary["saved_to"] = args.save
    if args.bundle:
        summary["bundle"] = dep.export_bundle(args.bundle)
        summary["bundle_path"] = args.bundle
    _print(summary)
    return 0 if summary["audit_ok"] else 1


def _route(args):
    from das.platform import deploy
    dep = deploy(args.spec, secret=args.secret)
    _print(dep.route(args.actor, args.query))
    return 0


def _offboard(args):
    from das.platform import deploy
    dep = deploy(args.spec, secret=args.secret)
    result = dep.offboard(args.tenant, actor=args.actor)
    out = {"offboarded": args.tenant, **result, "audit_ok": dep.verify()["ok"]}
    if args.bundle:
        out["bundle"] = dep.export_bundle(args.bundle)
        out["bundle_path"] = args.bundle
    _print(out)
    return 0 if result.get("non_interference") else 1


def _bundle(args):
    from das.platform import deploy
    dep = deploy(args.spec, secret=args.secret)
    manifest = dep.export_bundle(args.out)
    _print({"bundle_path": args.out, "manifest": manifest})
    return 0


def _verify(args):
    # Delegate to the existing offline verifier — a bundle is a valid audit doc.
    from das.audit_verify import main as verify_main
    argv = [args.file]
    if args.secret:
        argv += ["--secret", args.secret]
    if args.pubkey:
        argv += ["--pubkey", args.pubkey]
    return verify_main(argv)


# ── licensing (vendor + customer) ────────────────────────────────────
def _license_keygen(args):
    from das.audit import generate_keypair
    pem, pub_hex = generate_keypair()
    with open(args.out, "wb") as fh:
        fh.write(pem)
    os.chmod(args.out, 0o600)
    _print({"private_key": args.out,
            "public_key_hex": pub_hex,
            "note": "keep the PEM secret (vendor-side only); ship/pin the public "
                    "hex in customer environments via DAS_LICENSE_PUBKEY"})
    return 0


def _license_issue(args):
    from das.platform.license import issue_license
    with open(args.key, "rb") as fh:
        pem = fh.read()
    doc = issue_license(
        pem, customer=args.customer, days=args.days, tier=args.tier,
        max_deployments=args.max_deployments, max_tenants=args.max_tenants,
        max_experts=args.max_experts, features=args.feature or [],
    )
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)
    _print({"license": args.out, "customer": doc["customer"], "tier": doc["tier"],
            "expires_at": doc["expires_at"], "entitlements": doc["entitlements"]})
    return 0


def _license_verify(args):
    from das.platform.license import resolve_public_key, verify_license
    with open(args.file, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    ok, issues = verify_license(doc, resolve_public_key(args.pubkey) or "")
    out = {"file": args.file, "customer": doc.get("customer"),
           "expires_at": doc.get("expires_at"),
           "result": "VALID" if ok else "INVALID", "issues": issues}
    _print(out)
    return 0 if ok else 1


def _license_show(args):
    with open(args.file, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    doc.pop("signature", None)
    doc["note"] = "claims shown WITHOUT verification — run `das license verify`"
    _print(doc)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="das", description="DAS FDE deployment engine.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("deploy", help="stand up a governed fleet from a client spec")
    d.add_argument("spec", help="path to client.yaml / client.json")
    d.add_argument("--secret", default=None, help="audit secret (else read per the spec)")
    d.add_argument("--save", default=None, metavar="DIR", help="persist state to DIR")
    d.add_argument("--bundle", default=None, metavar="OUT", help="also write a compliance bundle")
    d.add_argument("--license", default=None, metavar="LIC",
                   help="license file (else $DAS_LICENSE; unset = evaluation mode)")
    d.set_defaults(func=_deploy)

    r = sub.add_parser("route", help="route a query with escalation policy applied")
    r.add_argument("spec")
    r.add_argument("--actor", required=True, help="acting principal (X-DAS-Actor)")
    r.add_argument("--query", required=True, help="the query text")
    r.add_argument("--secret", default=None)
    r.set_defaults(func=_route)

    o = sub.add_parser("offboard", help="delete a tenant (provable right-to-be-forgotten)")
    o.add_argument("spec")
    o.add_argument("--tenant", required=True)
    o.add_argument("--actor", default=None, help="defaults to the spec root admin")
    o.add_argument("--secret", default=None)
    o.add_argument("--bundle", default=None, metavar="OUT", help="write a post-deletion bundle")
    o.set_defaults(func=_offboard)

    b = sub.add_parser("bundle", help="write the offline-verifiable compliance bundle")
    b.add_argument("spec")
    b.add_argument("--out", required=True)
    b.add_argument("--secret", default=None)
    b.set_defaults(func=_bundle)

    v = sub.add_parser("verify", help="verify a compliance bundle offline")
    v.add_argument("file", help="path to a bundle / exported audit document")
    v.add_argument("--secret", default=None)
    v.add_argument("--pubkey", default=None)
    v.set_defaults(func=_verify)

    lic = sub.add_parser("license", help="offline subscription licensing")
    lsub = lic.add_subparsers(dest="license_cmd", required=True)

    lk = lsub.add_parser("keygen", help="vendor: generate an Ed25519 signing keypair")
    lk.add_argument("--out", required=True, metavar="PEM", help="private-key PEM path")
    lk.set_defaults(func=_license_keygen)

    li = lsub.add_parser("issue", help="vendor: sign a license file")
    li.add_argument("--key", required=True, metavar="PEM", help="vendor private-key PEM")
    li.add_argument("--customer", required=True)
    li.add_argument("--days", type=int, default=365, help="subscription length (default 365)")
    li.add_argument("--tier", default="platform", choices=["core", "control-plane", "platform"])
    li.add_argument("--max-deployments", type=int, default=None)
    li.add_argument("--max-tenants", type=int, default=None, help="per deployment")
    li.add_argument("--max-experts", type=int, default=None, help="per deployment")
    li.add_argument("--feature", action="append", help="repeatable feature flag")
    li.add_argument("--out", required=True, metavar="JSON")
    li.set_defaults(func=_license_issue)

    lv = lsub.add_parser("verify", help="customer: verify a license offline")
    lv.add_argument("file")
    lv.add_argument("--pubkey", default=None,
                    help="pinned vendor public key (hex or file; else $DAS_LICENSE_PUBKEY)")
    lv.set_defaults(func=_license_verify)

    ls = lsub.add_parser("show", help="print license claims (no verification)")
    ls.add_argument("file")
    ls.set_defaults(func=_license_show)

    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except Exception as e:  # LicenseError, SpecError, missing files — operator-facing
        from das.platform import LicenseError, SpecError
        if isinstance(e, (LicenseError, SpecError, OSError)):
            print(f"das: error: {e}", file=sys.stderr)
            return 1
        raise


if __name__ == "__main__":
    sys.exit(main())
