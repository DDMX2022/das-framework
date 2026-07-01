"""
das — the FDE deployment-engine CLI.

    das deploy   client.yaml [--save DIR] [--bundle OUT.json]
    das route    client.yaml --actor NAME --query "text"
    das offboard client.yaml --tenant NAME [--bundle OUT.json]
    das bundle   client.yaml --out OUT.json
    das verify   BUNDLE.json [--secret KEY | --pubkey HEX]

``deploy`` stands up a governed, multi-tenant expert fleet from a single spec.
The other verbs drive the lifecycle: route a query (with escalation), offboard a
tenant (provable deletion), emit the leave-behind compliance bundle, or verify a
bundle offline (delegates to the same engine as ``das-verify``).

The audit secret is read at runtime from the file/env the spec names, or from
``--secret``; it is never written to disk.
"""
from __future__ import annotations

import argparse
import json
import sys


def _print(obj):
    print(json.dumps(obj, indent=2, sort_keys=True))


def _deploy(args):
    from das.platform import deploy
    dep = deploy(args.spec, secret=args.secret)
    summary = dep.summary()
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


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="das", description="DAS FDE deployment engine.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("deploy", help="stand up a governed fleet from a client spec")
    d.add_argument("spec", help="path to client.yaml / client.json")
    d.add_argument("--secret", default=None, help="audit secret (else read per the spec)")
    d.add_argument("--save", default=None, metavar="DIR", help="persist state to DIR")
    d.add_argument("--bundle", default=None, metavar="OUT", help="also write a compliance bundle")
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

    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
