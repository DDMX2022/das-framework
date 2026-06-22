"""
das-verify — independently verify an exported DAS audit document.

    das-verify audit.json                 # keyless structural verification
    das-verify audit.json --secret KEY    # + HMAC authenticity check

Structural verification (no key) detects reordering, insertion, deletion, and any
edit to the recorded weight fingerprints — enough for a recipient to confirm the
document is internally consistent and unaltered. Authenticity (proving the log was
produced by the holder of the secret, not fabricated wholesale) needs the HMAC
key: hand a contracted auditor the key out-of-band, or see SECURITY_REVIEW.md for
the planned asymmetric (Ed25519) signing that makes authenticity verifiable with
only a public key.

Exit code 0 = verified, 1 = tampered/invalid, 2 = could not read the file.
"""
import argparse
import json
import sys

from das.audit import verify_document


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="das-verify",
        description="Independently verify an exported DAS audit document.",
    )
    ap.add_argument("file", help="path to an exported audit JSON document")
    ap.add_argument("--secret", default=None,
                    help="HMAC secret for the authenticity check "
                         "(omit for keyless structural verification)")
    args = ap.parse_args(argv)

    try:
        with open(args.file) as f:
            doc = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"could not read '{args.file}': {e}", file=sys.stderr)
        return 2

    ok, issues = verify_document(doc, secret=args.secret)
    d = doc if isinstance(doc, dict) else {}
    n = d.get("count", len(d.get("entries", [])))
    mode = "authenticated (HMAC + structural)" if args.secret else "structural (keyless)"

    print(f"document : {args.file}")
    print(f"scheme   : {d.get('scheme', '?')}")
    print(f"created  : {d.get('created', '?')}")
    print(f"entries  : {n}")
    print(f"head     : {d.get('head', '?')}")
    print(f"mode     : {mode}")

    if ok:
        detail = "chain intact, fingerprints consistent"
        if args.secret:
            detail += ", signatures valid"
        print(f"RESULT   : VERIFIED ✓ — {detail}")
        if not args.secret:
            print("note     : authenticity not checked — re-run with --secret to confirm authorship")
        return 0

    print(f"RESULT   : TAMPERED ✗ — {len(issues)} issue(s):")
    for idx, reason in issues:
        print(f"   entry {idx}: {reason}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
