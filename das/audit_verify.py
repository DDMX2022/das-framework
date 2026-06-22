"""
das-verify — independently verify an exported DAS audit document.

    das-verify audit.json                      # keyless structural verification
    das-verify audit.json --secret KEY         # + HMAC authenticity (symmetric docs)
    das-verify audit.json --pubkey KEY_OR_FILE # + Ed25519 authenticity (asymmetric docs)

Structural verification (no key) detects reordering, insertion, deletion, and any
edit to the recorded weight fingerprints — enough to confirm the document is
internally consistent and unaltered.

Authenticity proves authorship:
  * HMAC docs    need the shared --secret (symmetric).
  * Ed25519 docs need only the --pubkey (asymmetric) — give a regulator the public
    key out-of-band and they can verify your log with no secret and no system
    access. Pin --pubkey rather than trusting the key embedded in the document.

Exit code 0 = verified, 1 = tampered/invalid, 2 = could not read the file.
"""
import argparse
import json
import os
import sys

from das.audit import verify_document


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="das-verify",
        description="Independently verify an exported DAS audit document.",
    )
    ap.add_argument("file", help="path to an exported audit JSON document")
    ap.add_argument("--secret", default=None,
                    help="HMAC secret for hmac-signed documents")
    ap.add_argument("--pubkey", default=None,
                    help="Ed25519 public key (hex, or a path to a file containing it) "
                         "for ed25519-signed documents")
    args = ap.parse_args(argv)

    try:
        with open(args.file) as f:
            doc = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"could not read '{args.file}': {e}", file=sys.stderr)
        return 2

    pubkey = args.pubkey
    if pubkey and os.path.exists(pubkey):
        pubkey = open(pubkey).read().strip()

    ok, issues = verify_document(doc, secret=args.secret, public_key=pubkey)
    d = doc if isinstance(doc, dict) else {}
    scheme = d.get("scheme", "")
    n = d.get("count", len(d.get("entries", [])))

    if "ed25519" in scheme:
        if pubkey:
            mode = "authenticated (Ed25519, pinned public key)"
        elif d.get("public_key"):
            mode = "authenticated (Ed25519, key embedded in document)"
        else:
            mode = "structural only (no public key available)"
    elif args.secret:
        mode = "authenticated (HMAC + structural)"
    else:
        mode = "structural (keyless)"

    print(f"document : {args.file}")
    print(f"scheme   : {scheme or '?'}")
    print(f"created  : {d.get('created', '?')}")
    print(f"entries  : {n}")
    print(f"head     : {d.get('head', '?')}")
    print(f"mode     : {mode}")

    if ok:
        print(f"RESULT   : VERIFIED ✓ — {mode}")
        if "ed25519" in scheme and not pubkey and d.get("public_key"):
            print("note     : verified against the key embedded in the document — pin "
                  "--pubkey (obtained out-of-band) to prove authorship")
        elif "ed25519" not in scheme and not args.secret:
            print("note     : authenticity not checked — re-run with --secret to confirm authorship")
        return 0

    print(f"RESULT   : TAMPERED ✗ — {len(issues)} issue(s):")
    for idx, reason in issues:
        print(f"   entry {idx}: {reason}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
