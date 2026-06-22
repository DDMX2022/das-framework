"""
ed25519_audit_demo.py — public-key-verifiable audit (closes SECURITY_REVIEW F7)
-------------------------------------------------------------------------------
The HMAC audit log is tamper-evident, but proving *authorship* of an exported log
needs the shared secret — so a regulator can't independently confirm it without
holding your key. Ed25519 fixes that: sign with a private key the writer keeps,
and anyone verifies authenticity with only the PUBLIC key.

This demo:
  1. generates an Ed25519 keypair,
  2. runs real governed operations on a control plane signing with the private key,
  3. exports the document (it embeds the public key),
  4. verifies authenticity with the PUBLIC KEY ONLY — no secret, and
  5. shows a wrong key and a tampered entry both rejected.

    pip install -e ".[crypto]"
    python examples/ed25519_audit_demo.py

Equivalent CLI:  das-verify das_audit.json --pubkey <public-hex>
"""
import json
import os
import tempfile

import numpy as np
from das.model import DASForest
from das.functional import softmax
from das.governance import ControlPlane
from das.audit import generate_keypair, verify_document

rng = np.random.default_rng(0)
D, LEAF, N = 16, [16, 13, 8, 2], 300
DATA = {}


def expert_data(key, n):
    c = np.zeros(D); c[(abs(hash(key)) % (D // 2)) * 2] = 4.0
    rule = np.random.default_rng(abs(hash(key)) % (2**31)).normal(0, 1, D)
    X = c + rng.normal(0, 1.0, (n, D))
    return X, (X @ rule > 0).astype(int)


def ce_grad(logits, y):
    p = softmax(logits); oh = np.zeros_like(p); oh[np.arange(len(y)), y] = 1.0
    return (p - oh) / len(y)


def train_fn(key, cp):
    DATA[key] = expert_data(key, N)
    def fn(forest, idx):
        leaf = forest.leaves[idx]; leaf.frozen = False
        X, y = DATA[key]
        for _ in range(300):
            i = rng.integers(0, len(X), 64); leaf.backward(ce_grad(leaf.forward(X[i]), y[i]), 0.05)
        leaf.frozen = True
        keys = [r["name"] for r in cp.experts] + [key]
        Xr = np.vstack([DATA[k][0] for k in keys])
        dr = np.concatenate([np.full(N, s) for s, _ in enumerate(keys)])
        for _ in range(400):
            i = rng.integers(0, len(Xr), 64); forest.router.train_step(Xr[i], dr[i], lr=0.15)
    return fn


def rule(s):
    print("\n" + "─" * 72 + f"\n{s}\n" + "─" * 72)


def show(doc, **kw):
    ok, issues = verify_document(doc, **kw)
    label = ("pubkey=" + kw["public_key"][:12] + "…") if kw.get("public_key") else "no key"
    print(f"  verify ({label:20s}) → {'VERIFIED ✓' if ok else 'REJECTED ✗'}"
          + ("" if ok else f"  ({issues[0][1]})"))
    return ok


def main():
    rule("1 · Generate an Ed25519 keypair")
    priv_pem, pub_hex = generate_keypair()
    print(f"  private key: kept by the writer ({len(priv_pem)} bytes PEM, never shared)")
    print(f"  public key : {pub_hex}   ← give this to the regulator")

    rule("2 · Run governed ops on a control plane that signs with the private key")
    DATA["acme-tax"] = expert_data("acme-tax", N)
    forest = DASForest(D, LEAF, num_leaves=1, seed=7)
    leaf = forest.leaves[0]; leaf.frozen = False
    X, y = DATA["acme-tax"]
    for _ in range(300):
        i = rng.integers(0, N, 64); leaf.backward(ce_grad(leaf.forward(X[i]), y[i]), 0.05)
    leaf.frozen = True
    cp = ControlPlane(forest, seed_tenant="acme", seed_name="acme-tax", private_key=priv_pem)
    cp.register_tenant("root", "globex")
    eid = cp.graft("root", "globex", "globex-vision", train_fn("globex-vision", cp))
    cp.prune("root", eid)
    print(f"  audit scheme: {cp.audit.scheme}")
    print(f"  entries: {len(cp.audit.entries)} (init · graft · prune), signed with the private key")

    rule("3 · Export — the document carries the public key, never the private one")
    path = os.path.join(tempfile.mkdtemp(), "das_audit.json")
    cp.audit.export(path)
    doc = json.load(open(path))
    print(f"  wrote {path}")
    print(f"  scheme={doc['scheme']!r}")
    print(f"  embedded public_key={doc['public_key'][:24]}…   private key present? "
          f"{'private' in json.dumps(doc).lower()}")

    rule("4 · A regulator verifies authenticity with the PUBLIC KEY ONLY")
    show(doc, public_key=pub_hex)                 # pinned key, out-of-band — proves authorship
    print("  (no secret was needed — that's the F7 fix)")

    rule("5 · Forgery & tamper are rejected")
    _, other_pub = generate_keypair()             # wrong key
    show(doc, public_key=other_pub)               # → REJECTED
    bad = json.loads(json.dumps(doc))             # edited content
    bad["entries"][1]["detail"] += " (altered)"
    show(bad, public_key=pub_hex)                 # → REJECTED

    rule("Done — an audit log a third party can verify with only your public key")
    print(f"  CLI:  das-verify {os.path.basename(path)} --pubkey {pub_hex[:16]}…")
    print("  Closes SECURITY_REVIEW.md F7 (authenticity without sharing a secret).")

    ok = (show(doc, public_key=pub_hex)
          and not verify_document(doc, public_key=other_pub)[0]
          and not verify_document(bad, public_key=pub_hex)[0])
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
