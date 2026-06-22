"""
audit_export_demo.py — the audit log as an exportable compliance artifact
-------------------------------------------------------------------------
The governance moat isn't the model — it's being able to hand a regulator a
signed, self-contained document that proves exactly what changed, in what order,
and that no certified expert was disturbed. This demo:

  1. runs real governed operations (graft / prune) on a control plane, so the
     audit entries carry REAL SHA-256 weight fingerprints,
  2. exports the signed log to a portable JSON document,
  3. verifies it offline two ways — keyless (structural) and with the key
     (authenticated) — exactly what `das-verify` does, and
  4. shows what each verification mode does and does NOT catch when tampered.

Pure NumPy, no downloads.  Equivalent CLI:  das-verify das_audit.json [--secret KEY]
"""
import copy
import json
import tempfile
import os

import numpy as np
from das.model import DASForest
from das.functional import softmax
from das.governance import ControlPlane
from das.audit import verify_document

SECRET = "prod-secret"
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


def make_train_fn(key, cp):
    DATA[key] = expert_data(key, N)
    def train_fn(forest, idx):
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
    return train_fn


def rule(s):
    print("\n" + "─" * 72 + f"\n{s}\n" + "─" * 72)


def report(doc, secret=None):
    ok, issues = verify_document(doc, secret=secret)
    mode = "authenticated (key)" if secret else "keyless (structural)"
    tag = "VERIFIED ✓" if ok else "TAMPERED ✗"
    print(f"  {mode:24s} → {tag}")
    for idx, reason in issues:
        print(f"        entry {idx}: {reason}")
    return ok


def main():
    rule("1 · Run real governed operations (real weight fingerprints in the log)")
    DATA["acme-tax"] = expert_data("acme-tax", N)
    forest = DASForest(D, LEAF, num_leaves=1, seed=7)
    leaf = forest.leaves[0]; leaf.frozen = False
    X, y = DATA["acme-tax"]
    for _ in range(300):
        i = rng.integers(0, N, 64); leaf.backward(ce_grad(leaf.forward(X[i]), y[i]), 0.05)
    leaf.frozen = True
    cp = ControlPlane(forest, seed_tenant="acme", seed_name="acme-tax", secret=SECRET)
    cp.register_tenant("root", "globex")
    eid = cp.graft("root", "globex", "globex-vision", make_train_fn("globex-vision", cp))
    cp.prune("root", eid)            # right-to-be-forgotten
    print(f"  events: init seed expert · graft 'globex-vision' · prune it (eid {eid})")
    print(f"  audit entries: {len(cp.audit.entries)}  (each signed + chained)")

    rule("2 · Export the signed log to a portable compliance document")
    path = os.path.join(tempfile.mkdtemp(), "das_audit.json")
    cp.audit.export(path, meta={"note": "demo export"})
    doc = json.load(open(path))
    print(f"  wrote {path}")
    print(f"  scheme={doc['scheme']!r}  entries={doc['count']}  head={doc['head'][:16]}…")
    print(f"  a sample entry's recorded fingerprints: "
          f"{list(doc['entries'][-1]['payload'].items())[:1]}")

    rule("3 · Verify it offline — what a recipient runs (no system access)")
    report(doc, secret=None)        # keyless — chain + fingerprints
    report(doc, secret=SECRET)      # authenticated — + HMAC signatures
    print(f"\n  CLI equivalent:  das-verify {os.path.basename(path)} [--secret {SECRET}]")

    rule("4 · Tamper, and watch verification catch it")

    t = copy.deepcopy(doc)          # (a) edit a recorded weight fingerprint
    k = next(iter(t["entries"][0]["payload"]))
    t["entries"][0]["payload"][k] = "0000000000000000"
    print("  (a) edited a recorded weight fingerprint:")
    report(t, secret=None)          # keyless catches it (fingerprint ≠ signed hash)

    t = copy.deepcopy(doc)          # (b) edit an entry's description
    t["entries"][1]["detail"] += " (quietly altered)"
    print("  (b) edited an entry's description text:")
    keyless_ok = report(t, secret=None)     # keyless does NOT catch a pure text edit
    auth_ok = report(t, secret=SECRET)      # the key does
    print("      → text edits need the key (HMAC) to detect; structural checks alone don't.")

    t = copy.deepcopy(doc)          # (c) delete an entry
    if len(t["entries"]) >= 2:
        del t["entries"][1]
        print("  (c) deleted an entry:")
        report(t, secret=None)      # keyless catches it (chain broken)

    rule("Done — a signed log you can export and anyone can independently verify")
    print("  Keyless: chain + fingerprints (catches reorder/insert/delete/fingerprint edits).")
    print("  With the key: full authenticity (catches any content edit).")
    print("  Roadmap (SECURITY_REVIEW.md): Ed25519 so authenticity needs only a public key.")

    # exit non-zero if the guarantees we assert here don't hold
    healthy = report(doc) and report(doc, SECRET)
    ok = healthy and (keyless_ok is True) and (auth_ok is False)
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
