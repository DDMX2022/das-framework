"""
governance_api.py
-----------------
A REST API for the governance CONTROL PLANE (das/governance.py) — the deployable
unit. Unlike serve.py (a torch MNIST *inference* API), this serves the *governed
fleet*: routing with provenance, right-to-be-forgotten, and the tamper-evident
audit log. It is NumPy + Flask only (no torch), so it containerizes light.

State & secret:
  DAS_STATE         directory holding a saved control plane (forest.npz +
                    control_plane.json + audit.json). If set and present, it's
                    loaded; if set but empty, a demo fleet is bootstrapped and
                    saved there; if unset, a demo fleet runs in memory.
  DAS_AUDIT_SECRET  HMAC secret for the audit log. NEVER written to disk — it
                    must be supplied at boot to validate/extend the chain.
  DAS_AUDIT_PRIVKEY path to an Ed25519 private-key PEM. If set, the audit log is
                    signed asymmetrically so /audit/export is verifiable with only
                    the public key (no shared secret). Overrides HMAC. Optional.
  DAS_PORT          listen port (default 5070).

Identity:
  The acting principal is taken from the `X-DAS-Actor` header (or `actor` in the
  JSON body), default "root". This is identity-by-ASSERTION — RBAC is enforced by
  the control plane, but authentication is intentionally out of scope here; a
  real deployment puts an authn proxy (mTLS / OIDC) in front. We're honest about
  that rather than pretending the header is a credential.

Endpoints:
  GET  /health              fleet status + audit-chain health
  GET  /experts             registry (tenant-scoped to the actor)
  POST /predict             {"embedding":[d floats]} -> routing provenance
  POST /prune               {"eid":int}     remove one expert (survivors proven intact)
  POST /delete_tenant       {"tenant":str}  right-to-be-forgotten for a whole tenant
  GET  /audit?n=N           last N audit entries (default all)
  GET  /audit/verify        re-walk the signed chain
  GET  /audit/export        download a self-contained, verifiable audit document
  POST /save                persist current state to DAS_STATE (requires it set)

Run:
    conda run -n das python governance_api.py
    curl localhost:5070/health
"""
import os
import sys

import numpy as np
from flask import Flask, jsonify, request

sys.path.insert(0, ".")
from das.model import DASForest
from das.functional import softmax
from das.governance import ControlPlane, AccessDenied

STATE = os.environ.get("DAS_STATE")
SECRET = os.environ.get("DAS_AUDIT_SECRET", "das-dev-key")
PORT = int(os.environ.get("DAS_PORT", "5070"))

# Optional asymmetric audit signing (SECURITY_REVIEW F7): if DAS_AUDIT_PRIVKEY
# points to an Ed25519 private-key PEM, the audit log is signed so an exported
# document (GET /audit/export) is verifiable with only the PUBLIC key — no shared
# secret. Default stays HMAC. The private key is read at boot and never returned.
PRIVKEY = None
_pk_path = os.environ.get("DAS_AUDIT_PRIVKEY")
if _pk_path:
    with open(_pk_path, "rb") as _f:
        PRIVKEY = _f.read()


# ── bootstrap a small two-tenant fleet so the API serves something real ──
def _bootstrap():
    rng = np.random.default_rng(0)
    D, LEAF, N = 16, [16, 13, 8, 2], 160
    centers = {"acme-tax": 0, "acme-legal": 4, "globex-vision": 8, "globex-nlp": 12}
    data = {}

    def make(key):
        c = np.zeros(D); c[centers[key]] = 6.0
        rule = np.random.default_rng(abs(hash(key)) % (2**31)).normal(0, 1, D)
        X = c + rng.normal(0, 0.6, (N, D))
        data[key] = (X, (X @ rule > 0).astype(int))
        return data[key]

    def ce(logits, y):
        p = softmax(logits); oh = np.zeros_like(p); oh[np.arange(len(y)), y] = 1.0
        return (p - oh) / len(y)

    def train_fn(key, cp):
        X, y = make(key)
        def fn(forest, idx):
            leaf = forest.leaves[idx]; leaf.frozen = False
            for _ in range(150):
                i = rng.integers(0, N, 32); leaf.backward(ce(leaf.forward(X[i]), y[i]), 0.05)
            leaf.frozen = True
            keys = [r["name"] for r in cp.experts] + [key]   # include the one being grafted
            Xr = np.vstack([data[k][0] for k in keys])
            dr = np.concatenate([np.full(N, s) for s, _ in enumerate(keys)])
            for _ in range(900):
                i = rng.integers(0, len(Xr), 64); forest.router.train_step(Xr[i], dr[i], lr=0.25)
        return fn

    forest = DASForest(D, LEAF, num_leaves=1, seed=7)
    Xs, ys = make("acme-tax"); leaf = forest.leaves[0]; leaf.frozen = False
    for _ in range(250):
        i = rng.integers(0, N, 32); leaf.backward(ce(leaf.forward(Xs[i]), ys[i]), 0.05)
    leaf.frozen = True
    cp = ControlPlane(forest, seed_tenant="acme", seed_name="acme-tax",
                      secret=SECRET, private_key=PRIVKEY)
    cp.register_tenant("root", "globex")
    cp.add_user("root", "alice", "operator", tenant="acme")
    cp.add_user("root", "bob", "operator", tenant="globex")
    cp.add_user("root", "carol", "auditor")
    for tenant, name in [("acme", "acme-legal"), ("globex", "globex-vision"), ("globex", "globex-nlp")]:
        cp.graft("root", tenant, name, train_fn(name, cp))
    return cp


def _make_cp():
    if STATE and os.path.isdir(STATE) and os.path.exists(os.path.join(STATE, "control_plane.json")):
        print(f"Loading control plane from {STATE} ...")
        cp = ControlPlane.load(STATE, secret=SECRET, private_key=PRIVKEY)
        ok = cp.state_matches_audit()
        print(f"  loaded: {len(cp.experts)} experts, audit chain ok={cp.audit.verify()[0]}, "
              f"state↔audit bound={ok}")
        if not ok:
            print("  WARNING: forest weights do not match the signed audit fingerprint "
                  "(possible tampering of forest.npz).")
        return cp
    print("Bootstrapping a demo two-tenant fleet ...")
    cp = _bootstrap()
    if STATE:
        os.makedirs(STATE, exist_ok=True); cp.save(STATE)
        print(f"  saved bootstrapped state to {STATE}")
    return cp


cp = _make_cp()
DIM = cp.forest.d_model
app = Flask(__name__)


def _actor():
    body = request.get_json(silent=True) or {}
    return request.headers.get("X-DAS-Actor") or body.get("actor") or "root"


@app.errorhandler(AccessDenied)
def _denied(e):
    return jsonify({"error": "access denied", "reason": str(e)}), 403


@app.route("/health")
def health():
    ok, idx, reason = cp.audit.verify()
    return jsonify({
        "status": "ok", "backend": "numpy (control plane)",
        "experts": len(cp.experts), "tenants": sorted(cp.tenants),
        "embedding_dim": DIM, "audit_entries": len(cp.audit.entries),
        "audit_chain_ok": ok, "state_matches_audit": cp.state_matches_audit(),
        "audit_scheme": cp.audit.scheme,
    })


@app.route("/experts")
def experts():
    return jsonify({"actor": _actor(), "experts": cp.list_experts(_actor())})


@app.route("/predict", methods=["POST"])
def predict():
    body = request.get_json(silent=True) or {}
    emb = body.get("embedding")
    if not isinstance(emb, list) or len(emb) != DIM:
        return jsonify({"error": f"'embedding' must be a list of {DIM} floats"}), 400
    try:
        h = np.asarray([float(x) for x in emb], dtype=float)
    except (TypeError, ValueError):
        return jsonify({"error": "'embedding' must all be numbers"}), 400
    prov = cp.route_explain(_actor(), h)[0]
    return jsonify({"actor": _actor(), **prov})


@app.route("/prune", methods=["POST"])
def prune():
    body = request.get_json(silent=True) or {}
    if "eid" not in body:
        return jsonify({"error": "expected {\"eid\": int}"}), 400
    try:
        intact = cp.prune(_actor(), int(body["eid"]))
    except KeyError as e:
        return jsonify({"error": str(e)}), 404
    return jsonify({"pruned": int(body["eid"]), "survivors_byte_identical": intact})


@app.route("/delete_tenant", methods=["POST"])
def delete_tenant():
    body = request.get_json(silent=True) or {}
    if "tenant" not in body:
        return jsonify({"error": "expected {\"tenant\": str}"}), 400
    res = cp.delete_tenant(_actor(), body["tenant"])
    return jsonify({"tenant": body["tenant"], **res})


@app.route("/audit")
def audit():
    n = request.args.get("n", type=int)
    return jsonify({"entries": cp.read_audit(_actor(), n)})


@app.route("/audit/verify")
def audit_verify():
    return jsonify(cp.verify_audit(_actor()))


@app.route("/audit/export")
def audit_export():
    """The signed audit log as a downloadable, independently-verifiable
    compliance document. Verify offline with:  das-verify das_audit.json"""
    doc = cp.export_audit(_actor())
    resp = jsonify(doc)
    resp.headers["Content-Disposition"] = "attachment; filename=das_audit.json"
    return resp


@app.route("/save", methods=["POST"])
def save():
    if not STATE:
        return jsonify({"error": "DAS_STATE is not set; nowhere to persist"}), 400
    cp._check(_actor(), "manage")          # only managers may persist
    cp.save(STATE)
    return jsonify({"saved_to": STATE, "experts": len(cp.experts)})


if __name__ == "__main__":
    print(f"\n  -> DAS governance API: http://localhost:{PORT}")
    print(f"  -> GET /health   GET /experts   POST /predict   POST /prune")
    print(f"  -> POST /delete_tenant   GET /audit   GET /audit/verify   GET /audit/export   POST /save")
    print(f"  -> actor via 'X-DAS-Actor' header (default root); embedding dim = {DIM}\n")
    app.run(host="0.0.0.0", port=PORT, threaded=True)
