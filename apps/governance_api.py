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
  DAS_ANCHOR        path to an append-only freshness anchor (F1). If set, each save
                    records the chain tip and the API REFUSES TO START on a rolled-
                    back/forged snapshot. Must live OUTSIDE DAS_STATE. Optional.
  DAS_ENV           'production' turns on startup guards: refuse the default audit
                    secret (F3) and require DAS_TRUSTED_PROXY_SECRET (F2). Default
                    'development'.
  DAS_TRUSTED_PROXY_SECRET  shared secret the authn gateway sends as the
                    'X-DAS-Proxy-Auth' header. When set, requests lacking it get
                    401 (except /health), so X-DAS-Actor is only honoured via the
                    proxy. Optional in dev; required when DAS_ENV=production (F2).
  DAS_RATE_LIMIT    max requests/min per client (source IP / actor); 0 disables.
                    In-process backstop — keep real DoS controls at the proxy (F5).
  <VAR>_FILE        any secret above (DAS_AUDIT_SECRET, DAS_TRUSTED_PROXY_SECRET)
                    may instead be read from a mounted file via <VAR>_FILE — prefer
                    this over plain env in production (F6). DAS_AUDIT_PRIVKEY is
                    already a file path.
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
import hmac
import os
import sys
import threading
import time

import numpy as np
from flask import Flask, jsonify, request

sys.path.insert(0, ".")
from das.model import DASForest
from das.functional import softmax
from das.governance import ControlPlane, AccessDenied
from das.freshness import FreshnessAnchor, RollbackDetected

def _resolve_secret(name, default=None):
    """F6: prefer a mounted-file secret (`{name}_FILE`) over the env var, so
    secrets can come from a Docker/k8s secret mount rather than the environment
    (which can leak via /proc, crash dumps, or child processes)."""
    path = os.environ.get(name + "_FILE")
    if path:
        with open(path) as f:
            return f.read().strip()
    return os.environ.get(name, default)


STATE = os.environ.get("DAS_STATE")
SECRET = _resolve_secret("DAS_AUDIT_SECRET", "das-dev-key")
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

# Optional freshness/rollback anchor (SECURITY_REVIEW F1): if DAS_ANCHOR points to
# an append-only file, each save records the chain tip and each load refuses a
# rolled-back snapshot. It MUST live outside DAS_STATE (a separate, more-trusted
# store) — otherwise an attacker who can roll back DAS_STATE can roll it back too.
ANCHOR = None
_anchor_path = os.environ.get("DAS_ANCHOR")
if _anchor_path:
    if STATE and os.path.abspath(_anchor_path).startswith(os.path.abspath(STATE) + os.sep):
        print("  WARNING: DAS_ANCHOR is inside DAS_STATE — it provides no rollback "
              "protection there; put it on a separate, more-trusted store.")
    ANCHOR = FreshnessAnchor(_anchor_path)

# Identity & deployment hardening (SECURITY_REVIEW F2/F3).
ENV = os.environ.get("DAS_ENV", "development").lower()
# Trusted-proxy shared secret (F2). When set, every request except the liveness
# probe must carry a matching `X-DAS-Proxy-Auth` header, which the authn gateway
# (mTLS/OIDC) adds alongside the verified `X-DAS-Actor`. So the asserted identity
# is only honoured for requests that actually came through the gateway.
PROXY_SECRET = _resolve_secret("DAS_TRUSTED_PROXY_SECRET")


def _startup_errors(env, secret, has_privkey, proxy_secret):
    """Fatal misconfigurations to refuse in production. Returns a list of messages."""
    errs = []
    if env == "production":
        if not has_privkey and (not secret or secret == "das-dev-key"):
            errs.append("DAS_AUDIT_SECRET is unset or the dev default 'das-dev-key' — "
                        "set a strong secret (or DAS_AUDIT_PRIVKEY) before production (F3).")
        if not proxy_secret:
            errs.append("DAS_TRUSTED_PROXY_SECRET is not set — X-DAS-Actor would be "
                        "trusted without authentication. Front the API with an authn "
                        "proxy and set the shared secret (F2).")
    return errs


# Lightweight in-process rate limit (F5, defense-in-depth). Real DoS protection
# belongs at a reverse proxy; this is a backstop. DAS_RATE_LIMIT = requests/min per
# client (X-DAS-Actor, else source IP); 0/unset disables it.
RATE_LIMIT = int(os.environ.get("DAS_RATE_LIMIT", "0"))


class _RateLimiter:
    """Fixed-window per-client limiter (thread-safe)."""

    def __init__(self, per_min):
        self.per_min = per_min
        self._lock = threading.Lock()
        self._hits = {}

    def allow(self, client, now=None):
        if self.per_min <= 0:
            return True
        window = int((now if now is not None else time.time()) // 60)
        with self._lock:
            w, c = self._hits.get(client, (window, 0))
            if w != window:
                w, c = window, 0
            c += 1
            self._hits[client] = (w, c)
            return c <= self.per_min


RATE_LIMITER = _RateLimiter(RATE_LIMIT)


_fatal = _startup_errors(ENV, SECRET, PRIVKEY is not None, PROXY_SECRET)
if _fatal:
    print("  REFUSING TO START (DAS_ENV=production):")
    for _m in _fatal:
        print("   -", _m)
    sys.exit(2)


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
        try:
            cp = ControlPlane.load(STATE, secret=SECRET, private_key=PRIVKEY, anchor=ANCHOR)
        except RollbackDetected as e:
            print(f"  REFUSING TO START: rolled-back/forged state detected — {e}")
            sys.exit(1)
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
        os.makedirs(STATE, exist_ok=True); cp.save(STATE, anchor=ANCHOR)
        print(f"  saved bootstrapped state to {STATE}")
    return cp


cp = _make_cp()
DIM = cp.forest.d_model
app = Flask(__name__)


def _actor():
    body = request.get_json(silent=True) or {}
    return request.headers.get("X-DAS-Actor") or body.get("actor") or "root"


@app.before_request
def _rate_limit():
    """F5: cap requests/min per client (source IP, or actor) as a backstop. The
    liveness probe is exempt."""
    if request.path == "/health":
        return None
    client = request.remote_addr or request.headers.get("X-DAS-Actor") or "anon"
    if not RATE_LIMITER.allow(client):
        return jsonify({"error": "rate limited",
                        "reason": f"exceeded {RATE_LIMITER.per_min} requests/min"}), 429
    return None


@app.before_request
def _require_trusted_proxy():
    """Enforce the authn-proxy contract (F2). If DAS_TRUSTED_PROXY_SECRET is set,
    every request except the liveness probe must present the matching
    X-DAS-Proxy-Auth header (added by the gateway). This makes the X-DAS-Actor
    identity trustworthy, since only the gateway can produce a valid request."""
    if PROXY_SECRET is None or request.path == "/health":
        return None
    if not hmac.compare_digest(request.headers.get("X-DAS-Proxy-Auth", ""), PROXY_SECRET):
        return jsonify({"error": "unauthenticated",
                        "reason": "missing or invalid trusted-proxy credential"}), 401
    return None


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
    cp.save(STATE, anchor=ANCHOR)
    return jsonify({"saved_to": STATE, "experts": len(cp.experts),
                    "freshness_anchored": ANCHOR is not None})


if __name__ == "__main__":
    print(f"\n  -> DAS governance API: http://localhost:{PORT}")
    print(f"  -> GET /health   GET /experts   POST /predict   POST /prune")
    print(f"  -> POST /delete_tenant   GET /audit   GET /audit/verify   GET /audit/export   POST /save")
    print(f"  -> actor via 'X-DAS-Actor' header (default root); embedding dim = {DIM}\n")
    app.run(host="0.0.0.0", port=PORT, threaded=True)
