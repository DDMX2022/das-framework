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
import json
import os
import re
import sys
import tempfile
import threading
import time

import numpy as np
from flask import Flask, Response, jsonify, request, render_template

sys.path.insert(0, ".")
from das.model import DASForest
from das.functional import softmax
from das.governance import ControlPlane, AccessDenied
from das.hub import list_leaves as hub_list, publish as hub_publish, pull as hub_pull
from das.mobile_store import GB, MobileModelStore
from das.freshness import FreshnessAnchor, RollbackDetected
from das.training import (
    ExpertEvalSet,
    GrowthManager,
    GrowthPolicy,
    HashingTextEncoder,
    LLMTeacherError,
    VectorTeacher,
    teacher_from_config,
    train_leaf,
)

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
MOBILE_MODEL_DIR = os.environ.get("DAS_MOBILE_MODEL_DIR") or (
    os.path.join(STATE, "mobile_models") if STATE else os.path.join(tempfile.gettempdir(), "das_mobile_models")
)
MOBILE_WARNING_GB = float(os.environ.get("DAS_MOBILE_WARNING_GB", "2.5"))
SHARED_EXPERT_DIR = os.environ.get("DAS_SHARED_EXPERT_DIR") or (
    os.path.join(STATE, "shared_experts") if STATE else os.path.join(tempfile.gettempdir(), "das_shared_experts")
)

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
# Per-expert cluster centers for the demo fleet — also surfaced to the dashboard so
# its "simulate query" buttons can build a probe vector that routes to that expert.
DEMO_CENTERS = {"acme-tax": 0, "acme-legal": 4, "globex-vision": 8, "globex-nlp": 12}


def _bootstrap():
    rng = np.random.default_rng(0)
    D, LEAF, N = 16, [16, 13, 8, 2], 160
    centers = DEMO_CENTERS
    bootstrap_teacher = VectorTeacher("bootstrap", D, centers=centers, noise=0.6)
    data = {}

    def make(key):
        c = np.zeros(D); c[centers[key]] = 6.0
        rule = bootstrap_teacher.rule_for(key)
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


growth = GrowthManager(cp)
mobile_store = MobileModelStore(MOBILE_MODEL_DIR, warning_step_bytes=int(MOBILE_WARNING_GB * GB))
GROWTH_POLICY = GrowthPolicy(
    min_accuracy=0.55,
    min_delta=0.0,
    max_previous_regression=0.03,
)
GROWTH_TEACHERS = {
    "qwen-8b-teacher": VectorTeacher(
        "qwen-8b-teacher", DIM, centers=DEMO_CENTERS, noise=0.72, shift=0.18,
        label="Qwen 8B teacher (local mock)",
    ),
    "llama-teacher": VectorTeacher(
        "llama-teacher", DIM, centers=DEMO_CENTERS, noise=0.82, shift=0.26,
        label="Llama teacher (local mock)",
    ),
    "mistral-teacher": VectorTeacher(
        "mistral-teacher", DIM, centers=DEMO_CENTERS, noise=0.66, shift=0.12,
        label="Mistral teacher (local mock)",
    ),
}
TEACHER_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{2,64}$")
GROWTH_PROVIDER_OPTIONS = [
    {
        "id": "local-vector",
        "name": "Local vector mock",
        "needs_endpoint": False,
        "hint": "Offline deterministic teacher for tests and demos.",
    },
    {
        "id": "openai-compatible",
        "name": "OpenAI-compatible API",
        "needs_endpoint": True,
        "hint": "Use /v1 base URL from OpenAI-compatible, vLLM, or llama.cpp servers.",
    },
    {
        "id": "ollama",
        "name": "Ollama",
        "needs_endpoint": True,
        "hint": "Example endpoint: http://phone-or-laptop:11434",
    },
    {
        "id": "custom-json",
        "name": "Custom JSON endpoint",
        "needs_endpoint": True,
        "hint": "Any endpoint that returns the DAS lesson JSON contract.",
    },
]


def _growth_eval_sets(n=120):
    """Frozen previous-accuracy probes for the current demo-style experts."""
    baseline = VectorTeacher("baseline-evaluator", DIM, centers=DEMO_CENTERS, noise=0.6)
    sets = []
    for rec in cp.experts:
        lessons = baseline.generate(rec["name"], n_train=1, n_eval=n,
                                    dataset_version=f"baseline:{rec['name']}:n{n}")
        sets.append(ExpertEvalSet(rec["eid"], rec["name"], lessons.X_eval, lessons.y_eval))
    return sets


def _teacher_options():
    rows = []
    for key, teacher in GROWTH_TEACHERS.items():
        if hasattr(teacher, "describe"):
            row = dict(teacher.describe())
        else:
            row = {"id": key, "name": getattr(teacher, "label", key), "provider": "unknown"}
        row["id"] = key
        row.pop("api_key", None)
        rows.append(row)
    return rows


def _teacher_payload():
    body = request.get_json(silent=True) or {}
    tid = str(body.get("id") or "").strip()
    if not TEACHER_ID_RE.match(tid):
        raise ValueError("teacher id must be 2-64 letters/numbers/dot/dash/underscore")
    provider = str(body.get("provider") or "local-vector").strip()
    known = {p["id"] for p in GROWTH_PROVIDER_OPTIONS}
    aliases = {"openai": "openai-compatible", "vector": "local-vector", "mock": "local-vector"}
    provider = aliases.get(provider, provider)
    if provider not in known and provider not in ("llama.cpp", "vllm"):
        raise ValueError(f"unknown provider '{provider}'")
    payload = dict(body, id=tid, provider=provider)
    payload["label"] = str(body.get("label") or tid).strip()[:80]
    return payload


def _mobile_payload(extra=None):
    status = mobile_store.status()
    if extra:
        status.update(extra)
    return status


def _save_mobile_expert(eid):
    try:
        row = mobile_store.export_expert(cp, eid)
        return _mobile_payload({"last_saved": row})
    except Exception as e:
        return _mobile_payload({"error": str(e)})


def _sync_mobile_models():
    try:
        return mobile_store.export_all(cp)
    except Exception as e:
        return {"saved": [], "status": _mobile_payload({"error": str(e)})}


def _slug(value, fallback="player"):
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value).strip().lower())
    text = text.strip("-._")
    return text[:64] or fallback


def _player_rows():
    rows = []
    shared = hub_list(SHARED_EXPERT_DIR)
    for actor, user in sorted(cp.users.items()):
        if user.get("role") != "operator" or not user.get("tenant"):
            continue
        tenant = user["tenant"]
        experts = [r for r in cp.experts if r["tenant"] == tenant]
        rows.append({
            "actor": actor,
            "tenant": tenant,
            "display_name": actor.replace("-", " ").title(),
            "experts": len(experts),
            "shared": sum(1 for row in shared.values() if row.get("author") == actor),
            "score": len(experts) * 10 + sum(1 for row in shared.values() if row.get("author") == actor) * 25,
        })
    return rows


def _actor_profile(actor):
    user = cp.users.get(actor)
    if not user:
        return {"actor": actor, "role": "unknown", "tenant": None}
    return {"actor": actor, "role": user.get("role"), "tenant": user.get("tenant")}


def _shared_rows():
    rows = []
    for key, row in hub_list(SHARED_EXPERT_DIR).items():
        item = dict(row)
        item["id"] = key
        rows.append(item)
    rows.sort(key=lambda r: (str(r.get("domain", "")), str(r.get("name", ""))))
    return rows


def _block_rows():
    rows = []
    for row in _shared_rows():
        if row.get("kind") == "knowledge_block":
            rows.append(row)
    rows.sort(key=lambda r: (str(r.get("material", "")), str(r.get("name", ""))))
    return rows


def _building_file():
    return os.path.join(SHARED_EXPERT_DIR, "buildings.json")


def _load_buildings():
    path = _building_file()
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _save_buildings(rows):
    os.makedirs(SHARED_EXPERT_DIR, exist_ok=True)
    with open(_building_file(), "w") as f:
        json.dump(rows, f, indent=2, sort_keys=True)


def _building_rows():
    rows = []
    for key, row in _load_buildings().items():
        item = dict(row)
        item["id"] = key
        rows.append(item)
    rows.sort(key=lambda r: (str(r.get("building_type", "")), str(r.get("name", ""))))
    return rows


def _visible_expert(actor, eid):
    visible = {int(r["eid"]) for r in cp.list_experts(actor)}
    if int(eid) not in visible:
        cp._deny(actor, "read_audit", f"expert eid={eid} is not visible")
    return cp._find(int(eid))


def _default_specialty(rec):
    name = rec.get("name", "")
    if "-" in name:
        return name.split("-", 1)[0]
    return rec.get("tenant", "general")


def _ensure_tree_metadata():
    for rec in cp.experts:
        rec.setdefault("specialty", _default_specialty(rec))
        rec.setdefault("parent", rec["specialty"])
        rec.setdefault("growth_status", "stable")
        rec.setdefault("created_by", "bootstrap")


def _ce_grad(logits, y):
    p = softmax(logits)
    oh = np.zeros_like(p)
    oh[np.arange(len(y)), y] = 1.0
    return (p - oh) / len(y)


def _router_training_batches(new_name=None, new_lessons=None, n=140):
    baseline = VectorTeacher("router-baseline", DIM, centers=DEMO_CENTERS, noise=0.65)
    batches = []
    for rec in cp.experts:
        lessons = baseline.generate(rec["name"], n_train=n, n_eval=1,
                                    dataset_version=f"router:{rec['name']}:n{n}")
        batches.append((rec["name"], lessons.X_train))
    if new_name is not None and new_lessons is not None:
        batches.append((new_name, new_lessons.X_train))
    return batches


def _train_router_for_batches(forest, batches, steps=900, lr=0.25, seed=0):
    rng = np.random.default_rng(seed)
    Xr = np.vstack([X for _, X in batches])
    dr = np.concatenate([np.full(len(X), s, dtype=int) for s, (_, X) in enumerate(batches)])
    for _ in range(int(steps)):
        i = rng.integers(0, len(Xr), min(64, len(Xr)))
        forest.router.train_step(Xr[i], dr[i], lr=lr)


def _latest_growth_for_eids():
    rows = {}
    for r in growth.history:
        rows[r.eid] = r
    return rows


def _tree_payload(actor=None):
    _ensure_tree_metadata()
    latest = _latest_growth_for_eids()
    records = cp.list_experts(actor) if actor is not None else [dict(r) for r in cp.experts]
    grouped = {}
    for rec in records:
        g = latest.get(rec["eid"])
        status = rec.get("growth_status", "stable")
        if g is not None:
            status = "growing" if g.accepted else "needs_attention"
        grouped.setdefault(rec["specialty"], []).append({
            "eid": rec["eid"],
            "name": rec["name"],
            "tenant": rec["tenant"],
            "specialty": rec["specialty"],
            "parent": rec.get("parent") or rec["specialty"],
            "status": status,
            "created_by": rec.get("created_by", "unknown"),
            "teacher": rec.get("teacher") if g is None else g.teacher,
            "accuracy": None if g is None else g.target_accuracy_after,
            "delta": None if g is None else g.target_delta,
            "hash": cp.forest.leaves[cp._find(rec["eid"])[0]].weight_hash(),
        })
    return {
        "name": "DAS",
        "children": [
            {
                "name": specialty,
                "status": "branch",
                "children": sorted(children, key=lambda x: x["name"]),
            }
            for specialty, children in sorted(grouped.items())
        ],
    }


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
    if PROXY_SECRET is None or request.path in ("/health", "/", "/favicon.ico"):
        return None
    if not hmac.compare_digest(request.headers.get("X-DAS-Proxy-Auth", ""), PROXY_SECRET):
        return jsonify({"error": "unauthenticated",
                        "reason": "missing or invalid trusted-proxy credential"}), 401
    return None


@app.errorhandler(AccessDenied)
def _denied(e):
    return jsonify({"error": "access denied", "reason": str(e)}), 403


@app.route("/")
def dashboard():
    """Tenant operational dashboard — a thin client over the JSON API below. The
    page is the static shell; every privileged call it makes carries X-DAS-Actor
    (and X-DAS-Proxy-Auth when the gateway contract is enabled)."""
    return render_template(
        "dashboard.html",
        dim=DIM,
        demo_centers=DEMO_CENTERS,
        proxy_required=PROXY_SECRET is not None,
        audit_scheme=cp.audit.scheme,
    )


@app.route("/growth")
def growth_dashboard():
    """Growing-child dashboard: teacher lessons -> candidate -> tests -> audit."""
    return render_template(
        "growth.html",
        proxy_required=PROXY_SECRET is not None,
        policy={
            "min_accuracy": GROWTH_POLICY.min_accuracy,
            "min_delta": GROWTH_POLICY.min_delta,
            "max_previous_regression": GROWTH_POLICY.max_previous_regression,
        },
    )


@app.route("/growth/mobile/trainer")
def mobile_trainer_dashboard():
    """Mobile-first expert trainer: connect teacher -> train expert -> test prompt."""
    return render_template(
        "mobile_trainer.html",
        proxy_required=PROXY_SECRET is not None,
    )


@app.route("/growth/manifest.json")
def growth_manifest():
    return jsonify({
        "name": "DAS Growing Child",
        "short_name": "DAS Forest",
        "start_url": "/growth",
        "display": "standalone",
        "background_color": "#0a0d0f",
        "theme_color": "#07100c",
        "icons": [],
    })


@app.route("/growth/sw.js")
def growth_service_worker():
    js = """
self.addEventListener('install', event => {
  event.waitUntil(caches.open('das-growth-v1').then(cache => cache.addAll(['/growth'])));
  self.skipWaiting();
});
self.addEventListener('activate', event => event.waitUntil(self.clients.claim()));
self.addEventListener('fetch', event => {
  const url = new URL(event.request.url);
  if (event.request.method === 'GET' && url.pathname === '/growth') {
    event.respondWith(fetch(event.request).catch(() => caches.match('/growth')));
  }
});
""".strip()
    return Response(js, mimetype="application/javascript")


@app.route("/favicon.ico")
def favicon():
    return ("", 204)


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


@app.route("/growth/status")
def growth_status():
    _ensure_tree_metadata()
    eval_sets = _growth_eval_sets(n=80)
    return jsonify({
        "actor": _actor(),
        "teachers": _teacher_options(),
        "providers": GROWTH_PROVIDER_OPTIONS,
        "lesson_contract": {
            "train": [{"input": "short text", "label": 1}],
            "eval": [{"input": "nearby confusing negative", "label": 0}],
        },
        "policy": {
            "min_accuracy": GROWTH_POLICY.min_accuracy,
            "min_delta": GROWTH_POLICY.min_delta,
            "max_previous_regression": GROWTH_POLICY.max_previous_regression,
        },
        "experts": cp.list_experts(_actor()),
        "history": growth.recent(20),
        "cycles": growth.recent_cycles(10),
        "automation": {
            "default_max_attempts": len(cp.list_experts(_actor())),
            "endpoint": "/growth/auto/run",
        },
        "mobile_store": _mobile_payload(),
        "players": _player_rows(),
        "actor_profile": _actor_profile(_actor()),
        "shared_arena": {
            "path": SHARED_EXPERT_DIR,
            "experts": len(_shared_rows()),
            "blocks": len(_block_rows()),
            "buildings": len(_building_rows()),
        },
        "router_accuracy": growth.history[-1].router_accuracy if growth.history else None,
        "baseline_probe_count": sum(len(ev.X) for ev in eval_sets),
    })


@app.route("/growth/players", methods=["POST"])
def growth_create_player():
    actor = _actor()
    body = request.get_json(silent=True) or {}
    player = _slug(body.get("player") or body.get("actor") or body.get("name"), "player")
    tenant = _slug(body.get("tenant") or f"player-{player}", f"player-{player}")
    display = str(body.get("display_name") or player.replace("-", " ").title()).strip()[:80]
    try:
        cp._check(actor, "manage")
        if player in cp.users and cp.users[player].get("tenant") != tenant:
            return jsonify({"error": f"player '{player}' already exists with another forest"}), 409
        if tenant not in cp.tenants:
            cp.register_tenant(actor, tenant)
        if player not in cp.users:
            cp.add_user(actor, player, "operator", tenant=tenant)
        cp.audit.append(
            "growth_player_forest",
            f"{actor} prepared player forest '{tenant}' for player '{player}'",
            payload=cp._hashes(),
        )
    except AccessDenied:
        raise
    return jsonify({
        "player": {
            "actor": player,
            "tenant": tenant,
            "display_name": display,
            "role": cp.users[player]["role"],
        },
        "players": _player_rows(),
    }), 201


@app.route("/growth/shared")
def growth_shared():
    return jsonify({
        "path": SHARED_EXPERT_DIR,
        "experts": _shared_rows(),
        "players": _player_rows(),
    })


@app.route("/growth/blocks")
def growth_blocks():
    return jsonify({
        "path": SHARED_EXPERT_DIR,
        "blocks": _block_rows(),
        "buildings": _building_rows(),
    })


@app.route("/growth/blocks/harvest", methods=["POST"])
def growth_harvest_block():
    actor = _actor()
    body = request.get_json(silent=True) or {}
    if "eid" not in body:
        return jsonify({"error": "expected {\"eid\": int}"}), 400
    try:
        idx, rec = _visible_expert(actor, int(body["eid"]))
    except KeyError as e:
        return jsonify({"error": str(e)}), 404
    leaf = cp.forest.leaves[idx]
    material = _slug(body.get("material") or rec.get("specialty") or _default_specialty(rec), "general")
    base = body.get("block_name") or body.get("name") or f"{rec['tenant']}-{rec['name']}-{material}-block"
    block_id = _slug(base, f"eid{rec['eid']}-block")
    row = hub_publish(
        leaf,
        block_id,
        SHARED_EXPERT_DIR,
        domain=str(body.get("domain") or material),
        author=actor,
        metadata={
            "kind": "knowledge_block",
            "material": material,
            "source_eid": rec["eid"],
            "source_tenant": rec["tenant"],
            "source_name": rec["name"],
            "harvested_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )
    cp.audit.append(
        "growth_block_harvested",
        f"{actor} harvested expert eid={rec['eid']} ('{rec['tenant']}/{rec['name']}') as block '{block_id}'",
        payload=cp._hashes(),
    )
    return jsonify({
        "block": {"id": block_id, **row},
        "blocks": _block_rows(),
        "buildings": _building_rows(),
    }), 201


@app.route("/growth/buildings", methods=["POST"])
def growth_create_building():
    actor = _actor()
    body = request.get_json(silent=True) or {}
    name = str(body.get("name") or "").strip()
    if not name:
        return jsonify({"error": "expected {\"name\": str, \"blocks\": [str]}"}), 400
    raw_blocks = body.get("blocks") or []
    if isinstance(raw_blocks, str):
        raw_blocks = [b.strip() for b in raw_blocks.split(",")]
    block_ids = [_slug(b, "") for b in raw_blocks if str(b).strip()]
    if not block_ids:
        return jsonify({"error": "at least one harvested block is required"}), 400

    index = hub_list(SHARED_EXPERT_DIR)
    missing = [bid for bid in block_ids if bid not in index]
    if missing:
        return jsonify({"error": f"unknown block(s): {', '.join(missing)}"}), 404
    not_blocks = [bid for bid in block_ids if index[bid].get("kind") != "knowledge_block"]
    if not_blocks:
        return jsonify({"error": f"not harvested block(s): {', '.join(not_blocks)}"}), 400

    building_id = _slug(body.get("id") or name, "building")
    rows = _load_buildings()
    if building_id in rows and not body.get("replace"):
        return jsonify({"error": f"building '{building_id}' already exists"}), 409

    profile = _actor_profile(actor)
    materials = []
    for bid in block_ids:
        block = index[bid]
        materials.append({
            "id": bid,
            "name": block.get("source_name") or block.get("name") or bid,
            "material": block.get("material") or block.get("domain") or "general",
            "author": block.get("author"),
            "source_eid": block.get("source_eid"),
        })
    row = {
        "id": building_id,
        "kind": "knowledge_building",
        "name": name,
        "building_type": str(body.get("building_type") or body.get("type") or "general").strip() or "general",
        "author": actor,
        "tenant": profile.get("tenant"),
        "blocks": block_ids,
        "block_count": len(block_ids),
        "materials": materials,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    rows[building_id] = row
    _save_buildings(rows)
    cp.audit.append(
        "growth_building_assembled",
        f"{actor} assembled building '{building_id}' from {len(block_ids)} knowledge block(s)",
        payload=cp._hashes(),
    )
    return jsonify({"building": row, "buildings": _building_rows(), "blocks": _block_rows()}), 201


@app.route("/growth/share", methods=["POST"])
def growth_share_expert():
    actor = _actor()
    body = request.get_json(silent=True) or {}
    if "eid" not in body:
        return jsonify({"error": "expected {\"eid\": int}"}), 400
    try:
        idx, rec = _visible_expert(actor, int(body["eid"]))
    except KeyError as e:
        return jsonify({"error": str(e)}), 404
    leaf = cp.forest.leaves[idx]
    base = body.get("shared_name") or f"{rec['tenant']}-{rec['name']}-eid{rec['eid']}"
    shared_id = _slug(base, f"eid{rec['eid']}")
    domain = str(body.get("domain") or rec.get("specialty") or _default_specialty(rec))
    row = hub_publish(
        leaf,
        shared_id,
        SHARED_EXPERT_DIR,
        domain=domain,
        author=actor,
        metadata={
            "source_eid": rec["eid"],
            "source_tenant": rec["tenant"],
            "source_name": rec["name"],
            "published_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )
    cp.audit.append(
        "growth_expert_shared",
        f"{actor} shared expert eid={rec['eid']} ('{rec['tenant']}/{rec['name']}') as '{shared_id}'",
        payload=cp._hashes(),
    )
    return jsonify({"shared": {"id": shared_id, **row}, "experts": _shared_rows()}), 201


@app.route("/growth/import", methods=["POST"])
def growth_import_shared_expert():
    actor = _actor()
    body = request.get_json(silent=True) or {}
    shared_id = str(body.get("shared_id") or body.get("id") or "").strip()
    if not shared_id:
        return jsonify({"error": "expected {\"shared_id\": str}"}), 400
    shared = hub_list(SHARED_EXPERT_DIR).get(shared_id)
    if shared is None:
        return jsonify({"error": f"shared expert '{shared_id}' was not found"}), 404
    try:
        leaf = hub_pull(shared_id, SHARED_EXPERT_DIR)
    except (KeyError, ValueError) as e:
        return jsonify({"error": str(e)}), 400
    if leaf.dims[0] != DIM or leaf.dims[-1] != cp.forest.leaf_dims[-1]:
        return jsonify({"error": "shared expert shape is not compatible with this forest"}), 400

    profile = _actor_profile(actor)
    tenant = str(body.get("tenant") or profile.get("tenant") or "learning").strip()
    name = str(body.get("name") or f"imported-{shared.get('source_name') or shared_id}").strip()
    specialty = str(body.get("specialty") or shared.get("domain") or _default_specialty({"name": name, "tenant": tenant})).strip()
    parent = str(body.get("parent") or specialty).strip()

    if any(r["tenant"] == tenant and r["name"] == name for r in cp.experts):
        return jsonify({"error": f"expert '{tenant}/{name}' already exists"}), 409
    if tenant not in cp.tenants:
        cp.register_tenant(actor, tenant)

    baseline = VectorTeacher("multiplayer-router", DIM, centers=DEMO_CENTERS, noise=0.65)
    router_lessons = baseline.generate(name, n_train=160, n_eval=1)

    def graft_import(forest, idx):
        forest.leaves[idx] = leaf
        _train_router_for_batches(
            forest,
            _router_training_batches(new_name=name, new_lessons=router_lessons),
            seed=len(cp.experts) + 31,
        )

    try:
        eid = cp.graft(actor, tenant, name, graft_import)
    except AccessDenied:
        raise
    _, rec = cp._find(eid)
    rec.update({
        "specialty": specialty,
        "parent": parent,
        "growth_status": "new",
        "created_by": "multiplayer_import",
        "teacher": f"shared:{shared_id}",
        "source_shared": shared_id,
        "source_author": shared.get("author"),
    })
    cp.audit.append(
        "growth_expert_imported",
        f"{actor} imported shared expert '{shared_id}' into '{tenant}/{name}' as eid={eid}",
        payload=cp._hashes(),
    )
    mobile_status = _save_mobile_expert(eid)
    return jsonify({
        "expert": dict(rec),
        "source": {"id": shared_id, **shared},
        "tree": _tree_payload(actor),
        "mobile_store": mobile_status,
    }), 201


@app.route("/growth/mobile/memory")
def growth_mobile_memory():
    return jsonify({
        "status": _mobile_payload(),
        "manifest": mobile_store.manifest() if os.path.exists(mobile_store.manifest_path) else {"version": 1, "models": []},
    })


@app.route("/growth/mobile/save", methods=["POST"])
def growth_mobile_save():
    actor = _actor()
    cp._check(actor, "manage")
    synced = _sync_mobile_models()
    cp.audit.append(
        "growth_mobile_models_saved",
        f"{actor} synced {len(synced.get('saved', []))} compact expert models to '{mobile_store.path}'",
        payload=cp._hashes(),
    )
    return jsonify(synced)


@app.route("/growth/teachers", methods=["POST"])
def growth_register_teacher():
    actor = _actor()
    try:
        cp._check(actor, "manage")
        payload = _teacher_payload()
        replace = bool(payload.get("replace", False))
        if payload["id"] in GROWTH_TEACHERS and not replace:
            return jsonify({"error": f"teacher '{payload['id']}' already exists"}), 409
        teacher = teacher_from_config(payload, DIM, centers=DEMO_CENTERS)
    except AccessDenied:
        raise
    except (TypeError, ValueError) as e:
        return jsonify({"error": str(e)}), 400
    GROWTH_TEACHERS[payload["id"]] = teacher
    cp.audit.append(
        "growth_teacher_registered",
        (
            f"{actor} registered teacher '{payload['id']}' "
            f"provider='{payload['provider']}' model='{payload.get('model', '')}'"
        ),
        payload=cp._hashes(),
    )
    return jsonify({"teacher": teacher.describe(), "teachers": _teacher_options()}), 201


@app.route("/growth/tree")
def growth_tree():
    return jsonify({"tree": _tree_payload(_actor()), "history": growth.recent(20)})


@app.route("/growth/run", methods=["POST"])
def growth_run():
    body = request.get_json(silent=True) or {}
    if "eid" not in body:
        return jsonify({"error": "expected {\"eid\": int}"}), 400
    teacher_id = body.get("teacher", "qwen-8b-teacher")
    if teacher_id not in GROWTH_TEACHERS:
        return jsonify({"error": f"unknown teacher '{teacher_id}'"}), 400
    try:
        steps = max(1, min(int(body.get("steps", 140)), 1000))
        lr = max(0.001, min(float(body.get("lr", 0.05)), 1.0))
        n_train = max(16, min(int(body.get("n_train", 180)), 1200))
        n_eval = max(16, min(int(body.get("n_eval", 120)), 1200))
        eid = int(body["eid"])
    except (TypeError, ValueError):
        return jsonify({"error": "eid, steps, lr, n_train, and n_eval must be numeric"}), 400
    try:
        result = growth.improve_expert(
            _actor(),
            eid,
            GROWTH_TEACHERS[teacher_id],
            topic=body.get("topic") or None,
            eval_sets=_growth_eval_sets(),
            policy=GROWTH_POLICY,
            steps=steps,
            lr=lr,
            n_train=n_train,
            n_eval=n_eval,
            seed=len(growth.history) + 1,
        )
    except KeyError as e:
        return jsonify({"error": str(e)}), 404
    except LLMTeacherError as e:
        return jsonify({"error": "teacher failed", "reason": str(e)}), 502
    mobile_status = _save_mobile_expert(result.eid) if result.accepted else _mobile_payload()
    return jsonify({
        "result": result.to_dict(),
        "history": growth.recent(20),
        "mobile_store": mobile_status,
    })


@app.route("/growth/create_expert", methods=["POST"])
def growth_create_expert():
    body = request.get_json(silent=True) or {}
    name = str(body.get("name", "")).strip()
    if not name:
        return jsonify({"error": "expected {\"name\": str}"}), 400
    tenant = str(body.get("tenant") or "learning").strip()
    specialty = str(body.get("specialty") or _default_specialty({"name": name, "tenant": tenant})).strip()
    parent = str(body.get("parent") or specialty).strip()
    teacher_id = body.get("teacher", "qwen-8b-teacher")
    if teacher_id not in GROWTH_TEACHERS:
        return jsonify({"error": f"unknown teacher '{teacher_id}'"}), 400
    if any(r["tenant"] == tenant and r["name"] == name for r in cp.experts):
        return jsonify({"error": f"expert '{tenant}/{name}' already exists"}), 409
    try:
        steps = max(1, min(int(body.get("steps", 180)), 1200))
        lr = max(0.001, min(float(body.get("lr", 0.05)), 1.0))
        n_train = max(32, min(int(body.get("n_train", 220)), 1600))
        n_eval = max(16, min(int(body.get("n_eval", 120)), 1200))
    except (TypeError, ValueError):
        return jsonify({"error": "steps, lr, n_train, and n_eval must be numeric"}), 400

    actor = _actor()
    if tenant not in cp.tenants:
        cp.register_tenant(actor, tenant)

    teacher = GROWTH_TEACHERS[teacher_id]
    try:
        lessons = teacher.generate(name, n_train=n_train, n_eval=n_eval)
    except LLMTeacherError as e:
        return jsonify({"error": "teacher failed", "reason": str(e)}), 502

    def train_new_expert(forest, idx):
        train_leaf(
            forest.leaves[idx],
            lessons.X_train,
            lessons.y_train,
            steps=steps,
            lr=lr,
            batch=32,
            seed=len(growth.history) + len(cp.experts) + 1,
        )
        _train_router_for_batches(
            forest,
            _router_training_batches(new_name=name, new_lessons=lessons),
            seed=len(cp.experts) + 17,
        )

    eid = cp.graft(actor, tenant, name, train_new_expert)
    _, rec = cp._find(eid)
    rec.update({
        "specialty": specialty,
        "parent": parent,
        "growth_status": "new",
        "created_by": "growth_create_expert",
        "teacher": teacher_id,
    })
    eval_acc = float((cp.forest.leaves[cp._find(eid)[0]].forward(lessons.X_eval).argmax(1) == lessons.y_eval).mean())
    cp.audit.append(
        "growth_create_expert",
        (
            f"{actor} created expert eid={eid} ('{tenant}/{name}') under "
            f"specialty '{specialty}' using teacher '{teacher_id}'; seed acc {eval_acc:.3f}"
        ),
        payload=cp._hashes(),
    )
    mobile_status = _save_mobile_expert(eid)
    return jsonify({
        "expert": dict(rec),
        "seed_accuracy": round(eval_acc, 6),
        "tree": _tree_payload(actor),
        "mobile_store": mobile_status,
    }), 201


@app.route("/growth/mobile/test_prompt", methods=["POST"])
def growth_mobile_test_prompt():
    actor = _actor()
    body = request.get_json(silent=True) or {}
    prompt = str(body.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "expected {\"prompt\": str}"}), 400
    if "eid" not in body:
        return jsonify({"error": "expected {\"eid\": int, \"prompt\": str}"}), 400
    try:
        idx, rec = _visible_expert(actor, int(body["eid"]))
    except (TypeError, ValueError):
        return jsonify({"error": "eid must be numeric"}), 400
    except KeyError as e:
        return jsonify({"error": str(e)}), 404

    topic = str(body.get("topic") or rec.get("specialty") or rec.get("name") or "").strip()
    encoder = HashingTextEncoder(DIM)
    h = encoder.encode_one(prompt, topic=topic)
    direct_logits = cp.forest.leaves[idx].forward(h[None, :])
    direct_probs = softmax(direct_logits)[0]
    try:
        routed = cp.route_explain(actor, h)[0]
    except AccessDenied:
        raise
    result = {
        "prompt": prompt,
        "topic": topic,
        "expert": {
            "eid": rec["eid"],
            "tenant": rec["tenant"],
            "name": rec["name"],
            "specialty": rec.get("specialty"),
        },
        "direct": {
            "label": int(np.argmax(direct_probs)),
            "belongs": bool(int(np.argmax(direct_probs)) == 1),
            "confidence": float(direct_probs[int(np.argmax(direct_probs))]),
            "probabilities": [float(x) for x in direct_probs.tolist()],
        },
        "router": routed,
    }
    return jsonify(result)


@app.route("/growth/auto/run", methods=["POST"])
def growth_auto_run():
    body = request.get_json(silent=True) or {}
    teachers = body.get("teachers")
    if isinstance(teachers, str):
        teachers = [teachers]
    if teachers is not None and not isinstance(teachers, list):
        return jsonify({"error": "'teachers' must be a string or list of strings"}), 400
    try:
        max_attempts = body.get("max_attempts")
        max_attempts = None if max_attempts is None else max(0, min(int(max_attempts), 50))
        steps = max(1, min(int(body.get("steps", 120)), 1000))
        lr = max(0.001, min(float(body.get("lr", 0.05)), 1.0))
        n_train = max(16, min(int(body.get("n_train", 180)), 1200))
        n_eval = max(16, min(int(body.get("n_eval", 120)), 1200))
    except (TypeError, ValueError):
        return jsonify({"error": "max_attempts, steps, lr, n_train, and n_eval must be numeric"}), 400
    try:
        cycle = growth.auto_cycle(
            _actor(),
            GROWTH_TEACHERS,
            teacher_names=teachers,
            eval_sets=_growth_eval_sets(),
            policy=GROWTH_POLICY,
            max_attempts=max_attempts,
            steps=steps,
            lr=lr,
            n_train=n_train,
            n_eval=n_eval,
            seed_base=len(growth.history) + 1,
        )
    except KeyError as e:
        return jsonify({"error": str(e)}), 400
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except LLMTeacherError as e:
        return jsonify({"error": "teacher failed", "reason": str(e)}), 502
    synced = []
    for row in cycle.results:
        if row.get("accepted"):
            synced.append(_save_mobile_expert(row["eid"]))
    return jsonify({
        "cycle": cycle.to_dict(),
        "history": growth.recent(20),
        "cycles": growth.recent_cycles(10),
        "mobile_store": _mobile_payload({"synced": synced}),
    })


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
    print(f"  -> Growth dashboard: http://localhost:{PORT}/growth")
    print(f"  -> Mobile trainer: http://localhost:{PORT}/growth/mobile/trainer")
    print(f"  -> GET /health   GET /experts   POST /predict   POST /prune")
    print(f"  -> POST /delete_tenant   GET /audit   GET /audit/verify   GET /audit/export   POST /save")
    print(f"  -> GET /growth/status   GET /growth/tree   POST /growth/run")
    print(f"  -> POST /growth/teachers   POST /growth/create_expert   POST /growth/auto/run")
    print(f"  -> POST /growth/players   GET /growth/shared   POST /growth/share   POST /growth/import")
    print(f"  -> GET /growth/blocks   POST /growth/blocks/harvest   POST /growth/buildings")
    print(f"  -> GET /growth/mobile/memory   POST /growth/mobile/save")
    print(f"  -> actor via 'X-DAS-Actor' header (default root); embedding dim = {DIM}\n")
    app.run(host="0.0.0.0", port=PORT, threaded=True)
