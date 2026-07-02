"""
DAS Platform Console — the FDE's "Rancher for model fleets".

One pane of glass over MANY client deployments (Rancher: clusters -> here:
clients). Each deployment is a live das.platform.Deployment stood up from a
declarative spec, so everything the console shows is backed by the real
ControlPlane — the same guarantees as `das deploy`, with a UI on top.

    Sidebar        = client deployments (Rancher's cluster list)
    Overview       = tenants / experts / RBAC / audit health per client
    Route console  = send a query, watch routing + the escalate/local decision
                     (the cost-deflection signal, tallied live)
    Audit          = the signed chain + verify + download the compliance bundle
    Actions        = deploy a new client spec, grow a specialist, offboard a tenant

Run:
    python apps/platform_console.py           # http://localhost:5090

Authentication (PLATFORM_PLAN §10.2): set DAS_CONSOLE_USERS(_FILE) to a JSON
object {username: password_hash} (mint hashes with
`python apps/platform_console.py hash-password`) plus DAS_CONSOLE_SECRET(_FILE)
for session signing, and the console requires login — the logged-in user IS
the RBAC principal on every governed call (a client-supplied "actor" field is
ignored). Unset = the open local demo mode it always was, loudly logged;
DAS_ENV=production refuses to start that way. The hardened, single-fleet
serving unit remains apps/governance_api.py — see docs/SECURITY_REVIEW.md.
"""
import io
import json
import os
import threading

from flask import (Flask, jsonify, redirect, render_template,
                   render_template_string, request, send_file, session)
from werkzeug.security import check_password_hash, generate_password_hash

from das.governance import AccessDenied
from das.platform import ClientSpec, LicenseError, SpecError, deploy
from das.platform.bundle import build_bundle
from das.platform.license import evaluation_notice, load_license

app = Flask(__name__)

# LOCK guards the REGISTRY (create/list/lookup); each deployment then has its
# own lock in LOCKS, so one client's multi-second training op (grow/germinate)
# no longer serializes every other client's requests (review finding #4).
LOCK = threading.Lock()
LOCKS = {}         # client name -> threading.Lock
PENDING = set()    # client names mid-deploy (name reserved, trained outside LOCK)
DEPLOYMENTS = {}   # client name -> Deployment
STATS = {}         # client name -> {"local": int, "escalate": int}

DEMO_SPECS = [
    {
        "client": "northwind",
        "d_model": 18,
        "escalation": {"frontier_llm": "claude-sonnet-5", "confidence_threshold": 0.7},
        "tenants": [
            {"name": "careplus", "experts": [
                {"name": "medical-claim", "keywords": ["claim", "mri", "denied", "appeal", "insurance"]},
                {"name": "prior-auth", "keywords": ["authorization", "referral", "approval"]},
            ]},
            {"name": "fintrust", "experts": [
                {"name": "card-dispute", "keywords": ["charge", "card", "merchant", "dispute", "fraud"]},
                {"name": "loan-status", "keywords": ["loan", "payment", "balance", "mortgage"]},
            ]},
        ],
        "users": [
            {"name": "care-agent", "role": "operator", "tenant": "careplus"},
            {"name": "bank-agent", "role": "operator", "tenant": "fintrust"},
            {"name": "auditor-jane", "role": "auditor"},
        ],
    },
    {
        "client": "meridian",
        "d_model": 18,
        "escalation": {"frontier_llm": "claude-sonnet-5", "confidence_threshold": 0.8},
        "tenants": [
            {"name": "logistics", "experts": [
                {"name": "shipment-eta", "keywords": ["shipment", "tracking", "delivery", "eta"]},
                {"name": "customs-docs", "keywords": ["customs", "declaration", "tariff", "import"]},
            ]},
        ],
        "users": [{"name": "ops-agent", "role": "operator", "tenant": "logistics"}],
    },
]

SECRET = os.environ.get("DAS_AUDIT_SECRET", "console-dev-secret")

# Resolved once at boot per the trust model: a verified License, or None for
# evaluation mode. A configured-but-invalid license fails the console closed.
LICENSE = load_license()

# ── authentication (SECURITY_REVIEW / PLATFORM_PLAN §10.2) ────────────
# The console principal IS the RBAC principal: when auth is configured, every
# governed call runs as the LOGGED-IN user (the session), never as a
# client-supplied "actor" field — the same stance the gateway takes with
# X-DAS-Actor. Without configured users the console stays the open local demo
# surface it always was, and the boot log says so loudly; DAS_ENV=production
# refuses to start that way.


def _read_secret(name):
    """F6: prefer a mounted-file secret (`{name}_FILE`) over the env var."""
    path = os.environ.get(name + "_FILE")
    if path:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read().strip()
    return os.environ.get(name)


def _load_users():
    """DAS_CONSOLE_USERS(_FILE): a JSON object {username: password_hash} with
    werkzeug hashes — mint entries with `python apps/platform_console.py
    hash-password`. Usernames should match the RBAC users in the client specs
    (that is what tenant scoping is enforced on)."""
    raw = _read_secret("DAS_CONSOLE_USERS")
    if not raw:
        return {}
    users = json.loads(raw)
    if not isinstance(users, dict) or not users or \
            not all(isinstance(v, str) and v for v in users.values()):
        raise SystemExit("DAS_CONSOLE_USERS must be a non-empty JSON object "
                         "{username: password_hash}")
    return users


ENV = os.environ.get("DAS_ENV", "development").lower()
USERS = _load_users()
AUTH = bool(USERS)
_session_secret = _read_secret("DAS_CONSOLE_SECRET")
app.secret_key = _session_secret or os.urandom(32)   # ephemeral: dev only
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax")

if ENV == "production" and not (AUTH and _session_secret):
    raise SystemExit(
        "platform console REFUSES TO START (DAS_ENV=production): set "
        "DAS_CONSOLE_USERS(_FILE) [JSON {user: password_hash}] and "
        "DAS_CONSOLE_SECRET(_FILE) [session-signing secret] — an "
        "unauthenticated console must never be exposed (PLATFORM_PLAN §10.2)")

LOGIN_HTML = """<!doctype html><html><head><title>DAS Console — sign in</title>
<style>body{background:#0d1117;color:#e6edf3;font:14px -apple-system,sans-serif;
display:flex;align-items:center;justify-content:center;height:100vh;margin:0}
form{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:28px;
width:300px}h1{font-size:16px;margin:0 0 14px}input{width:100%;margin:5px 0;
padding:8px;background:#21262d;color:#e6edf3;border:1px solid #30363d;
border-radius:6px;box-sizing:border-box}button{width:100%;margin-top:10px;
padding:8px;background:#1f6feb;color:#fff;border:0;border-radius:6px;
cursor:pointer}.err{color:#f85149;font-size:13px;min-height:18px}</style>
</head><body><form method="post" action="/login">
<h1>DAS Platform Console</h1><div class="err">{{ error or "" }}</div>
<input name="username" placeholder="username" autofocus autocomplete="username">
<input name="password" type="password" placeholder="password"
       autocomplete="current-password">
<button type="submit">Sign in</button></form></body></html>"""


def _actor(body, default="root"):
    """The RBAC principal for a governed call: the session user when auth is
    on (client-supplied 'actor' is IGNORED — identity is not a request
    parameter), else the demo behaviour (body actor, defaulting to root)."""
    if AUTH:
        return session["actor"]
    a = body.get("actor", default)
    return a if a is not None else default


@app.before_request
def _require_login():
    if not AUTH:
        return None
    if request.path == "/login" or "actor" in session:
        return None
    if request.path.startswith("/api/"):
        return jsonify({"error": "authentication required", "login": "/login"}), 401
    return redirect("/login")


@app.route("/login", methods=["GET", "POST"])
def login():
    if not AUTH:
        return redirect("/")
    if request.method == "GET":
        return render_template_string(LOGIN_HTML, error=None)
    user = request.form.get("username", "")
    pw = request.form.get("password", "")
    stored = USERS.get(user)
    if stored is None or not check_password_hash(stored, pw):
        return render_template_string(LOGIN_HTML, error="invalid credentials"), 401
    session.clear()
    session["actor"] = user
    return redirect("/")


@app.post("/logout")
def logout():
    session.clear()
    return redirect("/login") if AUTH else redirect("/")


@app.get("/api/me")
def me():
    return jsonify({"auth": AUTH,
                    "actor": session.get("actor") if AUTH else None})


def _register(dep):
    DEPLOYMENTS[dep.spec.client] = dep
    LOCKS.setdefault(dep.spec.client, threading.Lock())
    STATS.setdefault(dep.spec.client, {"local": 0, "escalate": 0})


def _bootstrap():
    for spec in DEMO_SPECS:
        _register(deploy(spec, secret=SECRET))


def _get(client):
    """Look up a deployment and ITS lock under the registry lock. Returns
    (dep, lock) or (None, http-error-response)."""
    with LOCK:
        dep = DEPLOYMENTS.get(client)
        if dep is None:
            return None, (jsonify({"error": f"no deployment named '{client}'"}), 404)
        return dep, LOCKS[client]


def _expert_rows(dep):
    """Expert registry with live weight fingerprints plus germination stage and
    parameter count (registry order == leaf order)."""
    germ = {g["eid"]: g for g in dep.growth_report()}
    return [
        {**r, "hash": dep.cp.forest.leaves[i].weight_hash()[:16],
         "stage": germ[r["eid"]]["stage"], "params": germ[r["eid"]]["params"]}
        for i, r in enumerate(dep.cp.experts)
    ]


# ── pages ────────────────────────────────────────────────────────────
@app.get("/")
def index():
    return render_template("platform_console.html")


# ── fleet-level API ──────────────────────────────────────────────────
@app.get("/api/license")
def license_status():
    if LICENSE is None:
        return jsonify({"licensed": False, "notice": evaluation_notice()})
    return jsonify(LICENSE.status())


@app.get("/api/deployments")
def list_deployments():
    with LOCK:
        items = [(name, dep, LOCKS[name]) for name, dep in DEPLOYMENTS.items()]
    rows = []
    for name, dep, dlock in items:
        with dlock:
            s = dep.summary()
        rows.append({
                "client": name,
                "tenants": len(s["tenants"]),
                "experts": len(s["experts"]),
                "users": len(s["users"]),
                "audit_ok": s["audit_ok"],
                "audit_entries": s["audit_entries"],
                "threshold": s["escalation"]["confidence_threshold"],
                "stats": dict(STATS[name]),
            })
    return jsonify({"deployments": rows})


@app.post("/api/deployments")
def deploy_client():
    body = request.get_json(silent=True) or {}
    spec_data = body.get("spec")
    if not isinstance(spec_data, dict):
        return jsonify({"error": "body must be {\"spec\": {...client spec...}}"}), 400
    try:
        spec = ClientSpec.from_dict(spec_data)
    except SpecError as e:
        return jsonify({"error": str(e)}), 400
    with LOCK:
        if spec.client in DEPLOYMENTS or spec.client in PENDING:
            return jsonify({"error": f"deployment '{spec.client}' already exists"}), 409
        try:
            if LICENSE is not None:
                LICENSE.check_fleet(len(DEPLOYMENTS) + len(PENDING) + 1)
        except LicenseError as e:
            return jsonify({"error": str(e), "license": True}), 403
        PENDING.add(spec.client)          # reserve the name, train outside LOCK
    try:
        dep = deploy(spec, secret=SECRET, license=LICENSE)
    except LicenseError as e:
        return jsonify({"error": str(e), "license": True}), 403
    finally:
        with LOCK:
            PENDING.discard(spec.client)
    with LOCK:
        _register(dep)
    return jsonify(dep.summary()), 201


# ── per-deployment API ───────────────────────────────────────────────
@app.get("/api/deployments/<client>")
def deployment_detail(client):
    dep, lock = _get(client)
    if dep is None:
        return lock
    with lock:
        s = dep.summary()
        s["experts"] = _expert_rows(dep)
        s["stats"] = dict(STATS[client])
        s["teachers"] = [t.describe() for t in dep.teachers.values()]
        s["growth_history"] = dep.growth.recent(10)
        return jsonify(s)


@app.post("/api/deployments/<client>/route")
def route(client):
    body = request.get_json(silent=True) or {}
    actor = _actor(body)
    query = body.get("query", "")
    if not query:
        return jsonify({"error": "missing 'query'"}), 400
    dep, lock = _get(client)
    if dep is None:
        return lock
    with lock:
        try:
            r = dep.route(actor, query)
        except AccessDenied as e:
            return jsonify({"error": str(e), "denied": True}), 403
        STATS[client][r["decision"]] += 1
        r["stats"] = dict(STATS[client])
        return jsonify(r)


@app.post("/api/deployments/<client>/grow")
def grow(client):
    body = request.get_json(silent=True) or {}
    actor = _actor(body)
    tenant, name = body.get("tenant"), body.get("name")
    if not tenant or not name:
        return jsonify({"error": "need 'tenant' and 'name'"}), 400
    dep, lock = _get(client)
    if dep is None:
        return lock
    with lock:
        if any(r["name"] == name for r in dep.cp.experts):
            return jsonify({"error": f"expert '{name}' already exists"}), 409
        before = {r["eid"]: dep.cp.forest.leaves[i].weight_hash()
                  for i, r in enumerate(dep.cp.experts)}
        try:
            eid = dep.grow(actor, tenant, name, keywords=body.get("keywords") or [],
                           teacher=body.get("teacher") or None,
                           stage=body.get("stage") or None)
        except AccessDenied as e:
            return jsonify({"error": str(e), "denied": True}), 403
        except LicenseError as e:
            return jsonify({"error": str(e), "license": True}), 403
        except (KeyError, ValueError) as e:
            return jsonify({"error": str(e)}), 400
        intact = all(dep.cp.forest.leaves[i].weight_hash() == before[r["eid"]]
                     for i, r in enumerate(dep.cp.experts) if r["eid"] in before)
        return jsonify({"eid": eid, "others_byte_identical": intact,
                        "audit_ok": dep.verify()["ok"],
                        "teacher_report": dep.teacher_trainer.reports.get(name)})


@app.post("/api/deployments/<client>/improve")
def improve(client):
    body = request.get_json(silent=True) or {}
    actor = _actor(body)
    if body.get("eid") is None:
        return jsonify({"error": "need 'eid'"}), 400
    dep, lock = _get(client)
    if dep is None:
        return lock
    with lock:
        try:
            result = dep.improve(actor, int(body["eid"]),
                                 teacher=body.get("teacher") or None)
        except AccessDenied as e:
            return jsonify({"error": str(e), "denied": True}), 403
        except (KeyError, ValueError) as e:
            return jsonify({"error": str(e)}), 400
        result["audit_ok"] = dep.verify()["ok"]
        return jsonify(result)


@app.post("/api/deployments/<client>/germinate")
def germinate(client):
    body = request.get_json(silent=True) or {}
    actor = _actor(body)
    if body.get("eid") is None:
        return jsonify({"error": "need 'eid'"}), 400
    dep, lock = _get(client)
    if dep is None:
        return lock
    with lock:
        try:
            result = dep.germinate(actor, int(body["eid"]),
                                   teacher=body.get("teacher") or None,
                                   target_acc=float(body.get("target_acc", 0.85)))
        except AccessDenied as e:
            return jsonify({"error": str(e), "denied": True}), 403
        except (KeyError, ValueError) as e:
            return jsonify({"error": str(e)}), 400
        result["audit_ok"] = dep.verify()["ok"]
        return jsonify(result)


@app.post("/api/deployments/<client>/teachers")
def register_teacher(client):
    body = request.get_json(silent=True) or {}
    dep, lock = _get(client)
    if dep is None:
        return lock
    with lock:
        try:
            return jsonify(dep.register_teacher(body)), 201
        except (ValueError, TypeError) as e:
            return jsonify({"error": str(e)}), 400


@app.post("/api/deployments/<client>/offboard")
def offboard(client):
    body = request.get_json(silent=True) or {}
    tenant = body.get("tenant")
    if not tenant:
        return jsonify({"error": "need 'tenant'"}), 400
    dep, lock = _get(client)
    if dep is None:
        return lock
    with lock:
        try:
            result = dep.offboard(tenant, actor=_actor(body, default=None))
        except AccessDenied as e:
            return jsonify({"error": str(e), "denied": True}), 403
        return jsonify({**result, "audit_ok": dep.verify()["ok"]})


@app.get("/api/deployments/<client>/audit")
def audit(client):
    dep, lock = _get(client)
    if dep is None:
        return lock
    with lock:
        entries = dep.cp.read_audit(dep.spec.root)
        v = dep.verify()
        return jsonify({"entries": entries[-50:], "verify": v})


@app.get("/api/deployments/<client>/bundle")
def bundle(client):
    dep, lock = _get(client)
    if dep is None:
        return lock
    with lock:
        doc = build_bundle(dep)
    buf = io.BytesIO(json.dumps(doc, indent=2, sort_keys=True).encode("utf-8"))
    return send_file(buf, mimetype="application/json", as_attachment=True,
                     download_name=f"{client}_compliance_bundle.json")


_bootstrap()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "hash-password":
        # mint a DAS_CONSOLE_USERS entry: {"<user>": "<printed hash>"}
        import getpass
        pw = getpass.getpass("password: ")
        if pw != getpass.getpass("again: "):
            raise SystemExit("passwords do not match")
        print(generate_password_hash(pw))
        raise SystemExit(0)
    port = int(os.environ.get("DAS_CONSOLE_PORT", "5090"))
    print(f"DAS Platform Console — http://localhost:{port}")
    if AUTH:
        print(f"  authentication: ON ({len(USERS)} console users; "
              f"session actor = RBAC principal)")
    else:
        print("  authentication: OFF — open local demo mode. Anyone reaching "
              "this port acts as any RBAC principal. Set "
              "DAS_CONSOLE_USERS(_FILE) + DAS_CONSOLE_SECRET(_FILE) before "
              "any network exposure (DAS_ENV=production enforces this).")
    app.run(host="127.0.0.1", port=port, debug=False)
