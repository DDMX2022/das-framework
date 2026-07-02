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

This is the FDE's local console over in-process deployments (a POC/demo surface,
like the other apps/ demos). The hardened, single-fleet serving unit remains
apps/governance_api.py — see docs/SECURITY_REVIEW.md before exposing anything.
"""
import io
import json
import os
import threading

from flask import Flask, jsonify, render_template, request, send_file

from das.governance import AccessDenied
from das.platform import ClientSpec, LicenseError, SpecError, deploy
from das.platform.bundle import build_bundle
from das.platform.license import evaluation_notice, load_license

app = Flask(__name__)

LOCK = threading.Lock()
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


def _register(dep):
    DEPLOYMENTS[dep.spec.client] = dep
    STATS.setdefault(dep.spec.client, {"local": 0, "escalate": 0})


def _bootstrap():
    for spec in DEMO_SPECS:
        _register(deploy(spec, secret=SECRET))


def _dep_or_404(client):
    dep = DEPLOYMENTS.get(client)
    if dep is None:
        return None, (jsonify({"error": f"no deployment named '{client}'"}), 404)
    return dep, None


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
        rows = []
        for name, dep in DEPLOYMENTS.items():
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
        if spec.client in DEPLOYMENTS:
            return jsonify({"error": f"deployment '{spec.client}' already exists"}), 409
        try:
            if LICENSE is not None:
                LICENSE.check_fleet(len(DEPLOYMENTS) + 1)
            dep = deploy(spec, secret=SECRET, license=LICENSE)
        except LicenseError as e:
            return jsonify({"error": str(e), "license": True}), 403
        _register(dep)
        return jsonify(dep.summary()), 201


# ── per-deployment API ───────────────────────────────────────────────
@app.get("/api/deployments/<client>")
def deployment_detail(client):
    with LOCK:
        dep, err = _dep_or_404(client)
        if err:
            return err
        s = dep.summary()
        s["experts"] = _expert_rows(dep)
        s["stats"] = dict(STATS[client])
        s["teachers"] = [t.describe() for t in dep.teachers.values()]
        s["growth_history"] = dep.growth.recent(10)
        return jsonify(s)


@app.post("/api/deployments/<client>/route")
def route(client):
    body = request.get_json(silent=True) or {}
    actor = body.get("actor", "root")
    query = body.get("query", "")
    if not query:
        return jsonify({"error": "missing 'query'"}), 400
    with LOCK:
        dep, err = _dep_or_404(client)
        if err:
            return err
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
    actor = body.get("actor", "root")
    tenant, name = body.get("tenant"), body.get("name")
    if not tenant or not name:
        return jsonify({"error": "need 'tenant' and 'name'"}), 400
    with LOCK:
        dep, err = _dep_or_404(client)
        if err:
            return err
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
    actor = body.get("actor", "root")
    if body.get("eid") is None:
        return jsonify({"error": "need 'eid'"}), 400
    with LOCK:
        dep, err = _dep_or_404(client)
        if err:
            return err
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
    actor = body.get("actor", "root")
    if body.get("eid") is None:
        return jsonify({"error": "need 'eid'"}), 400
    with LOCK:
        dep, err = _dep_or_404(client)
        if err:
            return err
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
    with LOCK:
        dep, err = _dep_or_404(client)
        if err:
            return err
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
    with LOCK:
        dep, err = _dep_or_404(client)
        if err:
            return err
        try:
            result = dep.offboard(tenant, actor=body.get("actor"))
        except AccessDenied as e:
            return jsonify({"error": str(e), "denied": True}), 403
        return jsonify({**result, "audit_ok": dep.verify()["ok"]})


@app.get("/api/deployments/<client>/audit")
def audit(client):
    with LOCK:
        dep, err = _dep_or_404(client)
        if err:
            return err
        entries = dep.cp.read_audit(dep.spec.root)
        v = dep.verify()
        return jsonify({"entries": entries[-50:], "verify": v})


@app.get("/api/deployments/<client>/bundle")
def bundle(client):
    with LOCK:
        dep, err = _dep_or_404(client)
        if err:
            return err
        doc = build_bundle(dep)
    buf = io.BytesIO(json.dumps(doc, indent=2, sort_keys=True).encode("utf-8"))
    return send_file(buf, mimetype="application/json", as_attachment=True,
                     download_name=f"{client}_compliance_bundle.json")


_bootstrap()

if __name__ == "__main__":
    port = int(os.environ.get("DAS_CONSOLE_PORT", "5090"))
    print(f"DAS Platform Console — http://localhost:{port}")
    app.run(host="127.0.0.1", port=port, debug=False)
