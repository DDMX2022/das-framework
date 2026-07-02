"""
The demo moment, in a browser (PLATFORM_PLAN §12 step 5, dashboard version).

A deliberately small Flask app over ONE ``das.platform.deploy`` call with
``backend: lora-minilm``: every expert is a LoRA adapter on the frozen
MiniLM's own attention layers, and the page shows the loop that closes deals
(§8): route real text to a specialist, grow a NEW specialist live while every
existing expert's weight hash is streamed unchanged, watch the parsimony gate
refuse capacity a seed doesn't need, and read the signed audit trail.

    pip install -e ".[hf,platform]"
    python apps/lora_growth_app.py          # http://127.0.0.1:5099

Local demo surface: binds loopback, single root actor, no authn — the
hardened multi-user path is apps/governance_api.py (PLATFORM_PLAN §10).
CLI twin: examples/lora_growth_demo.py.
"""
import threading

from flask import Flask, jsonify, render_template, request

from das.platform import deploy

SPEC = {
    "client": "northwind-demo",
    "backend": "lora-minilm",
    "tenants": [
        {"name": "legalco", "experts": [{"name": "legal"}]},
        {"name": "medico", "experts": [{"name": "medical"}]},
    ],
}

app = Flask(__name__, template_folder="templates")

_DEP = None
_LOCK = threading.Lock()


def _dep():
    global _DEP
    with _LOCK:
        if _DEP is None:
            _DEP = deploy(SPEC, secret="demo-secret")
        return _DEP


def _fleet(dep):
    rows = dep.growth_report()
    for i, row in enumerate(rows):
        row["hash"] = dep.cp.forest.leaves[i].weight_hash()[:16]
    return rows


def _state(dep):
    v = dep.verify()
    return {
        "client": dep.spec.client,
        "backend": dep.backend,
        "backbone": dep.cp.forest.backbone.model_name,
        "fleet": _fleet(dep),
        "audit_ok": v["ok"],
        "audit_entries": v["entries"],
        "audit_tail": [
            {"seq": e["seq"], "event": e["event"], "detail": e["detail"]}
            for e in dep.cp.read_audit("root", n=6)
        ],
    }


@app.route("/")
def page():
    return render_template("lora_growth.html")


@app.route("/api/state")
def state():
    return jsonify(_state(_dep()))


@app.route("/api/route", methods=["POST"])
def route():
    body = request.get_json(silent=True) or {}
    query = str(body.get("query", "")).strip()
    if not query:
        return jsonify({"error": "query required"}), 400
    return jsonify(_dep().route("root", query))


@app.route("/api/grow", methods=["POST"])
def grow():
    """THE demo action: graft a specialist live and return the byte-identity
    proof — every pre-existing expert's hash before and after, compared."""
    dep = _dep()
    body = request.get_json(silent=True) or {}
    name = str(body.get("name", "")).strip()
    tenant = str(body.get("tenant", "medico")).strip()
    stage = str(body.get("stage", "seed"))
    if not name:
        return jsonify({"error": "expert name required"}), 400
    if any(r["name"] == name for r in dep.cp.experts):
        return jsonify({"error": f"expert '{name}' already exists"}), 400
    before = {r["name"]: dep.cp.forest.leaves[i].weight_hash()
              for i, r in enumerate(dep.cp.experts)}
    eid = dep.grow("root", tenant, name, stage=stage)
    after = {r["name"]: dep.cp.forest.leaves[i].weight_hash()
             for i, r in enumerate(dep.cp.experts)}
    return jsonify({
        "grown": {"eid": eid, "name": name, "tenant": tenant, "stage": stage,
                  "report": dep.trainer.reports.get(name, {})},
        "proof": [{"name": n, "before": h[:16], "after": after[n][:16],
                   "unchanged": after[n] == h} for n, h in before.items()],
        "all_unchanged": all(after[n] == h for n, h in before.items()),
        "state": _state(dep),
    })


@app.route("/api/germinate", methods=["POST"])
def germinate():
    dep = _dep()
    body = request.get_json(silent=True) or {}
    try:
        eid = int(body.get("eid"))
    except (TypeError, ValueError):
        return jsonify({"error": "eid required"}), 400
    target = float(body.get("target_acc", 0.85))
    result = dep.germinate("root", eid, target_acc=target)
    return jsonify({"result": result, "state": _state(dep)})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5099, threaded=True)
