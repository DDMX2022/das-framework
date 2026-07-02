"""
DAS Vendor Console — the SUPERADMIN licensing UI.

Vendor-side only; never ship this to customers. It fronts das.platform.vendor:
the Ed25519 signing key, and issue / renew / revoke / verify over the license
registry. Customers receive only the downloaded license file and pin the public
key shown in the header.

Run:
    DAS_VENDOR_TOKEN=change-me python apps/vendor_console.py   # http://localhost:5095

Access: when DAS_VENDOR_TOKEN is set, every /api request must carry it in the
X-DAS-Vendor-Token header (the UI prompts once and remembers). Unset = local
dev mode, open — the boot log says so loudly. This is a local admin surface;
put real authn (mTLS/OIDC gateway) in front of it before any network exposure.
"""
import io
import json
import os
import tempfile
import threading

from flask import Flask, jsonify, render_template, request, send_file

from das.platform import LicenseError
from das.platform.vendor import VendorStore

app = Flask(__name__)
LOCK = threading.Lock()

VENDOR_DIR = os.environ.get("DAS_VENDOR_DIR") or os.path.join(
    tempfile.gettempdir(), "das_vendor")
TOKEN = os.environ.get("DAS_VENDOR_TOKEN")

store = VendorStore(VENDOR_DIR)


@app.before_request
def _gate():
    if TOKEN and request.path.startswith("/api/"):
        if request.headers.get("X-DAS-Vendor-Token") != TOKEN:
            return jsonify({"error": "vendor token required"}), 401


@app.get("/")
def index():
    return render_template("vendor_console.html")


@app.get("/api/vendor/status")
def status():
    with LOCK:
        pub = store.ensure_keypair()
        rows = store.list()
    counts = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    return jsonify({
        "public_key_hex": pub,
        "vendor_dir": VENDOR_DIR,
        "token_gate": bool(TOKEN),
        "counts": {"total": len(rows), **counts},
    })


@app.get("/api/vendor/licenses")
def list_licenses():
    with LOCK:
        return jsonify({"licenses": store.list()})


@app.post("/api/vendor/licenses")
def issue():
    b = request.get_json(silent=True) or {}
    try:
        days = int(b.get("days", 365))
        limits = {k: (int(b[k]) if b.get(k) not in (None, "",) else None)
                  for k in ("max_deployments", "max_tenants", "max_experts")}
        with LOCK:
            record = store.issue(
                customer=b.get("customer", ""), days=days,
                tier=b.get("tier", "platform"),
                features=[f.strip() for f in (b.get("features") or []) if f.strip()],
                **limits)
        return jsonify(record), 201
    except (LicenseError, ValueError, TypeError) as e:
        return jsonify({"error": str(e)}), 400


@app.post("/api/vendor/licenses/<license_id>/revoke")
def revoke(license_id):
    b = request.get_json(silent=True) or {}
    try:
        with LOCK:
            record = store.revoke(license_id, reason=b.get("reason", ""))
        return jsonify(record)
    except LicenseError as e:
        return jsonify({"error": str(e)}), 400


@app.post("/api/vendor/licenses/<license_id>/renew")
def renew(license_id):
    b = request.get_json(silent=True) or {}
    try:
        with LOCK:
            record = store.renew(license_id, days=int(b.get("days", 365)))
        return jsonify(record), 201
    except (LicenseError, ValueError) as e:
        return jsonify({"error": str(e)}), 400


@app.get("/api/vendor/licenses/<license_id>/download")
def download(license_id):
    try:
        with LOCK:
            record = store.get(license_id)
    except LicenseError as e:
        return jsonify({"error": str(e)}), 404
    # the customer receives ONLY the signed license document, not our registry record
    doc = record["license"]
    buf = io.BytesIO(json.dumps(doc, indent=2, sort_keys=True).encode("utf-8"))
    safe_customer = "".join(c if c.isalnum() else "-" for c in doc["customer"]).strip("-").lower()
    return send_file(buf, mimetype="application/json", as_attachment=True,
                     download_name=f"das_license_{safe_customer}_{license_id}.json")


@app.post("/api/vendor/verify")
def verify():
    b = request.get_json(silent=True) or {}
    doc = b.get("license")
    if not isinstance(doc, dict):
        return jsonify({"error": "body must be {\"license\": {...signed doc...}}"}), 400
    with LOCK:
        ok, issues = store.verify_doc(doc)
    return jsonify({"result": "VALID" if ok else "INVALID", "issues": issues,
                    "customer": doc.get("customer"), "expires_at": doc.get("expires_at")})


if __name__ == "__main__":
    port = int(os.environ.get("DAS_VENDOR_PORT", "5095"))
    store.ensure_keypair()
    gate = "token-gated" if TOKEN else "OPEN (dev mode — set DAS_VENDOR_TOKEN)"
    print(f"DAS Vendor Console (superadmin) — http://localhost:{port} [{gate}]")
    print(f"  registry: {VENDOR_DIR}")
    print(f"  public key (pin in customer envs): {store.public_key_hex}")
    app.run(host="127.0.0.1", port=port, debug=False)
