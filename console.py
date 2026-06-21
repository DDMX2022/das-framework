"""
console.py — the DAS Console (product UI)
-----------------------------------------
An interactive console for operating a live forest, built around DAS's real
value proposition: a governed fleet of isolated, auditable experts. You can
route queries, graft a new expert, prune one, and watch the SHA-256 audit trail
prove non-interference in real time.

This is the PRODUCT-facing app (port 5070), separate from the benchmark
visualizer (app.py, 5050) and the inference API (serve.py, 5060).

Run:  python console.py   ->  http://localhost:5070
"""
import hashlib
import time
import threading
import numpy as np
from flask import Flask, request, jsonify, render_template

from das.model import DASForest
from das.functional import softmax
from das.lifecycle import ForestLifecycle

app = Flask(__name__)

@app.after_request
def _no_cache(resp):
    resp.headers["Cache-Control"] = "no-store, must-revalidate"
    return resp
D, LEAF, N = 21, [21, 13, 8, 2], 300
ALL_DOMAINS = ["math", "vision", "language", "audio", "finance", "medical", "legal", "weather"]
CENTER = {name: i * 2 for i, name in enumerate(ALL_DOMAINS)}   # distinct cluster per domain
LOCK = threading.Lock()

def ce_grad(logits, y):
    p = softmax(logits); oh = np.zeros_like(p); oh[np.arange(len(y)), y] = 1.0
    return (p - oh) / len(y)

class Service:
    def __init__(self): self.reset()

    def reset(self):
        self.rng = np.random.default_rng(0)
        self.active = ["math", "vision"]
        self.forest = DASForest(D, LEAF, num_leaves=2, seed=7)
        for i, name in enumerate(self.active):
            self._train_leaf(i, name)
        self._train_router()
        self.life = ForestLifecycle(self.forest)
        self.audit = []
        self._log("init", f"forest created with experts: {', '.join(self.active)}")

    def _gen(self, name, n):
        center = np.zeros(D); center[CENTER[name]] = 4.0
        seed = int(hashlib.md5(name.encode()).hexdigest(), 16) % (2**31)
        rule = np.random.default_rng(seed).normal(0, 1, D)
        X = center + self.rng.normal(0, 1.0, (n, D))
        return X, (X @ rule > 0).astype(int)

    def _train_leaf(self, idx, name):
        X, y = self._gen(name, N); leaf = self.forest.leaves[idx]; leaf.frozen = False
        for _ in range(400):
            i = self.rng.integers(0, N, 64); leaf.backward(ce_grad(leaf.forward(X[i]), y[i]), 0.05)
        leaf.frozen = True

    def _train_router(self):
        Xs, ds = [], []
        for slot, name in enumerate(self.active):
            X, _ = self._gen(name, N); Xs.append(X); ds.append(np.full(N, slot))
        Xr, dr = np.vstack(Xs), np.concatenate(ds)
        for _ in range(700):
            i = self.rng.integers(0, len(Xr), 64); self.forest.router.train_step(Xr[i], dr[i], lr=0.15)

    def _log(self, event, detail):
        self.audit.append({"t": time.strftime("%H:%M:%S"), "event": event, "detail": detail,
                           "hashes": {self.active[i]: self.forest.leaves[i].weight_hash()
                                      for i in range(len(self.active))}})

    def state(self):
        mon = self.life.monitor() if hasattr(self, "life") else {}
        return {"experts": [
            {"name": self.active[i], "hash": self.forest.leaves[i].weight_hash(),
             "share": round(mon.get(i, {}).get("share", 0.0), 3)} for i in range(len(self.active))]}

    def predict(self, name):
        X, _ = self._gen(name, 1)
        out, idx = self.life.predict(X)
        routed = self.active[int(idx[0])]
        return {"sent_from": name, "routed_to": routed, "correct": routed == name,
                "confidence": round(float(softmax(out)[0].max()), 3)}

    def graft(self, name):
        if name in self.active: return {"error": f"'{name}' already deployed"}
        before = {self.active[i]: self.forest.leaves[i].weight_hash() for i in range(len(self.active))}
        nid = self.life.graft(seed=hash(name) % 1000)
        self.active.append(name)
        self._train_leaf(nid, name); self._train_router()
        intact = all(self.forest.leaves[i].weight_hash() == before[self.active[i]] for i in range(len(before)))
        self._log("graft", f"added expert '{name}' — existing experts unchanged: {intact}")
        return {"ok": True, "non_interference": intact}

    def prune(self, name):
        if name not in self.active or len(self.active) <= 1:
            return {"error": "cannot prune"}
        idx = self.active.index(name)
        before = {self.active[i]: self.forest.leaves[i].weight_hash() for i in range(len(self.active)) if i != idx}
        self.life.prune(idx); self.active.pop(idx); self._train_router()
        intact = all(self.forest.leaves[self.active.index(n)].weight_hash() == before[n] for n in before)
        self._log("prune", f"removed expert '{name}' (right-to-be-forgotten) — others unchanged: {intact}")
        return {"ok": True, "non_interference": intact}

svc = Service()

@app.route('/')
def index(): return render_template('console.html', domains=ALL_DOMAINS)

@app.route('/favicon.ico')
def favicon(): return ('', 204)

@app.route('/api/state')
def api_state():
    with LOCK: return jsonify(svc.state())

@app.route('/api/audit')
def api_audit():
    with LOCK: return jsonify({"audit": svc.audit[-12:][::-1]})

@app.route('/api/predict', methods=['POST'])
def api_predict():
    with LOCK: return jsonify(svc.predict((request.get_json(silent=True) or {}).get("domain","math")))

@app.route('/api/graft', methods=['POST'])
def api_graft():
    with LOCK: return jsonify(svc.graft((request.get_json(silent=True) or {}).get("domain")))

@app.route('/api/prune', methods=['POST'])
def api_prune():
    with LOCK: return jsonify(svc.prune((request.get_json(silent=True) or {}).get("domain")))

@app.route('/api/reset', methods=['POST'])
def api_reset():
    with LOCK: svc.reset(); return jsonify({"ok": True})

if __name__ == '__main__':
    print("\n  → DAS Console (product UI): http://localhost:5070\n")
    app.run(debug=False, port=5070, threaded=True)
