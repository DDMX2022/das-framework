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
from das.audit import AuditLog

# Optional torch backend: experts as real LoRA adapters on a shared frozen
# backbone (the production expert format — PRODUCT_PLAN.md Phase 1). If torch
# is unavailable the console transparently falls back to the NumPy Service.
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from das_torch import LoRAForest, train_leaf_isolated_lora, train_router, leaf_hash
    TORCH = True
except Exception:
    TORCH = False

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
        self.alog = AuditLog()
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
        payload = {self.active[i]: self.forest.leaves[i].weight_hash() for i in range(len(self.active))}
        self.alog.append(event, detail, payload=payload)

    def audit_view(self):
        return [{"t": e["ts"][11:], "event": e["event"], "detail": e["detail"], "sig": e["sig"][:12]}
                for e in self.alog.entries[-12:][::-1]]

    def audit_verify(self):
        ok, idx, reason = self.alog.verify()
        return {"ok": ok, "broken_index": idx, "reason": reason, "entries": len(self.alog.entries)}

    def state(self):
        mon = self.life.monitor() if hasattr(self, "life") else {}
        return {"backend": "numpy", "experts": [
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

FEAT = 32   # shared backbone feature width for the LoRA backend

class LoRAService:
    """Console backend where every expert is a REAL LoRA adapter on one shared,
    frozen backbone (das_torch.LoRAForest) — the production expert format. Same
    operations and audit surface as the NumPy Service, but the 'grow a forest'
    UI is now driving genuine torch LoRA experts: each graft trains an isolated
    adapter, each prune drops one, and the SHA-256 hashes are over the adapter
    weights that actually moved (or provably didn't)."""

    def __init__(self):
        self.device = "cpu"
        self.reset()

    # ── synthetic, clustered domain data (matches the NumPy Service) ──
    def _gen(self, name, n):
        center = np.zeros(D); center[CENTER[name]] = 4.0
        seed = int(hashlib.md5(name.encode()).hexdigest(), 16) % (2**31)
        rule = np.random.default_rng(seed).normal(0, 1, D)
        X = center + self.rng.normal(0, 1.0, (n, D))
        y = (X @ rule > 0).astype(np.int64)
        return (torch.tensor(X, dtype=torch.float32),
                torch.tensor(y, dtype=torch.long))

    def _pretrain_backbone(self):
        """Pretrain the shared backbone ONCE over all domains (a stand-in for a
        frozen, broadly-pretrained encoder), then freeze it for good. Every
        expert is added as a LoRA adapter on top; the backbone never moves
        again — that's what makes the per-expert hashes meaningful."""
        torch.manual_seed(7)
        Xs, ds = [], []
        for did, name in enumerate(ALL_DOMAINS):
            X, _ = self._gen(name, N)
            Xs.append(X); ds.append(torch.full((N,), did, dtype=torch.long))
        Xp, dp = torch.cat(Xs), torch.cat(ds)
        tmp = nn.Linear(FEAT, len(ALL_DOMAINS))
        opt = torch.optim.Adam(list(self.forest.backbone.parameters()) + list(tmp.parameters()), lr=1e-3)
        for _ in range(600):
            i = torch.randint(0, len(Xp), (128,))
            opt.zero_grad()
            F.cross_entropy(tmp(self.forest.backbone(Xp[i])), dp[i]).backward()
            opt.step()
        self.forest.freeze_backbone()

    def _train_router(self):
        Xs, ds = [], []
        for slot, name in enumerate(self.active):
            X, _ = self._gen(name, N); Xs.append(X)
            ds.append(torch.full((N,), slot, dtype=torch.long))
        feats = self.forest.features(torch.cat(Xs)).detach()
        train_router(self.forest, feats, torch.cat(ds), steps=700, lr=0.05, device=self.device)

    def reset(self):
        self.rng = np.random.default_rng(0)
        self.active = ["math", "vision"]
        torch.manual_seed(7)
        self.forest = LoRAForest(D, FEAT, out_dim=2, num_leaves=len(self.active), rank=8).to(self.device)
        self._pretrain_backbone()
        for i, name in enumerate(self.active):
            train_leaf_isolated_lora(self.forest, i, *self._gen(name, N), steps=400, device=self.device)
        self._train_router()
        self.alog = AuditLog()
        self._log("init", f"forest created with LoRA experts: {', '.join(self.active)}")

    def _hashes(self):
        return {self.active[i]: leaf_hash(self.forest.leaves[i]) for i in range(len(self.active))}

    def _log(self, event, detail):
        self.alog.append(event, detail, payload=self._hashes())

    def audit_view(self):
        return [{"t": e["ts"][11:], "event": e["event"], "detail": e["detail"], "sig": e["sig"][:12]}
                for e in self.alog.entries[-12:][::-1]]

    def audit_verify(self):
        ok, idx, reason = self.alog.verify()
        return {"ok": ok, "broken_index": idx, "reason": reason, "entries": len(self.alog.entries)}

    def _shares(self):
        """Routing share: send an even mix of queries from every active domain
        and count where the router actually sends them (real top-1 dispatch)."""
        counts = np.zeros(len(self.active))
        with torch.no_grad():
            for name in self.active:
                X, _ = self._gen(name, 40)
                _, idx = self.forest.predict(X)
                for j in idx.tolist():
                    if 0 <= j < len(self.active): counts[j] += 1
        tot = counts.sum()
        return counts / tot if tot else counts

    def state(self):
        shares = self._shares()
        return {"backend": "lora (torch)", "experts": [
            {"name": self.active[i], "hash": leaf_hash(self.forest.leaves[i]),
             "share": round(float(shares[i]), 3)} for i in range(len(self.active))]}

    def predict(self, name):
        X, _ = self._gen(name, 1)
        with torch.no_grad():
            out, idx = self.forest.predict(X)
        j = int(idx[0])
        routed = self.active[j] if 0 <= j < len(self.active) else "?"
        return {"sent_from": name, "routed_to": routed, "correct": routed == name,
                "confidence": round(float(F.softmax(out, dim=-1)[0].max()), 3)}

    def graft(self, name):
        if name in self.active: return {"error": f"'{name}' already deployed"}
        before = self._hashes()
        nid = self.forest.graft_leaf()
        self.active.append(name)
        train_leaf_isolated_lora(self.forest, nid, *self._gen(name, N), steps=400, device=self.device)
        self._train_router()
        intact = all(leaf_hash(self.forest.leaves[i]) == before[self.active[i]] for i in range(len(before)))
        self._log("graft", f"added LoRA expert '{name}' — existing experts unchanged: {intact}")
        return {"ok": True, "non_interference": intact}

    def prune(self, name):
        if name not in self.active or len(self.active) <= 1:
            return {"error": "cannot prune"}
        idx = self.active.index(name)
        before = {self.active[i]: leaf_hash(self.forest.leaves[i]) for i in range(len(self.active)) if i != idx}
        keep = [i for i in range(len(self.active)) if i != idx]
        self.forest.leaves = nn.ModuleList([self.forest.leaves[i] for i in keep])
        old = self.forest.router.gate                       # drop the pruned expert's router column
        new = nn.Linear(old.in_features, len(keep)).to(old.weight.device)
        with torch.no_grad():
            new.weight.copy_(old.weight[keep]); new.bias.copy_(old.bias[keep])
        self.forest.router.gate = new
        self.active.pop(idx)
        self._train_router()
        intact = all(leaf_hash(self.forest.leaves[self.active.index(n)]) == before[n] for n in before)
        self._log("prune", f"removed LoRA expert '{name}' (right-to-be-forgotten) — others unchanged: {intact}")
        return {"ok": True, "non_interference": intact}

svc = LoRAService() if TORCH else Service()
print(f"  → console backend: {'LoRA experts (torch)' if TORCH else 'NumPy DASForest'}")

@app.route('/')
def index(): return render_template('console.html', domains=ALL_DOMAINS)

@app.route('/favicon.ico')
def favicon(): return ('', 204)

@app.route('/api/state')
def api_state():
    with LOCK: return jsonify(svc.state())

@app.route('/api/audit')
def api_audit():
    with LOCK: return jsonify({"audit": svc.audit_view()})

@app.route('/api/audit/verify')
def api_audit_verify():
    with LOCK: return jsonify(svc.audit_verify())

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
