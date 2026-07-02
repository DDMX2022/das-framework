"""
Mobile DAS supercharger demo.

This module is intentionally a consumer of the local ``das`` package. The UI is
just a control surface; every proof it displays comes from DASForest,
ControlPlane, and the LangGraph-compatible DASExpertNode.
"""
import hashlib
import threading

import numpy as np
from flask import jsonify, render_template, request

from das.functional import softmax
from das.governance import AccessDenied, ControlPlane
from das.integrations import DASExpertNode
from das.model import DASForest


D_MODEL = 16
LEAF_DIMS = [D_MODEL, 13, 8, 2]
N = 180
CONCURRENT_AGENTS = 10_000

EXPERT_SPECS = [
    {
        "tenant": "shopstream",
        "name": "return-policy",
        "title": "Smart Return Policy",
        "center": 0,
        "responses": ["manual review", "instant refund"],
    },
    {
        "tenant": "shopstream",
        "name": "user-8842-memory",
        "title": "User 8842 Leaf",
        "center": 4,
        "responses": ["standard path", "loyalty exception"],
        "forget_target": True,
    },
    {
        "tenant": "mednorth",
        "name": "claim-triage",
        "title": "Healthcare Claim Triage",
        "center": 8,
        "responses": ["human review", "covered path"],
    },
    {
        "tenant": "finwise",
        "name": "chargeback-risk",
        "title": "Finance Chargeback Risk",
        "center": 12,
        "responses": ["hold evidence", "file dispute pack"],
    },
]

SCENARIOS = {
    "return": {
        "tenant": "shopstream",
        "actor": "agent-shop",
        "expert": "return-policy",
        "text": "Bootcamp return agent: damaged headphones, order delivered yesterday, customer asks for refund.",
    },
    "user_memory": {
        "tenant": "shopstream",
        "actor": "agent-shop",
        "expert": "user-8842-memory",
        "text": "User 8842 return request: prior loyalty exception and escalation preference should be used.",
    },
    "claim": {
        "tenant": "mednorth",
        "actor": "agent-med",
        "expert": "claim-triage",
        "text": "Healthcare support request: claim denied after an in-network procedure.",
    },
    "chargeback": {
        "tenant": "finwise",
        "actor": "agent-fin",
        "expert": "chargeback-risk",
        "text": "Finance dispute request: cardholder asks about an unknown recurring charge.",
    },
}


def _seed(text):
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def _count_params(leaf):
    return int(sum(w.size for w in leaf.W) + sum(b.size for b in leaf.b))


def _ce_grad(logits, y):
    p = softmax(logits)
    oh = np.zeros_like(p)
    oh[np.arange(len(y)), y] = 1.0
    return (p - oh) / len(y)


class SuperchargerLab:
    """Small deterministic fleet used by the mobile demo."""

    def __init__(self):
        self.lock = threading.RLock()
        self.rng = np.random.default_rng(42)
        self.specs = {s["name"]: dict(s) for s in EXPERT_SPECS}
        self.order = [s["name"] for s in EXPERT_SPECS]
        self.data = {name: self._make_domain(name) for name in self.order}
        self.cp = self._build_control_plane()
        self.node = DASExpertNode(self.cp, default_actor="agent-shop")
        self.request_seq = 0

    def _make_domain(self, name):
        spec = self.specs[name]
        center = np.zeros(D_MODEL)
        center[spec["center"]] = 6.0
        center[(spec["center"] + 1) % D_MODEL] = 2.0
        rule_rng = np.random.default_rng(_seed("rule:" + name))
        rule = rule_rng.normal(0, 1, D_MODEL)
        sample_rng = np.random.default_rng(_seed("sample:" + name))
        x = center + sample_rng.normal(0, 0.55, (N, D_MODEL))
        y = (x @ rule > 0).astype(int)
        return x, y

    def _train_leaf(self, forest, idx, name, steps=260):
        x, y = self.data[name]
        leaf = forest.leaves[idx]
        leaf.frozen = False
        for _ in range(steps):
            batch = self.rng.integers(0, len(x), 36)
            leaf.backward(_ce_grad(leaf.forward(x[batch]), y[batch]), 0.045)
        leaf.frozen = True

    def _train_router(self, forest, names, steps=900):
        x = np.vstack([self.data[name][0] for name in names])
        d = np.concatenate([np.full(N, i) for i, _ in enumerate(names)])
        for _ in range(steps):
            batch = self.rng.integers(0, len(x), 64)
            forest.router.train_step(x[batch], d[batch], lr=0.18)

    def _train_fn(self, name):
        def train(forest, idx):
            self._train_leaf(forest, idx, name)
            names = [r["name"] for r in self.cp.experts] + [name]
            self._train_router(forest, names)
        return train

    def _build_control_plane(self):
        seed_name = self.order[0]
        forest = DASForest(D_MODEL, LEAF_DIMS, num_leaves=1, seed=7)
        self._train_leaf(forest, 0, seed_name, steps=320)
        cp = ControlPlane(
            forest,
            seed_tenant=self.specs[seed_name]["tenant"],
            seed_name=seed_name,
            secret="supercharger-demo-key",
        )
        for tenant in sorted({s["tenant"] for s in EXPERT_SPECS}):
            if tenant != self.specs[seed_name]["tenant"]:
                cp.register_tenant("root", tenant)
        cp.add_user("root", "agent-shop", role="operator", tenant="shopstream")
        cp.add_user("root", "agent-med", role="operator", tenant="mednorth")
        cp.add_user("root", "agent-fin", role="operator", tenant="finwise")
        cp.add_user("root", "auditor", role="auditor")
        self.cp = cp
        for name in self.order[1:]:
            spec = self.specs[name]
            cp.graft("root", spec["tenant"], name, self._train_fn(name), seed=_seed("leaf:" + name))
        return cp

    def _embedding(self, scenario_id, text):
        scenario = SCENARIOS.get(scenario_id, SCENARIOS["return"])
        name = scenario["expert"]
        x, _ = self.data[name]
        row = x[(self.request_seq * 17 + 5) % len(x)].copy()
        noise_rng = np.random.default_rng(_seed(f"{scenario_id}:{text}:{self.request_seq}"))
        return row + noise_rng.normal(0, 0.025, D_MODEL)

    def _expert_rows(self):
        rows = []
        for idx, rec in enumerate(self.cp.experts):
            spec = self.specs.get(rec["name"], {})
            rows.append({
                "index": idx,
                "eid": rec["eid"],
                "tenant": rec["tenant"],
                "name": rec["name"],
                "title": spec.get("title", rec["name"]),
                "hash": self.cp.forest.leaves[idx].weight_hash(),
                "params": _count_params(self.cp.forest.leaves[idx]),
                "forgetTarget": bool(spec.get("forget_target", False)),
            })
        return rows

    def _hashes_by_eid(self):
        return {
            rec["eid"]: self.cp.forest.leaves[idx].weight_hash()
            for idx, rec in enumerate(self.cp.experts)
        }

    def _cost(self, leaf_idx=None):
        router_params = int(self.cp.forest.router.W.size + self.cp.forest.router.b.size)
        leaf_params = [_count_params(leaf) for leaf in self.cp.forest.leaves]
        total_leaf = int(sum(leaf_params))
        if leaf_idx is None or leaf_idx >= len(leaf_params):
            selected = max(leaf_params) if leaf_params else 0
        else:
            selected = leaf_params[leaf_idx]
        naive_hot = router_params + total_leaf
        das_hot = router_params + selected
        reduction = 0.0 if naive_hot == 0 else 1.0 - (das_hot / naive_hot)
        total_leaves = max(len(leaf_params), 1)
        return {
            "routerParams": router_params,
            "totalLeafParams": total_leaf,
            "activeLeafParams": selected,
            "activeLeaves": 1 if leaf_params else 0,
            "totalLeaves": len(leaf_params),
            "hotSetPct": round(100.0 * selected / total_leaf, 1) if total_leaf else 0.0,
            "hotReductionPct": round(100.0 * reduction, 1),
            "concurrentAgents": CONCURRENT_AGENTS,
            "naiveHotLeafSlots": CONCURRENT_AGENTS * total_leaves,
            "dasHotLeafSlots": CONCURRENT_AGENTS,
            "leafSlotsSavedPct": round(100.0 * (1.0 - 1.0 / total_leaves), 1),
        }

    def state(self):
        with self.lock:
            ok, broken_idx, reason = self.cp.audit.verify()
            experts = self._expert_rows()
            return {
                "experts": experts,
                "tenants": sorted(self.cp.tenants),
                "auditEntries": len(self.cp.audit.entries),
                "audit": {"ok": ok, "brokenIndex": broken_idx, "reason": reason},
                "cost": self._cost(),
                "package": {
                    "forest": "das.model.DASForest",
                    "controlPlane": "das.governance.ControlPlane",
                    "node": "das.integrations.DASExpertNode",
                },
                "scenarios": SCENARIOS,
                "deletedUserLeaf": not any(r["name"] == "user-8842-memory" for r in self.cp.experts),
            }

    def route(self, scenario_id, tenant_token=None, actor=None, text=None):
        with self.lock:
            scenario = SCENARIOS.get(scenario_id, SCENARIOS["return"])
            tenant_token = tenant_token or scenario["tenant"]
            actor = actor or scenario["actor"]
            text = text or scenario["text"]
            self.request_seq += 1
            h = self._embedding(scenario_id, text)
            before_audit = len(self.cp.audit.entries)

            leaf_probe, tau = self.cp.forest.router.route(h.reshape(1, -1))
            probed_idx = int(leaf_probe[0])
            probed_rec = self.cp.experts[probed_idx]
            if probed_rec["tenant"] != tenant_token:
                self.cp.audit.append(
                    "tenant_firewall_block",
                    f"blocked tenant token '{tenant_token}' from routed tenant '{probed_rec['tenant']}'",
                    payload={
                        "eid": probed_rec["eid"],
                        "tenant_token": tenant_token,
                        "confidence": float(tau[0, probed_idx]),
                    },
                )
                return {
                    "denied": False,
                    "tokenOk": False,
                    "served": False,
                    "tenantToken": tenant_token,
                    "actor": actor,
                    "scenario": scenario_id,
                    "text": text,
                    "routed": {
                        "eid": probed_rec["eid"],
                        "tenant": probed_rec["tenant"],
                        "expert": probed_rec["name"],
                        "confidence": round(float(tau[0, probed_idx]), 4),
                        "leafIndex": probed_idx,
                        "response": "redacted",
                    },
                    "prediction": None,
                    "cost": self._cost(probed_idx),
                    "auditEntries": len(self.cp.audit.entries),
                    "auditDelta": len(self.cp.audit.entries) - before_audit,
                    "otherHashes": [
                        {"eid": rec["eid"], "hash": self.cp.forest.leaves[i].weight_hash()}
                        for i, rec in enumerate(self.cp.experts)
                        if i != probed_idx
                    ],
                }

            try:
                out = self.node({"embedding": h.tolist(), "actor": actor})
            except AccessDenied as ex:
                return {"denied": True, "reason": str(ex), "auditEntries": len(self.cp.audit.entries)}
            if out.get("das_denied"):
                return {
                    "denied": True,
                    "reason": out.get("das_denied_reason"),
                    "auditEntries": len(self.cp.audit.entries),
                }

            leaf_idx = next(
                (i for i, rec in enumerate(self.cp.experts) if rec["eid"] == out["das_eid"]),
                None,
            )
            prediction_class = int(np.argmax(out["das_prediction"]))
            spec = self.specs.get(out["das_expert"], {})
            responses = spec.get("responses", ["class 0", "class 1"])
            response = responses[prediction_class]
            other_hashes = [
                {"eid": rec["eid"], "hash": self.cp.forest.leaves[i].weight_hash()}
                for i, rec in enumerate(self.cp.experts)
                if i != leaf_idx
            ]
            return {
                "denied": False,
                "tokenOk": True,
                "served": True,
                "tenantToken": tenant_token,
                "actor": actor,
                "scenario": scenario_id,
                "text": text,
                "routed": {
                    "eid": out["das_eid"],
                    "tenant": out["das_tenant"],
                    "expert": out["das_expert"],
                    "confidence": round(float(out["das_confidence"]), 4),
                    "leafIndex": leaf_idx,
                    "response": response,
                },
                "prediction": out["das_prediction"],
                "cost": self._cost(leaf_idx),
                "auditEntries": len(self.cp.audit.entries),
                "auditDelta": len(self.cp.audit.entries) - before_audit,
                "otherHashes": other_hashes,
            }

    def delete_user_leaf(self):
        with self.lock:
            target = next((r for r in self.cp.experts if r["name"] == "user-8842-memory"), None)
            if target is None:
                return {
                    "alreadyDeleted": True,
                    "experts": self._expert_rows(),
                    "auditEntries": len(self.cp.audit.entries),
                }
            before = self._hashes_by_eid()
            intact = self.cp.prune("root", target["eid"])
            after = self._hashes_by_eid()
            survivors = [
                {
                    "eid": eid,
                    "before": before[eid],
                    "after": after[eid],
                    "same": before[eid] == after[eid],
                }
                for eid in sorted(after)
            ]
            ok, broken_idx, reason = self.cp.audit.verify()
            return {
                "alreadyDeleted": False,
                "removedEid": target["eid"],
                "removedName": target["name"],
                "survivorsByteIdentical": bool(intact),
                "survivors": survivors,
                "experts": self._expert_rows(),
                "audit": {"ok": ok, "brokenIndex": broken_idx, "reason": reason},
                "auditEntries": len(self.cp.audit.entries),
                "cost": self._cost(),
            }


_LAB = None
_LAB_LOCK = threading.Lock()


def _lab():
    global _LAB
    with _LAB_LOCK:
        if _LAB is None:
            _LAB = SuperchargerLab()
        return _LAB


def _reset_lab():
    global _LAB
    with _LAB_LOCK:
        _LAB = SuperchargerLab()
        return _LAB


def register_supercharger_routes(app):
    @app.route("/supercharger")
    def supercharger():
        return render_template("supercharger.html")

    @app.route("/api/supercharger/state")
    def supercharger_state():
        return jsonify(_lab().state())

    @app.route("/api/supercharger/route", methods=["POST"])
    def supercharger_route():
        body = request.get_json(silent=True) or {}
        return jsonify(_lab().route(
            body.get("scenario", "return"),
            tenant_token=body.get("tenant"),
            actor=body.get("actor"),
            text=body.get("text"),
        ))

    @app.route("/api/supercharger/delete-user", methods=["POST"])
    def supercharger_delete_user():
        return jsonify(_lab().delete_user_leaf())

    @app.route("/api/supercharger/reset", methods=["POST"])
    def supercharger_reset():
        return jsonify(_reset_lab().state())
