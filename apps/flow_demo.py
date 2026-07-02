"""
DAS flow explainer demo.

This page answers a different question than the supercharger demo:
"Where does DAS fit in the AI application flow?"

The backend still uses the local das package so the diagram is backed by a real
ControlPlane, DASForest, and LangGraph-compatible node.
"""
import hashlib
import threading

import numpy as np
from flask import jsonify, render_template, request

from das.functional import softmax
from das.governance import ControlPlane
from das.integrations import DASExpertNode
from das.model import DASForest


D_MODEL = 18
LEAF_DIMS = [D_MODEL, 13, 8, 2]
N = 150

EXPERTS = [
    {
        "tenant": "careplus",
        "name": "medical-claim-review",
        "title": "Medical claim review",
        "center": 0,
        "answers": ["send to nurse reviewer", "prepare appeal packet"],
    },
    {
        "tenant": "fintrust",
        "name": "card-dispute-risk",
        "title": "Card dispute risk",
        "center": 6,
        "answers": ["request more evidence", "open chargeback case"],
    },
    {
        "tenant": "retailops",
        "name": "warranty-return",
        "title": "Warranty return check",
        "center": 12,
        "answers": ["manual warranty review", "approve replacement"],
    },
]

CASES = {
    "claim": {
        "tenant": "careplus",
        "actor": "care-agent",
        "expert": "medical-claim-review",
        "person": "Priya",
        "text": "Priya says her MRI claim was denied even though the clinic was in network.",
    },
    "dispute": {
        "tenant": "fintrust",
        "actor": "bank-agent",
        "expert": "card-dispute-risk",
        "person": "Jon",
        "text": "Jon sees an unknown recurring charge and asks the bank to dispute it.",
    },
    "warranty": {
        "tenant": "retailops",
        "actor": "retail-agent",
        "expert": "warranty-return",
        "person": "Ava",
        "text": "Ava's phone failed during the warranty period and she wants a replacement.",
    },
}


def _seed(text):
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def _ce_grad(logits, y):
    p = softmax(logits)
    oh = np.zeros_like(p)
    oh[np.arange(len(y)), y] = 1.0
    return (p - oh) / len(y)


class FlowLab:
    def __init__(self):
        self.lock = threading.RLock()
        self.rng = np.random.default_rng(10)
        self.specs = {x["name"]: dict(x) for x in EXPERTS}
        self.order = [x["name"] for x in EXPERTS]
        self.data = {name: self._make_domain(name) for name in self.order}
        self.cp = self._build_control_plane()
        self.node = DASExpertNode(self.cp, default_actor="care-agent")
        self.seq = 0

    def _make_domain(self, name):
        spec = self.specs[name]
        center = np.zeros(D_MODEL)
        center[spec["center"]] = 6.0
        center[(spec["center"] + 2) % D_MODEL] = 2.4
        rule_rng = np.random.default_rng(_seed("flow-rule:" + name))
        rule = rule_rng.normal(0, 1, D_MODEL)
        sample_rng = np.random.default_rng(_seed("flow-sample:" + name))
        x = center + sample_rng.normal(0, 0.52, (N, D_MODEL))
        y = (x @ rule > 0).astype(int)
        return x, y

    def _train_leaf(self, forest, idx, name, steps=240):
        x, y = self.data[name]
        leaf = forest.leaves[idx]
        leaf.frozen = False
        for _ in range(steps):
            batch = self.rng.integers(0, len(x), 32)
            leaf.backward(_ce_grad(leaf.forward(x[batch]), y[batch]), 0.045)
        leaf.frozen = True

    def _train_router(self, forest, names, steps=700):
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
        forest = DASForest(D_MODEL, LEAF_DIMS, num_leaves=1, seed=12)
        self._train_leaf(forest, 0, seed_name, steps=300)
        cp = ControlPlane(
            forest,
            seed_tenant=self.specs[seed_name]["tenant"],
            seed_name=seed_name,
            secret="das-flow-demo-key",
        )
        for tenant in sorted({x["tenant"] for x in EXPERTS}):
            if tenant != self.specs[seed_name]["tenant"]:
                cp.register_tenant("root", tenant)
        cp.add_user("root", "care-agent", role="operator", tenant="careplus")
        cp.add_user("root", "bank-agent", role="operator", tenant="fintrust")
        cp.add_user("root", "retail-agent", role="operator", tenant="retailops")
        self.cp = cp
        for name in self.order[1:]:
            spec = self.specs[name]
            cp.graft("root", spec["tenant"], name, self._train_fn(name), seed=_seed("flow-leaf:" + name))
        return cp

    def _embedding(self, case_id, text):
        case = CASES.get(case_id, CASES["claim"])
        x, _ = self.data[case["expert"]]
        row = x[(self.seq * 19 + 3) % len(x)].copy()
        noise_rng = np.random.default_rng(_seed(f"flow:{case_id}:{text}:{self.seq}"))
        return row + noise_rng.normal(0, 0.025, D_MODEL)

    def _experts(self):
        return [
            {
                "eid": rec["eid"],
                "tenant": rec["tenant"],
                "name": rec["name"],
                "title": self.specs[rec["name"]]["title"],
                "hash": self.cp.forest.leaves[idx].weight_hash(),
            }
            for idx, rec in enumerate(self.cp.experts)
        ]

    def state(self):
        with self.lock:
            ok, broken_idx, reason = self.cp.audit.verify()
            return {
                "cases": CASES,
                "experts": self._experts(),
                "audit": {"ok": ok, "brokenIndex": broken_idx, "reason": reason},
                "auditEntries": len(self.cp.audit.entries),
                "package": {
                    "forest": "das.model.DASForest",
                    "controlPlane": "das.governance.ControlPlane",
                    "node": "das.integrations.DASExpertNode",
                },
            }

    def run(self, case_id, token=None):
        with self.lock:
            case = CASES.get(case_id, CASES["claim"])
            token = token or case["tenant"]
            self.seq += 1
            h = self._embedding(case_id, case["text"])
            leaf_idx, tau = self.cp.forest.router.route(h.reshape(1, -1))
            idx = int(leaf_idx[0])
            rec = self.cp.experts[idx]
            confidence = float(tau[0, idx])

            base_steps = [
                {"id": "user", "title": "Real person", "owner": "Customer", "status": "done",
                 "detail": case["text"]},
                {"id": "agent", "title": "Agent workflow", "owner": "Bootcamp code / LangGraph", "status": "done",
                 "detail": "The app turns the message into a task and attaches the company identity."},
                {"id": "das", "title": "DAS safety layer", "owner": "DAS", "status": "active",
                 "detail": f"Checks tenant token: {token}."},
            ]

            if rec["tenant"] != token:
                self.cp.audit.append(
                    "flow_demo_block",
                    f"blocked token '{token}' before tenant '{rec['tenant']}' expert could run",
                    payload={"eid": rec["eid"], "token": token, "confidence": confidence},
                )
                steps = base_steps + [
                    {"id": "router", "title": "Stem Router", "owner": "DAS", "status": "blocked",
                     "detail": f"Route pointed at {rec['tenant']}, so DAS refused the mismatched token."},
                    {"id": "leaf", "title": "Specialist LLM safe box", "owner": "DAS expert", "status": "closed",
                     "detail": "No specialist LLM, adapter, or RAG index was allowed to answer."},
                    {"id": "answer", "title": "Answer", "owner": "Agent app", "status": "blocked",
                     "detail": "Blocked before private knowledge could leave the safe box."},
                ]
                return {
                    "ok": False,
                    "blocked": True,
                    "case": case_id,
                    "tenantToken": token,
                    "routedTenant": rec["tenant"],
                    "expert": rec["name"],
                    "confidence": round(confidence, 4),
                    "message": "DAS blocked the request before the specialist ran.",
                    "steps": steps,
                    "auditEntries": len(self.cp.audit.entries),
                }

            out = self.node({"embedding": h.tolist(), "actor": case["actor"]})
            answer_idx = int(np.argmax(out["das_prediction"]))
            spec = self.specs[out["das_expert"]]
            response = spec["answers"][answer_idx]
            steps = base_steps + [
                {"id": "router", "title": "Stem Router", "owner": "DAS", "status": "done",
                 "detail": f"Picked {spec['title']} with {confidence * 100:.0f}% confidence."},
                {"id": "leaf", "title": "Specialist LLM safe box", "owner": "DAS expert", "status": "done",
                 "detail": f"Only the {spec['title']} LLM specialist opened. Other tenants stayed closed."},
                {"id": "answer", "title": "Answer + proof", "owner": "Agent app", "status": "done",
                 "detail": response},
            ]
            return {
                "ok": True,
                "blocked": False,
                "case": case_id,
                "tenantToken": token,
                "routedTenant": out["das_tenant"],
                "expert": out["das_expert"],
                "expertTitle": spec["title"],
                "confidence": round(float(out["das_confidence"]), 4),
                "message": response,
                "steps": steps,
                "auditEntries": len(self.cp.audit.entries),
            }


_LAB = None
_LOCK = threading.Lock()


def _lab():
    global _LAB
    with _LOCK:
        if _LAB is None:
            _LAB = FlowLab()
        return _LAB


def register_flow_routes(app):
    @app.route("/das-flow")
    def das_flow():
        return render_template("das_flow.html")

    @app.route("/api/das-flow/state")
    def das_flow_state():
        return jsonify(_lab().state())

    @app.route("/api/das-flow/run", methods=["POST"])
    def das_flow_run():
        body = request.get_json(silent=True) or {}
        return jsonify(_lab().run(body.get("case", "claim"), token=body.get("tenant")))
