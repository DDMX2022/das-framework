"""
langgraph_demo.py
-----------------
DAS as a node UNDER an orchestrator (das/integrations/langgraph_node.py).

The pitch DAS actually earns isn't "a better agent framework" — it's the
*governed, auditable layer* an orchestrator routes into. This demo builds a tiny
two-tenant fleet, wraps it as a LangGraph-style node, and shows that every routed
answer comes back with provenance — which tenant/expert served it, the router's
confidence, the acting principal — and that an RBAC denial surfaces as state, not
a crash. The denial is recorded in the same tamper-evident audit log.

Runs with NumPy only. If `langgraph` is installed it also drives the node through
a compiled StateGraph; otherwise it calls the node directly (identical contract).
Pure synthetic data, no downloads.
"""
import numpy as np

from das.model import DASForest
from das.functional import softmax
from das.governance import ControlPlane
from das.integrations import DASExpertNode, build_graph

rng = np.random.default_rng(0)
D, LEAF, N = 16, [16, 13, 8, 2], 160
CENTERS = {"acme-tax": 0, "acme-legal": 4, "globex-vision": 8, "globex-nlp": 12}
DATA = {}


def _data(key):
    c = np.zeros(D); c[CENTERS[key]] = 6.0
    rule = np.random.default_rng(abs(hash(key)) % (2**31)).normal(0, 1, D)
    X = c + rng.normal(0, 0.6, (N, D))
    return X, (X @ rule > 0).astype(int)


def _ce(logits, y):
    p = softmax(logits); oh = np.zeros_like(p); oh[np.arange(len(y)), y] = 1.0
    return (p - oh) / len(y)


def _train_fn(key, cp):
    X, y = _data(key); DATA[key] = (X, y)

    def fn(forest, idx):
        leaf = forest.leaves[idx]; leaf.frozen = False
        for _ in range(150):
            i = rng.integers(0, N, 32); leaf.backward(_ce(leaf.forward(X[i]), y[i]), 0.05)
        leaf.frozen = True
        keys = [r["name"] for r in cp.experts] + [key]   # include expert being grafted
        Xr = np.vstack([DATA[k][0] for k in keys])
        dr = np.concatenate([np.full(N, s) for s, _ in enumerate(keys)])
        for _ in range(900):
            i = rng.integers(0, len(Xr), 64); forest.router.train_step(Xr[i], dr[i], lr=0.25)
    return fn


print("=" * 68)
print(" DAS under an orchestrator — a governed expert node with provenance")
print("=" * 68)

# ── build a two-tenant fleet ────────────────────────────────────────
forest = DASForest(D, LEAF, num_leaves=1, seed=7)
DATA["acme-tax"] = _data("acme-tax")
leaf = forest.leaves[0]; leaf.frozen = False
X, y = DATA["acme-tax"]
for _ in range(250):
    i = rng.integers(0, N, 32); leaf.backward(_ce(leaf.forward(X[i]), y[i]), 0.05)
leaf.frozen = True

cp = ControlPlane(forest, seed_tenant="acme", seed_name="acme-tax")
cp.register_tenant("root", "globex")
cp.add_user("root", "svc", role="operator")          # service principal for the graph
cp.add_user("root", "carol", role="auditor")         # auditor: no 'predict'
for tenant, name in [("acme", "acme-legal"), ("globex", "globex-vision"), ("globex", "globex-nlp")]:
    cp.graft("root", tenant, name, _train_fn(name, cp))
print(f"\n[fleet] {len(cp.experts)} experts across tenants "
      f"{sorted(cp.tenants)}: {[r['name'] for r in cp.experts]}")

# ── pick the invocation path (compiled graph if langgraph is present) ─
try:
    invoke = build_graph(cp, default_actor="svc").invoke
    mode = "compiled langgraph StateGraph"
except Exception:
    node = DASExpertNode(cp, default_actor="svc")
    invoke = node.__call__
    mode = "DASExpertNode (langgraph not installed — identical node contract)"
print(f"[graph] invocation path: {mode}")

# ── route a query per domain; every answer is attributable ───────────
print("\n  query domain      ->  served by (tenant/expert)        conf   actor")
print("  " + "-" * 66)
ok = True
for key in CENTERS:
    q = DATA[key][0][0]
    out = invoke({"embedding": list(map(float, q)), "actor": "svc"})
    served = f"{out['das_tenant']}/{out['das_expert']}"
    print(f"  {key:16s} ->  {served:32s} {out['das_confidence']:.2f}   {out['das_actor']}")
    # provenance must name a real registered expert
    ok &= any(r["name"] == out["das_expert"] and r["tenant"] == out["das_tenant"]
              for r in cp.experts)

# ── an RBAC denial surfaces as STATE, and is itself audited ──────────
n0 = len(cp.audit.entries)
denied = invoke({"embedding": list(map(float, DATA["acme-tax"][0][0])), "actor": "carol"})
audited = len(cp.audit.entries) == n0 + 1 and cp.audit.entries[-1]["event"] == "denied"
print(f"\n[rbac]  auditor 'carol' invokes node -> denied={denied['das_denied']} "
      f"({denied.get('das_denied_reason','')})")
print(f"[rbac]  denial recorded in tamper-evident log: {audited}; "
      f"chain verifies: {cp.audit.verify()[0]}")

print("\n" + "=" * 68)
passed = ok and denied["das_denied"] and audited and cp.audit.verify()[0]
print(f"  Every routed answer carries valid provenance: {'PASS' if ok else 'FAIL'}")
print(f"  RBAC denial surfaced as state + audited:      "
      f"{'PASS' if denied['das_denied'] and audited else 'FAIL'}")
print(f"  Overall integration proof:                    {'PASS' if passed else 'FAIL'}")
print("=" * 68)
import sys; sys.exit(0 if passed else 1)
