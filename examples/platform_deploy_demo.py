"""
platform_deploy_demo.py
-----------------------
The FDE deployment engine, end to end — the 5-day POC compressed into one script.
Pure NumPy, no downloads, no secrets on disk (a demo secret is passed in-process).

    python examples/platform_deploy_demo.py

It builds the spec inline (as a dict) so it runs with zero extra dependencies;
in the field the same spec lives in `client.yaml` and you run `das deploy`.
"""
from das.platform import deploy

SPEC = {
    "client": "northwind",
    "d_model": 18,
    "escalation": {"frontier_llm": "claude-sonnet-5", "confidence_threshold": 0.7},
    "tenants": [
        {"name": "careplus", "experts": [
            {"name": "medical-claim", "keywords": ["claim", "mri", "denied", "appeal"]},
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
}

print("=" * 70)
print(" DAS platform — one spec -> a governed, audited, multi-tenant fleet")
print("=" * 70)

# ── Day 1: deploy ────────────────────────────────────────────────────
dep = deploy(SPEC, secret="demo-secret")
s = dep.summary()
print(f"\n[day 1] deployed '{s['client']}': {len(s['experts'])} experts across "
      f"{len(s['tenants'])} tenants; audit chain ok: {s['audit_ok']}")
for e in s["experts"]:
    print(f"          eid {e['eid']}: {e['name']:<16} (tenant {e['tenant']})")

# ── Day 2: route a real query -> stays local ─────────────────────────
r = dep.route("bank-agent", "I see an unknown recurring card charge from StreamBox")
print(f"\n[day 2] route 'unknown card charge' -> {r['expert']} "
      f"(tenant {r['tenant']}, confidence {r['confidence']:.2f}) -> {r['decision'].upper()}")

# ── Day 3: escalation policy in action ───────────────────────────────
# The linear router is overconfident on out-of-domain inputs (the known routing
# bottleneck), so escalation is driven by the confidence *threshold*. Under a
# strict policy, a borderline query routes to the frontier LLM instead.
r2 = dep.route("bank-agent", "can you also help me with my taxes?", threshold=0.999)
print(f"[day 3] borderline query @ strict policy -> confidence {r2['confidence']:.2f} "
      f"-> {r2['decision'].upper()}"
      + (f" to {r2['frontier_llm']}" if r2["decision"] == "escalate" else ""))

# ── Day 4: grow a specialist live -> others byte-identical ───────────
eid = dep.grow("root", "careplus", "pharmacy-claim", keywords=["pharmacy", "prescription", "rx"])
print(f"\n[day 4] grew 'pharmacy-claim' (eid {eid}); audit chain still ok: "
      f"{dep.verify()['ok']}")

# ── Day 5: offboard a tenant + export the leave-behind ───────────────
result = dep.offboard("fintrust")
print(f"\n[day 5] offboarded 'fintrust': removed {result['removed']} experts, "
      f"survivors byte-identical: {result['non_interference']}")
manifest = dep.export_bundle("northwind_bundle.json")
print(f"          wrote northwind_bundle.json — {manifest['audit_entries']} audit "
      f"entries, chain ok: {manifest['audit_chain_ok']}")
print("          verify offline with:  das-verify northwind_bundle.json")
