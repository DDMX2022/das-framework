# Case study — a multi-tenant regulated-AI provider

> **Status: illustrative.** This is a worked scenario, not a deployed customer —
> DAS does not yet have one (see the [roadmap](README.md#roadmap-to-production)).
> But every capability claimed below is backed by a runnable script and a measured
> number, not a slide. Where DAS *can't* do something, this says so.

## The customer profile

"Northwind AI" (fictional) is a B2B SaaS that hosts **per-customer specialist
models**. Each of its enterprise customers (a *tenant*) gets one or more
fine-tuned capabilities — a contracts classifier, a support triage model, a
fraud scorer. Northwind's problems are not accuracy problems; they are
**governance** problems:

| Requirement | Driver |
|---|---|
| One customer's model must never be affected by changes to another's | contractual isolation, SOC 2 |
| "Delete everything you learned from us" must be provable | **GDPR Art. 17** (right to erasure), DPA exit clauses |
| Every change to a customer's model fleet must be attributable, after the fact | audit / compliance |
| Different staff roles may do different things (ship vs. read-only) | least-privilege, SOC 2 CC6 |
| When a model answers, Northwind must know *which* model and *whose* it was | incident response, billing |

Northwind's incumbent approach is one shared model that they keep fine-tuning as
customers come and go. The benchmark below is exactly why that hurts.

## What the measurements say

All numbers from [`governance_benchmark.py`](governance_benchmark.py) (6
capabilities, 2 tenants, deterministic). "Isolated experts" = the LoRA-per-task
equivalent; "DAS-CP" = DAS control plane.

| Governance axis | Monolith (incumbent) | Isolated experts | **DAS control plane** |
|---|---|---|---|
| Mean task accuracy | 0.55 | 0.95 | 0.94 |
| Forgetting (BWT) | **−0.467** | 0.000 | **0.000** |
| Add a capability: others byte-identical | 0% | 100% | **100%** |
| Delete: survivors byte-identical | 0% | 100% | **100%** |
| Capability actually removable | ✗ | ✓ | **✓** |
| Tamper-evident audit log | ✗ | ✗ | **✓** |
| RBAC enforced | ✗ | ✗ | **✓** |
| Per-query provenance | ✗ | ✗ | **✓** |

The shared-model incumbent fails every isolation requirement: a new capability
drags accuracy on existing ones down by ~0.47, and there is no clean way to
remove one customer's influence short of a full retrain. **Isolated adapters fix
the top half** — and DAS ties them there, because that's just what isolation buys
(DAS ≈ LoRA + a router). DAS's distinct, measured contribution is the **bottom
half: audit, RBAC, provenance** — the layer Northwind would otherwise have to
build itself.

## How each requirement is met (and proven)

1. **Cross-tenant isolation.** Each capability is a frozen expert; adding or
   retraining one leaves every other byte-identical (SHA-256 weight hash
   unchanged). *Proof:* `governance_benchmark.py` → "others byte-identical 100%";
   `control_plane_demo.py` → tenant-delete leaves the other tenant byte-identical.

2. **Provable right-to-be-forgotten.** `delete_tenant` removes exactly that
   tenant's experts and returns `non_interference: true` only if every survivor
   hashes identically. Deletion is structural (the weights are gone), not a
   filter. *Proof:* benchmark removes 3 experts, survivors intact ✓;
   `tests/test_governance.py::test_delete_tenant_isolation`.

3. **Attributable change history.** Every privileged action — and every *denied*
   attempt — is appended to an HMAC-signed, hash-chained log that also
   fingerprints the fleet state at each step. Any edit/reorder/insert is caught.
   *Proof:* `audit tamper caught 100%`; `tests/test_governance.py::test_audit_tamper_detected`.

4. **Least-privilege roles.** `admin / operator / auditor / viewer`, with
   operators scoped to a tenant. Unauthorized privileged ops are denied (and
   logged). *Proof:* `RBAC denials 100%`; the REST API returns HTTP 403 for them
   (`tests/test_governance_api.py`).

5. **Per-answer provenance.** Each routed answer carries `tenant / expert / eid /
   confidence`, so an incident or invoice can be traced to one model.
   *Proof:* `langgraph_demo.py` (4/4 domains attributed); `POST /predict` returns
   the provenance record.

## How Northwind would deploy it

- Run the control plane as the container in [`Dockerfile`](Dockerfile) /
  [`deploy/k8s.yaml`](deploy/k8s.yaml); state on a volume, audit secret from a
  k8s `Secret`.
- Put it **under** their existing LangGraph orchestration via
  [`DASExpertNode`](das/integrations/langgraph_node.py) — DAS is the governed
  routing layer, not a new orchestrator.
- Export `audit.json` per period as the compliance artifact.

## What this case study does **not** claim

- **Not** that DAS is more accurate or cheaper than a frontier model — it isn't,
  and that's not the pitch.
- **Not** that DAS beats isolated LoRA adapters on isolation/forgetting — it
  *ties* them; the win is the governance plane on top.
- **Not** scale-proven: the benchmark is small and synthetic. Northwind's real
  workload (large HF-backed experts, production traffic) is roadmap **Phase 1**
  (real backend) + a genuine design partner, both still open.
- **Not** a substitute for authentication: the API enforces RBAC but trusts an
  asserted principal — see [`SECURITY_REVIEW.md`](SECURITY_REVIEW.md).
