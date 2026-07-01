# DAS Platform — the FDE deployment engine

> **What this document is.** The product definition for turning the DAS
> governance core into a *repeatable enterprise-AI deployment engine* — the
> infrastructure a Forward Deployed Engineer (FDE) stands up at client after
> client. It covers the positioning, the full product lifecycle (design →
> delivery), the SKUs and pricing, and where the thing runs.
>
> It sits above [PRODUCT_PLAN.md](PRODUCT_PLAN.md) (the research→product maturity
> plan) and is scoped to the *commercial platform* built on the proven core.

---

## 1. The one-sentence product

**DAS is the FDE's deployment engine: one config file stands up a governed,
audited, multi-tenant expert fleet at a new client — and the signed audit bundle
is what proves it was safe after the FDE has moved on.**

The unit of value is not a feature. It is **time-to-deploy at client N+1**. A
library helps you build one deployment; a platform makes the *tenth* deployment
cookie-cutter, so one FDE does what used to take a team.

## 2. Why this product exists (the FDE gap)

FDE work splits into a **repeatable infrastructure layer** and a
**non-repeatable judgement layer**. DAS productizes the first and gets out of the
way of the second.

| FDE does every engagement | Who owns it | In DAS? |
|---|---|---|
| Understand the business problem | FDE (human) | ✗ out of scope |
| Design the agent workflow | FDE + LangGraph | ✗ integrates under it |
| Wire real databases / APIs | FDE, via a fixed seam | 🟡 `ContextSource` interface |
| Decide where humans approve | FDE (policy) | 🟡 escalation threshold config |
| Route to the right specialist | **DAS** | ✅ stem router |
| Keep each client's model isolated | **DAS** | ✅ tenant control plane |
| Prove isolation / deletion to audit | **DAS** | ✅ signed audit log |
| Add a capability without re-certifying | **DAS** | ✅ graft |
| Delete a client cleanly (offboard) | **DAS** | ✅ delete_tenant |
| Reduce cost by deflecting the frontier LLM | **DAS** | 🟡 escalation + cost report |

Everything in the **DAS** rows is already proven in the core
([governance_benchmark](../benchmarks/governance_benchmark.py)). The platform is
the machinery that makes those guarantees *deployable in a day, repeatably*.

## 3. Ideal customer profile (ICP)

Not "enterprises doing AI." Specifically **teams that already have an FDE-shaped
function and do repeat deployments**:

1. **AI systems integrators / consultancies** — deploy the same governed stack
   across many clients; feel "repeatable FDE work" as margin pain.
2. **Internal platform teams** at regulated multi-tenant companies — one platform,
   many business units / customers, each needing isolation + audit.
3. **B2B SaaS hosting per-customer models** (the "Northwind" profile in
   [CASE_STUDY.md](CASE_STUDY.md)) — governance is their problem, not accuracy.

Common wedge: they must answer *"prove tenant A wasn't affected by B,"* *"delete
this client and prove it,"* and *"cut the frontier-LLM bill"* — on every account.

---

## 4. Product architecture — three tiers

```
DAS Core (open source)          the mechanics — ≈ LoRA + a router; give it away
  router · leaf · forest · graft/prune · LangGraph node

DAS Control Plane (commercial)  the governance you can prove and charge for
  signed audit · RBAC · multi-tenancy · provenance · persistence · hardening
  + cost-deflection reporting

DAS Platform (commercial)       the FDE deployment engine — repeatability
  das deploy client.yaml · ContextSource connectors · compliance bundle · CLI
```

The **Platform tier is what this plan builds.** It is thin orchestration over the
already-proven Control Plane, so its correctness rests on tested foundations.

### Platform components (this build)

| Component | Module | Job |
|---|---|---|
| Declarative spec | `das/platform/spec.py` | `client.yaml` → validated `ClientSpec` |
| Deployment engine | `das/platform/deploy.py` | spec → live `ControlPlane` in one call |
| Connector seam | `das/platform/connectors.py` | `ContextSource` — the last-mile integration point |
| Expert trainer | `das/platform/trainer.py` | pluggable `train_fn`; deterministic default |
| Compliance bundle | `das/platform/bundle.py` | the FDE "leave-behind" audit artifact |
| CLI | `das/platform/cli.py` | `das deploy / verify / offboard / bundle` |

---

## 5. Complete product lifecycle (design → delivery)

### Stage 0 — Design (pre-engagement)
- FDE + client scope the workflow bottleneck (human work — out of scope for DAS).
- Output: a list of **tenants**, the **specialists** each needs, and the
  **escalation policy** (when to fall back to a frontier LLM / a human).
- This becomes a `client.yaml` — the single artifact that drives everything else.

### Stage 1 — Build (declare, don't code)
- FDE writes `client.yaml` (tenants, experts, users/roles, escalation, signing).
- FDE implements one `ContextSource` per client data source against the fixed
  interface (~50 lines) — SQL, REST, or vector store. No forking DAS internals.

### Stage 2 — POC (5 days, the sales proof)
The measured POC sequence, each step a real command:
1. **Setup** — `das deploy client.yaml` → governed fleet up.
2. **Route** — real query → local specialist, high confidence, stays local.
3. **Escalate** — ambiguous query → below threshold → frontier LLM fallback.
4. **Grow** — graft a new specialist live; prove others byte-identical.
5. **Erase** — `das offboard --tenant X`; export the signed bundle; `das verify`.

### Stage 3 — Deploy (production)
- Container / k8s per client (see §7). State on a volume; secrets from files.
- `DAS_ENV=production` refuses to start without a real audit secret + trusted
  proxy (security hardening F1–F7, [SECURITY_REVIEW.md](SECURITY_REVIEW.md)).

### Stage 4 — Operate
- Add/retrain specialists via graft (no re-certification of the rest).
- Cost-deflection report: escalation rate, % kept local, $/1k queries.
- Observability on routing quality + drift.

### Stage 5 — Offboard / deliver
- `das offboard --tenant X` → provable deletion (GDPR Art. 17).
- `das bundle` → dated, self-contained compliance document the client's auditor
  verifies **offline** with `das-verify` — the deliverable that outlives the FDE.

---

## 6. Pricing

Open-core. Charge for exactly the two things that are provable — governance and
cost deflection — never for model quality.

| SKU | Who | What's included | Price point (design-partner era) |
|---|---|---|---|
| **Core** | everyone | router/leaf/forest, LangGraph node, NumPy+torch backends | **Free / OSS (MIT)** |
| **Control Plane** | production teams | signed audit, RBAC, multi-tenancy, provenance, persistence, hardening | **$2.5–5k / month** per deployment (annual), or $30–60k/yr |
| **Platform** | FDE teams / integrators | `das deploy`, connectors, compliance bundles, multi-client CLI, priority support | **$60–120k / yr** per FDE team seat-pack (up to N client deployments) |
| **Design partner** | first 1–3 logos | Platform + hands-on deployment + roadmap input | **Discounted / co-development** (credibility > revenue at this stage) |

Value-metric options to test with the design partner (pick one, don't stack):
- **Per governed deployment** (per client/tenant fleet under management) — cleanest.
- **Per audited action volume** (graft/prune/delete events) — aligns to governance use.
- **Frontier-LLM $ deflected** — share of measured savings; strongest ROI story
  but requires the cost benchmark to be live first.

Expansion path: land on Control Plane for one deployment → expand to Platform once
they do a *second* client and feel the repeatability pain.

---

## 7. Where to deploy it

DAS runs **inside the client's trust boundary** — that is the privacy pitch. It is
a torch-free NumPy+Flask control plane, so it fits almost anywhere.

| Target | When | How |
|---|---|---|
| **Client VPC / on-prem k8s** | regulated, data-residency, default | `deploy/k8s.yaml` — Deployment + Service + PVC + audit Secret |
| **Single container** | POC, small edge box | `docker run das-governance` with state volume |
| **Client's own cloud (BYOC)** | SaaS integrators | one namespace per client, single-replica audit writer |
| **Air-gapped / offline** | high-security | NumPy core only; audit verified offline with `das-verify` |
| **Edge / mobile** | field / disconnected | compact expert store (`das/mobile_store.py`) syncs to a folder |

Non-negotiables everywhere: **single audit writer** (one replica owns the chain),
**state on a durable volume**, **secrets from mounted files not env** in prod,
and the API behind a gateway that authenticates `X-DAS-Actor` (mTLS/OIDC).

We do **not** run it as a shared multi-tenant SaaS we operate — the whole value is
that each client's fleet lives in *their* boundary. We ship the engine; the FDE
deploys it there.

---

## 8. Go-to-market sequence

1. **Design partner before more building** (still the real blocker — Phase 0 exit).
   Target an integrator or internal platform team, not an end-enterprise.
2. **Lead with two provable axes** — governance *and* cost deflection — explicitly
   not capability. Matches the repo's "credibility is the product" stance.
3. **The demo moment is the growth loop** — grow a specialist live, prove the
   others didn't move. No competitor demos that.
4. **The leave-behind closes it** — hand the auditor a bundle they verify offline.
5. **Open-core flywheel** — Core OSS drives adoption; Platform captures the FDEs.

## 9. Success metrics

- **Time-to-deploy at client N+1** < 1 day (from `client.yaml` to routing traffic).
- 1 design partner running a real governed workload.
- Cost benchmark live: measured % traffic deflected + $/1k on a real query mix.
- Audit bundle accepted as a compliance artifact by the partner.
- Green CI; semver PyPI releases; the platform CLI documented.

## 10. Honest gaps (what this build does and does not do)

- **Deterministic synthetic trainer** ships as the default so `das deploy` runs
  end-to-end with zero heavy deps and is fully tested. Production swaps in a
  teacher-backed trainer (`das/training/`, [GROWING_CHILD.md](GROWING_CHILD.md)).
- **Reference connectors** (`Static`, `Callable`, `Rest`) define the seam; the
  client-specific SQL/vector integration is still the FDE's ~50 lines.
- **Cost deflection is measured at route time** (confidence < threshold) but the
  end-to-end `$ / 1k` benchmark (`benchmarks/cost_bench.py`) is the next build.
- Real large-model backend + latency SLA remain roadmap Phase 1, unchanged.
