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

### Vertical: creative-IP fleets (second sales area)

Studios and creative-tooling vendors using AI art manage their real assets as
**fleets of LoRA adapters** — one per character, one per style. That fleet has
exactly the governance shape DAS sells:

- add a character mid-series → graft; established characters provably cannot drift
- a character license expires / an artist revokes style permission → prune, with
  a cryptographic proof the style is STRUCTURALLY GONE (not filtered) — legal
  right-to-be-forgotten for creative IP
- per-panel provenance: which character/style expert produced what, audited

The wedge is NOT "make comics" (crowded generation market) — it is the
**rights-and-provenance registrar** for teams already generating and worried
about legal exposure. Honestly gated: requires the Phase-1 LoRA leaf format
first, then a diffusion-LoRA backend behind the same `train_fn` seam
(architecturally the same move — the control plane never cared what an expert
is made of; unbuilt, unpromised until measured). Candidate design partner
profile: an AI-art tooling vendor or a studio pipeline team, not end artists.

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
| Germination | `das/platform/germination.py` | Fibonacci seed→tree capacity ladder; promotions must earn their params (audited); multi-stage search takes the smallest qualifying stage; distillation transfer; fleet-wide sweep; lesson store persisted with `Deployment.save/load` |
| Console UI | `apps/platform_console.py` | Rancher-style multi-client dashboard over the engine |
| Vendor console | `apps/vendor_console.py` | superadmin licensing UI (issue/renew/revoke) — vendor-side only, never shipped |

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

### Licensing mechanics (built)

Subscriptions are **offline Ed25519-signed license files**
([`das/platform/license.py`](../das/platform/license.py)): the vendor signs
entitlement claims (customer, expiry, max deployments/tenants/experts); the
shipped code verifies against a **pinned** vendor public key — no license server,
no phone-home, so it works in the air-gapped deployments the product targets.
`das license keygen / issue / verify / show`; `das deploy` enforces automatically
via `DAS_LICENSE`. Trust model: nothing configured = evaluation mode (noticed);
a configured-but-invalid/expired/tampered license **fails closed**;
`DAS_LICENSE_REQUIRED=1` is the commercial build's hard switch. Honestly stated:
a key check in source-available Python is a compliance mechanism, not DRM — the
EULA does the legal work (industry-standard posture: GitLab EE, Rancher).

**Open legal prerequisite:** the repo is currently MIT end-to-end. Before selling,
split open-core for real — core stays MIT; `das/platform` + console move to a
commercial license — and bake the vendor public key into the commercial build.

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

**The container front door.** The image boots from the spec: set `DAS_SPEC` to a
mounted `client.yaml` and the entrypoint runs `das deploy` on first start
(materializing the fleet to `DAS_STATE`), then serves it — restart-safe, since a
present state skips the deploy. In k8s the spec is a `ConfigMap`, so changing the
fleet is editing declarative config, not rebuilding an image. Unset `DAS_SPEC`
and behaviour is unchanged (the API bootstraps a demo fleet).

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

## 10. Backlog — server/production hardening (before any hosted sale)

1. ~~Gateway reference deployment~~ **— done**
   ([`deploy/gateway/`](../deploy/gateway/README.md)): nginx TLS + oauth2-proxy
   OIDC via the canonical auth_request pattern, `DAS_TRUSTED_PROXY_SECRET`
   wired end-to-end from file-mounted secrets, identity headers overwritten
   unconditionally at the gateway; `deploy/k8s.yaml` upgraded to the same
   contract (production guards, file-mounted secrets, anchor on its own PVC,
   Ingress stub with the header-injection snippet).
2. ~~Platform-console authentication~~ **— done**: session login
   (`DAS_CONSOLE_USERS(_FILE)` with werkzeug hashes + `DAS_CONSOLE_SECRET`),
   and the logged-in user IS the RBAC principal — the client-supplied
   "actor" field is ignored whenever auth is on (tested: a tenant-scoped
   operator claiming root in the body is still denied cross-tenant).
   Open mode survives for local demos; `DAS_ENV=production` refuses it.
3. ~~Vendor console stays internal-only~~ **— documented**
   ([RUNBOOK.md §5](RUNBOOK.md)): never leaves the vendor network, the token
   is defense-in-depth not the boundary, key custody per the runbook's §4.
4. ~~Backup/restore runbook~~ **— done** ([`docs/RUNBOOK.md`](RUNBOOK.md)):
   what the state is, the three separations (state ≠ anchor ≠ secrets),
   backup/restore procedure with the `RollbackDetected` semantics spelled
   out, restore proof via `/health`'s `audit_chain_ok` +
   `state_matches_audit`, and key-custody notes (HMAC rotation, Ed25519,
   anchor credentials).
5. **Independent security audit** (self-review exists; buyers will ask for third-party).
6. **Legal open-core split** (MIT → core-MIT + commercial platform) + EULA — the
   hard prerequisite to any sale.
7. Managed-cloud operator layer (dedicated instances / BYOC provisioning,
   monitoring, SLA) and marketplace packaging (AWS/Azure container listing).

## 11. Honest gaps (what this build does and does not do)

- **Deterministic synthetic trainer** ships as the default so `das deploy` runs
  end-to-end with zero heavy deps and is fully tested. **The teacher bridge is
  now built** ([`das/platform/teacher_trainer.py`](../das/platform/teacher_trainer.py)):
  `dep.grow(teacher=…)` trains new experts on teacher lessons (offline
  `AlignedVectorTeacher` by default; Ollama/OpenAI-compatible LLM teachers via
  `register_teacher`), and `dep.improve(…)` runs the full Growing-Child loop —
  candidate in quarantine, accuracy floor + no-regression policy, accept/reject,
  audited as `growth_update`/`growth_rejected`. The *substance* objection —
  "the expert is a scoring head, not a language model" — is answered by the
  pulled-forward Phase-1 item below: the default trainer stays NumPy, and the
  LoRA-on-MiniLM trainer plugs in behind the same seam.
- **Pulled forward from PRODUCT_PLAN Phase 1 — now built**
  ([`das/platform/lora_expert.py`](../das/platform/lora_expert.py)): experts as
  **LoRA adapters on MiniLM's own attention layers** (all 6 layers'
  query/value projections) behind the same `train_fn` seam. Growing an expert
  is real adapter fine-tuning (teacher generates a text corpus → adapter
  trains → policy gates), and every governance guarantee is unchanged —
  `MiniLMLoRAForest` duck-types `DASForest`, so the UNMODIFIED ControlPlane
  proves byte-identical isolation on graft/prune over transformer experts
  (tests/test_lora_minilm.py). Germination is a **rank ladder** (seed r=0
  head-only → sprout r=1 → … → tree r=8) with the same earned-promotion gate,
  measured honestly (benchmarks/lora_rank_bench.py): topical text saturates at
  rank 0 (growth refused — parsimony); compositional labels (word order,
  negation×valence) are the real first rung (0.71→1.00, 0.90→1.00); above
  rank 1 buys nothing (pre-α/r-scaling it even destabilized — rank 8 worst
  seed 0.47; the standard scaling fixed it, re-measured all-stable). Lessons
  are template-teacher text — real labeled design-partner data remains open.
- **Reference connectors** (`Static`, `Callable`, `Rest`) define the seam; the
  client-specific SQL/vector integration is still the FDE's ~50 lines.
  **Real-text semantics now exist behind the same seam** (`[hf]` extra):
  `MiniLMContextSource` routes real text through a frozen MiniLM encoder and
  `RealTextLessonEncoder` puts LLM-teacher lessons in the same geometry — plus
  `HierarchicalDASNode` slots the specialty tree under LangGraph with two-level
  provenance and per-branch escalation. With attention-layer LoRA experts now
  built (above), the remaining Phase-1 substance is real labeled lesson data
  (design-partner gated) and a latency SLA at scale.
- **Cost deflection is now measured**: [`benchmarks/cost_bench.py`](../benchmarks/cost_bench.py)
  sweeps the escalation threshold and reports deflection %, answer quality, and
  `$ / 1k` vs a 100%-frontier baseline — including the honest trade-off (aggressive
  deflection misroutes novel queries as the over-confident router keeps them local).
  Constants are illustrative; a real-traffic study on a design partner's mix remains.
- Real large-model backend + latency SLA remain roadmap Phase 1, unchanged.

## 12. Next steps — making the LoRA backend deployable

The attention-LoRA expert (§11) proves the substance; this sequence makes it
something an FDE can actually stand up. Each step has an exit proof, in
dependency order; none of it outranks the §8 rule that a design partner comes
before more building.

1. **Adapter persistence** — **done**: `MiniLMLoRAForest.save/load` (adapters
   + heads + ranks in a manifest, backbone excluded — pinned by model name);
   leaf `weight_hash`es byte-identical across a reload (tested).
2. **Backend switch in the deployment engine** — **done**: `backend:
   lora-minilm` in the spec; `deploy()` stands up a governed transformer
   fleet end-to-end (`ControlPlane` delegates forest persistence via
   `forest_loader`); `grow(stage=…)` grafts rank-ladder seeds, `germinate`
   runs the parsimony gate, save/load survives restarts with the chain
   verified (tested). `improve()` on this backend IS `germinate()` and says
   so.
3. **LLM-teacher → text-lesson bridge** — **done**: `EndpointLLMTeacher`
   grew a public `fetch_rows` seam and `LLMTextLessonTeacher` emits the raw
   sentences as `TextLessonBatch`; `dep.register_teacher` on the LoRA backend
   bridges endpoint teachers automatically (and refuses vector teachers
   honestly). An endpoint-taught adapter passes the same promotion gate
   (tested).
4. **Latency numbers** — **done**: [`benchmarks/lora_latency_bench.py`](../benchmarks/lora_latency_bench.py)
   (CPU, p50): routing floor ~7 ms/query; a seed expert costs exactly the
   floor (its head reuses the routing embedding); an adapted expert ~14 ms
   (its own encoder pass); batch-16 amortizes to ~1 / ~2.4 ms per text. A
   real SLA at scale (bigger encoder, GPU, concurrency) still waits for
   partner pull, per PRODUCT_PLAN Phase 1.
5. **The demo moment on the LoRA fleet** — **done**, twice: the CLI version
   ([`examples/lora_growth_demo.py`](../examples/lora_growth_demo.py)) and the
   dashboard ([`apps/lora_growth_app.py`](../apps/lora_growth_app.py),
   `python apps/lora_growth_app.py` → 127.0.0.1:5099): deploy → real-text
   routing → grow a specialist live with every other expert's hash shown
   before/after, UNCHANGED → the parsimony gate refusing a saturated seed →
   the signed audit trail, all in the browser. Local demo surface (loopback,
   single actor); the hardened multi-user path stays the governance API
   (§10).
6. **Design partner** (the standing blocker) — real labeled lessons replace
   template teachers; the cost benchmark reruns on their traffic mix.
