# DAS — Product Maturity Plan

From research prototype → mature, production framework. This plan is deliberately
**narrow**: it commits to the one lane the evidence supports and drops the rest.

## North star
**"Governed AI capabilities you can add, remove, and audit without ever touching
what's already certified."** DAS is a governance / model-fleet layer — not a
"better/cheaper model." Maturity = doing that one thing excellently.

## What we deliberately DROP (focus is maturity)
- "Beats frontier models", "100B on a laptop", "90% cost cut" — unsupported by
  our own measurements; they dilute the product and burn credibility.
- Competing with serving stacks (vLLM) or orchestration (LangGraph) — we
  **integrate under** them, not replace them.
- Bespoke expert formats — adopt **LoRA/PEFT adapters** as the expert format
  (we measured DAS ≈ LoRA + a router, so use LoRA).

## Success metrics (how we know it matured)
- 1 design partner running it on real, governed workloads.
- Reproducible public benchmark: DAS vs LoRA/PEFT/Avalanche on **governance
  axes** (auditable isolation, deletion, multi-tenant), not accuracy.
- p95 inference latency + throughput SLA met at real model scale.
- Tamper-evident audit log accepted as a compliance artifact by the partner.
- PyPI releases on semver; green CI; docs site.

---

## Phase 0 — Foundation & focus  *(weeks 0–3)*
**Goal:** credible engineering baseline + a committed wedge.
- Pick the wedge use case + secure one design partner (regulated / multi-tenant).
- Tests (pytest) over the core proofs (forgetting, restore, routing); GitHub CI.
- Types + `ruff`/`mypy`/pre-commit; structured logging; input validation (no asserts).
- Semver, changelog, PyPI release; mkdocs docs site.
**Exit:** `pip install das-engine` is versioned, CI-green, documented; partner + wedge defined.

## Phase 1 — Real backend  *(weeks 3–8)*
**Goal:** stop being toy-scale; handle real models and real load.
- Experts = **HuggingFace model + LoRA/PEFT adapters** (real, not synthetic leaves).
- Fix routing: shared-backbone / attention router, benchmarked (raw-pixel router is the known bottleneck).
- Serving: batching, async, GPU; integrate **vLLM / Ray Serve** rather than reinventing.
- OpenAI-compatible inference API.
**Exit:** a forest of real LoRA experts serves traffic under a latency SLA.

## Phase 2 — Governance control plane (the product)  *(weeks 8–16)*
**Goal:** the differentiator, production-grade.
- **Tamper-evident audit log** — signed fingerprints, exportable compliance artifact.
- **Multi-tenancy + RBAC + authn/authz** — enforced data isolation, not just shown.
- **Expert registry** — versioning, provenance, rollback, access control (the "marketplace" as a real service).
- **Persistence + lifecycle ops** — forests survive restarts; deploy/retire/rollback experts.
- **Observability** — usage, routing quality, drift dashboards (extend the console).
**Exit:** partner can add/remove/audit experts with provable non-interference + deletion.

## Phase 3 — Integrations  *(weeks 12–18, overlaps)*
**Goal:** be the governance layer *under* existing stacks.
- LangGraph / LangChain node that routes into a DAS forest.
- HF Hub interop for pulling/pushing adapter experts.
- Reference deployment: Docker + Helm/k8s.
**Exit:** DAS drops into an existing LLM app as the governed-expert layer.

## Phase 4 — Prove & launch  *(weeks 16–24)*
**Goal:** evidence + go-to-market.
- Reproducible public benchmark suite (governance axes) + write-up.
- Partner case study; security review of the isolation/audit guarantees.
- Open-core: framework OSS, **governed control plane** as the commercial layer.
- 1.0 release, docs, examples, quickstart.
**Exit:** a real user in production + a credible public story.

---

## Sequencing logic
Phase 0 buys credibility; Phase 1 removes the toy-scale objection; **Phase 2 is the
actual product** (everything else exists to support it); Phase 3 makes it adoptable;
Phase 4 proves it. Don't start Phase 2 polish before Phase 1 makes it real, and don't
build any of it without the Phase 0 design partner pulling.

## Top risks (honest)
- **No real demand** → it stays a demo. Mitigate: design partner before Phase 1.
- **Router quality** → if routing can't be made reliable on real data, the end-to-end
  story weakens; the governance value (known-task-id routing) still holds.
- **"DAS ≈ LoRA + router"** → the moat is governance + audit, not the architecture;
  lead with that or competitors (PEFT + a registry) close the gap.
- **Scope creep** → every dropped item above is a temptation; maturity is saying no.

## First concrete milestone (this is where to start)
1. Back the console with **real LoRA-adapter experts on a HF model** (kills "it's synthetic").
2. **Tests + CI + PyPI** release (credibility).
3. **Signed, exportable audit log** (the governance differentiator made real).
