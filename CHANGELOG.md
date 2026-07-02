# Changelog

## 0.2.0 — 2026-07-03

The FDE platform release: experts became real transformer adapters, the
deployment engine became a product, and the hardening backlog's engineering
items closed. Every claim below is backed by a test, a benchmark, or a
runnable demo.

### The LoRA expert backend
- Experts as **LoRA adapters on frozen MiniLM's own attention layers**
  (`das/platform/lora_expert.py`) behind the existing `train_fn` seam — the
  unmodified ControlPlane proves byte-identical isolation on graft/prune over
  transformer experts. α/r scaling; standard target set (all query/value
  projections).
- **Rank-ladder germination**: experts start as seeds (rank 0, a 770-param
  head) and earn capacity (r=1→8) through the accuracy-gated quarantine loop,
  with best-of-restarts candidates; audited either way.
- Honestly measured (`benchmarks/lora_rank_bench.py`): lexical text saturates
  at rank 0 (growth refused — parsimony); compositional labels are the real
  first rung (word order 0.71→1.00, negation×valence 0.90→1.00); nothing
  above rank 1 qualifies. Latency (`benchmarks/lora_latency_bench.py`):
  ~7 ms/query routing floor on CPU; seeds ride the routing embedding free;
  adapted experts ~14 ms.
- Teacher-driven text lessons: deterministic template teachers offline, and
  the **LLM-teacher bridge** (`EndpointLLMTeacher.fetch_rows` →
  `LLMTextLessonTeacher`) so a real endpoint LLM writes the corpus.

### Platform / deployment engine
- `backend: lora-minilm` in `client.yaml` — `deploy()` stands up a governed
  transformer fleet end-to-end; forest persistence delegates through
  `ControlPlane.save/load(forest_loader=…)` with byte-identical hashes across
  restarts.
- **Fibonacci germination** for the NumPy backend (width ladder), the
  teacher bridge (`grow`/`improve` Growing-Child loop), specialty trees under
  LangGraph, offline Ed25519 licensing with lifetime entitlement checks, the
  vendor console, and the measured cost-deflection benchmark.
- The **demo moment**, twice: `examples/lora_growth_demo.py` (CLI) and
  `apps/lora_growth_app.py` (browser) — grow a specialist live while every
  other expert's hash streams unchanged, parsimony gate, signed audit tail.

### Hardening (PLATFORM_PLAN §10 engineering items — all closed)
- **Reference authn gateway** (`deploy/gateway/`): nginx TLS + oauth2-proxy
  OIDC, the trusted-proxy contract wired end-to-end; `deploy/k8s.yaml`
  upgraded (production guards, file-mounted secrets, freshness anchor on its
  own PVC, Ingress stub).
- **Platform-console login**: session auth where the logged-in user IS the
  RBAC principal — client-supplied "actor" fields are ignored under auth;
  `DAS_ENV=production` refuses the open demo mode.
- **Backup/restore runbook** (`docs/RUNBOOK.md`): state ≠ anchor ≠ secrets,
  `RollbackDetected` semantics, restore proof, key custody; vendor-console
  network policy documented.
- Docker: `--build-arg DAS_EXTRAS=web,platform,hf` variant for the LoRA
  backend.

### Also
- Mobile store + growth worker + tenant/flow/live-LLM/supercharger demo apps.
- Suite: 225 passed, 3 skipped; NumPy-only CI path stays torch-free.

## 0.1.0

Initial versioned release: the NumPy DAS core (forest, router, isolated
leaves), the governance ControlPlane (RBAC, multi-tenancy, signed audit,
right-to-be-forgotten, persistence), the governance API + console demos, and
the governance benchmark.
