# DAS — a governance layer for fleets of AI experts

**v0.2.0** · MIT core · 225 tests passing · NumPy-only core, optional torch/HF backend

> **Governed AI capabilities you can add, remove, and audit — without ever
> touching what's already certified.**

DAS runs your specialists as a **forest**: a router in front of isolated
expert leaves, wrapped in a control plane that makes every change provable.
Grow a new expert and the others are proven byte-identical. Delete a tenant
and the survivors are proven untouched. Every privileged action lands in a
tamper-evident, offline-verifiable audit chain. Experts can be real **LoRA
adapters on a real transformer's own attention layers** — and growing one is
real adapter fine-tuning, gated by policy, not vibes.

**DAS is not** a smarter or cheaper model. Measured against per-task LoRA
adapters it *ties* on isolation and forgetting — that's what isolation buys,
and we publish the benchmark that says so. Its contribution is everything
around the adapters: routing, RBAC, multi-tenancy, provenance, provable
deletion, gated growth, and the audit artifact a compliance team can verify
without trusting you.

---

## How it works

```
                      client.yaml  ──  das deploy
                                           │
            ┌───────────── ControlPlane (governance) ─────────────┐
            │  RBAC · tenants · signed audit chain · persistence  │
            │                                                     │
 query ──►  │   embed ──► StemRouter ──► one expert leaf ──► out  │ ──► answer
            │                │                                    │   or escalate
            │       confidence < threshold ──► frontier LLM       │
            └─────────────────────────────────────────────────────┘

 leaves (pick a backend, same guarantees):
   synthetic     NumPy heads — zero heavy deps, deterministic, CI-fast
   lora-minilm   LoRA adapters on frozen MiniLM's q/v attention projections
                 (rank ladder: seed r=0 → sprout 1 → sapling 2 → … → tree 8)
```

Three properties hold by construction and are re-proven on every change:

1. **Isolation** — experts share nothing trainable; grafting or pruning one
   leaves every other expert's SHA-256 weight fingerprint unchanged, and the
   control plane records that proof in the audit entry.
2. **Earned growth** — a new or bigger expert trains as a *candidate* in
   quarantine and replaces the live expert only if it clears an accuracy
   floor and a minimum improvement (the parsimony gate refuses capacity that
   isn't needed — most experts live and die as seeds).
3. **Verifiable history** — the audit chain is HMAC- or Ed25519-signed,
   survives restarts bound to the weights (`state_matches_audit`), detects
   snapshot rollback via a freshness anchor, and exports as a bundle an
   auditor verifies offline.

## Quickstart

```bash
pip install -e ".[platform]"        # NumPy core + the `das` CLI
# extras: [hf] LoRA-on-MiniLM backend · [web] serving · [crypto] Ed25519 · [all]
```

**Create a forest** — one spec, one command:

```yaml
# client.yaml
client: acme
backend: lora-minilm          # or omit for the NumPy backend
tenants:
  - name: legalco
    experts: [{name: legal}]
  - name: medico
    experts: [{name: medical}]
users:
  - {name: care-agent, role: operator, tenant: medico}
escalation: {frontier_llm: claude-sonnet-5, confidence_threshold: 0.7}
```

```bash
das deploy client.yaml --save state/
```

Or in Python:

```python
from das.platform import deploy

dep = deploy("client.yaml")
dep.route("care-agent", "the MRI scan was blocked pending urgent review")
#  {'expert': 'medical', 'confidence': 0.71, 'decision': 'local', ...}

dep.grow("root", "medico", "insurance", stage="seed")   # others proven untouched
dep.germinate("root", eid=2, target_acc=0.85)           # promote ONLY if earned
dep.offboard("legalco")                                 # provable right-to-be-forgotten
dep.export_bundle("acme_audit.json")                    # the auditor's leave-behind
dep.save("state/")                                      # byte-exact across restarts
```

Teachers write the lessons experts learn from: deterministic template
teachers offline, or a real LLM over Ollama / OpenAI-compatible endpoints
(`dep.register_teacher({...})`) writing the text corpus an adapter trains on.

## See it

```bash
python apps/lora_growth_app.py       # :5099 — grow a specialist live in the
                                     # browser; watch every other expert's hash
                                     # stay byte-identical; parsimony gate; audit
python apps/platform_console.py     # :5090 — Rancher-style console over many
                                     # fleets (session login: the logged-in user
                                     # IS the RBAC principal)
python examples/lora_growth_demo.py  # the same demo moment, in a terminal
```

## Measured, honestly

**Latency** ([lora_latency_bench.py](benchmarks/lora_latency_bench.py), CPU p50):
routing floor ~7 ms/query; a seed expert costs exactly the floor (its head
reuses the routing embedding); an adapted expert ~14 ms (its own encoder
pass). Batch-16: ~1 / ~2.4 ms per text.

**The rank ladder** ([lora_rank_bench.py](benchmarks/lora_rank_bench.py), 3 seeds,
train/eval vocabulary disjoint):

| curriculum | rank 0 (seed, 770 params) | rank 1 (~10K) | above |
|---|---|---|---|
| topical (lexical label) | **1.00** | 1.00 | pure cost — growth refused |
| word order (identical vocab) | 0.71 | **1.00** | buys nothing |
| negation × valence (XOR) | 0.90 | **1.00** | buys nothing |

The first rung is real on compositional labels; everything above it is
cosmetic, and the promotion gate enforces exactly that. Two failed designs
are documented in the bench so they don't come back (negation *parity* is
lexically countable under mean pooling; a 16-sentence corpus can't support a
held-out eval).

**Governance** ([governance_benchmark.py](benchmarks/governance_benchmark.py)):
against a monolith and per-task isolated adapters, DAS ties the adapters on
isolation/forgetting/deletion — and is alone on audit, RBAC, and provenance.
**Cost deflection** ([cost_bench.py](benchmarks/cost_bench.py)): the
deflection-vs-quality curve across escalation thresholds, including the
failure mode (aggressive deflection misroutes novel queries). Dollar
constants are illustrative until measured on a real traffic mix.

## Production

```bash
docker build -t das-governance .                                  # torch-free image
docker build --build-arg DAS_EXTRAS=web,platform,hf -t das:hf .   # + LoRA backend
kubectl apply -f deploy/k8s.yaml                                  # single writer + PVCs
```

`DAS_ENV=production` refuses to start without real secrets and a trusted-proxy
contract; secrets are file-mounted; the audit chain can be Ed25519-signed and
freshness-anchored. Put the reference **authn gateway** in front —
[deploy/gateway/](deploy/gateway/README.md) (nginx TLS + oauth2-proxy OIDC;
the gateway is the only thing allowed to assert identity) — and operate it
per the **runbook** ([docs/RUNBOOK.md](docs/RUNBOOK.md): backup/restore,
rollback detection, key custody).

## Documentation

| | |
|---|---|
| [PLATFORM_PLAN.md](docs/PLATFORM_PLAN.md) | the FDE platform: strategy, §10 hardening, §11 honest gaps, §12 execution |
| [PRODUCT_PLAN.md](docs/PRODUCT_PLAN.md) | the maturity roadmap (Phase 0→4) and what we deliberately dropped |
| [GROWING_CHILD.md](docs/GROWING_CHILD.md) | the growth loop: teachers, quarantine, germination |
| [SPECIALTY_FORESTS.md](docs/SPECIALTY_FORESTS.md) | hierarchical specialty trees under LangGraph |
| [SECURITY_REVIEW.md](docs/SECURITY_REVIEW.md) · [THREAT_MODEL.md](docs/THREAT_MODEL.md) | findings F1–F7, mitigations, control→test traceability |
| [RUNBOOK.md](docs/RUNBOOK.md) | backup, restore, key custody, vendor-console policy |
| [CHANGELOG.md](CHANGELOG.md) | release history (current: v0.2.0) |

## Repository map

```
das/                 the MIT core: forest, router, leaves, lifecycle,
                     governance ControlPlane, audit, hierarchy, training
das/platform/        the deployment engine: spec → deploy, LoRA experts,
                     germination, teachers, licensing, bundles, CLI
das_torch.py         torch backend (autograd leaves, LoRA-on-MLP, checkpoints)
das_text.py          the frozen MiniLM text encoder (+ demo corpus)
apps/                consoles, dashboards, demo surfaces
benchmarks/          every number this README cites, reproducible
deploy/              Dockerfile entrypoint, k8s, the reference authn gateway
tests/               225 tests — isolation proofs, gates, RBAC, persistence
```

## Honest limits

Experts are MiniLM-class adapters, not large generative models — GPU/serving
scale is deliberately deferred until a design partner's workload demands it.
Lessons come from template or endpoint-LLM teachers, not yet real labeled
production data. The control plane is a single writer by design (one audit
chain, one owner). The open-core/commercial split is planned, not yet
executed. What's next, in order: a design partner, their real data, their
traffic on the cost benchmark, an independent security audit.

## License

MIT — see [LICENSE](LICENSE). The `das/platform` deployment engine is
intended to become the commercial layer of an open-core split; today the
whole repository is MIT.
