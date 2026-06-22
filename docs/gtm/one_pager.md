# DAS — governance for fleets of AI models

**Add, remove, and audit AI capabilities without touching what's already certified.**

---

### The problem

Teams shipping AI into **regulated or multi-tenant** settings can't answer three questions their own compliance, security, and legal teams ask:

1. *"Prove adding this feature didn't change the model we already certified."*
2. *"This customer invoked their right to be forgotten — delete their data's influence and prove it's gone."*
3. *"Show a regulator that tenant A's data never influenced tenant B's model."*

Monolithic models and standard fine-tuning **cannot** answer these. One shared set of weights means every change re-opens validation of the whole model, deletion is unsolved, and isolation is unprovable. RAG and guardrails filter *text*; they don't give you provable isolation of *weights*.

### What DAS is

A **governance control plane** that sits **under** your orchestrator (LangGraph, etc.) and serving stack — not a replacement for them. Each capability is an **isolated expert** (a LoRA adapter). DAS routes each request to exactly one expert, lets you **graft** a new one without disturbing the others, **prune** one to delete it cleanly, and emits a **tamper-evident, signed audit trail** that is itself the compliance evidence.

### What's proven today (measured, not asserted)

| Guarantee | Evidence |
|---|---|
| Add a capability → every other expert **byte-identical** (SHA-256) | BWT **0.000**; 100% isolation on graft |
| Delete an expert / a whole tenant → survivors **byte-identical**, capability gone | right-to-be-forgotten, 100% |
| **Tamper-evident audit log** — every action (and every denial) signed + hash-chained | 100% of edits/reorders/deletes caught |
| **Exportable compliance artifact** — verify offline with `das-verify`, no system access | keyless + signed verification |
| RBAC + multi-tenancy + per-query provenance | role/tenant denials enforced + logged |
| Runs on a **real frozen pretrained encoder over real text** | experts route real queries correctly |

Reproducible head-to-head vs. a monolith and isolated adapters: the monolith fails every isolation test (BWT **−0.467**, 0% isolation); DAS ties isolated adapters on isolation/deletion and **adds** the governance plane (audit, RBAC, provenance) you'd otherwise build yourself.

### What DAS is *not* (so you can trust the rest)

Not a "better" or "cheaper" frontier model. On raw capability it ties LoRA adapters — its edge is the **governance layer**, not the architecture. Today it's proven at small scale; large-real-model scale is the open work. We lead with this because **credibility is the product.**

### Who it's for

B2B SaaS serving enterprise tenants · regulated AI (fintech, health, legal, public sector) · anyone facing GDPR/EU-AI-Act deletion and isolation obligations on a fleet of fine-tuned models.

### The ask — a design partner, not a sale

We're looking for **one** team with a real multi-tenant or regulated AI workload to run the control plane on real adapters, treat the exported audit log as a compliance artifact with us, and co-define the production SLA. In return: hands-on integration, direct influence on the roadmap, and the governance evidence your auditors want.

**Try it in 10 minutes:** `pip install -e ".[hf]"` → `python examples/hf_governance_demo.py` · `python examples/audit_export_demo.py`
**Repo:** https://github.com/DDMX2022/das-framework
