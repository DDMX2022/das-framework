# DAS — 3-minute design-partner demo script

Goal: in three minutes, make a technical buyer (CTO / Chief Risk / Head of Platform)
believe **provable isolation, clean deletion, and an auditable trail** are real and
running — on a real model, not slides. Every step below is a command in this repo.

**Before the call:** `pip install -e ".[hf,web]"`, run each demo once so the MiniLM
weights are cached and nothing downloads live. Have two terminals + a browser ready.

---

## The arc (what you're proving)

> "You can add, remove, and audit AI capabilities without ever touching what's
> already certified — and hand your auditor a document they can verify themselves."

---

### 0 · Frame (20s)
"Production AI is one shared set of weights. That means you can't prove a new feature
didn't change the certified model, you can't cleanly delete one customer's influence,
and you can't prove tenant A never affected tenant B. DAS fixes that structurally.
Let me show you — on a real model, live."

### 1 · Isolation + deletion on real text (70s) — `examples/hf_governance_demo.py`
Run it and narrate the sections as they print:
- "Real frozen encoder, real English sentences — not synthetic vectors."
- "Three experts trained; watch unseen sentences route to the right one." → point at the ✓ provenance lines.
- **Graft:** "I add a new capability — the existing experts are now **byte-identical**, SHA-256 verified. Adding a feature didn't touch the certified ones."
- **Prune:** "Right-to-be-forgotten: I delete one — survivors **byte-identical**, the capability is gone."
- "Every step is in a signed, hash-chained audit log. Chain valid."

*Land it:* "Monolithic fine-tuning can't do any of that without re-validating the whole model."

### 2 · The auditor's artifact (60s) — `examples/audit_export_demo.py` + `das-verify`
- "Here's the part compliance cares about. I export the signed log to a portable document."
- Show it verify **keyless**: "Your auditor runs `das-verify` with **no access to our system and no secret** — chain intact, weight fingerprints consistent."
- Show a **tamper** caught: edit a fingerprint → `das-verify` flags it.
- "With the key, full authenticity. This document *is* the compliance evidence."

### 3 · It fits under your stack (30s) — the console + LangGraph
- `python apps/console.py` → graft/prune live, watch the SHA-256 trail update; OR
- one line: "It's a drop-in node under LangGraph — `DASExpertNode` routes to a governed
  expert and writes provenance back into graph state. You don't rip anything out."

### 4 · The honest close (20s)
"To be straight: on raw model quality this ties LoRA adapters — the moat is the
governance plane, and we're small-scale today. That's exactly why we want a design
partner: to prove it on **your** real, governed workload. Is forgetting-proof
isolation or hard cryptographic deletion the sharper pain for you right now?"

---

## If you only have 60 seconds
Run `examples/audit_export_demo.py`, say: *"Real governed operations → a signed
document your auditor verifies offline, and tamper is caught. That's the whole pitch."*

## Likely questions (have these ready)
- **"How is this different from RAG?"** RAG changes the *text* in the prompt; DAS changes
  *which isolated weights* run, and proves it. They compose — DAS routes to the expert, RAG feeds it the document.
- **"Isn't this just MoE / LoRA?"** Mechanically yes — hard-routed LoRA experts. The
  product is the governance plane on top: signed audit, RBAC, multi-tenancy, provable deletion.
- **"Does the API need a GPU / transformer?"** No — the control plane is NumPy + Flask,
  torch-free and containerized. You encode client-side and POST the embedding.
- **"Scale?"** Honest answer: unproven at large-LLM size today; the sparse mechanic scales,
  real-LLM quality is the open work — and a reason to partner now.
