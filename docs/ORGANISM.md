# The Organism Branch — research charter

> **Branch:** `research/organism`, forked from `feat/das-fde-platform`.
> **Status:** research. Nothing here is product, nothing here is sold, and
> nothing here feeds the compliance pitch. Findings flow back to the trunk only
> as measured results.

## The fork, stated as the measured fact it comes from

The repo's own benchmark ([`benchmarks/unsupervised_routing.py`](../benchmarks/unsupervised_routing.py))
established the tension this branch exists to explore:

> *Self-organising routing and auditable isolation are in tension: co-training
> the router to discover domains unsupervised destroys the byte-identical
> guarantee. You get one or the other.*

The trunk chose the audit trail: a **garden** — everything lives, grows, and
dies, but only through policy-gated, hash-proven, human-set acts. That is the
commercial product, and this branch does not touch it.

This branch chooses the other fork: the **organism** — what DAS becomes when
the guarantees are deliberately relaxed and the lifecycle machinery is allowed
to run itself.

## What the organism gives up (eyes open)

- **Byte-identical proofs.** Co-adaptation means frozen things move.
- **Per-action auditability.** A continuously learning system has no discrete
  "acts" to sign.
- **The compliance market.** Nothing grown here is certifiable in the trunk's
  sense; that is the price of admission and it is paid knowingly.

## What it might gain (the research questions)

1. **Self-organising specialization.** Do domains *emerge* from traffic without
   labeled routing supervision? (The trunk's known bottleneck — routing — solved
   by letting the router live.)
2. **Population dynamics.** The lifecycle primitives already look Darwinian:
   `graft` = birth, teacher lessons = nutrition, `prune_dormant` = natural
   selection by traffic, the share/import arena = reproduction. Add mutation
   (perturbed candidates) and fitness (routing share × accuracy) — does a
   *population* of experts find a better division of labour than a designed one?
3. **Continuous adaptation.** No accept/reject gate — the organism drifts with
   its data. When does drift help, and when does it eat itself (catastrophic
   forgetting is back on the table, deliberately)?

## The epoch hypothesis — the possible middle ground

The most valuable outcome would be a *partial* reconciliation:

> **You cannot audit the organism's every heartbeat — but you might audit its
> generations.**

Let the system live wild *within an epoch*; at epoch boundaries, snapshot,
fingerprint, and sign. Provenance becomes **lineage** ("this expert descends
from that one, through these generations") rather than per-action attribution.
Deletion becomes **extinction with a signed death certificate** at the next
boundary. The research question: which of the trunk's guarantees survive in
generational form, and are they worth anything to anyone? (Honest prior: maybe.
Regulated buyers likely still need the garden; research/creative/personal
fleets might not.)

## Research plan

| Phase | Question | Exit |
|---|---|---|
| O-1 | Epoch harness: run the existing fleet unguarded between signed generation snapshots | Lineage chain verifies across N generations |
| O-2 | Revisit self-organising routing at today's substance (real embeddings, teacher traffic) | Measured: does specialization emerge? At what isolation cost? |
| O-3 | Population dynamics: mutation + traffic-fitness selection over the germination ladder | Measured: designed fleet vs evolved fleet on the same curriculum |
| O-4 | Write up honestly, including the failures | A benchmark table the trunk's README could cite |

## Non-goals

- No RBAC, licensing, or console work here — the trunk owns operability.
- No claims. This branch earns sentences the same way the trunk did: a hash, a
  benchmark, or a test underneath every one — the standard is inherited even
  where the guarantees are not.

## Relationship to the trunk

The platform sells the garden. This branch explores what the garden deliberately
forbids — so that when a buyer asks *"why can't it just learn on its own?"*, the
answer is a measured document instead of a shrug.
