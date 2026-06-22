# DAS — design-partner targeting & outreach

The single goal of this phase: **one** design partner with a real, governed AI
workload. Not revenue, not ten logos — one team whose compliance pain is sharp
enough that provable isolation + deletion + an auditable trail is worth their time.

---

## Ideal design partner (in priority order)

1. **B2B SaaS that fine-tunes / adapts a model per enterprise customer.**
   They already feel multi-tenancy pain: "prove customer A's data didn't leak into
   customer B's model." Legal-tech, HR-tech, support automation, vertical copilots.
   *Why first:* the isolation + per-tenant-deletion story maps 1:1 to a contract blocker they hit in security review.

2. **Regulated AI teams** — fintech, health/med-tech, insurance, public sector.
   Driven by GDPR / EU AI Act / sector audits. They need *evidence*, not assurances.
   *Why:* the exportable signed audit artifact is the exact thing their auditor asks for.

3. **Platform/ML-infra teams** running many fine-tuned adapters who keep rebuilding
   homegrown audit + access control around them.
   *Why:* DAS replaces the governance glue they maintain by hand.

**Disqualifiers (don't chase):** teams who want a better/cheaper base model (not us),
single-tenant consumer apps with no compliance driver, anyone needing proven large-LLM
scale *today*.

### Qualifying questions (first call)
- "When a customer asks you to prove their data didn't influence another tenant's model, what do you do today?"
- "What happens operationally when someone exercises a deletion / right-to-be-forgotten request?"
- "When you ship a model change, how much re-validation does it trigger, and who signs off?"
- If these land with a wince → strong fit. If shrugs → wrong partner.

## Where to find them
- Warm intros first: your network, design-partner asks in relevant founder/Slack/Discord communities.
- Regulated-AI and MLOps communities; LangChain/LangGraph ecosystem (you integrate *under* them).
- Posts that lead with the **honest** framing (proof + limits) — credibility is the differentiator; the
  README's "Honest evaluation" section is your filter for serious people.

---

## Cold outreach — email template

> **Subject:** prove your AI change didn't touch the certified model
>
> Hi {name},
>
> {Company} adapts models per customer — which means at some point a security or
> compliance team asks you to *prove* one tenant's data never influenced another's,
> or to delete a customer's influence and show it's gone. With one shared model that's
> effectively unanswerable.
>
> I built DAS to make it structural: each capability is an isolated adapter, adding or
> deleting one leaves every other **byte-identical** (SHA-256 verified), and every
> action lands in a signed audit log your auditor can verify *offline, without access
> to your system*. It runs **under** your existing stack (e.g. LangGraph), not instead of it.
>
> It's an honest research-preview — proven at small scale, and I'm upfront about what
> isn't. I'm looking for one design partner with a real multi-tenant or regulated
> workload to prove it on. 20 minutes to see if the pain is real for you?
>
> 90-second proof you can run yourself: {repo link} → `python examples/audit_export_demo.py`
>
> — {you}

### Warm-intro version (2 lines)
> Working on DAS — a governance layer that gives a fleet of fine-tuned models provable
> per-tenant isolation, clean deletion, and an auditor-verifiable trail. Looking for one
> design partner with a real regulated/multi-tenant workload — does {company/person} fit?

---

## What you offer a partner (and what you ask)

**You give:** hands-on integration, the control plane on their real adapters, a
compliance artifact their auditors accept, direct roadmap influence, no charge during the partnership.

**You ask:** a real (even small) governed workload to run it on, honest feedback,
a willingness to put the exported audit log in front of their compliance team, and —
if it delivers — a reference / case study.

## Success criteria for this phase
- 1 partner running the control plane on real adapters on a real workload.
- Their compliance/security person looks at an exported `das-verify` artifact and says it's useful.
- One concrete SLA / scale requirement *they* hand you → that becomes the Phase-1 scale roadmap.

## What NOT to do
Don't build more features hoping demand appears (PRODUCT_PLAN's #1 risk). Don't pitch
"beats frontier models / 90% cheaper" — your own benchmarks contradict it and it burns the credibility that *is* the moat. Lead with proof and limits.
