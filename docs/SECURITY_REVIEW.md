# Security review — DAS governance control plane

> **Scope:** the governance layer that is DAS's actual product —
> [`das/governance.py`](../das/governance.py), [`das/audit.py`](../das/audit.py), the
> persistence format, and the REST surface in
> [`apps/governance_api.py`](../apps/governance_api.py). This is a **self-review** by the
> authors, written to be honest about gaps rather than to reassure. It is **not**
> an independent third-party audit; a real launch needs one (roadmap Phase 4).
>
> **Last reviewed:** 2026-06-21 · against commit at time of writing.

## Assets & trust boundaries

| Asset | Why it matters |
|---|---|
| Expert weights (`forest.npz`) | the customers' models; isolation + deletion guarantees attach here |
| Audit log (`audit.json`) | the compliance record; must be tamper-evident |
| HMAC secret (`DAS_AUDIT_SECRET`) | the root of trust for the whole audit chain |
| Governance state (`control_plane.json`) | tenants, users, roles, registry |

**Trust boundary:** everything inside the process is trusted; the disk
(`DAS_STATE`) and the network are **not**. The secret is supplied at runtime
(env / k8s `Secret`) and is **never written to disk** — that separation is the
design's main strength and is what makes a swapped weights file detectable.

## What is protected (and verified)

- **Audit integrity.** Each entry is HMAC-SHA256 signed and hash-chained to the
  previous; `verify()` re-walks both. Edit/reorder/insert/delete is caught.
  *Tested:* `test_audit_tamper_detected`, benchmark "tamper caught 100%".
- **State ↔ audit binding.** The last entry fingerprints the fleet; `load()`
  recomputes it from the restored weights (`state_matches_audit()`). So replacing
  the *unsigned* `forest.npz` with a different forest is detected even though the
  npz itself isn't signed. *Tested:* `test_state_matches_audit_detects_weight_swap`.
- **Access control.** Every privileged op is `_check`ed against role + tenant
  scope; denials are themselves logged before the exception. *Tested:*
  `test_rbac_role_denials`, `test_rbac_tenant_scope`; API returns 403.
- **No code-execution via state files.** Persistence is `np.savez` + JSON, not
  pickle; `np.load` runs with `allow_pickle=False` (default). Loading a malicious
  state file cannot execute code — at worst it fails to parse or fails the audit
  binding.
- **Input validation at the API.** `/predict` checks embedding type/length;
  malformed bodies get 400, unknown experts 404, denials 403.

## Findings — real gaps (ranked)

### F1 · Rollback / freshness is NOT proven (HIGH)
The audit chain proves *internal* consistency and ordering, **not freshness**. An
attacker with write access to `DAS_STATE` can restore an *older, self-consistent*
snapshot (matching `forest.npz` + `control_plane.json` + `audit.json` from a
previous `save()`). Both `verify()` and `state_matches_audit()` pass, because the
old triple is internally valid. This could silently undo a deletion or
role-revocation.
**Mitigation (not yet built):** anchor the latest `(seq, last_sig)` to an
append-only external store (a monotonic counter, a notarization service, or a
transparency log) and refuse to load a chain shorter/older than the anchor.

### F2 · Identity is asserted, not authenticated (HIGH, API only)
`apps/governance_api.py` takes the principal from the `X-DAS-Actor` header. RBAC is
enforced on that principal, but nothing proves the caller *is* that principal.
This is stated plainly in the module docstring and is acceptable only behind a
trusted authn proxy.
**Mitigation:** terminate mTLS/OIDC at a gateway and inject a verified identity;
never expose the API directly. Treat the header as trusted *only* from the proxy.

### F3 · Default secret is a footgun (MEDIUM)
`DAS_AUDIT_SECRET` defaults to `das-dev-key`. If unset in production, anyone who
knows the (open-source) default can forge a valid-looking chain.
**Mitigation:** the process should refuse to start with the default secret when a
"production" flag is set. *Not yet implemented* — currently only documented.

### F4 · Self-reported timestamps (MEDIUM)
Entry `ts` comes from local `time.strftime` with no trusted time source. The
chain proves *relative order*, not wall-clock time, and a compromised host could
backdate entries within a freshly forged chain (see F1/F3).
**Mitigation:** include an external trusted-timestamp / RFC 3161 token, or anchor
as in F1.

### F5 · No rate limiting / DoS controls (LOW–MEDIUM)
The Flask app has no throttling; `/predict` does a forest forward per call and
`delete_tenant` is O(experts). Flask's dev server is also not a hardened
production server.
**Mitigation:** run behind a real WSGI server (gunicorn/uvicorn) + reverse proxy
with rate limits; the container should not expose the dev server directly.

### F6 · Secret lives in process memory / env (LOW, inherent)
By design the secret is in memory and the environment. Env vars can leak via
crash dumps, `/proc`, or child processes.
**Mitigation:** prefer mounted-file secrets over env where the platform supports
it; scrub on shutdown is out of scope for Python.

## Non-findings (checked, currently fine)

- **Tenant scope bypass:** operators are checked against the *expert's* tenant on
  prune and the *target* tenant on graft/delete; cross-tenant attempts are denied
  and logged (`test_rbac_tenant_scope`).
- **Hash collision for isolation:** isolation uses SHA-256 over raw weight bytes;
  forging a colliding-but-different weight set is not a practical threat.
- **Arbitrary file write on load:** `load()` reads fixed filenames inside the
  given dir; it does not take paths from the (untrusted) state file.

## Recommendations, in priority order

1. **Build F1 (freshness anchoring)** — it's the one gap that defeats the
   deletion/revocation guarantees the product is sold on.
2. **Enforce F3** (refuse default secret in prod) — cheap, high value.
3. **Document/enforce F2** deployment contract (authn proxy mandatory).
4. Harden the serving stack (F5) and adopt trusted timestamps (F4).
5. Commission an **independent** security audit before any 1.0 / GA.

## How to reproduce the protections

```bash
python control_plane_demo.py        # RBAC denials, tenant-delete isolation, tamper detection, persistence
python governance_benchmark.py      # audit/RBAC/provenance vs baselines, with numbers
pytest tests/test_governance.py tests/test_governance_api.py -q
```
