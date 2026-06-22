# Security review — DAS governance control plane

> **Scope:** the governance layer that is DAS's actual product —
> [`das/governance.py`](../das/governance.py), [`das/audit.py`](../das/audit.py), the
> persistence format, and the REST surface in
> [`apps/governance_api.py`](../apps/governance_api.py). This is a **self-review** by the
> authors, written to be honest about gaps rather than to reassure. It is **not**
> an independent third-party audit; a real launch needs one (roadmap Phase 4).
>
> **Last reviewed:** 2026-06-22 · against commit at time of writing.

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
- **Exportable & independently verifiable.** `to_document()` / `GET /audit/export`
  emit a self-contained document — the full chain plus the actual weight
  fingerprints, never the secret. `das-verify` re-checks it offline: chain
  continuity + fingerprint-vs-signed-hash need **no secret** (catch
  reorder/insert/delete and any edit to the recorded fingerprints); full
  authenticity uses the HMAC key. *Tested:* `test_audit_export.py`.
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

### F1 · Rollback / freshness — anchor now available (was HIGH; addressed, opt-in)
The audit chain proves *internal* consistency and ordering, **not freshness**. An
attacker with write access to `DAS_STATE` can restore an *older, self-consistent*
snapshot (matching `forest.npz` + `control_plane.json` + `audit.json` from a
previous `save()`). Both `verify()` and `state_matches_audit()` pass, because the
old triple is internally valid — silently undoing a deletion or role-revocation.
**Now shipped (opt-in):** a `FreshnessAnchor` (`das/freshness.py`) records the
chain's latest `(seq, head)` on every `save()`; `load()` refuses any restored chain
that doesn't contain the anchored head at the anchored position (shorter/older/
forked) with `RollbackDetected`. Wire it via `DAS_ANCHOR` on the API (it then
*refuses to start* on a rolled-back snapshot). Forging a longer valid chain needs
the signing key, so a `DAS_STATE`-only attacker can't defeat it. *Tested:*
`test_freshness.py`; demo `examples/freshness_demo.py`.
**Residual (important):** the anchor only helps if it lives on a store the
`DAS_STATE` writer **cannot also roll back** — a separate volume, an append-only/
WORM store, a monotonic counter, or a transparency log. The file backend is the
reference; the API warns if `DAS_ANCHOR` is placed inside `DAS_STATE`. This is
opt-in; the default (no anchor) still has the gap.

### F2 · Identity is asserted, not authenticated (was HIGH; enforced via proxy contract)
`apps/governance_api.py` takes the principal from `X-DAS-Actor`; RBAC is enforced on
it, but the header alone doesn't prove the caller is that principal.
**Now enforced:** set `DAS_TRUSTED_PROXY_SECRET` and the API rejects (401) any
request lacking the matching `X-DAS-Proxy-Auth` header (except the `/health` probe),
so `X-DAS-Actor` is only honoured for requests that came through the authn gateway —
which adds both headers. With `DAS_ENV=production` the API *refuses to start* unless
this is configured. *Tested:* `test_api_hardening.py`.
**Residual:** this proves traffic transited the gateway; the **gateway** is still
responsible for the actual user authentication (mTLS/OIDC) and for setting a
*verified* `X-DAS-Actor`. Don't expose the API directly; keep the proxy secret out
of client reach (and pair with TLS so it isn't sniffable).

### F3 · Default secret is a footgun (was MEDIUM; enforced)
`DAS_AUDIT_SECRET` defaults to `das-dev-key`; with it, anyone who knows the
open-source default can forge a valid HMAC chain.
**Now enforced:** with `DAS_ENV=production` the API refuses to start on the
default/unset secret — unless Ed25519 signing (`DAS_AUDIT_PRIVKEY`) is configured
instead (which doesn't use the HMAC secret). *Tested:* `test_api_hardening.py`.

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

### F7 · Third-party authenticity — Ed25519 signing now available (was MEDIUM; addressed, opt-in)
By default the signatures are HMAC (symmetric), so confirming *authorship* requires
the same `DAS_AUDIT_SECRET` that produced the log — fine for a contracted auditor
given the key, weak for an arms-length regulator.
**Now shipped (opt-in):** sign with an **Ed25519** private key instead —
`DAS_AUDIT_PRIVKEY` on the API, or `private_key=` on `AuditLog`/`ControlPlane`
(`pip install -e ".[crypto]"`). Entries are then verifiable by anyone holding only
the **public key**: `das-verify doc.json --pubkey <hex>`. The private key never
enters the exported document. *Tested:* `test_audit_ed25519.py`; demo
`examples/ed25519_audit_demo.py`; verified end-to-end through `GET /audit/export`.
**Residual:** the export embeds the public key for convenience — **pin the expected
key out-of-band** to prove authorship (don't trust the embedded key alone). And
authenticity is orthogonal to **freshness (F1)**: an Ed25519 signature doesn't prove
the log is the *latest*, so pair it with the F1 anchor. HMAC remains the
zero-dependency default.

## Non-findings (checked, currently fine)

- **Tenant scope bypass:** operators are checked against the *expert's* tenant on
  prune and the *target* tenant on graft/delete; cross-tenant attempts are denied
  and logged (`test_rbac_tenant_scope`).
- **Hash collision for isolation:** isolation uses SHA-256 over raw weight bytes;
  forging a colliding-but-different weight set is not a practical threat.
- **Arbitrary file write on load:** `load()` reads fixed filenames inside the
  given dir; it does not take paths from the (untrusted) state file.

## Recommendations, in priority order

1. ~~Build F1 (freshness anchoring)~~ **— done (opt-in `FreshnessAnchor`).** Make
   it the default for regulated deployments and document anchor-store custody (it
   must be on a store the `DAS_STATE` writer cannot roll back).
2. ~~Enforce F3~~ **— done.** `DAS_ENV=production` refuses the default audit secret.
3. ~~Document/enforce F2~~ **— done.** `DAS_TRUSTED_PROXY_SECRET` enforces the authn-
   proxy contract; production refuses to start without it. (Gateway still does authn.)
4. ~~Adopt asymmetric signing (F7)~~ **— done (opt-in Ed25519).** Make it the
   default for regulated deployments and document key custody.
5. Harden the serving stack (F5) and adopt trusted timestamps (F4).
6. Commission an **independent** security audit before any 1.0 / GA.

## How to reproduce the protections

```bash
python examples/control_plane_demo.py     # RBAC denials, tenant-delete isolation, tamper detection, persistence
python examples/audit_export_demo.py      # export the signed log + verify it offline (das-verify)
python examples/ed25519_audit_demo.py     # public-key-verifiable audit (F7); needs .[crypto]
python examples/freshness_demo.py         # refuse a rolled-back snapshot (F1)
python benchmarks/governance_benchmark.py # audit/RBAC/provenance vs baselines, with numbers
pytest tests/test_governance.py tests/test_governance_api.py tests/test_api_hardening.py tests/test_audit_export.py tests/test_audit_ed25519.py tests/test_freshness.py -q
```
