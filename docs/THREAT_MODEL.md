# Threat model — DAS governance control plane

> **Purpose.** Give an external security auditor (and a buyer's security review) a
> ready-made scope: what the system is trying to protect, against whom, where the
> trust boundaries are, which controls exist, and what risk is knowingly left on
> the table. It pairs with [SECURITY_REVIEW.md](SECURITY_REVIEW.md) (findings F1–F7,
> all addressed) and [TEST_COVERAGE_MAP.md](TEST_COVERAGE_MAP.md) (the test that
> proves each control). This is an **authors' threat model**, not an independent
> assessment — commissioning that assessment is the point of writing this.
>
> **Scope of system under analysis:** [`das/governance.py`](../das/governance.py),
> [`das/audit.py`](../das/audit.py), [`das/freshness.py`](../das/freshness.py), the
> on-disk persistence format (`forest.npz` + `control_plane.json` + `audit.json` +
> the freshness anchor), and the REST surface in
> [`apps/governance_api.py`](../apps/governance_api.py).
>
> **Out of scope (explicit):** the ML capability of the experts themselves, the
> authn gateway in front of the API (its mTLS/OIDC is assumed and is the deployer's
> responsibility — see F2), the host OS / container runtime, and the secret-store
> backend (k8s `Secret`, Vault, etc.). These are dependencies we rely on, not
> things this code implements.
>
> **Last updated:** 2026-06-22.

## 1. What we are protecting (assets)

| # | Asset | On disk as | Why it matters | Security property required |
|---|---|---|---|---|
| A1 | Expert weights | `forest.npz` | the customers' models; the isolation + deletion guarantees attach here | confidentiality (tenant scope), integrity (no silent swap) |
| A2 | Audit log | `audit.json` | the compliance record of every privileged action | integrity, non-repudiation, freshness |
| A3 | Governance state | `control_plane.json` | tenants, users, roles, the expert registry | integrity (no silent role/tenant edit) |
| A4 | Signing root of trust | `DAS_AUDIT_SECRET` (HMAC) **or** `DAS_AUDIT_PRIVKEY` (Ed25519) | forging it forges the whole audit chain | confidentiality |
| A5 | Freshness anchor | append-only `(seq, head)` store (`DAS_ANCHOR`) | proves the loaded snapshot is the *latest*, not a rolled-back one | integrity, append-only / monotonicity |
| A6 | Proxy contract secret | `DAS_TRUSTED_PROXY_SECRET` | proves a request actually transited the authn gateway | confidentiality |

## 2. Trust boundaries

```
        ┌─────────────────────── TRUSTED ───────────────────────┐
 client │  authn gateway        DAS API process                 │  disk (DAS_STATE)   anchor store
   ──────────────────────►  (mTLS/OIDC,    ──►  RBAC + audit  ──►│  forest.npz         (separate,
        │  sets X-DAS-Actor   + proxy secret)   + signing       │  control_plane.json  append-only /
        │                                                        │  audit.json          WORM)
        └────────────────────────────────────────────────────────┘
  ▲ network: UNTRUSTED                          ▲ disk: UNTRUSTED       ▲ anchor: must be on a store
  (F2 proxy contract gates it)                  (signing + binding      the DAS_STATE writer cannot
                                                 + anchor defend it)    also roll back (F1 residual)
```

- **Inside the process is trusted.** Everything else — the network, the disk under
  `DAS_STATE`, and any actor who can write files — is **untrusted**.
- **The secret is never written to disk.** It is injected at runtime (env, or a
  mounted file via `<VAR>_FILE` — F6) and held only in process memory. That
  separation is what makes a swapped-but-unsigned `forest.npz` detectable (A1↔A2
  binding).
- **The disk is hostile.** The design assumes an attacker can read and rewrite
  every file under `DAS_STATE`. The defenses are: signatures (can't forge entries
  without A4), state↔audit binding (can't swap weights undetected), and the
  freshness anchor (can't roll back to an older valid snapshot — provided A5 lives
  somewhere the disk attacker can't also rewrite).

## 3. Adversaries

| Adversary | Capability assumed | Primary goal |
|---|---|---|
| **Net-1 — Unauthenticated network caller** | can reach the API port | issue privileged ops without credentials |
| **Net-2 — Authenticated but under-privileged tenant** | a valid actor scoped to tenant X | read/modify/delete another tenant's experts; tamper with the record of their own actions |
| **Disk-1 — Storage-tier attacker** | read+write to all files under `DAS_STATE` (compromised volume, backup, insider) | rewrite weights/state/log undetectably, or **roll back** to undo a deletion or role-revocation |
| **Reg — Arms-length regulator/auditor** | holds only an exported document + a public key | *not an attacker* — a party we must let **independently verify** without trusting us or sharing A4 |

## 4. Attack surface → controls → residual (STRIDE per asset)

| STRIDE | Concrete attack | Control in code | Finding | Residual risk |
|---|---|---|---|---|
| **S**poofing | Net-1 sets `X-DAS-Actor: root` directly | F2 proxy contract: 401 unless `X-DAS-Proxy-Auth` matches; production refuses to boot if unset | F2 | gateway still owns real user authn; don't expose API directly |
| **S**poofing | Net-1 forges audit entries with the public default key | F3: production refuses to start on `das-dev-key`/unset secret | F3 | none if deployed in `production` mode (operator must set it) |
| **T**ampering | Disk-1 edits/reorders/inserts/deletes an audit entry (A2) | HMAC-SHA256 sign + hash-chain; `verify()` re-walks both | core | needs A4 secrecy; a key holder can rewrite history |
| **T**ampering | Disk-1 swaps `forest.npz` (A1) for a different forest | state↔audit binding: `load()` recomputes fingerprints from restored weights vs the signed last entry | core | binding only as strong as A4 + the anchor |
| **T**ampering | Net-2 edits another tenant's experts | RBAC `_check` on the expert's tenant; denial logged then 403 | core | correctness of the role/tenant model itself |
| **R**epudiation | An actor denies having done a privileged op | every op (incl. denials) is appended + signed *before* the exception | core | wall-clock `ts` is advisory (F4) |
| **R**epudiation | Reg can't confirm authorship without trusting us | F7: opt-in Ed25519 → verify with public key only, secret never exported | F7 | pin the expected pubkey out-of-band; embedded key alone isn't proof |
| **I**nfo disclosure | Net-2 lists/reads another tenant's experts | `list_experts` + `/predict` tenant-scoped | core | model the role grants themselves |
| **I**nfo disclosure | A4 leaks via `/proc`, crash dump, child proc (env var) | F6: read secret from a mounted file (`<VAR>_FILE`), preferred over env | F6 | secret still in process memory while running (inherent) |
| **D**enial of service | Net-1 floods the API | F5: gunicorn (not the dev server) + optional in-process per-client rate limit (429), `/health` exempt | F5 | real DoS belongs at proxy/WAF; `delete_tenant` is O(experts) |
| **E**levation | Code execution via a malicious state file | persistence is `np.savez`+JSON, `np.load(allow_pickle=False)`; no pickle | core | parser bugs in numpy/json (third-party) |
| **E**levation (freshness) | Disk-1 restores an *older, self-consistent* snapshot to undo a deletion | F1: `FreshnessAnchor` records `(seq, head)` per save; `load()` refuses any chain not containing the anchored head at its position (`RollbackDetected`) | F1 | **the anchor must live where Disk-1 cannot also roll it back** — separate/WORM/monotonic store. Default (no anchor) still has the gap. |

## 5. Key design decisions an auditor should scrutinize

1. **Symmetric default, asymmetric opt-in.** HMAC is the zero-dependency default
   (good DX, but a key holder can rewrite history); Ed25519 (F7) is opt-in for the
   regulator-verifiable case. *Question to test:* is the org deploying in the mode
   their compliance story claims?
2. **Freshness is opt-in and custody-dependent (F1).** The anchor defeats rollback
   **only** if its store is outside Disk-1's reach. The single most important
   deployment review item.
3. **Identity is delegated (F2).** DAS authenticates the *gateway*, not the *user*.
   The chain is only as trustworthy as the gateway that sets `X-DAS-Actor`.
4. **Timestamps are advisory (F4).** Ordering is proven by the chain + anchor, not
   by `ts`. Trusted wall-clock time needs a TSA plugged into `time_fn`.
5. **`delete_tenant` is O(experts) (F5 residual)** and is a deletion path — worth a
   look for both DoS and for completeness of the right-to-be-forgotten guarantee.

## 6. What an independent audit should cover (scope handed to the auditor)

- [ ] Re-derive the A1↔A2 binding and attempt an undetected weight swap.
- [ ] Attempt the F1 rollback with and without an anchor; test anchor custody
      assumptions (can the `DAS_STATE` writer also rewrite the anchor?).
- [ ] Review the RBAC/tenant model for scope-bypass (graft vs prune vs delete
      check the *correct* tenant — see "Non-findings" in the review).
- [ ] Fuzz the `/predict` and load paths (malformed embeddings, malformed state).
- [ ] Confirm F2/F3 production guards cannot be bypassed by env manipulation.
- [ ] Independent crypto review of the Ed25519 export/verify path (F7).
- [ ] Supply-chain review of the dependency set (numpy, flask, gunicorn,
      cryptography, sentence-transformers/torch when the `hf` extra is used).

Each control above is mapped to the specific test that exercises it in
[TEST_COVERAGE_MAP.md](TEST_COVERAGE_MAP.md).
