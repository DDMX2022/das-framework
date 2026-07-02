# Runbook — backup, restore, and key custody (PLATFORM_PLAN §10.4)

What has to survive a disaster, how to save it, how to bring it back, and how
to prove the thing you brought back is the thing you saved. Written for the
containerized governance API (`apps/governance_api.py` behind
`deploy/gateway/` or `deploy/k8s.yaml`); the library paths are named where
they differ.

## 1. What the state actually is

| Artifact | Where | Loss means |
|---|---|---|
| State directory (`DAS_STATE`) — forest weights (`forest.npz`, or `forest/` for the LoRA backend), `control_plane.json` (tenants/users/experts), `audit.json` (the signed chain), `lessons/`, `keywords.json` | state volume | the fleet: every expert's weights and the audit history |
| Freshness anchor (`DAS_ANCHOR`) — append-only `(seq, head)` records | its **own** volume | rollback DETECTION: a restored-but-stale snapshot would load silently |
| Audit secret (HMAC) | secret store (file-mounted, `DAS_AUDIT_SECRET_FILE`) | the chain can no longer be **verified** (entries remain readable) |
| Ed25519 private key (`DAS_AUDIT_PRIVKEY`, if the deployment signs asymmetrically) | secret store | no NEW entries can be signed; old ones still verify with the public key |
| Proxy secret (`DAS_TRUSTED_PROXY_SECRET_FILE`) | secret store, shared with the gateway | re-issue freely — it authenticates the gateway, not the data |
| `client.yaml` | config (ConfigMap / repo) | nothing at restore time — the STATE is authoritative once deployed; the spec is only the first-boot front door |

Three separations matter: **state ≠ anchor** (whoever can roll back the state
must not be able to roll back the evidence), **state ≠ secrets** (a stolen
state backup without the secret can't forge a verifying chain), and **spec ≠
state** (`das deploy` is idempotent: with state present the spec is ignored).

## 2. Backup

1. Quiesce writes if you can (a grow/germinate in flight is the only risk —
   reads don't touch state), then persist the live control plane:
   ```sh
   curl -s -X POST https://<host>/save \
     -H "X-DAS-Actor: <admin>"        # via the gateway, which adds proxy auth
   ```
   Every `/save` also appends to the anchor when `DAS_ANCHOR` is set.
2. Snapshot the **state volume** (filesystem copy / volume snapshot of
   `DAS_STATE`). The directory is self-contained and byte-exact — weight
   hashes are preserved through save/load (tested).
3. Snapshot the **anchor** to a different destination on a different
   credential. It is small (one line per save) and append-only — prefer
   object-lock / WORM storage.
4. Secrets are NOT part of state backups. They live (and are rotated) in your
   secret manager; back them up there. Never store the audit secret alongside
   a state snapshot — that pairing is exactly what lets a thief rewrite
   history convincingly.

Frequency: after every accepted growth event at minimum (`/save` is cheap);
volume snapshots on your normal schedule.

## 3. Restore

1. Provision the container as usual (`DAS_ENV=production`, secrets mounted,
   `DAS_ANCHOR` pointing at the **current** anchor — not a copy restored from
   the same backup as the state).
2. Restore the state snapshot into `DAS_STATE`. Do NOT set `DAS_SPEC` to
   anything you don't want deployed if the state directory turns out empty —
   an empty state plus a spec means a fresh fleet, not a restore.
3. Boot. Loading verifies the chain signature and, with the anchor, its
   freshness — a snapshot older than the last anchored save **refuses to
   load** with `RollbackDetected`. That refusal is the feature working, not a
   fault: someone restored a stale (or tampered-by-rollback) state. Find the
   newer snapshot, or make an explicit, logged decision to accept data loss
   by re-anchoring.
4. Prove the restore:
   ```sh
   curl -s https://<host>/health
   ```
   must report `"audit_chain_ok": true` **and** `"state_matches_audit": true`
   — the second binds the restored weights to the signed log (a swapped
   `forest.npz` is detectable even though weights aren't themselves signed).
   Library equivalent: `cp.verify_audit(actor)` + `cp.state_matches_audit()`.
5. The LoRA backend additionally re-downloads nothing at restore: the
   backbone is pinned by model name in the forest manifest and fetched from
   the HF cache/hub, while every adapter loads byte-exact from the snapshot.

## 4. Key custody

* **Audit secret (HMAC):** rotation = start a new chain era; document the
  cutover entry. Anyone holding secret + state can forge a chain, so scope
  read access to the verifier role only.
* **Ed25519 (`DAS_AUDIT_PRIVKEY`):** private key stays in the secret store;
  the PUBLIC key is what auditors get (offline bundle verification works
  without any shared secret — that is the point of using it). Custody note
  for regulated deployments: keep the private key in an HSM/KMS-backed store
  if available; the code reads a PEM path, so a mounted agent path works.
* **Anchor storage credential:** must not be writable by the state writer's
  identity — separate IAM principal, ideally WORM.

## 5. Vendor console — network policy (PLATFORM_PLAN §10.3)

`apps/vendor_console.py` holds the **license-signing Ed25519 private key** —
compromise means an attacker can mint valid licenses for any entitlement.
Policy, in one breath: **it never leaves the vendor's own network.**

* Run it on an internal-only host or VPN segment; never on a customer site,
  never on the same host as a customer-facing deployment, never behind a
  public DNS name. Customers only ever receive the license FILE and pin the
  public key — nothing about their deployment talks to this console.
* `DAS_VENDOR_TOKEN` must be set outside local dev (the console gates every
  `/api` call on it and boots loudly when it's absent); the token is defense
  in depth, not the boundary — the network is the boundary.
* The signing key directory (`DAS_VENDOR_DIR`) follows the same custody rules
  as §4: secret store or encrypted volume, backed up separately from any
  state, access scoped to the licensing operator role.
* Revocation and re-issue happen here and ship to customers as files — so
  losing this host loses convenience, not correctness, PROVIDED the key
  material was backed up per §4.

## 6. What this runbook does not cover

Multi-writer HA (the control plane is deliberately a single writer — one
audit chain, one owner), cross-region replication (snapshot shipping works;
anchor consistency rules above still apply), and IdP/gateway recovery
(standard infra, see `deploy/gateway/README.md`).
