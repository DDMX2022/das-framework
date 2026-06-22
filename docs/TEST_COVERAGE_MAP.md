# Control → test traceability map

> Every security-relevant control in the DAS governance control plane, mapped to the
> specific test(s) that exercise it. Pairs with [THREAT_MODEL.md](THREAT_MODEL.md)
> (the controls in context) and [SECURITY_REVIEW.md](SECURITY_REVIEW.md) (findings
> F1–F7). Use this to (a) hand an auditor a verifiable claim set and (b) catch
> regressions: if a row's test disappears, a control lost its proof.
>
> **Suite:** 64 tests across 11 files. CI runs the numpy-only core; the rows needing
> `flask`/`cryptography`/`torch` are gated with `importorskip` and run where the
> matching extra is installed (`pip install -e ".[web,crypto,hf]"`).
>
> **Last updated:** 2026-06-22.

## How to reproduce

```bash
pip install -e ".[web,crypto,hf]"     # all extras → no skips
pytest -q                              # full suite
# or the governance/security subset only:
pytest tests/test_governance.py tests/test_governance_api.py \
       tests/test_api_hardening.py tests/test_audit_export.py \
       tests/test_audit_ed25519.py tests/test_freshness.py -q
```

## Findings (F1–F7) → proof

| Finding | Control | Proving test(s) | Extra |
|---|---|---|---|
| **F1** freshness/rollback anchor | anchor records `(seq, head)`; rollback & fork detected; load refuses stale; append-only | `test_freshness::anchor_record_latest_and_check`, `anchor_detects_rollback_and_fork`, `load_refuses_rolled_back_state`, `load_current_state_with_anchor_ok`, `anchor_is_append_only_across_saves` | core (numpy) |
| **F2** authn-proxy contract | 401 without proxy header; honored with it; rejected on wrong; `/health` exempt; off when unset | `test_api_hardening::proxy_auth_enforced_when_configured`, `no_enforcement_when_unset` | flask |
| **F2/F3** production startup guard | dev never fatal; prod refuses default secret + missing proxy; passes with privkey or strong secret + proxy | `test_api_hardening::dev_config_is_never_fatal`, `prod_refuses_default_secret_and_missing_proxy`, `prod_ok_with_privkey_or_strong_secret_plus_proxy`, `prod_strong_secret_but_no_proxy_still_fails_f2` | flask |
| **F4** UTC + pluggable timestamps | default `ts` ends in `Z`; injected `time_fn` token is signed in | `test_audit_export::timestamps_are_utc_and_pluggable` | core |
| **F5** rate limiting | fixed-window per-client allow/deny + reset; 429 enforced; `/health` exempt | `test_api_hardening::rate_limiter_fixed_window`, `rate_limit_enforced` | flask |
| **F6** mounted-file secrets | `<VAR>_FILE` wins over env, stripped; falls back to env then default | `test_api_hardening::resolve_secret_prefers_file` | flask |
| **F7** Ed25519 public-key verification | scheme + embedded pubkey; verify with pubkey only; wrong key rejected; content edit caught; structural checks stay keyless; PEM string & object accepted | `test_audit_ed25519::*` (6 tests) | cryptography |

## Core audit integrity → proof

| Control | Proving test(s) | Extra |
|---|---|---|
| HMAC sign + hash-chain; edit/reorder/insert/delete caught | `test_governance::audit_tamper_detected`, `test_das::audit_detects_tampering`, `test_das::audit_detects_deletion`, `test_audit_export::live_log_verify_catches_payload_tamper` | core |
| Exportable, self-contained document (chain + real fingerprints, no secret) | `test_audit_export::document_is_self_contained`, `export_load_roundtrip` | core |
| Offline keyless structural verify (reorder/delete/fingerprint-edit) | `test_audit_export::verify_keyless_and_authenticated`, `keyless_catches_fingerprint_edit`, `keyless_catches_reorder_and_delete` | core |
| Keyed authenticity catches pure content edits invisible to structural check | `test_audit_export::content_edit_needs_the_key` | core |
| State↔audit binding detects a swapped `forest.npz` | `test_governance::state_matches_audit_detects_weight_swap` | core |
| API `/audit/export` round-trips and verifies | `test_governance_api::audit_verify` | flask |

## Access control & multi-tenancy → proof

| Control | Proving test(s) | Extra |
|---|---|---|
| RBAC role denials (op not allowed for role) | `test_governance::rbac_role_denials`, `test_governance_api::predict_denied_for_auditor`, `prune_denied_for_auditor` | core / flask |
| Tenant-scope enforcement (cross-tenant op denied) | `test_governance::rbac_tenant_scope`, `list_experts_tenant_scoped`, `test_governance_api::experts_tenant_scoped` | core / flask |
| Denials are themselves audited and the chain still holds | `test_governance::denials_are_audited_and_chain_holds` | core |
| API input validation (bad embedding dim → 400) | `test_governance_api::predict_bad_dim` | flask |

## Isolation, deletion & persistence → proof

| Control | Proving test(s) | Extra |
|---|---|---|
| Zero-forgetting on graft (byte-identical survivors) | `test_das::zero_forgetting_on_graft`, `test_governance::graft_non_interference` | core |
| Prune preserves survivors | `test_das::prune_preserves_survivors` | core |
| Tenant deletion leaves other tenants byte-identical | `test_governance::delete_tenant_isolation` | core |
| Save→reload byte-identical (control-plane) | `test_governance::persistence_roundtrip` | core |
| LoRA expert isolation byte-identical on graft | `test_lora::lora_isolation_on_graft` | core |
| LoRA checkpoint byte-exact; frozen backbone untrained | `test_lora::lora_checkpoint_byte_exact`, `lora_backbone_not_trained` | core |
| Hub/persistence round-trip + integrity | `test_das::hub_roundtrip_and_integrity` | core |

## Real-encoder (Phase 1) & orchestration → proof

| Control | Proving test(s) | Extra |
|---|---|---|
| Real frozen MiniLM embeddings route real text; graft/prune byte-identical on real text | `test_hf_text::real_encoder_routing_and_isolation`, `demo_corpus_wellformed`, `embed_domains_rejects_unknown_domain` | hf (torch) |
| Provenance per query under LangGraph; denial surfaces as state | `test_integrations::node_writes_provenance`, `node_surfaces_denial_as_state`, `node_can_raise_on_denied`, `default_actor_used_when_absent`, `build_graph_when_langgraph_present` | core |
| API `/predict` returns provenance; `/health` reports scheme | `test_governance_api::predict_returns_provenance`, `health` | flask |

## Benchmark claims → proof

| Claim | Proving test(s) |
|---|---|
| Monolith forgets + has no governance | `test_governance_benchmark::monolith_forgets_and_has_no_governance` |
| Isolated experts isolate but lack governance | `test_governance_benchmark::isolated_experts_isolate_but_lack_governance` |
| DAS matches isolation **and** adds governance | `test_governance_benchmark::das_matches_isolation_and_adds_governance` |

## Coverage gaps (honest)

These are exercised by demos/benchmarks or by manual review, **not** by an automated
assertion — candidates for the independent audit to formalize:

- **Anchor custody** (F1 residual): no test asserts the anchor store is outside the
  `DAS_STATE` writer's reach — that's a deployment property, not a code property.
- **Gateway authn** (F2 residual): the gateway's mTLS/OIDC is out of scope here; we
  test only that the proxy *contract* is enforced.
- **Trusted wall-clock time** (F4 residual): `time_fn` is tested with an injected
  token, not against a real RFC 3161 TSA.
- **Load/DoS behavior under real concurrency** (F5 residual): the rate limiter is
  unit-tested; sustained-load and `delete_tenant` O(experts) cost are not.
- **Fuzzing**: `/predict` and the load path get type/dim checks, not fuzzed inputs.
