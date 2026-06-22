"""
das/freshness.py — rollback / freshness anchor (SECURITY_REVIEW F1)
-------------------------------------------------------------------
The signed audit chain proves *internal* consistency and ordering, but NOT
freshness. An attacker with write access to `DAS_STATE` can restore an older,
self-consistent snapshot (`forest.npz` + `control_plane.json` + `audit.json` from a
previous `save()`): `verify()` and `state_matches_audit()` both pass, because the
old triple is valid on its own — silently undoing a deletion or role-revocation.

A `FreshnessAnchor` closes this. On every `save()` the control plane records the
chain's latest `(seq, head)` to an append-only anchor. On `load()` the restored
chain must contain the anchored `head` at the anchored position `seq` and be at
least that long; otherwise it is an old or forked snapshot and the load is refused
(`RollbackDetected`). Forging a *longer* valid chain needs the signing key, so an
attacker who only controls `DAS_STATE` cannot defeat this.

TRUST REQUIREMENT (read this): the anchor only helps if it lives somewhere the
`DAS_STATE` writer cannot also roll back — a SEPARATE volume, an append-only/WORM
store, a monotonic counter, or a transparency log. The file backend here is the
reference implementation; point `path` at a more-trusted store in production, or
subclass and override `record`/`latest`. Pairs with Ed25519 signing (F7): a
signature proves authorship, the anchor proves recency.

Stdlib only (json + os + time).
"""
import json
import os
import time


class RollbackDetected(Exception):
    """Raised when a loaded audit chain is older/shorter/forked vs. the anchor."""


class FreshnessAnchor:
    """Append-only record of the audit chain's latest (seq, head).

    `seq` is the index of the last audit entry at record time; `head` is that
    entry's signature. `check()` enforces that a candidate chain extends the
    anchored head."""

    def __init__(self, path):
        self.path = path

    def record(self, seq, head):
        """Append the current chain tip. Append-only by construction — never
        rewrites earlier lines."""
        entry = {"seq": int(seq), "head": head,
                 "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        parent = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(parent, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def latest(self):
        """The most recent (seq, head), or None if no anchor exists yet."""
        if not os.path.exists(self.path):
            return None
        last = None
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    last = json.loads(line)
        return None if last is None else (last["seq"], last["head"])

    def check(self, entries):
        """Returns (ok, reason). ok=True if there is no anchor yet (first run) or
        the chain contains the anchored head at the anchored position and is at
        least that long; ok=False on rollback (shorter/older) or fork (divergent
        history)."""
        a = self.latest()
        if a is None:
            return True, "no anchor yet (first run)"
        seq, head = a
        if seq >= len(entries):
            return False, (f"rollback: anchor expects entry {seq} but the loaded "
                           f"chain has only {len(entries)} entries (older snapshot)")
        if entries[seq].get("sig") != head:
            return False, (f"fork: entry {seq} does not match the anchored head "
                           f"(divergent or substituted history)")
        return True, "fresh (chain extends the anchored head)"

    def enforce(self, entries):
        """check() but raise RollbackDetected on failure."""
        ok, reason = self.check(entries)
        if not ok:
            raise RollbackDetected(reason)
        return reason
