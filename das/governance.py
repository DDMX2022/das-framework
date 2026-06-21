"""
das/governance.py
-----------------
A governance CONTROL PLANE over a DAS forest — the layer that turns "isolated,
auditable experts" into an operable, multi-tenant product.

It adds three things on top of the forest + lifecycle + signed audit log:

  1. MULTI-TENANCY — every expert belongs to a tenant. A tenant's experts are
     isolated (the forest already guarantees byte-level isolation); the control
     plane adds the *ownership* model and a one-call "delete this tenant"
     (right-to-be-forgotten) that removes only their experts and proves the rest
     are byte-identical.

  2. RBAC — every privileged operation (graft / prune / delete-tenant /
     manage-users / read-audit) is checked against the actor's role, and an
     operator can be *scoped to a tenant* so they cannot touch another tenant's
     experts. Denied attempts are themselves recorded in the audit log.

  3. AUDIT — every operation (allowed OR denied) is appended to the tamper-evident
     signed log (das.audit.AuditLog), fingerprinting the forest state each step.
     The log is the compliance record: who did what, to whom, in what order.

This is deliberately backend-agnostic: the control plane never trains anything
itself. Graft takes a `train_fn(forest, leaf_index)` callback, so it works
identically over the NumPy DASForest or (wrapped) the torch LoRAForest — the
governance guarantees don't depend on the expert format.

Only depends on numpy (for forest persistence).
"""
import hashlib
import json
import os

import numpy as np

from das.audit import AuditLog
from das.lifecycle import ForestLifecycle
from das.model import DASForest
from das.functional import FibonacciLeaf

# role -> set of permitted actions. Tenant scoping is enforced separately:
# a user with tenant=None is global (e.g. admin); a user bound to a tenant may
# only act on that tenant's experts.
ROLES = {
    "admin":    {"manage", "graft", "prune", "delete_tenant", "predict", "read_audit", "verify_audit"},
    "operator": {"graft", "prune", "predict", "read_audit", "verify_audit"},
    "auditor":  {"read_audit", "verify_audit"},
    "viewer":   {"predict", "read_audit"},
}


class AccessDenied(Exception):
    """Raised when an actor lacks the role/tenant scope for an operation.
    The denial is recorded in the audit log before this is raised."""


class ControlPlane:
    """Governed wrapper around a DAS forest.

    Construct it over a forest that already has exactly one trained expert (the
    seed tenant); grow it with `graft`, shrink it with `prune` / `delete_tenant`.
    Expert records are kept parallel to `forest.leaves` (so index = leaf
    position, which the lifecycle keeps compact on prune), but each record also
    carries a stable monotonic `eid` so audit references survive index shifts.
    """

    def __init__(self, forest, seed_tenant, seed_name, secret="das-dev-key", root="root"):
        self.forest = forest
        self.life = ForestLifecycle(forest)
        self.audit = AuditLog(secret)
        self.users = {root: {"role": "admin", "tenant": None}}
        self.tenants = {seed_tenant}
        # one record per existing leaf, in leaf order
        self.experts = [{"eid": 0, "tenant": seed_tenant, "name": seed_name}]
        self._next_eid = 1
        self.audit.append(
            "init",
            f"control plane created; root admin '{root}'; seed expert '{seed_name}' (tenant '{seed_tenant}')",
            payload=self._hashes(),
        )

    # ── internals ──────────────────────────────────────────────────
    def _hashes(self):
        """Fingerprint of forest state: {eid: weight_hash} for the audit payload."""
        return {f"eid{r['eid']}": self.forest.leaves[i].weight_hash()
                for i, r in enumerate(self.experts)}

    def _find(self, eid):
        for i, r in enumerate(self.experts):
            if r["eid"] == eid:
                return i, r
        raise KeyError(f"no expert with eid={eid}")

    def _deny(self, actor, action, reason):
        self.audit.append("denied", f"{actor} attempted '{action}' — DENIED: {reason}",
                          payload=self._hashes())
        raise AccessDenied(f"{actor}: {reason}")

    def _check(self, actor, action, tenant=None):
        u = self.users.get(actor)
        if u is None:
            self._deny(actor, action, "unknown user")
        if action not in ROLES[u["role"]]:
            self._deny(actor, action, f"role '{u['role']}' lacks '{action}'")
        if tenant is not None and u["tenant"] is not None and u["tenant"] != tenant:
            self._deny(actor, action, f"tenant scope '{u['tenant']}' may not act on '{tenant}'")
        return u

    # ── user / tenant administration ───────────────────────────────
    def add_user(self, actor, name, role, tenant=None):
        self._check(actor, "manage")
        if role not in ROLES:
            raise ValueError(f"unknown role '{role}' (choose from {sorted(ROLES)})")
        self.users[name] = {"role": role, "tenant": tenant}
        self.audit.append("add_user", f"{actor} added user '{name}' (role={role}, tenant={tenant})",
                          payload=self._hashes())

    def register_tenant(self, actor, tenant):
        self._check(actor, "manage")
        self.tenants.add(tenant)
        self.audit.append("register_tenant", f"{actor} registered tenant '{tenant}'",
                          payload=self._hashes())

    # ── expert lifecycle (governed) ────────────────────────────────
    def graft(self, actor, tenant, name, train_fn, seed=None):
        """Add + train a new expert for `tenant`. `train_fn(forest, leaf_index)`
        does the actual (isolated) training and any router update. Proves the
        existing experts stay byte-identical and records it."""
        self._check(actor, "graft", tenant)
        if tenant not in self.tenants:
            self._deny(actor, "graft", f"unknown tenant '{tenant}' (register it first)")
        before = {r["eid"]: self.forest.leaves[i].weight_hash() for i, r in enumerate(self.experts)}
        idx = self.life.graft(seed=seed if seed is not None else (abs(hash(name)) % 1000))
        train_fn(self.forest, idx)
        rec = {"eid": self._next_eid, "tenant": tenant, "name": name}
        self._next_eid += 1
        self.experts.append(rec)
        intact = all(self.forest.leaves[i].weight_hash() == before[r["eid"]]
                     for i, r in enumerate(self.experts) if r["eid"] in before)
        self.audit.append(
            "graft",
            f"{actor} grafted '{name}' for tenant '{tenant}' (eid={rec['eid']}); prior experts unchanged: {intact}",
            payload=self._hashes(),
        )
        return rec["eid"]

    def prune(self, actor, eid):
        """Remove one expert (right-to-be-forgotten at expert granularity).
        Survivors proven byte-identical."""
        i, rec = self._find(eid)
        self._check(actor, "prune", rec["tenant"])
        before = {r["eid"]: self.forest.leaves[j].weight_hash()
                  for j, r in enumerate(self.experts) if r["eid"] != eid}
        self.life.prune(i)
        self.experts.pop(i)
        intact = all(self.forest.leaves[j].weight_hash() == before[r["eid"]]
                     for j, r in enumerate(self.experts))
        self.audit.append(
            "prune",
            f"{actor} pruned eid={eid} ('{rec['name']}', tenant '{rec['tenant']}'); others unchanged: {intact}",
            payload=self._hashes(),
        )
        return intact

    def delete_tenant(self, actor, tenant):
        """Right-to-be-forgotten at TENANT granularity: remove every expert the
        tenant owns and prove all other tenants are byte-identical afterwards."""
        self._check(actor, "delete_tenant", tenant)
        eids = [r["eid"] for r in self.experts if r["tenant"] == tenant]
        if not eids:
            self._deny(actor, "delete_tenant", f"tenant '{tenant}' owns no experts")
        survivors = {r["eid"]: self.forest.leaves[j].weight_hash()
                     for j, r in enumerate(self.experts) if r["tenant"] != tenant}
        for eid in eids:
            i, _ = self._find(eid)
            self.life.prune(i)
            self.experts.pop(i)
        self.tenants.discard(tenant)
        intact = all(self.forest.leaves[j].weight_hash() == survivors[r["eid"]]
                     for j, r in enumerate(self.experts))
        self.audit.append(
            "delete_tenant",
            f"{actor} deleted tenant '{tenant}' ({len(eids)} experts removed); other tenants unchanged: {intact}",
            payload=self._hashes(),
        )
        return {"removed": len(eids), "non_interference": intact}

    # ── read paths ─────────────────────────────────────────────────
    def predict(self, actor, h):
        """Route + predict through the forest. Requires the 'predict' permission."""
        self._check(actor, "predict")
        return self.forest.predict(h)

    def route_explain(self, actor, h):
        """Governed predict that returns routing PROVENANCE for each input: which
        expert (eid / tenant / name) the hard router chose, the router's
        confidence in that choice, and the prediction. Permission-checked
        ('predict') — and a denied attempt is logged like any other — but allowed
        predictions are NOT appended to the audit log (they're high-volume reads;
        the log stays the record of privileged *changes*). This is the hook an
        orchestrator uses to make a routed answer attributable."""
        self._check(actor, "predict")
        h = np.asarray(h, dtype=float)
        if h.ndim == 1:
            h = h[None, :]
        leaf_idx, tau = self.forest.router.route(h)
        out, _ = self.forest.predict(h)
        rows = []
        for n in range(h.shape[0]):
            i = int(leaf_idx[n])
            rec = self.experts[i]
            rows.append({
                "eid": rec["eid"], "tenant": rec["tenant"], "name": rec["name"],
                "confidence": float(tau[n, i]), "prediction": out[n].tolist(),
            })
        return rows

    def list_experts(self, actor):
        """Tenant-scoped view: global users see all experts; a tenant-bound user
        sees only their own."""
        u = self._check(actor, "read_audit")
        scope = u["tenant"]
        return [dict(r) for r in self.experts if scope is None or r["tenant"] == scope]

    def read_audit(self, actor, n=None):
        self._check(actor, "read_audit")
        entries = self.audit.entries if n is None else self.audit.entries[-n:]
        return [dict(e) for e in entries]

    def verify_audit(self, actor):
        self._check(actor, "verify_audit")
        ok, idx, reason = self.audit.verify()
        return {"ok": ok, "broken_index": idx, "reason": reason, "entries": len(self.audit.entries)}

    # ── persistence ────────────────────────────────────────────────
    def save(self, path):
        """Persist the whole control plane to a directory: the forest weights
        (NumPy), the governance state (tenants/users/registry, JSON), and the
        signed audit log (JSON). The forest is saved verbatim so every expert's
        weight_hash is preserved byte-for-byte across a restart."""
        os.makedirs(path, exist_ok=True)
        f = self.forest
        arrays = {"router_W": f.router.W, "router_b": f.router.b}
        leaf_meta = []
        for li, leaf in enumerate(f.leaves):
            leaf_meta.append({"dims": leaf.dims, "frozen": bool(leaf.frozen), "depth": len(leaf.W)})
            for wi, (W, b) in enumerate(zip(leaf.W, leaf.b)):
                arrays[f"leaf{li}_W{wi}"] = W
                arrays[f"leaf{li}_b{wi}"] = b
        np.savez(os.path.join(path, "forest.npz"), **arrays)
        meta = {
            "d_model": f.d_model,
            "leaf_dims": f.leaf_dims,
            "leaves": leaf_meta,
            "tenants": sorted(self.tenants),
            "users": self.users,
            "experts": self.experts,
            "next_eid": self._next_eid,
        }
        with open(os.path.join(path, "control_plane.json"), "w") as fh:
            json.dump(meta, fh, indent=2)
        self.audit.export(os.path.join(path, "audit.json"))

    @classmethod
    def load(cls, path, secret="das-dev-key"):
        """Reconstruct a control plane saved by `save`. The same `secret` is
        required to validate the audit log (it is never written to disk). Does
        NOT append to the log — the restored chain is exactly as saved; verify
        it with `verify_audit(...)` and link it to the weights with
        `state_matches_audit()`."""
        with open(os.path.join(path, "control_plane.json")) as fh:
            meta = json.load(fh)
        npz = np.load(os.path.join(path, "forest.npz"))
        n = len(meta["leaves"])
        forest = DASForest(meta["d_model"], meta["leaf_dims"], num_leaves=max(n, 1))
        forest.leaves = []
        for li, lm in enumerate(meta["leaves"]):
            leaf = FibonacciLeaf(lm["dims"])
            leaf.W = [npz[f"leaf{li}_W{wi}"] for wi in range(lm["depth"])]
            leaf.b = [npz[f"leaf{li}_b{wi}"] for wi in range(lm["depth"])]
            leaf.frozen = bool(lm["frozen"])
            forest.leaves.append(leaf)
        forest.router.W = npz["router_W"]
        forest.router.b = npz["router_b"]
        forest.router.num_leaves = forest.router.W.shape[1]

        cp = cls.__new__(cls)                       # bypass __init__'s seeding
        cp.forest = forest
        cp.life = ForestLifecycle(forest)
        cp.audit = AuditLog.load(os.path.join(path, "audit.json"), secret=secret)
        cp.users = meta["users"]
        cp.tenants = set(meta["tenants"])
        cp.experts = meta["experts"]
        cp._next_eid = meta["next_eid"]
        return cp

    def state_matches_audit(self):
        """The last audit entry fingerprints the forest state at that moment.
        Recompute that fingerprint from the (restored) forest and compare —
        this binds the loaded weights to the signed log, so a swapped-out
        forest.npz is detectable even though it isn't itself signed."""
        if not self.audit.entries:
            return True
        want = self.audit.entries[-1]["payload_hash"]
        got = hashlib.sha256(json.dumps(self._hashes(), sort_keys=True).encode()).hexdigest()
        return got == want
