"""
das/platform/deploy.py
----------------------
The deployment engine: a validated ``ClientSpec`` in, a live governed
``ControlPlane`` out — in one call. This is ``das deploy`` in library form and
the piece that makes DAS an *FDE deployment engine* rather than a library: the
same guarantees stood up identically at client after client from one config file.

    from das.platform import deploy
    dep = deploy("client.yaml")
    dep.summary()                        # what got stood up
    dep.route("bank-agent", "unknown card charge from StreamBox")
    dep.offboard("fintrust")             # provable right-to-be-forgotten
    dep.export_bundle("northwind.json")  # the leave-behind

Nothing here re-implements governance — it wires tenants, users, and grafts onto
``das.governance.ControlPlane`` in spec order, then exposes query-time routing
with the spec's escalation policy applied.
"""
from __future__ import annotations

from typing import Optional

from das.governance import ControlPlane

from .spec import ClientSpec
from .trainer import SyntheticTrainer
from .connectors import ContextSource, SpecKeywordConnector
from .bundle import write_bundle


class Deployment:
    """A stood-up client fleet: the ControlPlane plus the spec and connector that
    produced it, with FDE-facing operations (route, offboard, verify, bundle)."""

    def __init__(self, spec: ClientSpec, cp: ControlPlane, trainer: SyntheticTrainer,
                 connector: Optional[ContextSource] = None):
        self.spec = spec
        self.cp = cp
        self.trainer = trainer
        self.connector = connector or SpecKeywordConnector(spec, trainer)

    # ── introspection ────────────────────────────────────────────────
    def summary(self) -> dict:
        """Everything the deploy stood up — the POC "day 1" evidence."""
        v = self.cp.verify_audit(self.spec.root)
        return {
            "client": self.spec.client,
            "tenants": sorted(self.cp.tenants),
            "experts": [{"eid": r["eid"], "tenant": r["tenant"], "name": r["name"]}
                        for r in self.cp.experts],
            "users": {name: {"role": u["role"], "tenant": u["tenant"]}
                      for name, u in self.cp.users.items()},
            "escalation": {
                "frontier_llm": self.spec.escalation.frontier_llm,
                "confidence_threshold": self.spec.escalation.confidence_threshold,
            },
            "audit_ok": v["ok"],
            "audit_entries": v["entries"],
        }

    def verify(self) -> dict:
        """Re-walk the signed audit chain."""
        return self.cp.verify_audit(self.spec.root)

    # ── query time: route + escalation ───────────────────────────────
    def route(self, actor: str, query, connector: Optional[ContextSource] = None,
              threshold: Optional[float] = None) -> dict:
        """Embed a query through the connector, route it through the governed
        forest, and apply the escalation policy. Returns provenance plus the
        decision (``local`` vs ``escalate``) — the per-query record that is both
        the attribution story and the cost-deflection signal. ``threshold``
        overrides the spec's escalation threshold for what-if analysis."""
        src = connector or self.connector
        h = src.embed(query)
        rows = self.cp.route_explain(actor, h)
        row = rows[0]
        thr = self.spec.escalation.confidence_threshold if threshold is None else threshold
        escalate = row["confidence"] < thr
        return {
            "tenant": row["tenant"],
            "expert": row["name"],
            "eid": row["eid"],
            "confidence": row["confidence"],
            "decision": "escalate" if escalate else "local",
            "frontier_llm": self.spec.escalation.frontier_llm if escalate else None,
        }

    # ── lifecycle ops ────────────────────────────────────────────────
    def grow(self, actor: str, tenant: str, name: str, keywords=None) -> int:
        """Graft a new specialist live (proves the others stay byte-identical).
        The trainer already keys data by name, so a fresh name just works."""
        if tenant not in self.cp.tenants:
            self.cp.register_tenant(actor, tenant)
        eid = self.cp.graft(actor, tenant, name, self.trainer.train_fn(name, self.cp),
                            seed=self.trainer.seed_for(name))
        if keywords:
            # extend the demo connector's keyword map so the new expert routes
            if isinstance(self.connector, SpecKeywordConnector):
                for kw in keywords:
                    self.connector._kw.setdefault(kw.lower(), name)
        return eid

    def offboard(self, tenant: str, actor: Optional[str] = None) -> dict:
        """Right-to-be-forgotten at tenant granularity. Survivors proven intact."""
        return self.cp.delete_tenant(actor or self.spec.root, tenant)

    # ── delivery ─────────────────────────────────────────────────────
    def export_bundle(self, path: str, actor: Optional[str] = None) -> dict:
        """Write the self-contained, offline-verifiable compliance bundle — the
        artifact the FDE leaves behind for the client's auditor."""
        return write_bundle(self, path, actor=actor or self.spec.root)

    def save(self, path: str, anchor: Optional[str] = None):
        """Persist the fleet + signed log so it survives restarts."""
        self.cp.save(path, anchor=anchor)


def deploy(source, secret: Optional[str] = None,
           connector: Optional[ContextSource] = None) -> Deployment:
    """Stand up a governed fleet from a client spec.

    ``source`` is a path to ``client.yaml`` / ``.json``, a ``dict``, or an
    already-parsed ``ClientSpec``. ``secret`` overrides the spec's audit-secret
    resolution (useful in tests); otherwise the secret is read at runtime from the
    file/env the spec names and never stored.
    """
    if isinstance(source, ClientSpec):
        spec = source
    elif isinstance(source, dict):
        spec = ClientSpec.from_dict(source)
    else:
        spec = ClientSpec.from_file(source)

    if not spec.experts:
        raise ValueError(f"spec for client '{spec.client}' declares no experts")

    audit_secret = secret if secret is not None else spec.resolve_secret()
    private_key = _load_private_key(spec)

    trainer = SyntheticTrainer(spec.d_model, spec.resolved_leaf_dims())

    # Seed the ControlPlane over the first declared expert of the first tenant.
    seed_tenant, seed_expert = spec.experts[0]
    forest = trainer.seed_forest(seed_expert.name)
    cp = ControlPlane(forest, seed_tenant=seed_tenant, seed_name=seed_expert.name,
                      secret=audit_secret, root=spec.root, private_key=private_key)

    # Register remaining tenants (seed tenant registered by the constructor).
    for tname in spec.tenant_names:
        if tname not in cp.tenants:
            cp.register_tenant(spec.root, tname)

    # Add declared users (root admin already exists).
    for u in spec.users:
        if u.name != spec.root:
            cp.add_user(spec.root, u.name, role=u.role, tenant=u.tenant)

    # Graft every remaining expert in declaration order.
    for tenant, e in spec.experts[1:]:
        cp.graft(spec.root, tenant, e.name, trainer.train_fn(e.name, cp),
                 seed=e.seed if e.seed is not None else trainer.seed_for(e.name))

    return Deployment(spec, cp, trainer, connector=connector)


def _load_private_key(spec: ClientSpec):
    if not spec.audit.private_key_file:
        return None
    try:
        from cryptography.hazmat.primitives.serialization import load_pem_private_key
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "audit.private_key_file set but 'cryptography' not installed — "
            "`pip install das-engine[crypto]`"
        ) from exc
    with open(spec.audit.private_key_file, "rb") as fh:
        return load_pem_private_key(fh.read(), password=None)
