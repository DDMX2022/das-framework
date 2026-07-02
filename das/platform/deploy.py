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

import json
import os
from typing import Optional

from das.governance import ControlPlane
from das.training.growth import GrowthManager, GrowthPolicy
from das.training.teachers import teacher_from_config

from .spec import ClientSpec
from .trainer import SyntheticTrainer
from .teacher_trainer import TeacherTrainer
from .germination import Germinator, GerminationPolicy, stage_dims
from .connectors import ContextSource, SpecKeywordConnector
from .bundle import write_bundle
from .license import License, load_license


class Deployment:
    """A stood-up client fleet: the ControlPlane plus the spec and connector that
    produced it, with FDE-facing operations (route, offboard, verify, bundle)."""

    def __init__(self, spec: ClientSpec, cp: ControlPlane, trainer: SyntheticTrainer,
                 connector: Optional[ContextSource] = None,
                 license: Optional[License] = None):
        self.spec = spec
        self.cp = cp
        self.trainer = trainer
        # Held so entitlements are enforced across the deployment's LIFETIME
        # (grow re-checks limits + expiry), not only at deploy time.
        self.license = license
        # The teacher bridge: grow/improve via das.training teachers behind the
        # same train_fn seam. The connector consults IT for centers so that
        # teacher-grown experts (whose geometry is their actual lessons) route
        # coherently alongside synthetic ones.
        self.teacher_trainer = TeacherTrainer(trainer)
        self.teachers = {"local-teacher": self.teacher_trainer.default_teacher}
        self.growth = GrowthManager(cp)
        self.germinator = Germinator(cp, self.teacher_trainer)
        self.connector = connector or SpecKeywordConnector(spec, self.teacher_trainer)

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
    def resolve_teacher(self, teacher):
        """A teacher object, a registered teacher id, or None (synthetic)."""
        if teacher is None or not isinstance(teacher, str):
            return teacher
        if teacher in ("synthetic", ""):
            return None
        if teacher not in self.teachers:
            raise KeyError(f"unknown teacher '{teacher}' "
                           f"(registered: {sorted(self.teachers)})")
        return self.teachers[teacher]

    def register_teacher(self, config: dict) -> dict:
        """Register a runtime teacher (local-vector, or an LLM over Ollama /
        OpenAI-compatible / custom-JSON endpoints). Returns sanitized metadata —
        API keys stay in memory, never in the description or on disk."""
        t = teacher_from_config(config, self.trainer.d_model)
        self.teachers[t.name] = t
        return t.describe()

    def grow(self, actor: str, tenant: str, name: str, keywords=None,
             teacher=None, stage: Optional[str] = None) -> int:
        """Graft a new specialist live (proves the others stay byte-identical).
        With ``teacher`` (an object or registered id), the new expert learns
        from teacher-generated lessons — the Growing-Child grow path; otherwise
        the deterministic synthetic trainer is used. Either way the router is
        retrained over every expert's real training distribution.

        ``stage`` starts the expert at a germination stage ('seed', 'sprout',
        'sapling', 'young-tree', 'tree') instead of the fleet default — pair
        with ``germinate`` to grow capacity only when learning demands it."""
        if self.license is not None:
            # entitlements hold for the deployment's lifetime, not just at boot
            self.license.check_expert_count(len(self.cp.experts) + 1)
        if tenant not in self.cp.tenants:
            self.cp.register_tenant(actor, tenant)
        t = self.resolve_teacher(teacher)
        train_fn = (self.teacher_trainer.train_fn(name, self.cp, teacher=t)
                    if t is not None else self.trainer.train_fn(name, self.cp))
        dims = None
        if stage is not None:
            out_dim = self.trainer.leaf_dims[-1]
            dims = stage_dims(self.trainer.d_model, out_dim, stage)
        eid = self.cp.graft(actor, tenant, name, train_fn,
                            seed=self.trainer.seed_for(name), leaf_dims=dims)
        if keywords and isinstance(self.connector, SpecKeywordConnector):
            # extend the demo connector's keyword map so the new expert routes
            self.connector.add_keywords(name, keywords)
        return eid

    def improve(self, actor: str, eid: int, teacher=None,
                policy: Optional[GrowthPolicy] = None,
                steps: int = 140, topic: Optional[str] = None) -> dict:
        """The full Growing-Child loop on an EXISTING expert: the teacher
        generates lessons, a CANDIDATE (clone) trains in quarantine, and it
        replaces the live expert only if it passes policy — accuracy floor, no
        regression on any expert's frozen eval set, non-target hashes unchanged.
        Accepted or rejected, the attempt is signed into the audit log."""
        t = self.resolve_teacher(teacher) or self.teacher_trainer.default_teacher
        result = self.growth.improve_expert(
            actor, eid, t, topic=topic,
            eval_sets=self.teacher_trainer.eval_sets(self.cp),
            policy=policy, steps=steps,
        )
        return result.to_dict()

    def germinate(self, actor: str, eid: int, teacher=None,
                  target_acc: float = 0.85,
                  policy: Optional[GerminationPolicy] = None,
                  search: bool = True) -> dict:
        """The Fibonacci germination step: if the expert meets ``target_acc`` at
        its current stage it stays small (parsimony); otherwise candidates at
        higher stages train on fresh teacher lessons and the SMALLEST stage
        that earns its parameters replaces the live expert — audited either
        way. ``search=False`` restricts to the single next stage."""
        t = self.resolve_teacher(teacher) or self.teacher_trainer.default_teacher
        return self.germinator.auto_germinate(actor, eid, t,
                                              target_acc=target_acc,
                                              policy=policy, search=search)

    def germinate_all(self, actor: str, teacher=None, target_acc: float = 0.85,
                      policy: Optional[GerminationPolicy] = None) -> dict:
        """Fleet-wide plateau sweep: saturated experts stay small, stuck ones
        attempt searched promotion; one `germination_sweep` audit summary."""
        t = self.resolve_teacher(teacher) or self.teacher_trainer.default_teacher
        return self.germinator.sweep(actor, t, target_acc=target_acc, policy=policy)

    def growth_report(self) -> list:
        """Per-expert germination metrics: stage, dims, parameter count."""
        return self.germinator.report()

    def offboard(self, tenant: str, actor: Optional[str] = None) -> dict:
        """Right-to-be-forgotten at tenant granularity. Survivors proven intact."""
        return self.cp.delete_tenant(actor or self.spec.root, tenant)

    # ── delivery ─────────────────────────────────────────────────────
    def export_bundle(self, path: str, actor: Optional[str] = None) -> dict:
        """Write the self-contained, offline-verifiable compliance bundle — the
        artifact the FDE leaves behind for the client's auditor."""
        return write_bundle(self, path, actor=actor or self.spec.root)

    def save(self, path: str, anchor: Optional[str] = None):
        """Persist the fleet + signed log + the teacher lesson store, so a
        restart keeps both the weights AND the routing geometry of
        teacher-grown experts. Grow-time keywords are saved too — they extend
        the connector beyond what the spec declares."""
        self.cp.save(path, anchor=anchor)
        self.teacher_trainer.save_lessons(os.path.join(path, "lessons"))
        if isinstance(self.connector, SpecKeywordConnector):
            with open(os.path.join(path, "keywords.json"), "w", encoding="utf-8") as fh:
                json.dump(self.connector.keyword_map(), fh, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: str, source, secret: Optional[str] = None,
             connector: Optional[ContextSource] = None,
             license: Optional[License] = None) -> "Deployment":
        """Reconstruct a saved deployment: the ControlPlane (weights + signed
        log, integrity-checked), the trainers, and the persisted lesson store.
        ``source`` is the same client spec that produced it (path/dict/spec) —
        the spec is configuration, the state directory is the data. The license
        resolves per the usual trust model so entitlements keep holding after a
        restart."""
        if isinstance(source, ClientSpec):
            spec = source
        elif isinstance(source, dict):
            spec = ClientSpec.from_dict(source)
        else:
            spec = ClientSpec.from_file(source)
        audit_secret = secret if secret is not None else spec.resolve_secret()
        cp = ControlPlane.load(path, secret=audit_secret,
                               private_key=_load_private_key(spec))
        trainer = SyntheticTrainer(spec.d_model, spec.resolved_leaf_dims())
        lic = license if license is not None else load_license()
        dep = cls(spec, cp, trainer, connector=connector, license=lic)
        dep.teacher_trainer.load_lessons(os.path.join(path, "lessons"))
        kw_path = os.path.join(path, "keywords.json")
        if os.path.exists(kw_path) and isinstance(dep.connector, SpecKeywordConnector):
            with open(kw_path, "r", encoding="utf-8") as fh:
                for kw, name in json.load(fh).items():
                    dep.connector.add_keywords(name, [kw])
        return dep


def deploy(source, secret: Optional[str] = None,
           connector: Optional[ContextSource] = None,
           license: Optional[License] = None) -> Deployment:
    """Stand up a governed fleet from a client spec.

    ``source`` is a path to ``client.yaml`` / ``.json``, a ``dict``, or an
    already-parsed ``ClientSpec``. ``secret`` overrides the spec's audit-secret
    resolution (useful in tests); otherwise the secret is read at runtime from the
    file/env the spec names and never stored.

    Licensing: pass a verified ``License`` to enforce its entitlements, or leave
    it ``None`` to resolve per the trust model (``DAS_LICENSE`` env; evaluation
    mode when nothing is configured; fail closed on an invalid/expired license
    or when ``DAS_LICENSE_REQUIRED=1``). See ``das.platform.license``.
    """
    if isinstance(source, ClientSpec):
        spec = source
    elif isinstance(source, dict):
        spec = ClientSpec.from_dict(source)
    else:
        spec = ClientSpec.from_file(source)

    if not spec.experts:
        raise ValueError(f"spec for client '{spec.client}' declares no experts")

    lic = license if license is not None else load_license()
    if lic is not None:
        lic.check_spec(spec)          # raises LicenseError on exceeded entitlements

    audit_secret = secret if secret is not None else spec.resolve_secret()
    private_key = _load_private_key(spec)

    trainer = SyntheticTrainer(spec.d_model, spec.resolved_leaf_dims())

    # Seed the ControlPlane over the first declared expert of the first tenant
    # (its spec-level seed override is honoured, same as grafted experts).
    seed_tenant, seed_expert = spec.experts[0]
    forest = trainer.seed_forest(seed_expert.name, seed=seed_expert.seed)
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

    return Deployment(spec, cp, trainer, connector=connector, license=lic)


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
