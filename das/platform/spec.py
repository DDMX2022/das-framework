"""
das/platform/spec.py
--------------------
The declarative deployment spec — the single artifact an FDE fills in per client.

A ``client.yaml`` (or ``.json``) describes *what* fleet to stand up; the platform
figures out *how*. Everything downstream (deploy, connectors, bundle) is driven
from a validated ``ClientSpec``, so the spec is the contract.

Example (YAML)::

    client: northwind
    d_model: 18
    audit:
      secret_env: DAS_AUDIT_SECRET
    escalation:
      frontier_llm: claude-sonnet-5
      confidence_threshold: 0.7
    tenants:
      - name: careplus
        experts:
          - name: medical-claim
            keywords: [claim, clinic, mri, denied, appeal]
          - name: prior-auth
            keywords: [authorization, referral, approval]
      - name: fintrust
        experts:
          - name: card-dispute
            keywords: [charge, card, merchant, dispute, fraud]
    users:
      - name: care-agent
        role: operator
        tenant: careplus
      - name: auditor-jane
        role: auditor

Validation is strict and returns precise errors (no silent defaults for the
things that matter: tenant/expert names, roles). YAML support is optional
(``pip install das-engine[platform]``); JSON always works.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Roles must match das.governance.ROLES. Kept here as a literal so spec validation
# does not import the (numpy-dependent) governance module just to check a string.
VALID_ROLES = {"admin", "operator", "auditor", "viewer"}


class SpecError(ValueError):
    """Raised when a client spec is malformed. Message names the offending path."""


@dataclass(frozen=True)
class ExpertSpec:
    name: str
    keywords: List[str] = field(default_factory=list)
    # Optional deterministic seed override; otherwise derived from the name.
    seed: Optional[int] = None


@dataclass(frozen=True)
class TenantSpec:
    name: str
    experts: List[ExpertSpec]


@dataclass(frozen=True)
class UserSpec:
    name: str
    role: str
    tenant: Optional[str] = None


@dataclass(frozen=True)
class EscalationSpec:
    """When a routed query's confidence falls below ``confidence_threshold``, the
    orchestrator should escalate to ``frontier_llm`` (or a human) instead of
    trusting the local specialist. This is the cost/safety dial."""
    frontier_llm: Optional[str] = None
    confidence_threshold: float = 0.0  # 0.0 = never escalate on confidence


@dataclass(frozen=True)
class AuditSpec:
    # The audit secret is supplied at runtime and never stored in the spec. The
    # spec only names *where* to read it from, so client.yaml is safe to commit.
    secret_env: str = "DAS_AUDIT_SECRET"
    secret_file: Optional[str] = None
    # Path to an Ed25519 private-key PEM for public-key-verifiable audit (F7).
    private_key_file: Optional[str] = None


@dataclass(frozen=True)
class ClientSpec:
    client: str
    tenants: List[TenantSpec]
    users: List[UserSpec] = field(default_factory=list)
    escalation: EscalationSpec = field(default_factory=EscalationSpec)
    audit: AuditSpec = field(default_factory=AuditSpec)
    d_model: int = 18
    leaf_dims: Optional[List[int]] = None  # default derived from d_model
    root: str = "root"

    # ── convenience views ────────────────────────────────────────────
    @property
    def tenant_names(self) -> List[str]:
        return [t.name for t in self.tenants]

    @property
    def experts(self) -> List[tuple]:
        """Flat (tenant, ExpertSpec) list in declaration order — deploy order."""
        return [(t.name, e) for t in self.tenants for e in t.experts]

    def resolved_leaf_dims(self) -> List[int]:
        """Leaf shape for every expert. Fibonacci-ish descent to a binary head —
        the width schedule is cosmetic (see leaf_shapes_bench), the head size is
        what matters. Defaults derived from d_model, overridable in the spec."""
        if self.leaf_dims:
            return list(self.leaf_dims)
        return [self.d_model, 13, 8, 2]

    def resolve_secret(self) -> str:
        """Read the audit secret at runtime from file or env. Never stored."""
        if self.audit.secret_file:
            with open(self.audit.secret_file, "r", encoding="utf-8") as fh:
                return fh.read().strip()
        val = os.environ.get(self.audit.secret_env)
        if not val:
            raise SpecError(
                f"audit secret not found: set ${self.audit.secret_env} "
                f"or audit.secret_file in the spec"
            )
        return val

    # ── loading ──────────────────────────────────────────────────────
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClientSpec":
        return _parse(data)

    @classmethod
    def from_file(cls, path: str) -> "ClientSpec":
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
        if path.endswith((".yaml", ".yml")):
            data = _load_yaml(text, path)
        else:
            data = json.loads(text)
        if not isinstance(data, dict):
            raise SpecError(f"{path}: top level must be a mapping, got {type(data).__name__}")
        return _parse(data)


# ── internals ────────────────────────────────────────────────────────
def _load_yaml(text: str, path: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only without pyyaml
        raise SpecError(
            f"{path}: reading YAML needs PyYAML — `pip install das-engine[platform]`, "
            f"or convert the spec to JSON"
        ) from exc
    return yaml.safe_load(text)


def _require(data: Dict[str, Any], key: str, where: str):
    if key not in data:
        raise SpecError(f"{where}: missing required key '{key}'")
    return data[key]


def _parse(data: Dict[str, Any]) -> ClientSpec:
    client = _require(data, "client", "spec")
    if not isinstance(client, str) or not client.strip():
        raise SpecError("spec.client must be a non-empty string")

    raw_tenants = _require(data, "tenants", "spec")
    if not isinstance(raw_tenants, list) or not raw_tenants:
        raise SpecError("spec.tenants must be a non-empty list")

    tenants: List[TenantSpec] = []
    seen_tenants = set()
    seen_experts = set()
    for ti, rt in enumerate(raw_tenants):
        where = f"tenants[{ti}]"
        if not isinstance(rt, dict):
            raise SpecError(f"{where}: must be a mapping")
        tname = _require(rt, "name", where)
        if tname in seen_tenants:
            raise SpecError(f"{where}: duplicate tenant name '{tname}'")
        seen_tenants.add(tname)
        raw_experts = _require(rt, "experts", where)
        if not isinstance(raw_experts, list) or not raw_experts:
            raise SpecError(f"{where}.experts must be a non-empty list")
        experts: List[ExpertSpec] = []
        for ei, re_ in enumerate(raw_experts):
            ewhere = f"{where}.experts[{ei}]"
            if not isinstance(re_, dict):
                raise SpecError(f"{ewhere}: must be a mapping")
            ename = _require(re_, "name", ewhere)
            if ename in seen_experts:
                raise SpecError(f"{ewhere}: duplicate expert name '{ename}' (must be globally unique)")
            seen_experts.add(ename)
            kws = re_.get("keywords", [])
            if not isinstance(kws, list):
                raise SpecError(f"{ewhere}.keywords must be a list")
            experts.append(ExpertSpec(name=ename, keywords=[str(k) for k in kws], seed=re_.get("seed")))
        tenants.append(TenantSpec(name=tname, experts=experts))

    users: List[UserSpec] = []
    for ui, ru in enumerate(data.get("users", []) or []):
        uwhere = f"users[{ui}]"
        if not isinstance(ru, dict):
            raise SpecError(f"{uwhere}: must be a mapping")
        uname = _require(ru, "name", uwhere)
        role = _require(ru, "role", uwhere)
        if role not in VALID_ROLES:
            raise SpecError(f"{uwhere}.role '{role}' invalid (choose from {sorted(VALID_ROLES)})")
        utenant = ru.get("tenant")
        if utenant is not None and utenant not in seen_tenants:
            raise SpecError(f"{uwhere}.tenant '{utenant}' is not a declared tenant")
        users.append(UserSpec(name=uname, role=role, tenant=utenant))

    esc = data.get("escalation", {}) or {}
    if not isinstance(esc, dict):
        raise SpecError("spec.escalation must be a mapping")
    threshold = esc.get("confidence_threshold", 0.0)
    try:
        threshold = float(threshold)
    except (TypeError, ValueError):
        raise SpecError("spec.escalation.confidence_threshold must be a number")
    if not 0.0 <= threshold <= 1.0:
        raise SpecError("spec.escalation.confidence_threshold must be in [0, 1]")
    escalation = EscalationSpec(frontier_llm=esc.get("frontier_llm"), confidence_threshold=threshold)

    aud = data.get("audit", {}) or {}
    if not isinstance(aud, dict):
        raise SpecError("spec.audit must be a mapping")
    audit = AuditSpec(
        secret_env=aud.get("secret_env", "DAS_AUDIT_SECRET"),
        secret_file=aud.get("secret_file"),
        private_key_file=aud.get("private_key_file"),
    )

    d_model = data.get("d_model", 18)
    if not isinstance(d_model, int) or d_model < 2:
        raise SpecError("spec.d_model must be an integer >= 2")
    leaf_dims = data.get("leaf_dims")
    if leaf_dims is not None:
        if not isinstance(leaf_dims, list) or len(leaf_dims) < 2:
            raise SpecError("spec.leaf_dims must be a list of at least 2 ints")
        if leaf_dims[0] != d_model:
            raise SpecError(f"spec.leaf_dims[0] ({leaf_dims[0]}) must equal d_model ({d_model})")

    return ClientSpec(
        client=client,
        tenants=tenants,
        users=users,
        escalation=escalation,
        audit=audit,
        d_model=d_model,
        leaf_dims=leaf_dims,
        root=data.get("root", "root"),
    )
