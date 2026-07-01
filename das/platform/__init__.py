"""
das.platform — the FDE deployment engine.

A thin, repeatable orchestration layer over the proven governance Control Plane
(``das.governance``). It turns a single declarative ``client.yaml`` into a live,
governed, multi-tenant expert fleet — so a Forward Deployed Engineer stands up
the same guarantees at client after client without re-assembling them by hand.

The public surface:

    from das.platform import ClientSpec, Deployment, deploy

    dep = deploy("client.yaml")          # spec -> live ControlPlane
    print(dep.summary())                 # tenants, experts, users, audit_ok
    dep.route("bank-agent", "my card was double charged")   # route + escalate?
    dep.export_bundle("northwind_audit.json")               # the leave-behind

Design notes:
  * Correctness rests on ``das.governance.ControlPlane`` — the platform never
    re-implements isolation, RBAC, or the audit chain; it only wires them.
  * The default expert trainer is deterministic and dependency-free so the whole
    engine runs (and is tested) with only NumPy. Production swaps in a
    teacher-backed trainer via the same ``train_fn`` seam.
  * ``ContextSource`` is the one place a client's real data integration plugs in.
"""
from .spec import ClientSpec, ExpertSpec, TenantSpec, UserSpec, SpecError
from .connectors import (
    ContextSource,
    StaticContextSource,
    CallableContextSource,
    RestContextSource,
    SpecKeywordConnector,
)
from .trainer import SyntheticTrainer
from .deploy import Deployment, deploy
from .bundle import write_bundle

__all__ = [
    "ClientSpec",
    "ExpertSpec",
    "TenantSpec",
    "UserSpec",
    "SpecError",
    "ContextSource",
    "StaticContextSource",
    "CallableContextSource",
    "RestContextSource",
    "SpecKeywordConnector",
    "SyntheticTrainer",
    "Deployment",
    "deploy",
    "write_bundle",
]
