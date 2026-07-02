"""Audited growing-child loop for DAS experts."""

from dataclasses import asdict, dataclass, field
import time

from .evaluator import (
    clone_leaf,
    evaluate_candidate_suite,
    evaluate_leaf,
    evaluate_suite,
    router_accuracy,
    train_leaf,
)


@dataclass
class GrowthPolicy:
    """Acceptance thresholds for candidate expert updates."""

    min_accuracy: float = 0.55
    min_delta: float = 0.0
    max_previous_regression: float = 0.02
    require_other_hashes_unchanged: bool = True


@dataclass
class GrowthResult:
    """Serializable record of one teacher-driven growth attempt."""

    accepted: bool
    reason: str
    actor: str
    eid: int
    tenant: str
    expert: str
    teacher: str
    topic: str
    dataset_version: str
    target_accuracy_before: float
    target_accuracy_after: float
    target_delta: float
    previous_accuracy_before: dict = field(default_factory=dict)
    previous_accuracy_after: dict = field(default_factory=dict)
    max_previous_regression: float = 0.0
    router_accuracy: float = None
    old_experts_unchanged: bool = True
    candidate_hash: str = ""
    live_hash_after: str = ""
    audit_event: str = ""
    ts: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class GrowthCycle:
    """Serializable record of one automated sweep across experts."""

    actor: str
    strategy: str
    attempted: int
    accepted: int
    rejected: int
    teacher_names: list = field(default_factory=list)
    results: list = field(default_factory=list)
    audit_event: str = "growth_cycle"
    ts: str = ""

    def to_dict(self):
        return asdict(self)


class GrowthManager:
    """Train candidate experts, evaluate them, and audit accept/reject decisions."""

    def __init__(self, control_plane, history_limit=100):
        self.cp = control_plane
        self.history_limit = int(history_limit)
        self.history = []
        self.cycle_history = []

    def improve_expert(self, actor, eid, teacher, topic=None, eval_sets=None,
                       policy=None, steps=120, lr=0.05, batch=32, seed=0,
                       n_train=180, n_eval=120):
        policy = policy or GrowthPolicy()
        idx, rec = self.cp._find(int(eid))
        self.cp._check(actor, "graft", rec["tenant"])
        topic = topic or rec["name"]

        before_hashes = self.cp._hashes()
        target_key = f"eid{rec['eid']}"
        live_leaf = self.cp.forest.leaves[idx]
        lessons = teacher.generate(topic, n_train=n_train, n_eval=n_eval)

        target_before = evaluate_leaf(live_leaf, lessons.X_eval, lessons.y_eval)
        previous_before = evaluate_suite(self.cp, eval_sets)

        candidate = clone_leaf(live_leaf)
        train_leaf(candidate, lessons.X_train, lessons.y_train,
                   steps=steps, lr=lr, batch=batch, seed=seed)
        target_after = evaluate_leaf(candidate, lessons.X_eval, lessons.y_eval)
        previous_after = evaluate_candidate_suite(self.cp, rec["eid"], candidate, eval_sets)

        regressions = [
            previous_before[k] - previous_after.get(k, previous_before[k])
            for k in previous_before
        ]
        max_regression = max(regressions) if regressions else 0.0
        delta = target_after - target_before

        reasons = []
        if target_after < policy.min_accuracy:
            reasons.append(
                f"target accuracy {target_after:.3f} below minimum {policy.min_accuracy:.3f}"
            )
        if delta < policy.min_delta:
            reasons.append(
                f"delta {delta:+.3f} below required {policy.min_delta:+.3f}"
            )
        if max_regression > policy.max_previous_regression:
            reasons.append(
                f"previous accuracy regression {max_regression:.3f} exceeds "
                f"{policy.max_previous_regression:.3f}"
            )

        accepted = not reasons
        if accepted:
            self.cp.forest.leaves[idx] = candidate

        after_hashes = self.cp._hashes()
        old_unchanged = all(
            after_hashes.get(k) == v
            for k, v in before_hashes.items()
            if k != target_key
        )
        if accepted and policy.require_other_hashes_unchanged and not old_unchanged:
            self.cp.forest.leaves[idx] = live_leaf
            after_hashes = self.cp._hashes()
            accepted = False
            reasons.append("non-target expert hash changed")

        event = "growth_update" if accepted else "growth_rejected"
        reason = "accepted" if accepted else "; ".join(reasons)
        result = GrowthResult(
            accepted=accepted,
            reason=reason,
            actor=actor,
            eid=rec["eid"],
            tenant=rec["tenant"],
            expert=rec["name"],
            teacher=lessons.teacher,
            topic=lessons.topic,
            dataset_version=lessons.dataset_version,
            target_accuracy_before=round(float(target_before), 6),
            target_accuracy_after=round(float(target_after), 6),
            target_delta=round(float(delta), 6),
            previous_accuracy_before={k: round(float(v), 6) for k, v in previous_before.items()},
            previous_accuracy_after={k: round(float(v), 6) for k, v in previous_after.items()},
            max_previous_regression=round(float(max_regression), 6),
            router_accuracy=None if eval_sets is None else router_accuracy(self.cp, eval_sets),
            old_experts_unchanged=bool(old_unchanged),
            candidate_hash=candidate.weight_hash(),
            live_hash_after=after_hashes.get(target_key, ""),
            audit_event=event,
            ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

        self.cp.audit.append(
            event,
            (
                f"{actor} growth attempt for eid={rec['eid']} ('{rec['name']}', "
                f"tenant '{rec['tenant']}') using teacher '{lessons.teacher}'; "
                f"target acc {target_before:.3f}->{target_after:.3f}; "
                f"previous max regression {max_regression:.3f}; result: {reason}"
            ),
            payload=self.cp._hashes(),
        )
        self.history.append(result)
        self.history = self.history[-self.history_limit:]
        return result

    def recent(self, n=None):
        rows = self.history if n is None else self.history[-int(n):]
        return [r.to_dict() for r in rows]

    def recent_cycles(self, n=None):
        rows = self.cycle_history if n is None else self.cycle_history[-int(n):]
        return [r.to_dict() for r in rows]

    def _teacher_list(self, teachers, teacher_names=None):
        if isinstance(teachers, dict):
            names = teacher_names or list(teachers)
            rows = []
            for name in names:
                if name not in teachers:
                    raise KeyError(f"unknown teacher '{name}'")
                rows.append(teachers[name])
        else:
            rows = list(teachers)
            if teacher_names:
                wanted = set(teacher_names)
                rows = [t for t in rows if getattr(t, "name", None) in wanted]
        if not rows:
            raise ValueError("automation needs at least one teacher")
        return rows

    def auto_cycle(self, actor, teachers, eval_sets=None, policy=None,
                   max_attempts=None, teacher_names=None, strategy="round_robin",
                   steps=120, lr=0.05, batch=32, n_train=180, n_eval=120,
                   seed_base=0):
        """Run an automated growth sweep across the actor-visible experts.

        The cycle is intentionally conservative: it does the same candidate
        train/evaluate/accept flow as ``improve_expert`` for each expert, so every
        individual attempt is audited. A summary ``growth_cycle`` audit entry is
        appended at the end for operator visibility.
        """
        policy = policy or GrowthPolicy()
        self.cp._check(actor, "graft")
        teacher_rows = self._teacher_list(teachers, teacher_names=teacher_names)
        experts = self.cp.list_experts(actor)
        if max_attempts is not None:
            experts = experts[:max(0, int(max_attempts))]

        start = len(self.history)
        results = []
        for offset, rec in enumerate(experts):
            teacher = teacher_rows[(start + offset) % len(teacher_rows)]
            result = self.improve_expert(
                actor,
                rec["eid"],
                teacher,
                topic=rec["name"],
                eval_sets=eval_sets,
                policy=policy,
                steps=steps,
                lr=lr,
                batch=batch,
                seed=seed_base + offset,
                n_train=n_train,
                n_eval=n_eval,
            )
            results.append(result)

        accepted = sum(1 for r in results if r.accepted)
        rejected = len(results) - accepted
        cycle = GrowthCycle(
            actor=actor,
            strategy=strategy,
            attempted=len(results),
            accepted=accepted,
            rejected=rejected,
            teacher_names=[t.name for t in teacher_rows],
            results=[r.to_dict() for r in results],
            ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        self.cp.audit.append(
            "growth_cycle",
            (
                f"{actor} automated growth cycle completed; attempted={len(results)}, "
                f"accepted={accepted}, rejected={rejected}, teachers={cycle.teacher_names}"
            ),
            payload=self.cp._hashes(),
        )
        self.cycle_history.append(cycle)
        self.cycle_history = self.cycle_history[-self.history_limit:]
        return cycle
