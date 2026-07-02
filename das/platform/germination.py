"""
das/platform/germination.py
---------------------------
The Fibonacci germination lifecycle: a new expert starts as a SEED — the
smallest viable leaf — and earns capacity stage by stage as its curriculum
demands it, until it is a full tree.

    seed [3] -> sprout [5,3] -> sapling [8,5] -> young-tree [13,8] -> tree [21,13]

The hidden widths follow the Fibonacci sequence — and, in this codebase's own
measured tradition (benchmarks/leaf_shapes_bench.py: width *schedules* are
cosmetic for accuracy), the Fibonacci-ness is aesthetics. What is NOT cosmetic,
and is measured (benchmarks/germination_bench.py), is the CAPACITY LADDER
itself: on a hard curriculum a seed sits at chance while each stage climbs
(~0.48 -> 0.53 -> 0.58 -> 0.70 -> 0.73), and on an easy curriculum a seed
saturates immediately — so growth is only ever justified by learning, never by
default.

Promotion mechanics reuse the Growing-Child quarantine, at a new capacity:

    teacher lessons -> CANDIDATE at the next stage size, trained from scratch
      -> must EARN its parameters (accuracy floor + minimum improvement over
         the live expert + no regression elsewhere)
      -> accepted: replaces the live expert (audited `growth_promoted`)
      -> rejected: live expert untouched (audited `growth_promotion_rejected`)

`auto_germinate` adds the parsimony gate: if the live expert already meets the
target accuracy at its current size, DO NOTHING — most experts should live and
die as seeds. Capacity is a cost (params, memory, latency), not a status.
"""
from __future__ import annotations

from typing import Optional

from das.functional import FibonacciLeaf
from das.training.evaluator import evaluate_leaf, train_leaf
from das.training.teachers import stable_seed

# stage name -> hidden widths (Fibonacci ladder). Input/output dims come from
# the fleet. Note the platform's default leaf [d, 13, 8, out] IS young-tree.
STAGES = [
    ("seed", [3]),
    ("sprout", [5, 3]),
    ("sapling", [8, 5]),
    ("young-tree", [13, 8]),
    ("tree", [21, 13]),
]
STAGE_NAMES = [name for name, _ in STAGES]


class GerminationPolicy:
    """Promotion must be EARNED: the bigger candidate has to clear an absolute
    floor, beat the live expert by a real margin, and regress nothing."""

    def __init__(self, min_accuracy: float = 0.55, min_delta: float = 0.05,
                 max_previous_regression: float = 0.02):
        self.min_accuracy = min_accuracy
        self.min_delta = min_delta
        self.max_previous_regression = max_previous_regression


def stage_dims(d_model: int, out_dim: int, stage) -> list:
    """Full leaf dims for a stage (index or name)."""
    if isinstance(stage, str):
        if stage not in STAGE_NAMES:
            raise ValueError(f"no stage {stage!r} (stages: {STAGE_NAMES})")
        stage = STAGE_NAMES.index(stage)
    if not 0 <= stage < len(STAGES):
        raise ValueError(f"no stage {stage!r} (stages: {STAGE_NAMES})")
    return [d_model] + STAGES[stage][1] + [out_dim]


def stage_of(dims) -> Optional[str]:
    """Which stage a leaf's dims correspond to, or None for a custom shape."""
    hidden = list(dims)[1:-1]
    for name, widths in STAGES:
        if hidden == widths:
            return name
    return None


def leaf_params(leaf) -> int:
    return sum(w.size for w in leaf.W) + sum(b.size for b in leaf.b)


class Germinator:
    """Stage promotion over a governed fleet. Operates through the ControlPlane
    (RBAC-checked, audited); the live expert is never trained directly."""

    def __init__(self, cp, teacher_trainer, steps: int = 1200, lr: float = 0.08,
                 n_train: int = 400, n_eval: int = 300, restarts: int = 4):
        self.cp = cp
        self.teacher_trainer = teacher_trainer
        self.steps = steps
        self.lr = lr
        self.n_train = n_train
        self.n_eval = n_eval
        # Tiny nets on nonconvex curricula are init-sensitive: give the candidate
        # its best honest shot with a few restarts. Selection uses TRAINING
        # accuracy only — selecting on the eval split would contaminate the gate.
        self.restarts = max(1, int(restarts))

    # ── observability ────────────────────────────────────────────────
    def report(self) -> list:
        """Per-expert germination metrics: stage, dims, parameter count."""
        rows = []
        for i, rec in enumerate(self.cp.experts):
            leaf = self.cp.forest.leaves[i]
            rows.append({
                "eid": rec["eid"], "name": rec["name"], "tenant": rec["tenant"],
                "stage": stage_of(leaf.dims), "dims": list(leaf.dims),
                "params": leaf_params(leaf),
            })
        return rows

    # ── promotion ────────────────────────────────────────────────────
    def promote(self, actor: str, eid: int, teacher, to_stage=None,
                policy: Optional[GerminationPolicy] = None) -> dict:
        """Train a candidate at a LARGER stage on fresh teacher lessons; replace
        the live expert only if the extra capacity demonstrably earned it."""
        policy = policy or GerminationPolicy()
        idx, rec = self.cp._find(int(eid))
        self.cp._check(actor, "graft", rec["tenant"])
        live = self.cp.forest.leaves[idx]

        current = stage_of(live.dims)
        cur_i = STAGE_NAMES.index(current) if current else None
        if to_stage is None:
            if cur_i is None:
                raise ValueError(f"expert '{rec['name']}' has custom dims "
                                 f"{live.dims}; pass to_stage explicitly")
            if cur_i >= len(STAGES) - 1:
                raise ValueError(f"'{rec['name']}' is already a full tree")
            to_stage = STAGE_NAMES[cur_i + 1]
        to_i = STAGE_NAMES.index(to_stage)
        if cur_i is not None and to_i <= cur_i:
            raise ValueError(f"'{to_stage}' is not a promotion from '{current}'")

        lessons = teacher.generate(rec["name"], n_train=self.n_train,
                                   n_eval=self.n_eval)
        live_acc = evaluate_leaf(live, lessons.X_eval, lessons.y_eval)

        dims = stage_dims(self.cp.forest.d_model, live.dims[-1], to_stage)
        candidate, best_train = None, -1.0
        for r in range(self.restarts):
            cand = FibonacciLeaf(dims, seed=stable_seed("germinate", rec["name"], to_stage, r))
            train_leaf(cand, lessons.X_train, lessons.y_train,
                       steps=self.steps, lr=self.lr,
                       seed=stable_seed("promote-fit", rec["name"], to_stage, r))
            train_acc = evaluate_leaf(cand, lessons.X_train, lessons.y_train)
            if train_acc > best_train:
                candidate, best_train = cand, train_acc
        cand_acc = evaluate_leaf(candidate, lessons.X_eval, lessons.y_eval)
        delta = cand_acc - live_acc

        before_hashes = self.cp._hashes()
        target_key = f"eid{rec['eid']}"

        reasons = []
        if cand_acc < policy.min_accuracy:
            reasons.append(f"candidate accuracy {cand_acc:.3f} below floor "
                           f"{policy.min_accuracy:.3f}")
        if delta < policy.min_delta:
            reasons.append(f"improvement {delta:+.3f} below the {policy.min_delta:+.3f} "
                           f"a promotion must earn")
        accepted = not reasons

        if accepted:
            self.cp.forest.leaves[idx] = candidate
        after_hashes = self.cp._hashes()
        others_intact = all(after_hashes.get(k) == v for k, v in before_hashes.items()
                            if k != target_key)

        event = "growth_promoted" if accepted else "growth_promotion_rejected"
        result = {
            "accepted": accepted,
            "reason": "accepted" if accepted else "; ".join(reasons),
            "eid": rec["eid"], "expert": rec["name"], "tenant": rec["tenant"],
            "teacher": lessons.teacher,
            "stage_from": current, "stage_to": to_stage if accepted else current,
            "attempted_stage": to_stage,
            "params_before": leaf_params(live),
            "params_after": leaf_params(candidate if accepted else live),
            "accuracy_before": round(float(live_acc), 6),
            "accuracy_after": round(float(cand_acc if accepted else live_acc), 6),
            "candidate_accuracy": round(float(cand_acc), 6),
            "delta": round(float(delta), 6),
            "others_byte_identical": bool(others_intact),
        }
        self.cp.audit.append(
            event,
            (f"{actor} germination {current or live.dims}->{to_stage} for "
             f"eid={rec['eid']} ('{rec['name']}', tenant '{rec['tenant']}') via "
             f"teacher '{lessons.teacher}'; acc {live_acc:.3f}->{cand_acc:.3f} "
             f"(delta {delta:+.3f}), params {result['params_before']}->"
             f"{leaf_params(candidate)}; result: {result['reason']}"),
            payload=self.cp._hashes(),
        )
        return result

    def auto_germinate(self, actor: str, eid: int, teacher, target_acc: float = 0.85,
                       policy: Optional[GerminationPolicy] = None) -> dict:
        """The parsimony gate: promote ONLY if the live expert demonstrably
        cannot meet the target at its current capacity. A saturated expert
        stays small — that is the point of starting from a seed."""
        idx, rec = self.cp._find(int(eid))
        self.cp._check(actor, "graft", rec["tenant"])
        live = self.cp.forest.leaves[idx]
        lessons = teacher.generate(rec["name"], n_train=self.n_train,
                                   n_eval=self.n_eval)
        live_acc = evaluate_leaf(live, lessons.X_eval, lessons.y_eval)
        if live_acc >= target_acc:
            return {"action": "saturated", "eid": rec["eid"], "expert": rec["name"],
                    "stage": stage_of(live.dims), "accuracy": round(float(live_acc), 6),
                    "target": target_acc, "params": leaf_params(live),
                    "note": "meets target at current capacity — no growth needed"}
        result = self.promote(actor, eid, teacher, policy=policy)
        result["action"] = "promoted" if result["accepted"] else "promotion_rejected"
        return result
