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

import numpy as np

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

    def __init__(self, cp, teacher_trainer, steps: int = 1200, lr: float = 0.03,
                 n_train: int = 400, n_eval: int = 300, restarts: int = 4,
                 momentum: float = 0.9, transfer: str = "none"):
        self.cp = cp
        self.teacher_trainer = teacher_trainer
        self.steps = steps
        # momentum + a lower lr measurably beats plain SGD here (sapling on the
        # XOR curriculum: 0.74 vs 0.66 mean over seeds) — tiny nets on nonconvex
        # tasks are init-sensitive, momentum helps them escape.
        self.lr = lr
        self.momentum = momentum
        self.n_train = n_train
        self.n_eval = n_eval
        # Restarts give the candidate its best honest shot. Selection uses
        # TRAINING accuracy only — selecting on eval would contaminate the gate.
        self.restarts = max(1, int(restarts))
        # transfer="distill": warm-start each candidate by matching the LIVE
        # expert's logits before label training, so learned behaviour carries
        # across the capacity change (dims differ, weights can't be copied).
        if transfer not in ("none", "distill"):
            raise ValueError("transfer must be 'none' or 'distill'")
        self.transfer = transfer

    # ── candidate fitting ────────────────────────────────────────────
    def _distill(self, candidate, live, X, seed=0, temperature=2.0):
        """Pre-train the candidate to imitate the live expert — standard
        knowledge distillation toward its SOFT probabilities. (Matching raw
        logits by MSE collapses the candidate: the live logits are large, the
        early gradients explode, and the ReLUs die — measured, hence soft
        targets with a temperature.)"""
        from das.functional import softmax
        soft = softmax(live.forward(X) / temperature)
        rng = np.random.default_rng(seed)
        candidate.frozen = False
        for _ in range(max(400, self.steps // 2)):
            i = rng.integers(0, len(X), 32)
            d = (softmax(candidate.forward(X[i])) - soft[i]) / len(i)
            candidate.backward(d, 0.05)
        candidate.frozen = True

    def _fit_candidate(self, name: str, to_stage: str, out_dim: int, lessons,
                       live=None):
        """Best-of-restarts candidate at a stage (selected on TRAIN accuracy).
        Returns (candidate, eval_accuracy)."""
        best, best_train = None, -1.0
        for r in range(self.restarts):
            cand = FibonacciLeaf(
                stage_dims(self.cp.forest.d_model, out_dim, to_stage),
                seed=stable_seed("germinate", name, to_stage, r))
            if self.transfer == "distill" and live is not None:
                self._distill(cand, live, lessons.X_train,
                              seed=stable_seed("distill", name, to_stage, r))
            train_leaf(cand, lessons.X_train, lessons.y_train,
                       steps=self.steps, lr=self.lr, momentum=self.momentum,
                       seed=stable_seed("promote-fit", name, to_stage, r))
            train_acc = evaluate_leaf(cand, lessons.X_train, lessons.y_train)
            if train_acc > best_train:
                best, best_train = cand, train_acc
        return best, evaluate_leaf(best, lessons.X_eval, lessons.y_eval)

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
        candidate, cand_acc = self._fit_candidate(rec["name"], to_stage,
                                                  live.dims[-1], lessons, live=live)
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

    def promote_search(self, actor: str, eid: int, teacher,
                       policy: Optional[GerminationPolicy] = None) -> dict:
        """Multi-stage search: the next rung isn't always worth its params while
        a higher one is (measured: seed->sprout +0.02 rejected, seed->sapling
        +0.20 passes). Fit a candidate at EVERY higher stage on the same lesson
        batch, then accept the SMALLEST stage that satisfies policy — parsimony
        even in success. One audit event either way, listing what was searched."""
        policy = policy or GerminationPolicy()
        idx, rec = self.cp._find(int(eid))
        self.cp._check(actor, "graft", rec["tenant"])
        live = self.cp.forest.leaves[idx]
        current = stage_of(live.dims)
        if current is None:
            raise ValueError(f"expert '{rec['name']}' has custom dims {live.dims}")
        cur_i = STAGE_NAMES.index(current)
        if cur_i >= len(STAGES) - 1:
            raise ValueError(f"'{rec['name']}' is already a full tree")

        lessons = teacher.generate(rec["name"], n_train=self.n_train,
                                   n_eval=self.n_eval)
        live_acc = evaluate_leaf(live, lessons.X_eval, lessons.y_eval)

        searched, chosen = [], None
        for to_stage in STAGE_NAMES[cur_i + 1:]:
            cand, acc = self._fit_candidate(rec["name"], to_stage,
                                            live.dims[-1], lessons, live=live)
            ok = (acc >= policy.min_accuracy
                  and (acc - live_acc) >= policy.min_delta)
            searched.append({"stage": to_stage, "accuracy": round(float(acc), 6),
                             "delta": round(float(acc - live_acc), 6),
                             "qualifies": ok, "params": leaf_params(cand)})
            if ok and chosen is None:
                chosen = (to_stage, cand, acc)      # smallest qualifying stage

        before_hashes = self.cp._hashes()
        target_key = f"eid{rec['eid']}"
        accepted = chosen is not None
        if accepted:
            to_stage, candidate, cand_acc = chosen
            self.cp.forest.leaves[idx] = candidate
        else:
            best = max(searched, key=lambda s: s["accuracy"])
            to_stage, cand_acc = best["stage"], best["accuracy"]
        after_hashes = self.cp._hashes()
        others_intact = all(after_hashes.get(k) == v for k, v in before_hashes.items()
                            if k != target_key)

        event = "growth_promoted" if accepted else "growth_promotion_rejected"
        reason = ("accepted" if accepted else
                  f"no searched stage earned promotion (best {to_stage} "
                  f"acc {cand_acc:.3f} vs live {live_acc:.3f})")
        result = {
            "accepted": accepted, "reason": reason,
            "eid": rec["eid"], "expert": rec["name"], "tenant": rec["tenant"],
            "teacher": lessons.teacher,
            "stage_from": current,
            "stage_to": to_stage if accepted else current,
            "searched": searched,
            "params_before": leaf_params(live),
            "params_after": leaf_params(self.cp.forest.leaves[idx]),
            "accuracy_before": round(float(live_acc), 6),
            "accuracy_after": round(float(cand_acc if accepted else live_acc), 6),
            "delta": round(float(cand_acc - live_acc), 6) if accepted else 0.0,
            "others_byte_identical": bool(others_intact),
        }
        self.cp.audit.append(
            event,
            (f"{actor} germination search from {current} for eid={rec['eid']} "
             f"('{rec['name']}', tenant '{rec['tenant']}') via teacher "
             f"'{lessons.teacher}'; searched "
             f"{[(s['stage'], s['accuracy']) for s in searched]}; result: {reason}"),
            payload=self.cp._hashes(),
        )
        return result

    def auto_germinate(self, actor: str, eid: int, teacher, target_acc: float = 0.85,
                       policy: Optional[GerminationPolicy] = None,
                       search: bool = True) -> dict:
        """The parsimony gate: promote ONLY if the live expert demonstrably
        cannot meet the target at its current capacity. A saturated expert
        stays small — that is the point of starting from a seed. With
        ``search`` (default) the promotion considers every higher stage and
        takes the smallest that earns it."""
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
        if stage_of(live.dims) == STAGE_NAMES[-1]:
            return {"action": "at_capacity", "eid": rec["eid"], "expert": rec["name"],
                    "stage": STAGE_NAMES[-1], "accuracy": round(float(live_acc), 6),
                    "target": target_acc, "params": leaf_params(live),
                    "note": "below target but already a full tree — a better "
                            "teacher/curriculum is needed, not more capacity"}
        promote = self.promote_search if search else self.promote
        result = promote(actor, eid, teacher, policy=policy)
        result["action"] = "promoted" if result["accepted"] else "promotion_rejected"
        return result

    def sweep(self, actor: str, teacher, target_acc: float = 0.85,
              policy: Optional[GerminationPolicy] = None,
              search: bool = True) -> dict:
        """Plateau monitoring for the whole fleet: run the parsimony gate over
        every actor-visible expert. Saturated experts are left alone; stuck ones
        attempt (searched) promotion. Summarised in one `germination_sweep`
        audit entry; every individual promotion is audited as usual."""
        self.cp._check(actor, "graft")
        results = [self.auto_germinate(actor, rec["eid"], teacher,
                                       target_acc=target_acc, policy=policy,
                                       search=search)
                   for rec in self.cp.list_experts(actor)]
        counts = {}
        for r in results:
            counts[r["action"]] = counts.get(r["action"], 0) + 1
        self.cp.audit.append(
            "germination_sweep",
            (f"{actor} germination sweep: attempted={len(results)}, "
             + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))),
            payload=self.cp._hashes(),
        )
        return {"attempted": len(results), **counts, "results": results}
