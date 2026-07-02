"""
das/platform/teacher_trainer.py
-------------------------------
The bridge between the deployment engine and the Growing-Child teacher loop
(``das.training``). Both sides already speak ``train_fn``; this module makes a
*teacher* — the deterministic local one, or a real LLM over Ollama /
OpenAI-compatible endpoints — drive expert training behind the exact seam the
platform already uses, so ``dep.grow(..., teacher=...)`` and
``dep.improve(...)`` get real teacher→candidate→evaluate→accept/reject
semantics without touching a single governance guarantee.

Geometry coherence (the subtle part): the platform's router, connector, and
synthetic experts all live in ``SyntheticTrainer``'s dense-center geometry.
  * ``AlignedVectorTeacher`` generates lessons IN that geometry (same centers,
    same label rules), so offline teacher-grown experts route exactly like
    synthetic ones.
  * Endpoint LLM teachers encode text lessons with their hashing encoder — a
    different geometry — so for experts grown from those lessons the trainer
    records the LessonBatch: the router is retrained on the actual lessons and
    the connector embeds queries at the lessons' EMPIRICAL center. Coherent
    routing either way.

Two operations, two safety shapes:
  * grow  (new expert)      — graft proves existing experts byte-identical; the
    new expert's lesson eval accuracy is *reported*, not gated (there is no
    prior capability to regress).
  * improve (existing)      — full Growing-Child policy via
    ``das.training.GrowthManager``: candidate trained in quarantine, accuracy
    floor + no-regression checks, accept/reject, audited either way. The live
    expert is never trained directly.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from das.training.evaluator import ExpertEvalSet, evaluate_leaf, train_leaf
from das.training.teachers import VectorTeacher

from .trainer import SyntheticTrainer


class AlignedVectorTeacher(VectorTeacher):
    """A deterministic local teacher whose clusters and label rules follow the
    platform trainer's geometry — lessons are drawn from the same distribution
    the fleet's router and connector already understand, so offline
    teacher-driven growth stays route-coherent with zero external dependencies."""

    def __init__(self, name: str, base: SyntheticTrainer, noise: float = 0.7,
                 seed: int = 0, label: Optional[str] = None):
        super().__init__(name, base.d_model, noise=noise, seed=seed,
                         label=label or "Local aligned teacher")
        self._base = base

    def center_for(self, topic):
        return self._base.center(topic)

    def rule_for(self, topic):
        rule = self._base.rule(topic)
        norm = np.linalg.norm(rule)
        return rule if norm == 0 else rule / norm

    def describe(self):
        d = super().describe()
        d["provider"] = "local-aligned"
        return d


class TeacherTrainer:
    """Teacher-driven ``train_fn`` factory, duck-typed to the surfaces the
    platform already consumes (``d_model`` / ``center`` / ``data`` /
    ``seed_for``), so the keyword connector and router retraining work
    identically for synthetic-, local-teacher-, and LLM-teacher-grown experts."""

    def __init__(self, base: SyntheticTrainer, n_train: int = 180,
                 n_eval: int = 120, steps: int = 140, lr: float = 0.05):
        self.base = base
        self.n_train = n_train
        self.n_eval = n_eval
        self.steps = steps
        self.lr = lr
        self.lessons: Dict[str, object] = {}   # expert name -> LessonBatch
        self.reports: Dict[str, dict] = {}     # expert name -> last grow report
        self.default_teacher = AlignedVectorTeacher("local-teacher", base)

    # ── duck-typed trainer surface ───────────────────────────────────
    @property
    def d_model(self) -> int:
        return self.base.d_model

    def seed_for(self, name: str):
        return self.base.seed_for(name)

    def center(self, name: str) -> np.ndarray:
        """Where a query for this expert should embed: the empirical center of
        its teacher lessons if it was teacher-grown, else the base geometry."""
        batch = self.lessons.get(name)
        if batch is not None:
            return batch.X_train.mean(axis=0)
        return self.base.center(name)

    def data(self, name: str):
        """Training data for router retraining: real lessons when we have them,
        the base synthetic cluster otherwise."""
        batch = self.lessons.get(name)
        if batch is not None:
            return batch.X_train, batch.y_train
        return self.base.data(name)

    # ── grow: teacher-driven train_fn for graft ──────────────────────
    def train_fn(self, name: str, cp, teacher=None):
        """A graft ``train_fn`` that trains the new leaf on TEACHER lessons and
        retrains the router over every expert's actual training distribution."""
        t = teacher or self.default_teacher

        def _fn(forest, idx):
            batch = t.generate(name, n_train=self.n_train, n_eval=self.n_eval)
            self.lessons[name] = batch
            train_leaf(forest.leaves[idx], batch.X_train, batch.y_train,
                       steps=self.steps, lr=self.lr,
                       seed=self.base.seed_for("fit:" + name))
            self.reports[name] = {
                "teacher": batch.teacher,
                "dataset_version": batch.dataset_version,
                "train_examples": int(len(batch.X_train)),
                "eval_accuracy": evaluate_leaf(forest.leaves[idx],
                                               batch.X_eval, batch.y_eval),
                "notes": batch.notes,
            }
            # Router: retrain over all experts' REAL distributions (lessons where
            # they exist, base clusters where they don't), plus the new expert.
            names = [r["name"] for r in cp.experts] + [name]
            blocks = [self.data(n) for n in names]
            Xr = np.vstack([X for X, _y in blocks])
            dr = np.concatenate([np.full(len(X), slot, dtype=int)
                                 for slot, (X, _y) in enumerate(blocks)])
            rng = np.random.default_rng(self.base.seed_for("router:" + name))
            for _ in range(self.base.router_steps):
                i = rng.integers(0, len(Xr), min(64, len(Xr)))
                forest.router.train_step(Xr[i], dr[i], lr=self.base.router_lr)
        return _fn

    # ── improve: eval sets for the Growing-Child policy ─────────────
    def eval_sets(self, cp):
        """A frozen eval set per expert, for the no-regression check: the
        teacher lessons' eval split where the expert was teacher-grown, a fresh
        held-out synthetic draw otherwise."""
        sets = []
        for rec in cp.experts:
            batch = self.lessons.get(rec["name"])
            if batch is not None:
                X, y = batch.X_eval, batch.y_eval
            else:
                X, y = self.base.sample(rec["name"], self.n_eval)
            sets.append(ExpertEvalSet(eid=rec["eid"], name=rec["name"], X=X, y=y))
        return sets
