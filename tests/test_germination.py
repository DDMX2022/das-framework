"""Fibonacci germination — seeds, the capacity ladder, earned promotions,
parsimony, and the quarantine/audit guarantees."""
import pytest

from das.governance import AccessDenied
from das.platform import (
    GerminationPolicy,
    NonlinearVectorTeacher,
    deploy,
    stage_dims,
    stage_of,
)
from das.platform.germination import STAGE_NAMES, leaf_params

SPEC = {
    "client": "germ", "d_model": 18,
    "tenants": [{"name": "t", "experts": [{"name": "anchor"}]}],
    "users": [{"name": "auditor-jane", "role": "auditor"}],
}


@pytest.fixture
def dep():
    d = deploy(SPEC, secret="s")
    d.teachers["hard"] = NonlinearVectorTeacher("hard", d.trainer)
    # give grow-time training enough budget that a seed actually reaches its
    # capacity ceiling (the parsimony gate judges the live expert as-is)
    d.teacher_trainer.steps = 600
    return d


def _events(dep):
    return [e["event"] for e in dep.cp.audit.entries]


# ── stage arithmetic ─────────────────────────────────────────────────
def test_stage_dims_and_roundtrip():
    assert stage_dims(18, 2, "seed") == [18, 3, 2]
    assert stage_dims(18, 2, "tree") == [18, 21, 13, 2]
    for name in STAGE_NAMES:
        assert stage_of(stage_dims(18, 2, name)) == name
    assert stage_of([18, 99, 2]) is None
    with pytest.raises(ValueError, match="no stage"):
        stage_dims(18, 2, "bonsai")


def test_grow_as_seed_and_report(dep):
    dep.grow("root", "t", "tiny", teacher="hard", stage="seed")
    rows = {r["name"]: r for r in dep.growth_report()}
    assert rows["tiny"]["stage"] == "seed"
    assert rows["tiny"]["params"] == 65
    # the fleet default [18,13,8,2] IS the young-tree stage
    assert rows["anchor"]["stage"] == "young-tree"
    assert rows["anchor"]["params"] > 5 * rows["tiny"]["params"]


# ── promotion: capacity must earn its parameters ─────────────────────
def test_promotion_accepted_on_hard_curriculum(dep):
    eid = dep.grow("root", "t", "hard-domain", teacher="hard", stage="seed")
    anchor_hash = dep.cp.forest.leaves[0].weight_hash()
    result = dep.germinator.promote("root", eid, dep.teachers["hard"],
                                    to_stage="sapling")
    assert result["accepted"] is True
    assert result["stage_from"] == "seed" and result["stage_to"] == "sapling"
    assert result["params_before"] == 65 and result["params_after"] == 209
    assert result["delta"] >= 0.05                       # capacity earned it
    assert result["others_byte_identical"] is True
    assert dep.cp.forest.leaves[0].weight_hash() == anchor_hash
    assert "growth_promoted" in _events(dep)
    assert dep.verify()["ok"] and dep.cp.state_matches_audit()
    # the report reflects the new stage
    assert {r["name"]: r["stage"] for r in dep.growth_report()}["hard-domain"] == "sapling"


def test_promotion_rejected_leaves_live_untouched(dep):
    eid = dep.grow("root", "t", "hard-domain", teacher="hard", stage="seed")
    idx = next(i for i, r in enumerate(dep.cp.experts) if r["eid"] == eid)
    live_hash = dep.cp.forest.leaves[idx].weight_hash()
    result = dep.germinator.promote("root", eid, dep.teachers["hard"],
                                    to_stage="sapling",
                                    policy=GerminationPolicy(min_delta=0.9))
    assert result["accepted"] is False
    assert "must earn" in result["reason"]
    assert dep.cp.forest.leaves[idx].weight_hash() == live_hash   # quarantine held
    assert "growth_promotion_rejected" in _events(dep)
    assert dep.verify()["ok"] and dep.cp.state_matches_audit()


def test_parsimony_saturated_seed_stays_small(dep):
    # EASY curriculum: the default aligned teacher's linear rule — a seed nails it
    eid = dep.grow("root", "t", "easy-domain", teacher="local-teacher", stage="seed")
    result = dep.germinate("root", eid, teacher="local-teacher", target_acc=0.85)
    assert result["action"] == "saturated"
    assert result["stage"] == "seed" and result["params"] == 65
    assert not any(e.startswith("growth_promot") for e in _events(dep))


def test_auto_germinate_acts_when_below_target(dep):
    eid = dep.grow("root", "t", "hard-domain", teacher="hard", stage="seed")
    result = dep.germinate("root", eid, teacher="hard", target_acc=0.85)
    # a seed cannot meet 0.85 on the XOR curriculum -> the gate must attempt growth
    assert result["action"] in ("promoted", "promotion_rejected")
    assert any(e.startswith("growth_promot") for e in _events(dep))


def test_ladder_guards(dep):
    eid = dep.cp.experts[0]["eid"]                       # anchor: young-tree
    with pytest.raises(ValueError, match="not a promotion"):
        dep.germinator.promote("root", eid, dep.teachers["hard"], to_stage="seed")
    # walk anchor to the top, then confirm there is nowhere left to go
    dep.germinator.promote("root", eid, dep.teachers["hard"], to_stage="tree",
                           policy=GerminationPolicy(min_accuracy=0.0, min_delta=-1.0))
    with pytest.raises(ValueError, match="already a full tree"):
        dep.germinator.promote("root", eid, dep.teachers["hard"])


def test_promotion_rbac_denied(dep):
    with pytest.raises(AccessDenied):
        dep.germinator.promote("auditor-jane", dep.cp.experts[0]["eid"],
                               dep.teachers["hard"], to_stage="tree")
    assert "denied" in _events(dep)


# ── solution #1: multi-stage search ──────────────────────────────────
def test_promote_search_takes_smallest_earning_stage(dep):
    dep.germinator.steps, dep.germinator.restarts = 800, 2
    eid = dep.grow("root", "t", "hard-domain", teacher="hard", stage="seed")
    result = dep.germinator.promote_search("root", eid, dep.teachers["hard"])
    assert result["accepted"] is True
    assert len(result["searched"]) == 4              # sprout..tree all fitted
    chosen = result["stage_to"]
    order = [s["stage"] for s in result["searched"]]
    # every stage smaller than the chosen one failed to qualify
    for s in result["searched"][:order.index(chosen)]:
        assert s["qualifies"] is False
    assert result["others_byte_identical"] is True
    assert "growth_promoted" in _events(dep)
    assert dep.cp.state_matches_audit()


# ── solution #2: distillation transfer ───────────────────────────────
def test_distill_transfers_live_behaviour_across_dims(dep):
    import numpy as np
    from das.functional import FibonacciLeaf
    live = dep.cp.forest.leaves[0]                    # trained anchor
    X, _ = dep.trainer.sample("anchor", 200)
    cand = FibonacciLeaf([18, 8, 5, 2], seed=1)       # different capacity
    before = float((cand.forward(X).argmax(1) == live.forward(X).argmax(1)).mean())
    dep.germinator._distill(cand, live, X, seed=1)
    after = float((cand.forward(X).argmax(1) == live.forward(X).argmax(1)).mean())
    assert after >= 0.9 and after > before            # behaviour carried over


# ── solution #3: momentum ────────────────────────────────────────────
def test_momentum_training_path(dep):
    from das.functional import FibonacciLeaf
    from das.training.evaluator import evaluate_leaf, train_leaf
    lessons = dep.teachers["local-teacher"].generate("m-domain", 200, 150)
    leaf = FibonacciLeaf([18, 8, 5, 2], seed=0)
    train_leaf(leaf, lessons.X_train, lessons.y_train,
               steps=600, lr=0.03, momentum=0.9, seed=0)
    assert leaf.frozen is True
    assert evaluate_leaf(leaf, lessons.X_eval, lessons.y_eval) > 0.85


# ── solution #4: lesson persistence / Deployment.load ────────────────
def test_save_load_restores_lessons_and_routing(dep, tmp_path):
    dep.grow("root", "t", "pharma", keywords=["pharmacy", "rx"],
             teacher="local-teacher", stage="seed")
    before_route = dep.route("root", "pharmacy rx refill")
    assert before_route["expert"] == "pharma"
    dep.save(str(tmp_path / "state"))

    from das.platform import Deployment
    loaded = Deployment.load(str(tmp_path / "state"), SPEC, secret="s")
    assert loaded.verify()["ok"] and loaded.cp.state_matches_audit()
    assert "pharma" in loaded.teacher_trainer.lessons          # store restored
    assert loaded.teacher_trainer.reports["pharma"]["teacher"] == "local-teacher"
    # routing geometry AND grow-time keywords survived the restart
    r = loaded.route("root", "pharmacy rx refill")
    assert r["expert"] == "pharma" and r["confidence"] > 0.5
    stages = {g["name"]: g["stage"] for g in loaded.growth_report()}
    assert stages["pharma"] == "seed"                          # dims survived


# ── solution #5: fleet sweep ─────────────────────────────────────────
def test_sweep_gates_the_whole_fleet(dep):
    dep.germinator.steps, dep.germinator.restarts = 600, 2
    dep.grow("root", "t", "hard-domain", teacher="hard", stage="seed")
    summary = dep.germinate_all("root", teacher="hard", target_acc=0.85)
    assert summary["attempted"] == 2                  # anchor + hard-domain
    counted = sum(v for k, v in summary.items()
                  if k in ("saturated", "promoted", "promotion_rejected", "at_capacity"))
    assert counted == 2
    assert "germination_sweep" in _events(dep)
    assert dep.verify()["ok"] and dep.cp.state_matches_audit()
