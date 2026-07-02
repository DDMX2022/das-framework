import numpy as np

from das.governance import ControlPlane
from das.model import DASForest
from das.training import (
    EndpointLLMTeacher,
    ExpertEvalSet,
    GrowthManager,
    GrowthPolicy,
    VectorTeacher,
    teacher_from_config,
    train_leaf,
)
import das.training.teachers as teacher_mod


D, LEAF = 10, [10, 8, 5, 2]
CENTERS = {"math": 1, "physics": 6}


def _train_fn(teacher, topic, steps=80):
    lessons = teacher.generate(topic, n_train=180, n_eval=80)

    def fn(forest, idx):
        train_leaf(forest.leaves[idx], lessons.X_train, lessons.y_train,
                   steps=steps, lr=0.05, batch=32, seed=idx + 3)

    return fn


def _eval_sets(cp, teacher):
    rows = []
    for rec in cp.experts:
        lessons = teacher.generate(rec["name"], n_train=1, n_eval=100,
                                   dataset_version=f"eval:{rec['name']}")
        rows.append(ExpertEvalSet(rec["eid"], rec["name"], lessons.X_eval, lessons.y_eval))
    return rows


def _seed_cp():
    teacher = VectorTeacher("baseline", D, centers=CENTERS, noise=0.55)
    forest = DASForest(D, LEAF, num_leaves=1, seed=11)
    _train_fn(teacher, "math", steps=0)(forest, 0)
    cp = ControlPlane(forest, seed_tenant="school", seed_name="math")
    cp.register_tenant("root", "lab")
    cp.graft("root", "lab", "physics", _train_fn(teacher, "physics", steps=80))
    return cp, teacher


def test_growth_accepts_improving_candidate_and_audits_it():
    cp, teacher = _seed_cp()
    physics_hash = cp.forest.leaves[1].weight_hash()
    manager = GrowthManager(cp)
    result = manager.improve_expert(
        "root",
        eid=0,
        teacher=teacher,
        eval_sets=_eval_sets(cp, teacher),
        policy=GrowthPolicy(min_accuracy=0.55, min_delta=0.01, max_previous_regression=0.10),
        steps=180,
        seed=42,
    )

    assert result.accepted is True
    assert result.target_accuracy_after >= result.target_accuracy_before + 0.01
    assert result.audit_event == "growth_update"
    assert result.old_experts_unchanged is True
    assert cp.forest.leaves[1].weight_hash() == physics_hash
    assert cp.audit.entries[-1]["event"] == "growth_update"
    assert cp.audit.verify()[0] is True
    assert cp.state_matches_audit() is True


def test_growth_rejects_candidate_without_changing_live_weights():
    cp, teacher = _seed_cp()
    before = cp._hashes()
    manager = GrowthManager(cp)
    result = manager.improve_expert(
        "root",
        eid=0,
        teacher=teacher,
        eval_sets=_eval_sets(cp, teacher),
        policy=GrowthPolicy(min_accuracy=1.01, min_delta=0.0, max_previous_regression=0.10),
        steps=60,
        seed=7,
    )

    assert result.accepted is False
    assert "below minimum" in result.reason
    assert cp._hashes() == before
    assert cp.audit.entries[-1]["event"] == "growth_rejected"
    assert cp.audit.verify()[0] is True
    assert cp.state_matches_audit() is True


def test_auto_cycle_runs_visible_experts_and_audits_summary():
    cp, teacher = _seed_cp()
    before = cp._hashes()
    manager = GrowthManager(cp)
    cycle = manager.auto_cycle(
        "root",
        [teacher],
        eval_sets=_eval_sets(cp, teacher),
        policy=GrowthPolicy(min_accuracy=1.01, min_delta=0.0, max_previous_regression=0.10),
        max_attempts=1,
        steps=20,
        seed_base=10,
    )

    assert cycle.attempted == 1
    assert cycle.accepted == 0
    assert cycle.rejected == 1
    assert cycle.results[0]["audit_event"] == "growth_rejected"
    assert cp._hashes() == before
    assert cp.audit.entries[-1]["event"] == "growth_cycle"
    assert cp.audit.verify()[0] is True
    assert cp.state_matches_audit() is True


def test_endpoint_llm_teacher_turns_json_lessons_into_vectors(monkeypatch):
    def fake_post_json(url, body, headers=None, timeout=60):
        assert url == "http://phone.local/lessons"
        assert body["topic"] == "react"
        return {
            "dataset_version": "phone-react-v1",
            "train": [
                {"input": "useState stores component state", "label": 1},
                {"input": "SQL joins combine tables", "label": 0},
            ],
            "eval": [
                {"input": "useEffect handles side effects", "label": 1},
                {"input": "photosynthesis makes glucose", "label": 0},
            ],
            "notes": "phone teacher fixture",
        }

    monkeypatch.setattr(teacher_mod, "_post_json", fake_post_json)
    teacher = EndpointLLMTeacher(
        "phone-llm",
        D,
        provider="custom-json",
        endpoint="http://phone.local/lessons",
        model="tiny-mobile",
    )
    lessons = teacher.generate("react", n_train=8, n_eval=4)

    assert lessons.teacher == "phone-llm"
    assert lessons.X_train.shape == (2, D)
    assert lessons.X_eval.shape == (2, D)
    assert lessons.y_train.tolist() == [1, 0]
    assert "phone teacher" in lessons.notes


def test_teacher_from_config_builds_dynamic_local_teacher():
    teacher = teacher_from_config(
        {"id": "math-phone", "provider": "local-vector", "label": "Math phone"},
        D,
        centers=CENTERS,
    )

    assert teacher.describe()["id"] == "math-phone"
    assert teacher.describe()["provider"] == "local-vector"
