import pytest

pytest.importorskip("flask")

import governance_api as api
from das.mobile_store import MobileModelStore


@pytest.fixture()
def client():
    return api.app.test_client()


def test_growth_dashboard_loads(client):
    r = client.get("/growth")
    assert r.status_code == 200
    assert b"DAS Growing Child" in r.data
    assert b'id="forest3d"' in r.data
    assert b"LearningForestScene" in r.data
    assert b"Builder hotbar" in r.data
    assert b"Master Tree Toolbelt" in r.data
    assert b"Knowledge Blocks" in r.data
    assert b"Chop Tree" in r.data
    assert b"Connect LLM Teacher" in r.data
    assert b"Player Forest" in r.data
    assert b"Multiplayer Arena" in r.data
    assert b"/growth/manifest.json" in r.data


def test_mobile_trainer_dashboard_loads(client):
    r = client.get("/growth/mobile/trainer")
    assert r.status_code == 200
    assert b"Mobile Expert Trainer" in r.data
    assert b"Connect LLM" in r.data
    assert b"Train Expert" in r.data
    assert b"Test Prompt" in r.data
    assert b"/growth/mobile/test_prompt" in r.data


def test_growth_status_lists_teachers_and_experts(client):
    r = client.get("/growth/status", headers={"X-DAS-Actor": "root"})
    assert r.status_code == 200
    j = r.get_json()
    assert any(t["id"] == "qwen-8b-teacher" for t in j["teachers"])
    assert any(p["id"] == "openai-compatible" for p in j["providers"])
    assert "lesson_contract" in j
    assert "players" in j
    assert "shared_arena" in j
    assert "blocks" in j["shared_arena"]
    assert "buildings" in j["shared_arena"]
    assert len(j["experts"]) >= 1
    assert "policy" in j


def test_growth_registers_dynamic_teacher(client):
    tid = f"phone-test-{len(api.GROWTH_TEACHERS)}"
    r = client.post(
        "/growth/teachers",
        json={
            "id": tid,
            "provider": "local-vector",
            "label": "Phone test teacher",
            "replace": True,
        },
        headers={"X-DAS-Actor": "root"},
    )
    assert r.status_code == 201
    j = r.get_json()
    assert j["teacher"]["id"] == tid
    assert j["teacher"]["provider"] == "local-vector"
    assert "api_key" not in j["teacher"]
    assert tid in api.GROWTH_TEACHERS
    assert api.cp.audit.entries[-1]["event"] == "growth_teacher_registered"


def test_growth_register_teacher_denied_for_auditor(client):
    r = client.post(
        "/growth/teachers",
        json={"id": "auditor-llm", "provider": "local-vector"},
        headers={"X-DAS-Actor": "carol"},
    )
    assert r.status_code == 403


def test_growth_mobile_save_syncs_small_models(client, tmp_path, monkeypatch):
    monkeypatch.setattr(api, "mobile_store", MobileModelStore(str(tmp_path), warning_step_bytes=16))
    r = client.post("/growth/mobile/save", headers={"X-DAS-Actor": "root"}, json={})
    assert r.status_code == 200
    j = r.get_json()
    assert len(j["saved"]) == len(api.cp.experts)
    assert j["status"]["path"] == str(tmp_path)
    assert j["status"]["warning"] is True

    r = client.get("/growth/mobile/memory", headers={"X-DAS-Actor": "root"})
    assert r.status_code == 200
    j = r.get_json()
    assert len(j["manifest"]["models"]) == len(api.cp.experts)


def test_growth_mobile_save_denied_for_auditor(client):
    r = client.post("/growth/mobile/save", headers={"X-DAS-Actor": "carol"}, json={})
    assert r.status_code == 403


def test_growth_multiplayer_player_share_and_import(client, tmp_path, monkeypatch):
    monkeypatch.setattr(api, "SHARED_EXPERT_DIR", str(tmp_path))
    n = len(api.cp.users)
    p1 = f"player-{n}"
    p2 = f"player-{n + 1}"

    r = client.post(
        "/growth/players",
        json={"player": p1, "display_name": "Player One"},
        headers={"X-DAS-Actor": "root"},
    )
    assert r.status_code == 201
    tenant1 = r.get_json()["player"]["tenant"]
    assert api.cp.users[p1]["tenant"] == tenant1

    r = client.post(
        "/growth/create_expert",
        json={
            "tenant": tenant1,
            "name": f"{p1}-math",
            "specialty": "math",
            "parent": "math",
            "teacher": "qwen-8b-teacher",
            "steps": 6,
            "n_train": 48,
            "n_eval": 32,
        },
        headers={"X-DAS-Actor": p1},
    )
    assert r.status_code == 201
    eid = r.get_json()["expert"]["eid"]

    r = client.post(
        "/growth/share",
        json={"eid": eid, "shared_name": f"{p1}-math-share"},
        headers={"X-DAS-Actor": p1},
    )
    assert r.status_code == 201
    shared_id = r.get_json()["shared"]["id"]
    assert r.get_json()["shared"]["author"] == p1
    assert api.cp.audit.entries[-1]["event"] == "growth_expert_shared"

    r = client.post(
        "/growth/players",
        json={"player": p2, "display_name": "Player Two"},
        headers={"X-DAS-Actor": "root"},
    )
    assert r.status_code == 201
    tenant2 = r.get_json()["player"]["tenant"]

    r = client.post(
        "/growth/import",
        json={
            "shared_id": shared_id,
            "tenant": tenant2,
            "name": f"{p2}-imported-math",
            "specialty": "math",
        },
        headers={"X-DAS-Actor": p2},
    )
    assert r.status_code == 201
    j = r.get_json()
    assert j["expert"]["tenant"] == tenant2
    assert j["expert"]["source_shared"] == shared_id
    assert j["expert"]["source_author"] == p1
    assert api.cp.audit.entries[-1]["event"] == "growth_expert_imported"


def test_growth_share_denied_for_invisible_expert(client, tmp_path, monkeypatch):
    monkeypatch.setattr(api, "SHARED_EXPERT_DIR", str(tmp_path))
    eid = api.cp.experts[0]["eid"]
    r = client.post(
        "/growth/share",
        json={"eid": eid, "shared_name": "not-bobs-expert"},
        headers={"X-DAS-Actor": "bob"},
    )
    assert r.status_code == 403


def test_growth_harvests_blocks_and_assembles_building(client, tmp_path, monkeypatch):
    monkeypatch.setattr(api, "SHARED_EXPERT_DIR", str(tmp_path))
    eid = api.cp.list_experts("root")[0]["eid"]

    r = client.post(
        "/growth/blocks/harvest",
        json={"eid": eid, "block_name": "physics-force-block", "material": "physics"},
        headers={"X-DAS-Actor": "root"},
    )
    assert r.status_code == 201
    block = r.get_json()["block"]
    assert block["id"] == "physics-force-block"
    assert block["kind"] == "knowledge_block"
    assert block["material"] == "physics"

    r = client.post(
        "/growth/buildings",
        json={
            "name": "Physics Building",
            "building_type": "physics",
            "blocks": [block["id"]],
        },
        headers={"X-DAS-Actor": "root"},
    )
    assert r.status_code == 201
    building = r.get_json()["building"]
    assert building["id"] == "physics-building"
    assert building["block_count"] == 1
    assert building["materials"][0]["id"] == block["id"]

    r = client.get("/growth/blocks", headers={"X-DAS-Actor": "root"})
    assert r.status_code == 200
    j = r.get_json()
    assert any(row["id"] == block["id"] for row in j["blocks"])
    assert any(row["id"] == building["id"] for row in j["buildings"])

    r = client.get("/growth/status", headers={"X-DAS-Actor": "root"})
    assert r.status_code == 200
    arena = r.get_json()["shared_arena"]
    assert arena["blocks"] >= 1
    assert arena["buildings"] >= 1
    recent_events = [entry["event"] for entry in api.cp.audit.entries[-4:]]
    assert "growth_block_harvested" in recent_events
    assert "growth_building_assembled" in recent_events


def test_growth_mobile_prompt_test_scores_expert(client):
    eid = api.cp.list_experts("root")[0]["eid"]
    r = client.post(
        "/growth/mobile/test_prompt",
        json={"eid": eid, "prompt": "Explain contract tax rules in simple terms.", "topic": "legal"},
        headers={"X-DAS-Actor": "root"},
    )
    assert r.status_code == 200
    j = r.get_json()
    assert j["expert"]["eid"] == eid
    assert j["prompt"]
    assert j["direct"]["label"] in (0, 1)
    assert len(j["direct"]["probabilities"]) == 2
    assert "router" in j
    assert "confidence" in j["router"]


def test_growth_tree_returns_branch_structure(client):
    r = client.get("/growth/tree", headers={"X-DAS-Actor": "root"})
    assert r.status_code == 200
    tree = r.get_json()["tree"]
    assert tree["name"] == "DAS"
    assert tree["children"]
    assert all("children" in branch for branch in tree["children"])


def test_growth_create_expert_adds_tree_leaf(client):
    name = f"test-branch-{len(api.cp.experts)}"
    r = client.post(
        "/growth/create_expert",
        json={
            "tenant": "learning",
            "name": name,
            "specialty": "react",
            "parent": "react",
            "teacher": "qwen-8b-teacher",
            "steps": 8,
            "n_train": 48,
            "n_eval": 32,
        },
        headers={"X-DAS-Actor": "root"},
    )
    assert r.status_code == 201
    j = r.get_json()
    assert j["expert"]["name"] == name
    assert j["expert"]["specialty"] == "react"
    assert any(
        leaf["name"] == name
        for branch in j["tree"]["children"]
        for leaf in branch["children"]
    )
    assert api.cp.audit.entries[-1]["event"] == "growth_create_expert"


def test_growth_run_denied_for_auditor(client):
    eid = api.cp.experts[0]["eid"]
    r = client.post(
        "/growth/run",
        json={"eid": eid, "teacher": "qwen-8b-teacher", "steps": 1},
        headers={"X-DAS-Actor": "carol"},
    )
    assert r.status_code == 403


def test_growth_auto_run_can_do_empty_cycle(client):
    r = client.post(
        "/growth/auto/run",
        json={"max_attempts": 0},
        headers={"X-DAS-Actor": "root"},
    )
    assert r.status_code == 200
    j = r.get_json()
    assert j["cycle"]["attempted"] == 0
    assert j["cycle"]["audit_event"] == "growth_cycle"


def test_growth_auto_run_denied_for_auditor(client):
    r = client.post(
        "/growth/auto/run",
        json={"max_attempts": 0},
        headers={"X-DAS-Actor": "carol"},
    )
    assert r.status_code == 403
