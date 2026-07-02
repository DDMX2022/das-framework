import json
import os

from das.governance import ControlPlane
from das.mobile_store import MobileModelStore, memory_status
from das.model import DASForest


def test_memory_status_reports_warning_steps(tmp_path):
    (tmp_path / "blob.bin").write_bytes(b"x" * 25)

    status = memory_status(str(tmp_path), warning_step_bytes=10)

    assert status["size_bytes"] == 25
    assert status["warning"] is True
    assert status["warning_level_bytes"] == 20
    assert status["next_warning_bytes"] == 30


def test_mobile_model_store_exports_expert_and_manifest(tmp_path):
    forest = DASForest(4, [4, 3, 2], num_leaves=1, seed=3)
    cp = ControlPlane(forest, seed_tenant="mobile", seed_name="math")
    store = MobileModelStore(str(tmp_path), warning_step_bytes=1)

    row = store.export_expert(cp, 0)
    manifest = json.loads((tmp_path / "manifest.json").read_text())

    assert row["eid"] == 0
    assert row["hash"] == cp.forest.leaves[0].weight_hash()
    assert os.path.exists(tmp_path / row["file"])
    assert manifest["models"][0]["file"] == row["file"]
    assert store.status()["warning"] is True
