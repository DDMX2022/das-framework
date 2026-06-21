"""
Regression test for the governance benchmark (governance_benchmark.py). Pure
NumPy — locks in the headline comparison so it can't silently regress.
"""
import governance_benchmark as gb


def test_monolith_forgets_and_has_no_governance():
    m = gb.run_monolith()
    assert m["bwt"] < -0.01                       # sequential fine-tuning forgets
    assert m["isolation_on_add"] == 0.0           # shared weights: adds disturb all
    assert not (m["audit"] or m["rbac"] or m["provenance"])


def test_isolated_experts_isolate_but_lack_governance():
    iso = gb.run_isolated()
    assert iso["bwt"] == 0.0
    assert iso["isolation_on_add"] == 1.0
    assert iso["delete_survivors_identical"] == 1.0
    assert iso["delete_removes_capability"] is True
    assert not (iso["audit"] or iso["rbac"])      # the gap DAS fills


def test_das_matches_isolation_and_adds_governance():
    das = gb.run_das()
    # ties isolated experts on isolation/forgetting/deletion ...
    assert das["bwt"] == 0.0
    assert das["isolation_on_add"] == 1.0
    assert das["delete_survivors_identical"] == 1.0
    # ... and adds the governance plane, all measured
    assert das["audit"] and das["audit_rate"] == 1.0
    assert das["rbac"] and das["rbac_rate"] == 1.0
    assert das["provenance"] is True
    assert das["route_acc"] > 0.9
    assert das["delete_removed"] == 3 and das["delete_non_interference"] is True
