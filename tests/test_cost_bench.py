"""The cost benchmark's invariants — the honest trade-off must hold, not a
single flattering number."""
import cost_bench


def test_run_returns_a_row_per_threshold():
    res = cost_bench.run(k=4, m=2, n_per=30)
    assert len(res["rows"]) == len(cost_bench.THRESHOLDS)
    assert res["baseline_per_1k"] > 0
    assert res["n_queries"] == (4 + 2) * 30


def test_deflection_is_monotonic_in_threshold():
    rows = cost_bench.run(k=4, m=2, n_per=30)["rows"]
    defl = [r["deflection_pct"] for r in rows]         # thresholds ascending
    assert defl == sorted(defl, reverse=True)          # higher threshold -> less deflection
    assert defl[0] == 100.0                             # threshold 0 keeps everything local
    assert rows[-1]["deflection_pct"] == 0.0            # threshold 1.0 escalates everything


def test_cost_and_quality_rise_with_threshold():
    rows = cost_bench.run(k=4, m=2, n_per=30)["rows"]
    cost = [r["das_per_1k"] for r in rows]
    quality = [r["accuracy_pct"] for r in rows]
    assert cost == sorted(cost)                         # more escalation -> more cost
    assert quality == sorted(quality)                   # more escalation -> higher quality
    assert quality[-1] == 100.0                         # all-escalate == baseline quality


def test_thin_prompt_saves_even_with_zero_deflection():
    # the all-escalate row still beats the fat-prompt baseline purely on prompt size
    last = cost_bench.run(k=4, m=2, n_per=30)["rows"][-1]
    assert last["deflection_pct"] == 0.0
    assert last["savings_pct"] > 0.0


def test_full_deflection_makes_api_cost_zero():
    first = cost_bench.run(k=4, m=2, n_per=30)["rows"][0]
    assert first["das_per_1k"] == 0.0
    assert first["savings_pct"] == 100.0
