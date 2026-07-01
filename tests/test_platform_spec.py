"""Spec parsing + validation for the platform deployment engine."""
import json

import pytest

from das.platform import ClientSpec, SpecError


def _valid():
    return {
        "client": "acme-co",
        "d_model": 16,
        "escalation": {"frontier_llm": "claude-sonnet-5", "confidence_threshold": 0.6},
        "tenants": [
            {"name": "t1", "experts": [{"name": "e1", "keywords": ["a", "b"]},
                                       {"name": "e2"}]},
            {"name": "t2", "experts": [{"name": "e3"}]},
        ],
        "users": [{"name": "op1", "role": "operator", "tenant": "t1"},
                  {"name": "aud", "role": "auditor"}],
    }


def test_parses_valid_spec():
    spec = ClientSpec.from_dict(_valid())
    assert spec.client == "acme-co"
    assert spec.tenant_names == ["t1", "t2"]
    assert [name for _t, e in spec.experts for name in [e.name]] == ["e1", "e2", "e3"]
    assert spec.escalation.confidence_threshold == 0.6
    assert spec.resolved_leaf_dims()[0] == 16


def test_flat_expert_order_is_declaration_order():
    spec = ClientSpec.from_dict(_valid())
    assert [(t, e.name) for t, e in spec.experts] == [("t1", "e1"), ("t1", "e2"), ("t2", "e3")]


def test_from_json_file(tmp_path):
    p = tmp_path / "client.json"
    p.write_text(json.dumps(_valid()))
    spec = ClientSpec.from_file(str(p))
    assert spec.client == "acme-co"


@pytest.mark.parametrize("mutate,msg", [
    (lambda d: d.pop("client"), "client"),
    (lambda d: d.update(client=""), "client"),
    (lambda d: d.update(tenants=[]), "tenants"),
    (lambda d: d["tenants"][0].update(experts=[]), "experts"),
    (lambda d: d["users"][0].update(role="wizard"), "role"),
    (lambda d: d["users"][0].update(tenant="ghost"), "not a declared tenant"),
    (lambda d: d["escalation"].update(confidence_threshold=1.5), "confidence_threshold"),
    (lambda d: d.update(d_model=1), "d_model"),
])
def test_rejects_bad_specs(mutate, msg):
    d = _valid()
    mutate(d)
    with pytest.raises(SpecError) as exc:
        ClientSpec.from_dict(d)
    assert msg in str(exc.value)


def test_duplicate_tenant_and_expert_names_rejected():
    d = _valid()
    d["tenants"].append({"name": "t1", "experts": [{"name": "x"}]})
    with pytest.raises(SpecError, match="duplicate tenant"):
        ClientSpec.from_dict(d)

    d2 = _valid()
    d2["tenants"][1]["experts"][0]["name"] = "e1"  # collides with t1's e1
    with pytest.raises(SpecError, match="duplicate expert"):
        ClientSpec.from_dict(d2)


def test_secret_never_in_spec_read_from_env(monkeypatch):
    spec = ClientSpec.from_dict(_valid())
    monkeypatch.delenv("DAS_AUDIT_SECRET", raising=False)
    with pytest.raises(SpecError, match="audit secret not found"):
        spec.resolve_secret()
    monkeypatch.setenv("DAS_AUDIT_SECRET", "from-env")
    assert spec.resolve_secret() == "from-env"
