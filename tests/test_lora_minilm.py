"""
Phase-1 LoRA experts on MiniLM's own attention layers, governed by the
UNMODIFIED ControlPlane — needs the [hf] extra (torch + transformers) and the
cached MiniLM weights; skipped without them. Steps are kept tiny: these tests
prove the guarantees (no-op init, backbone immobility, byte-identity isolation,
earned promotion), not accuracy — accuracy is measured honestly in
benchmarks/lora_rank_bench.py.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from das.governance import ControlPlane
from das.platform.germination import GerminationPolicy
from das.platform.lora_expert import (
    MiniLMLoRABackbone,
    MiniLMLoRALeaf,
    MiniLMLoRATrainer,
    RankGerminator,
    RANK_STAGE_NAMES,
    TopicRiskTextTeacher,
    rank_stage_of,
    stage_rank,
)

TEXTS = ["the vendor accepted our limitation-of-liability language",
         "patient reports sudden chest pain radiating to the left arm"]


@pytest.fixture(scope="module")
def backbone():
    return MiniLMLoRABackbone.cached()


def _trainer(backbone, **kw):
    kw.setdefault("steps", 40)
    kw.setdefault("n_train", 32)
    kw.setdefault("n_eval", 24)
    kw.setdefault("batch", 8)
    kw.setdefault("router_steps", 150)
    return MiniLMLoRATrainer(backbone, **kw)


@pytest.fixture(scope="module")
def fleet(backbone):
    """A governed two-expert fleet (legal seed + medical graft), built through
    the SAME ControlPlane the NumPy backend uses."""
    tr = _trainer(backbone)
    cp = ControlPlane(tr.seed_forest("legal"), "acme", "legal")
    cp.register_tenant("root", "medico")
    cp.graft("root", "medico", "medical", tr.train_fn("medical", cp),
             seed=tr.seed_for("medical"))
    return cp, tr


# ── the adapter itself ───────────────────────────────────────────────────────

def test_targets_are_the_attention_projections(backbone):
    """LoRA must sit on the transformer's OWN attention layers — every target
    is a query/value projection and every encoder layer is covered."""
    assert len(backbone.targets) == 2 * backbone.model.config.num_hidden_layers
    assert all(t.endswith((".query", ".value", ".q_proj", ".v_proj"))
               for t in backbone.targets)


def test_fresh_adapter_is_a_noop(backbone):
    """B starts at zero: a fresh rank-8 leaf must equal the bare frozen
    encoder + its head, byte-for-byte in output."""
    lora = MiniLMLoRALeaf(backbone, rank=8, out_dim=2, seed=7)
    head_only = MiniLMLoRALeaf(backbone, rank=0, out_dim=2, seed=7)
    # same seed -> identical head init; rank must not perturb the output
    assert torch.allclose(lora.forward(TEXTS), head_only.forward(TEXTS),
                          atol=1e-6)


def test_trained_adapter_changes_output_and_hash(backbone, fleet):
    cp, tr = fleet
    batch = TopicRiskTextTeacher().generate("finance", n_train=32, n_eval=8)
    leaf = MiniLMLoRALeaf(backbone, rank=2, out_dim=2, seed=3)
    before_out = leaf.forward(TEXTS).detach().clone()
    before_hash = leaf.weight_hash()
    tr.fit_leaf(leaf, batch, seed=0)
    assert leaf.weight_hash() != before_hash
    assert not torch.allclose(leaf.forward(TEXTS), before_out)


def test_backbone_never_moves(backbone, fleet):
    """The seed and the graft both trained through the shared encoder — its
    parameters must be byte-identical to a freshly loaded copy and must all
    still refuse gradients."""
    assert all(not p.requires_grad for p in backbone.model.parameters())
    from transformers import AutoModel
    fresh = AutoModel.from_pretrained(backbone.model_name)
    for (name, p), (_, q) in zip(backbone.model.named_parameters(),
                                 fresh.named_parameters()):
        assert torch.equal(p.cpu(), q.cpu()), f"backbone moved at {name}"


# ── governance over transformer experts, unchanged ───────────────────────────

def test_graft_isolation_byte_identical(fleet):
    """Growing 'medical' must not move a byte of 'legal' — the same proof the
    NumPy backend makes, now over attention-layer adapters."""
    cp, tr = fleet
    assert len(cp.experts) == 2
    entry = [e for e in cp.audit.entries if e["event"] == "graft"][-1]
    assert "prior experts unchanged: True" in entry["detail"]
    assert cp.audit.verify()


def test_routing_separates_real_topics(fleet):
    cp, tr = fleet
    legal_q = "the counterparty added an uncapped indemnification obligation"
    medical_q = "severe allergic reaction with throat swelling after the dose"
    h = cp.forest.embed([legal_q, medical_q])
    leaf_idx, _ = cp.forest.router.route(h)
    assert leaf_idx[0] == 0 and leaf_idx[1] == 1


def test_prune_survivor_byte_identical(backbone):
    """The lifecycle's NumPy prune (leaf list + router column) works untouched
    on torch-backed leaves, with the survivor proven intact."""
    tr = _trainer(backbone)
    cp = ControlPlane(tr.seed_forest("legal"), "acme", "legal")
    cp.register_tenant("root", "medico")
    cp.graft("root", "medico", "medical", tr.train_fn("medical", cp),
             seed=tr.seed_for("medical"))
    legal_hash = cp.forest.leaves[0].weight_hash()
    assert cp.prune("root", eid=1)                       # survivors intact
    assert len(cp.forest.leaves) == 1
    assert cp.forest.leaves[0].weight_hash() == legal_hash
    assert cp.forest.router.W.shape[1] == 1


# ── the rank ladder ──────────────────────────────────────────────────────────

def test_alpha_over_rank_scaling(backbone):
    """α/r scaling: rank 1 keeps scale 1 (the measured-good regime), bigger
    adapters are damped proportionally — the fix for high-rank instability."""
    assert MiniLMLoRALeaf(backbone, rank=1).scale == 1.0
    assert MiniLMLoRALeaf(backbone, rank=8).scale == 0.125
    assert MiniLMLoRALeaf(backbone, rank=8, alpha=8.0).scale == 1.0
    assert MiniLMLoRALeaf(backbone, rank=0).scale == 1.0   # head-only, unused


def test_rank_ladder_shape():
    assert RANK_STAGE_NAMES == ["seed", "sprout", "sapling", "young-tree", "tree"]
    assert [stage_rank(s) for s in RANK_STAGE_NAMES] == [0, 1, 2, 4, 8]
    assert rank_stage_of(4) == "young-tree" and rank_stage_of(3) is None


def test_adapter_params_scale_with_rank(backbone):
    """Capacity is a cost: params must grow with rank, and the seed must be
    just the head."""
    sizes = [MiniLMLoRALeaf(backbone, rank=r).num_params() for r in (0, 1, 2, 8)]
    assert sizes == sorted(sizes) and sizes[0] < sizes[1]
    head_only = 2 * backbone.dim + 2
    per_rank = sum(fi + fo for fi, fo in backbone.target_shapes.values())
    assert sizes[0] == head_only
    assert sizes[1] == head_only + per_rank


def test_promotion_must_be_earned(fleet):
    """An impossible improvement bar -> rejected, live expert untouched,
    rejection audited."""
    cp, tr = fleet
    g = RankGerminator(cp, tr)
    live_hash = cp.forest.leaves[0].weight_hash()
    res = g.promote("root", eid=0, policy=GerminationPolicy(min_delta=2.0))
    assert not res["accepted"]
    assert res["rank_to"] == res["rank_from"]
    assert cp.forest.leaves[0].weight_hash() == live_hash
    assert cp.audit.entries[-1]["event"] == "growth_promotion_rejected"
    assert cp.audit.verify()


def test_promotion_replaces_only_the_target(fleet):
    """An always-pass gate -> promoted to the next rank, with the OTHER expert
    proven byte-identical through the swap."""
    cp, tr = fleet
    g = RankGerminator(cp, tr)
    other_hash = cp.forest.leaves[1].weight_hash()
    res = g.promote("root", eid=0,
                    policy=GerminationPolicy(min_accuracy=0.0, min_delta=-1.0))
    assert res["accepted"] and res["stage_to"] == "sprout"
    assert res["others_byte_identical"]
    assert cp.forest.leaves[0].rank == 1
    assert cp.forest.leaves[1].weight_hash() == other_hash
    assert cp.audit.entries[-1]["event"] == "growth_promoted"
    row = [r for r in g.report() if r["eid"] == 0][0]
    assert row["stage"] == "sprout"


def test_parsimony_gate_leaves_saturated_experts_alone(fleet):
    """auto_germinate with a target the head-only expert already meets must do
    nothing — most experts should live and die as seeds."""
    cp, tr = fleet
    g = RankGerminator(cp, tr)
    res = g.auto_germinate("root", eid=1, target_acc=0.85)
    assert res["action"] == "saturated"
    assert cp.forest.leaves[1].rank == stage_rank("seed")


def test_save_load_byte_identical(fleet, tmp_path):
    """Persistence without the backbone: every leaf's weight_hash and the
    router's routing must survive a save/load round-trip byte-for-byte —
    including the mixed-rank state left by the promotion tests."""
    cp, tr = fleet
    d = str(tmp_path / "fleet")
    cp.forest.save(d)
    g = type(cp.forest).load(d)
    assert [l.weight_hash() for l in g.leaves] == \
           [l.weight_hash() for l in cp.forest.leaves]
    assert [l.rank for l in g.leaves] == [l.rank for l in cp.forest.leaves]
    assert np.array_equal(g.router.W, cp.forest.router.W)
    texts = ["the arbitration clause was blocked pending urgent review",
             "the MRI scan completed on schedule"]
    out_a, idx_a = cp.forest.predict(texts)
    out_b, idx_b = g.predict(texts)
    assert np.array_equal(idx_a, idx_b) and np.allclose(out_a, out_b)


def test_rbac_gates_promotion(fleet):
    cp, tr = fleet
    cp.add_user("root", "viewer", "auditor")
    g = RankGerminator(cp, tr)
    from das.governance import AccessDenied
    with pytest.raises(AccessDenied):
        g.promote("viewer", eid=0)


# ── the deployment engine on the LoRA backend ────────────────────────────────

SPEC = {
    "client": "acme-lora",
    "backend": "lora-minilm",
    "tenants": [
        {"name": "legalco", "experts": [{"name": "legal"}]},
        {"name": "medico", "experts": [{"name": "medical"}]},
    ],
}


@pytest.fixture(scope="module")
def deployment(backbone):
    from das.platform import deploy
    return deploy(SPEC, secret="test-secret", trainer=_trainer(backbone))


def test_deploy_stands_up_a_lora_fleet(deployment):
    """client.yaml (backend: lora-minilm) -> a live governed transformer
    fleet: same summary, same audit chain, real-text routing."""
    s = deployment.summary()
    assert s["audit_ok"] and len(s["experts"]) == 2
    assert deployment.backend == "lora-minilm"
    r = deployment.route("root", "the arbitration clause triggered a security alert")
    assert r["expert"] == "legal" and r["decision"] in ("local", "escalate")
    r = deployment.route("root", "the MRI scan was blocked pending urgent review")
    assert r["expert"] == "medical"


def test_deploy_grow_and_germinate_live(deployment):
    """grow() grafts a rank-ladder seed through the ControlPlane; the
    parsimony gate then refuses to promote a saturated expert."""
    eid = deployment.grow("root", "medico", "finance", stage="seed")
    leaf = deployment.cp.forest.leaves[-1]
    assert leaf.rank == stage_rank("seed")
    res = deployment.germinate("root", eid, target_acc=0.85)
    assert res["action"] == "saturated"
    assert [r["stage"] for r in deployment.growth_report()].count("seed") >= 1


def test_deploy_honest_backend_limits(deployment):
    """improve() and runtime vector-teacher registration don't silently
    degrade — they say what to use instead."""
    with pytest.raises(NotImplementedError, match="germinate"):
        deployment.improve("root", 0)
    with pytest.raises(NotImplementedError, match="text-lesson bridge"):
        deployment.register_teacher({"kind": "local-vector", "name": "t"})


def test_deploy_save_load_round_trip(deployment, tmp_path):
    """Restart survival: ControlPlane delegates forest persistence to the
    backend; hashes, audit chain, and routing all survive byte-for-byte."""
    from das.platform import Deployment
    d = str(tmp_path / "state")
    deployment.save(d)
    dep2 = Deployment.load(d, SPEC, secret="test-secret")
    assert [l.weight_hash() for l in dep2.cp.forest.leaves] == \
           [l.weight_hash() for l in deployment.cp.forest.leaves]
    assert dep2.verify()["ok"] and dep2.cp.state_matches_audit()
    q = "the arbitration clause triggered a security alert"
    assert dep2.route("root", q)["expert"] == deployment.route("root", q)["expert"]
