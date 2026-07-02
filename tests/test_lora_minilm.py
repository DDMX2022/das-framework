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


def test_rbac_gates_promotion(fleet):
    cp, tr = fleet
    cp.add_user("root", "viewer", "auditor")
    g = RankGerminator(cp, tr)
    from das.governance import AccessDenied
    with pytest.raises(AccessDenied):
        g.promote("viewer", eid=0)
