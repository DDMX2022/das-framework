"""
das/platform/lora_expert.py
---------------------------
Phase-1 experts as LoRA adapters on a REAL frozen transformer's own attention
layers — the item PLATFORM_PLAN §11 pulled forward from PRODUCT_PLAN Phase 1.

Until now the platform's experts were NumPy scoring heads (honest, but the one
line a reviewer pulls on), and das_torch's ``LoRALeaf`` proved the adapter math
on a toy MLP backbone. This module closes the gap: an expert is now a low-rank
delta on MiniLM's OWN query/value projections (all 6 layers, 12 target modules)
plus a small classification head — growing an expert is real adapter
fine-tuning on real text, and every governance guarantee is unchanged because
the ControlPlane never knew what a leaf was made of:

  * the backbone is ONE shared, frozen ``BertModel`` — adapters attach through
    forward hooks, so no leaf can touch it even by accident (tested);
  * ``MiniLMLoRAForest`` duck-types ``DASForest`` (``leaves`` with
    ``weight_hash()``, a NumPy ``StemRouter``, ``graft``/``predict``), so
    ``ControlPlane``, ``ForestLifecycle``, RBAC, audit, prune and the
    byte-identity proofs run UNMODIFIED over transformer-backed experts;
  * ``MiniLMLoRATrainer`` sits behind the same ``train_fn(forest, idx)`` seam
    as ``SyntheticTrainer`` / ``TeacherTrainer``: teacher generates a text
    corpus -> adapter trains -> policy gates.

Germination here is a RANK ladder instead of a width ladder:

    seed r=0 (head-only) -> sprout r=1 -> sapling r=2 -> young-tree r=4 -> tree r=8

A seed expert trains NO adapter at all — just a head on the frozen embedding —
because on topical text that is already enough (measured, see below), and
"most experts should live and die as seeds" is the whole parsimony point.
Promotion to a higher rank goes through the same earned-capacity gate as
``das.platform.germination`` (accuracy floor + minimum improvement, audited
``growth_promoted`` / ``growth_promotion_rejected``).

HONEST MEASUREMENT (benchmarks/lora_rank_bench.py — CPU, 3 seeds, 400 steps,
n_train=320, train/eval vocabulary disjoint):

  * Easy curriculum (topical routine-vs-risk, lexical label): a head-only
    seed already sits at 1.00 — every LoRA rank is pure cost, and the
    parsimony gate correctly refuses to grow anything.
  * Word order (agent-patient reversal, identical vocabulary per class): the
    frozen mean-pooled embedding is the bottleneck — rank 0 scores 0.71
    while rank 1 scores 1.00. THIS is what re-weighting the encoder's own
    attention buys, and the FIRST rung buys all of it: rank 2 adds nothing
    and rank 8 is actively unstable at the same budget (mean 0.83, worst
    seed 0.47) — over-capacity is not merely cosmetic, it can hurt.
  * Negation XOR valence (interaction label, unigrams balanced): a second
    real rung — rank 0 ≈ 0.90, rank 1 = 1.00, at n_train=64 as well as 320.
  * Two traps the protocol caught, recorded so they aren't re-invented:
    negation PARITY measured easy (~1.00 head-only — counting "not" tokens
    is linear in the pooled embedding), and at 240 steps an under-trained
    rank-8 candidate looked like memorization (train 1.0 / eval below the
    seed) — the gate rejects such a candidate either way, which is the
    point: promotion is granted on demonstrated improvement, not intent.

Requires the ``[hf]`` extra (torch + transformers). All heavy imports are
lazy: importing this module — and ``das.platform`` — stays NumPy-only.
"""
from __future__ import annotations

import hashlib
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from das.routing import StemRouter
from das.training.teachers import stable_seed
from .germination import GerminationPolicy

# stage name -> LoRA rank. Same five stage names as the width ladder; the seed
# is rank 0 = a bare head on the frozen embedding (no adapter at all), because
# capacity is a cost and the frozen features are often already enough.
RANK_STAGES = [
    ("seed", 0),
    ("sprout", 1),
    ("sapling", 2),
    ("young-tree", 4),
    ("tree", 8),
]
RANK_STAGE_NAMES = [name for name, _ in RANK_STAGES]
_RANK_OF = dict(RANK_STAGES)


def stage_rank(stage) -> int:
    """LoRA rank for a stage (name or index)."""
    if isinstance(stage, str):
        if stage not in _RANK_OF:
            raise ValueError(f"no stage {stage!r} (stages: {RANK_STAGE_NAMES})")
        return _RANK_OF[stage]
    if not 0 <= stage < len(RANK_STAGES):
        raise ValueError(f"no stage {stage!r} (stages: {RANK_STAGE_NAMES})")
    return RANK_STAGES[stage][1]


def rank_stage_of(rank: int) -> Optional[str]:
    """Which stage a rank corresponds to, or None for a custom rank."""
    for name, r in RANK_STAGES:
        if r == rank:
            return name
    return None


# ── text lessons ─────────────────────────────────────────────────────────────

@dataclass
class TextLessonBatch:
    """A teacher-produced batch of RAW TEXT lessons for one expert topic.
    The vector-lesson ``LessonBatch`` pre-encodes rows because NumPy leaves eat
    vectors; LoRA experts must see the tokens themselves (the adapter lives
    inside the encoder), so here the texts survive to training."""

    teacher: str
    topic: str
    dataset_version: str
    texts_train: List[str]
    y_train: np.ndarray
    texts_eval: List[str]
    y_eval: np.ndarray
    notes: str = ""

    def summary(self):
        return {
            "teacher": self.teacher,
            "topic": self.topic,
            "dataset_version": self.dataset_version,
            "train_examples": len(self.texts_train),
            "eval_examples": len(self.texts_eval),
            "notes": self.notes,
        }


class TopicRiskTextTeacher:
    """Deterministic offline text teacher — the EASY curriculum. Sentences are
    composed from a per-topic subject vocabulary and shared routine-vs-risk
    predicates ("the lab panel was processed normally" vs "the wire transfer
    triggered a security alert"), so the label is carried by plain lexical
    semantics that a frozen MiniLM embedding + linear head separates cleanly
    (measured: benchmarks/lora_rank_bench.py) — which is exactly what makes it
    the parsimony-gate fixture: a seed saturates, growth is refused.

    Train and eval draw from DISJOINT subject pools, so evaluation never sees
    a training sentence's subject — the earlier curated-16-sentence corpus
    (das_text.DEMO_CORPUS) could not support that split honestly (a head
    memorizes 8 sentences and generalizes at chance; measured, hence this
    teacher). Swap in an ``EndpointLLMTeacher``-style generator for real
    teacher-written corpora behind the same ``generate`` contract."""

    SUBJECTS = {
        "legal": (["the NDA", "the master services agreement",
                   "the indemnity clause", "the licensing addendum",
                   "the settlement draft", "the engagement letter"],
                  ["the arbitration clause", "the renewal amendment",
                   "the non-compete provision", "the IP assignment"]),
        "medical": (["the lab panel", "the vaccination",
                     "the prescription refill", "the post-op check",
                     "the wellness visit", "the blood pressure reading"],
                    ["the MRI scan", "the biopsy result",
                     "the dosage adjustment", "the discharge summary"]),
        "finance": (["the wire transfer", "the expense report",
                     "the payroll run", "the quarterly forecast",
                     "the vendor invoice", "the account reconciliation"],
                    ["the reimbursement claim", "the purchase order",
                     "the margin account", "the tax filing"]),
        "support": (["the password reset", "the plan upgrade",
                     "the data export", "the billing address change",
                     "the login issue", "the mobile app install"],
                    ["the account recovery", "the refund request",
                     "the API key rotation", "the region migration"]),
    }
    _ROUTINE = ["was approved without issues", "completed on schedule",
                "was processed normally", "passed all standard checks",
                "was filed as part of the usual cycle",
                "went through with no discrepancies"]
    _RISK = ["was flagged as potentially fraudulent",
             "failed with a critical error", "triggered a security alert",
             "was blocked pending urgent review",
             "caused a serious outage downstream",
             "violated the compliance policy"]

    def __init__(self, name: str = "topic-risk-teacher", seed: int = 0,
                 subjects: dict = None):
        self.name = name
        self.seed = seed
        self.subjects = subjects or self.SUBJECTS

    def describe(self):
        return {"name": self.name, "kind": "topic-risk-template",
                "topics": sorted(self.subjects),
                "note": "label = risk predicate; train/eval subjects disjoint"}

    def _pools(self, topic):
        if topic in self.subjects:
            return self.subjects[topic]
        # unknown topic: generic subjects, topic named in the sentence so
        # routing still has something topical to hold on to
        return ([f"the {topic} request", f"the {topic} report",
                 f"the {topic} submission", f"the routine {topic} task",
                 f"the {topic} record", f"the {topic} update"],
                [f"the escalated {topic} case", f"the {topic} review",
                 f"the pending {topic} item", f"the {topic} filing"])

    def _sample(self, subjects, n, rng):
        texts, ys = [], []
        for _ in range(n):
            s = subjects[int(rng.integers(0, len(subjects)))]
            risky = int(rng.integers(0, 2))
            preds = self._RISK if risky else self._ROUTINE
            texts.append(f"{s} {preds[int(rng.integers(0, len(preds)))]}")
            ys.append(risky)
        return texts, np.asarray(ys, dtype=int)

    def generate(self, topic, n_train=96, n_eval=48, dataset_version=None):
        train_subj, eval_subj = self._pools(topic)
        rng = np.random.default_rng(stable_seed("topic-risk", self.name,
                                                topic, str(self.seed)))
        texts_tr, y_tr = self._sample(train_subj, n_train, rng)
        texts_ev, y_ev = self._sample(eval_subj, n_eval, rng)
        return TextLessonBatch(
            teacher=self.name, topic=topic,
            dataset_version=dataset_version or f"{self.name}-v1",
            texts_train=texts_tr, y_train=y_tr,
            texts_eval=texts_ev, y_eval=y_ev,
            notes="topical routine-vs-risk template lessons; "
                  "train/eval subjects disjoint",
        )


_ARTIFACTS_TRAIN = ["the wire transfer", "the refund request", "the deployment",
                    "the access request", "the shipment", "the invoice",
                    "the backup job", "the password reset"]
_ARTIFACTS_EVAL = ["the migration", "the purchase order",
                   "the firmware update", "the audit export"]


class WordOrderCurriculumTeacher:
    """The HARD curriculum: agent-patient role reversal. Label 1 iff the roles
    are anomalous — the artifact acts on the person:

        "the auditor cancelled the shipment"   -> 0
        "the shipment cancelled the auditor"   -> 1

    Both classes contain exactly the same words, so no bag-of-words statistic
    carries the label — it lives in word ORDER, which mean pooling discards
    unless the encoder's attention is re-weighted to keep it. This is the text
    analog of germination_bench's XOR curriculum, and it is where the rank
    ladder measurably earns its first rung (head-only 0.71 vs rank 1 = 1.00,
    while rank 8 destabilizes — benchmarks/lora_rank_bench.py). Artifacts are
    split disjointly between train and eval.

    A negation-PARITY curriculum was tried first and measured EASY (head-only
    ~1.00): counting "not" tokens is a linear function of the pooled
    embedding, so parity has a lexical shortcut after all. Kept out, noted
    here so it isn't re-invented."""

    _ACTORS = ["the auditor", "the operator", "the reviewer",
               "the administrator", "the compliance officer"]
    _VERBS = ["cancelled", "suspended", "flagged"]

    def __init__(self, name: str = "word-order-teacher", seed: int = 0):
        self.name = name
        self.seed = seed

    def describe(self):
        return {"name": self.name, "kind": "agent-patient-order",
                "note": "label = role reversal; identical vocabulary per class"}

    def _sample(self, artifacts, n, rng):
        texts, ys = [], []
        for _ in range(n):
            art = artifacts[int(rng.integers(0, len(artifacts)))]
            act = self._ACTORS[int(rng.integers(0, len(self._ACTORS)))]
            v = self._VERBS[int(rng.integers(0, len(self._VERBS)))]
            flip = int(rng.integers(0, 2))
            texts.append(f"{art} {v} {act}" if flip else f"{act} {v} {art}")
            ys.append(flip)
        return texts, np.asarray(ys, dtype=int)

    def generate(self, topic, n_train=96, n_eval=48, dataset_version=None):
        rng = np.random.default_rng(stable_seed("word-order", self.name, topic,
                                                str(self.seed)))
        texts_tr, y_tr = self._sample(_ARTIFACTS_TRAIN, n_train, rng)
        texts_ev, y_ev = self._sample(_ARTIFACTS_EVAL, n_eval, rng)
        return TextLessonBatch(
            teacher=self.name, topic=topic,
            dataset_version=dataset_version or f"{self.name}-v1",
            texts_train=texts_tr, y_train=y_tr,
            texts_eval=texts_ev, y_eval=y_ev,
            notes="agent-patient order curriculum; train/eval artifacts disjoint",
        )


class XorNegationTeacher:
    """The INTERACTION curriculum: negation XOR predicate valence. Label 1 iff
    the net outcome is bad — "was not approved" and "was delayed" are 1,
    "was approved" and "was not delayed" are 0. Every unigram ("not", each
    predicate) appears in both classes at the same rate, so the label is a
    pure interaction term.

    Measured (benchmarks/lora_rank_bench.py): a milder compositional rung
    than word order — the frozen head reaches ~0.90 (contextual embeddings
    partially bind negation to its predicate even under mean pooling), and
    rank 1 closes it to 1.00, at n_train=64 as well as 320. An earlier
    shorter run (240 steps) made adapters LOOK like memorizers (train 1.0,
    eval below the head); at 400 steps they generalize — the gate can't tell
    "won't learn" from "wasn't trained enough", and doesn't need to: either
    way the candidate failed to demonstrate improvement and is refused."""

    _GOOD = ["approved", "completed", "verified"]
    _BAD = ["delayed", "rejected", "flagged"]

    def __init__(self, name: str = "xor-negation-teacher", seed: int = 0):
        self.name = name
        self.seed = seed

    def describe(self):
        return {"name": self.name, "kind": "xor-negation",
                "note": "label = negation XOR valence; unigrams balanced"}

    def _sample(self, subjects, n, rng):
        texts, ys = [], []
        for _ in range(n):
            s = subjects[int(rng.integers(0, len(subjects)))]
            neg = int(rng.integers(0, 2))
            bad = int(rng.integers(0, 2))
            preds = self._BAD if bad else self._GOOD
            p = preds[int(rng.integers(0, len(preds)))]
            texts.append(f"{s} was {'not ' if neg else ''}{p}")
            ys.append(neg ^ bad)
        return texts, np.asarray(ys, dtype=int)

    def generate(self, topic, n_train=96, n_eval=48, dataset_version=None):
        rng = np.random.default_rng(stable_seed("xor-negation", self.name,
                                                topic, str(self.seed)))
        texts_tr, y_tr = self._sample(_ARTIFACTS_TRAIN, n_train, rng)
        texts_ev, y_ev = self._sample(_ARTIFACTS_EVAL, n_eval, rng)
        return TextLessonBatch(
            teacher=self.name, topic=topic,
            dataset_version=dataset_version or f"{self.name}-v1",
            texts_train=texts_tr, y_train=y_tr,
            texts_eval=texts_ev, y_eval=y_ev,
            notes="negation-xor-valence curriculum; train/eval subjects disjoint",
        )


# ── the shared frozen backbone ───────────────────────────────────────────────

_BACKBONE_CACHE: Dict[str, "MiniLMLoRABackbone"] = {}


class MiniLMLoRABackbone:
    """ONE frozen MiniLM (BertModel) shared by every leaf in a fleet, with
    LoRA attachment points on each layer's attention ``query`` and ``value``
    projections (the standard LoRA target set).

    Adapters attach through forward hooks: while a leaf is active (via the
    ``adapter`` context manager) each target Linear's output gains
    ``x @ Aᵀ @ Bᵀ`` from THAT leaf's matrices. The backbone's own parameters
    have ``requires_grad=False`` from the moment it loads and are never
    toggled — the same isolation stance as das_torch.LoRALeaf, on the real
    encoder."""

    DEFAULT = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, model_name: str = None, device: str = "cpu",
                 max_length: int = 64):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.model_name = model_name or self.DEFAULT
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.dim = int(self.model.config.hidden_size)
        # discover the attention q/v Linears (BERT names; *_proj as fallback)
        self.targets: List[str] = []
        for name, mod in self.model.named_modules():
            if isinstance(mod, torch.nn.Linear) and (
                    name.endswith(".query") or name.endswith(".value")
                    or name.endswith(".q_proj") or name.endswith(".v_proj")):
                self.targets.append(name)
                mod.register_forward_hook(self._make_hook(name))
        if not self.targets:
            raise RuntimeError(f"no attention query/value Linears found in "
                               f"{self.model_name} — cannot attach LoRA")
        self.target_shapes = {
            n: (m.in_features, m.out_features)
            for n, m in self.model.named_modules() if n in set(self.targets)
        }
        self._active: Dict[str, tuple] = {}

    @classmethod
    def cached(cls, model_name: str = None, device: str = "cpu"):
        """Process-level cache — the fleet, the trainer, the tests all share
        one ~90 MB model instead of loading it per call."""
        key = f"{model_name or cls.DEFAULT}@{device}"
        if key not in _BACKBONE_CACHE:
            _BACKBONE_CACHE[key] = cls(model_name, device)
        return _BACKBONE_CACHE[key]

    def _make_hook(self, name):
        def hook(_mod, inputs, output):
            ab = self._active.get(name)
            if ab is None:
                return output
            A, B = ab
            return output + inputs[0] @ A.T @ B.T
        return hook

    @contextmanager
    def adapter(self, leaf):
        """Attach ``leaf``'s LoRA matrices for the duration of the block.
        Single-active-adapter by design: the forest routes top-1, so exactly
        one expert's delta is ever live inside the encoder at a time."""
        if self._active:
            raise RuntimeError("an adapter is already active on this backbone")
        self._active = leaf.lora
        try:
            yield
        finally:
            self._active = {}

    def pool(self, texts, grad: bool = False):
        """texts -> L2-normalised mean-pooled embeddings [n, dim] (torch).
        ``grad=True`` keeps the graph so an ACTIVE adapter's A/B receive
        gradients; the backbone's own weights are requires_grad=False either
        way and receive none."""
        import torch
        import torch.nn.functional as F
        enc = self.tokenizer(list(texts), padding=True, truncation=True,
                             max_length=self.max_length, return_tensors="pt"
                             ).to(self.device)
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            out = self.model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            return F.normalize(emb, p=2, dim=1)

    def embed(self, texts) -> np.ndarray:
        """Routing-geometry embeddings: NumPy, no adapter, no grad — the same
        frozen mean-pooled MiniLM space a ``MiniLMContextSource`` produces, so
        connector queries and forest routing share one geometry."""
        if self._active:
            raise RuntimeError("routing embeddings must come from the bare "
                               "frozen encoder, not through an adapter")
        return self.pool(texts, grad=False).cpu().numpy().astype(float)


# ── the expert: adapter + head ───────────────────────────────────────────────

class MiniLMLoRALeaf:
    """One expert: LoRA matrices for every backbone target module (rank may be
    0 = no adapter, the seed stage) + a linear head over the pooled embedding.
    Holds plain torch tensors — not an nn.Module — so this file imports without
    torch and the trainable surface is exactly ``adapter_params()``, nothing
    hidden. B starts at zero, so a fresh leaf of ANY rank is byte-for-byte the
    frozen encoder + an untrained head."""

    leaf_type = "minilm-lora"

    def __init__(self, backbone: MiniLMLoRABackbone, rank: int = 0,
                 out_dim: int = 2, seed: int = 0):
        import torch
        if rank < 0:
            raise ValueError("rank must be >= 0")
        self.backbone = backbone            # shared, frozen — never trained here
        self.rank = int(rank)
        self.out_dim = int(out_dim)
        self.seed = int(seed)
        g = torch.Generator().manual_seed(seed)
        # head first, so same-seed leaves share a head regardless of rank —
        # that keeps "a fresh adapter is a no-op" testable byte-for-byte
        d = backbone.dim
        self.head_W = (torch.randn(out_dim, d, generator=g) * (1.0 / d ** 0.5)
                       ).requires_grad_(True)
        self.head_b = torch.zeros(out_dim).requires_grad_(True)
        self.lora: Dict[str, tuple] = {}
        if rank > 0:
            for name in backbone.targets:
                fan_in, fan_out = backbone.target_shapes[name]
                A = (torch.randn(rank, fan_in, generator=g) * 0.01
                     ).requires_grad_(True)
                B = torch.zeros(fan_out, rank).requires_grad_(True)
                self.lora[name] = (A, B)
        self.frozen = False

    # ── surface the platform relies on ───────────────────────────────
    @property
    def stage(self) -> Optional[str]:
        return rank_stage_of(self.rank)

    def adapter_params(self):
        """The ONLY trainable tensors — scopes the optimizer so the backbone
        cannot be touched even by accident."""
        ps = [self.head_W, self.head_b]
        for A, B in self.lora.values():
            ps.extend([A, B])
        return ps

    def num_params(self) -> int:
        return sum(int(p.numel()) for p in self.adapter_params())

    def weight_hash(self) -> str:
        """SHA-256 over every adapter/head tensor (name + shape + bytes) —
        the byte-identity fingerprint ControlPlane audits with."""
        h = hashlib.sha256()
        named = [("head_W", self.head_W), ("head_b", self.head_b)]
        for name in sorted(self.lora):
            A, B = self.lora[name]
            named += [(name + ".A", A), (name + ".B", B)]
        for name, t in named:
            h.update(name.encode())
            h.update(str(tuple(t.shape)).encode())
            h.update(t.detach().cpu().numpy().astype(np.float32).tobytes())
        return h.hexdigest()

    def freeze(self):
        for p in self.adapter_params():
            p.requires_grad_(False)
        self.frozen = True

    def unfreeze(self):
        for p in self.adapter_params():
            p.requires_grad_(True)
        self.frozen = False

    # ── forward ──────────────────────────────────────────────────────
    def forward(self, texts, grad: bool = False):
        """texts -> logits [n, out_dim] (torch). The pooled embedding runs
        through the backbone WITH this leaf's adapter attached (rank 0 = the
        bare frozen encoder)."""
        if self.rank > 0:
            with self.backbone.adapter(self):
                h = self.backbone.pool(texts, grad=grad)
        else:
            h = self.backbone.pool(texts, grad=grad)
        return h @ self.head_W.T + self.head_b

    def predict(self, texts) -> np.ndarray:
        """texts -> logits as NumPy (the forest's inference path)."""
        return self.forward(texts, grad=False).detach().cpu().numpy()

    def accuracy(self, texts, y) -> float:
        return float((self.predict(texts).argmax(1) == np.asarray(y)).mean())


# ── the forest ───────────────────────────────────────────────────────────────

class MiniLMLoRAForest:
    """A fleet of LoRA experts on ONE shared frozen MiniLM, duck-typed to
    ``DASForest``: ``leaves`` is a plain list of objects with ``weight_hash()``,
    ``router`` is the SAME NumPy ``StemRouter`` the whole stack already uses,
    and ``graft``/``predict`` match the lifecycle contract. ``ControlPlane``,
    ``ForestLifecycle``, RBAC and the audit chain therefore govern transformer
    experts without a line of change — the backend-agnostic seam, kept.

    ``predict`` takes TEXTS (the leaf needs tokens, not vectors): routing runs
    on the bare frozen embedding, then only the chosen leaf's adapter runs."""

    def __init__(self, backbone: MiniLMLoRABackbone = None, out_dim: int = 2,
                 num_leaves: int = 1, stage="seed", seed: int = 0):
        self.backbone = backbone or MiniLMLoRABackbone.cached()
        self.d_model = self.backbone.dim
        self.out_dim = int(out_dim)
        self.default_rank = stage if isinstance(stage, int) else stage_rank(stage)
        self.router = StemRouter(self.d_model, num_leaves, seed=seed)
        self.leaves = [MiniLMLoRALeaf(self.backbone, rank=self.default_rank,
                                      out_dim=out_dim, seed=seed + 1 + i)
                       for i in range(num_leaves)]

    def embed(self, texts) -> np.ndarray:
        return self.backbone.embed(texts)

    def predict(self, texts):
        """Route each text on the frozen embedding, run only the chosen
        expert's adapter. Returns (logits [n, out_dim], leaf_idx) as NumPy."""
        texts = [texts] if isinstance(texts, str) else list(texts)
        h = self.embed(texts)
        leaf_idx, _ = self.router.route(h)
        out = np.zeros((len(texts), self.out_dim))
        for i, leaf in enumerate(self.leaves):
            mask = leaf_idx == i
            if mask.any():
                out[mask] = leaf.predict([texts[j] for j in np.where(mask)[0]])
        return out, leaf_idx

    def graft(self, new_leaf_dims=None, seed=99):
        """Add a fresh expert + router slot. ``new_leaf_dims`` is this
        backend's capacity knob: a rank (int), a stage name, or None for the
        forest default — the ControlPlane passes it straight through, which is
        why the seam survives the backend swap."""
        rank = self.default_rank
        if isinstance(new_leaf_dims, str):
            rank = stage_rank(new_leaf_dims)
        elif new_leaf_dims is not None:
            rank = int(new_leaf_dims)
        self.leaves.append(MiniLMLoRALeaf(self.backbone, rank=rank,
                                          out_dim=self.out_dim, seed=seed))
        self.router.expand(seed=seed)
        return len(self.leaves) - 1

    def freeze_all_leaves(self):
        for leaf in self.leaves:
            leaf.freeze()

    def leaf_hashes(self):
        return {i: leaf.weight_hash() for i, leaf in enumerate(self.leaves)}


# ── the train_fn seam ────────────────────────────────────────────────────────

class MiniLMLoRATrainer:
    """Teacher-driven ``train_fn`` factory for LoRA experts — the same seam
    ``SyntheticTrainer`` and ``TeacherTrainer`` implement, so
    ``ControlPlane.graft(actor, tenant, name, trainer.train_fn(name, cp))``
    is the whole integration: teacher generates a text corpus, the adapter
    (+ head) trains on it in isolation, the router retrains over every
    expert's real lesson embeddings."""

    def __init__(self, backbone: MiniLMLoRABackbone = None,
                 teacher=None, out_dim: int = 2, stage="seed",
                 steps: int = 120, lr: float = 1e-2, batch: int = 16,
                 n_train: int = 96, n_eval: int = 48,
                 router_steps: int = 300, router_lr: float = 0.15):
        self.backbone = backbone or MiniLMLoRABackbone.cached()
        self.default_teacher = teacher or TopicRiskTextTeacher()
        self.out_dim = out_dim
        self.stage = stage
        self.steps = steps
        self.lr = lr
        self.batch = batch
        self.n_train = n_train
        self.n_eval = n_eval
        self.router_steps = router_steps
        self.router_lr = router_lr
        self.lessons: Dict[str, TextLessonBatch] = {}
        self.reports: Dict[str, dict] = {}
        self._emb_cache: Dict[str, np.ndarray] = {}

    @property
    def d_model(self) -> int:
        return self.backbone.dim

    def seed_for(self, name: str):
        return stable_seed("lora-leaf", name)

    # ── leaf fitting ─────────────────────────────────────────────────
    def fit_leaf(self, leaf: MiniLMLoRALeaf, batch: TextLessonBatch,
                 seed: int = 0):
        """Adam over ``adapter_params()`` ONLY, cross-entropy on the lesson
        texts. Deterministic given (leaf init, lessons, seed)."""
        import torch
        opt = torch.optim.Adam(leaf.adapter_params(), lr=self.lr)
        rng = np.random.default_rng(seed)
        y = torch.tensor(batch.y_train, dtype=torch.long,
                         device=self.backbone.device)
        n = len(batch.texts_train)
        leaf.unfreeze()
        for _ in range(self.steps):
            i = rng.integers(0, n, min(self.batch, n))
            logits = leaf.forward([batch.texts_train[j] for j in i], grad=True)
            loss = torch.nn.functional.cross_entropy(logits, y[i])
            opt.zero_grad()
            loss.backward()
            opt.step()
        leaf.freeze()

    def lesson_embeddings(self, name: str) -> np.ndarray:
        """Frozen-encoder embeddings of an expert's training texts (cached) —
        the router's view of that expert's real distribution."""
        if name not in self._emb_cache:
            self._emb_cache[name] = self.backbone.embed(
                self.lessons[name].texts_train)
        return self._emb_cache[name]

    def center(self, name: str) -> np.ndarray:
        """Where a query for this expert embeds — the empirical center of its
        lessons (connector-compatible surface)."""
        return self.lesson_embeddings(name).mean(axis=0)

    def _train_router(self, forest, names):
        blocks = [self.lesson_embeddings(n) for n in names]
        Xr = np.vstack(blocks)
        dr = np.concatenate([np.full(len(b), s, dtype=int)
                             for s, b in enumerate(blocks)])
        rng = np.random.default_rng(stable_seed("lora-router", *names))
        for _ in range(self.router_steps):
            i = rng.integers(0, len(Xr), min(64, len(Xr)))
            forest.router.train_step(Xr[i], dr[i], lr=self.router_lr)

    def _grow(self, name: str, leaf: MiniLMLoRALeaf, teacher) -> TextLessonBatch:
        t = teacher or self.default_teacher
        batch = t.generate(name, n_train=self.n_train, n_eval=self.n_eval)
        self.lessons[name] = batch
        self._emb_cache.pop(name, None)
        self.fit_leaf(leaf, batch, seed=stable_seed("lora-fit", name))
        self.reports[name] = {
            "teacher": batch.teacher,
            "dataset_version": batch.dataset_version,
            "train_examples": len(batch.texts_train),
            "eval_accuracy": leaf.accuracy(batch.texts_eval, batch.y_eval),
            "stage": leaf.stage, "rank": leaf.rank,
            "adapter_params": leaf.num_params(),
            "notes": batch.notes,
        }
        return batch

    # ── seed + graft hooks (the seam) ────────────────────────────────
    def seed_forest(self, seed_name: str, teacher=None) -> MiniLMLoRAForest:
        """One-leaf forest with the seed expert teacher-trained — what the
        ControlPlane is constructed over."""
        forest = MiniLMLoRAForest(self.backbone, out_dim=self.out_dim,
                                  num_leaves=1, stage=self.stage,
                                  seed=self.seed_for(seed_name))
        self._grow(seed_name, forest.leaves[0], teacher)
        self._train_router(forest, [seed_name])
        return forest

    def train_fn(self, name: str, cp, teacher=None):
        """The graft callback: train the new leaf on teacher lessons, retrain
        the router over every expert's real lesson embeddings."""
        def _fn(forest, idx):
            self._grow(name, forest.leaves[idx], teacher)
            names = [r["name"] for r in cp.experts] + [name]
            self._train_router(forest, names)
        return _fn


# ── germination: the rank ladder ─────────────────────────────────────────────

class RankGerminator:
    """Stage promotion for LoRA experts — the Growing-Child quarantine at a new
    RANK instead of a new width. A candidate adapter at the next rank trains
    from scratch on fresh teacher lessons and must EARN its parameters through
    the same ``GerminationPolicy`` gate (accuracy floor + minimum improvement);
    accepted candidates replace the live expert, rejects leave it untouched,
    and either way the event is audited with before/after params and the
    proof that every other expert stayed byte-identical.

    ``auto_germinate`` keeps the parsimony rule: an expert that already meets
    the target at its current rank is left alone — on topical text that is
    most experts, at rank 0 (measured: benchmarks/lora_rank_bench.py)."""

    def __init__(self, cp, trainer: MiniLMLoRATrainer):
        self.cp = cp
        self.trainer = trainer

    def report(self) -> list:
        rows = []
        for i, rec in enumerate(self.cp.experts):
            leaf = self.cp.forest.leaves[i]
            rows.append({
                "eid": rec["eid"], "name": rec["name"], "tenant": rec["tenant"],
                "stage": leaf.stage, "rank": leaf.rank,
                "params": leaf.num_params(),
            })
        return rows

    def promote(self, actor: str, eid: int, teacher=None, to_stage=None,
                policy: Optional[GerminationPolicy] = None) -> dict:
        policy = policy or GerminationPolicy()
        idx, rec = self.cp._find(int(eid))
        self.cp._check(actor, "graft", rec["tenant"])
        live = self.cp.forest.leaves[idx]

        current = live.stage
        cur_i = RANK_STAGE_NAMES.index(current) if current else None
        if to_stage is None:
            if cur_i is None:
                raise ValueError(f"expert '{rec['name']}' has custom rank "
                                 f"{live.rank}; pass to_stage explicitly")
            if cur_i >= len(RANK_STAGES) - 1:
                raise ValueError(f"'{rec['name']}' is already a full tree")
            to_stage = RANK_STAGE_NAMES[cur_i + 1]
        to_rank = stage_rank(to_stage)
        if to_rank <= live.rank:
            raise ValueError(f"'{to_stage}' (rank {to_rank}) is not a "
                             f"promotion from rank {live.rank}")

        t = teacher or self.trainer.default_teacher
        lessons = t.generate(rec["name"], n_train=self.trainer.n_train,
                             n_eval=self.trainer.n_eval)
        live_acc = live.accuracy(lessons.texts_eval, lessons.y_eval)
        candidate = MiniLMLoRALeaf(
            self.trainer.backbone, rank=to_rank, out_dim=live.out_dim,
            seed=stable_seed("rank-germinate", rec["name"], to_stage))
        self.trainer.fit_leaf(candidate, lessons,
                              seed=stable_seed("rank-fit", rec["name"], to_stage))
        cand_acc = candidate.accuracy(lessons.texts_eval, lessons.y_eval)
        delta = cand_acc - live_acc

        before_hashes = self.cp._hashes()
        target_key = f"eid{rec['eid']}"
        reasons = []
        if cand_acc < policy.min_accuracy:
            reasons.append(f"candidate accuracy {cand_acc:.3f} below floor "
                           f"{policy.min_accuracy:.3f}")
        if delta < policy.min_delta:
            reasons.append(f"improvement {delta:+.3f} below the "
                           f"{policy.min_delta:+.3f} a promotion must earn")
        accepted = not reasons
        if accepted:
            self.cp.forest.leaves[idx] = candidate
        after_hashes = self.cp._hashes()
        others_intact = all(after_hashes.get(k) == v
                            for k, v in before_hashes.items() if k != target_key)

        event = "growth_promoted" if accepted else "growth_promotion_rejected"
        result = {
            "accepted": accepted,
            "reason": "accepted" if accepted else "; ".join(reasons),
            "eid": rec["eid"], "expert": rec["name"], "tenant": rec["tenant"],
            "teacher": lessons.teacher,
            "stage_from": current, "stage_to": to_stage if accepted else current,
            "attempted_stage": to_stage,
            "rank_from": live.rank,
            "rank_to": to_rank if accepted else live.rank,
            "params_before": live.num_params(),
            "params_after": (candidate if accepted else live).num_params(),
            "accuracy_before": round(float(live_acc), 6),
            "accuracy_after": round(float(cand_acc if accepted else live_acc), 6),
            "candidate_accuracy": round(float(cand_acc), 6),
            "delta": round(float(delta), 6),
            "others_byte_identical": bool(others_intact),
        }
        self.cp.audit.append(
            event,
            (f"{actor} rank germination {current or live.rank}->{to_stage} "
             f"(r{live.rank}->r{to_rank}) for eid={rec['eid']} "
             f"('{rec['name']}', tenant '{rec['tenant']}') via teacher "
             f"'{lessons.teacher}'; acc {live_acc:.3f}->{cand_acc:.3f} "
             f"(delta {delta:+.3f}), params {result['params_before']}->"
             f"{candidate.num_params()}; result: {result['reason']}"),
            payload=self.cp._hashes(),
        )
        return result

    def auto_germinate(self, actor: str, eid: int, teacher=None,
                       target_acc: float = 0.85,
                       policy: Optional[GerminationPolicy] = None) -> dict:
        """The parsimony gate: promote ONLY if the live expert demonstrably
        cannot meet the target at its current rank."""
        idx, rec = self.cp._find(int(eid))
        self.cp._check(actor, "graft", rec["tenant"])
        live = self.cp.forest.leaves[idx]
        t = teacher or self.trainer.default_teacher
        lessons = t.generate(rec["name"], n_train=self.trainer.n_train,
                             n_eval=self.trainer.n_eval)
        live_acc = live.accuracy(lessons.texts_eval, lessons.y_eval)
        if live_acc >= target_acc:
            return {"action": "saturated", "eid": rec["eid"],
                    "expert": rec["name"], "stage": live.stage,
                    "rank": live.rank,
                    "accuracy": round(float(live_acc), 6),
                    "target": target_acc, "params": live.num_params(),
                    "note": "meets target at current rank — no growth needed"}
        if live.stage == RANK_STAGE_NAMES[-1]:
            return {"action": "at_capacity", "eid": rec["eid"],
                    "expert": rec["name"], "stage": live.stage,
                    "rank": live.rank,
                    "accuracy": round(float(live_acc), 6),
                    "target": target_acc, "params": live.num_params(),
                    "note": "below target but already a full tree — a better "
                            "teacher/curriculum is needed, not more rank"}
        result = self.promote(actor, eid, teacher=teacher, policy=policy)
        result["action"] = ("promoted" if result["accepted"]
                            else "promotion_rejected")
        return result

    def sweep(self, actor: str, teacher=None, target_acc: float = 0.85,
              policy: Optional[GerminationPolicy] = None) -> dict:
        """Plateau monitoring over every actor-visible expert, one audited
        summary — the fleet-level Growing-Child loop at the rank ladder."""
        self.cp._check(actor, "graft")
        results = [self.auto_germinate(actor, rec["eid"], teacher=teacher,
                                       target_acc=target_acc, policy=policy)
                   for rec in self.cp.list_experts(actor)]
        counts = {}
        for r in results:
            counts[r["action"]] = counts.get(r["action"], 0) + 1
        self.cp.audit.append(
            "germination_sweep",
            (f"{actor} rank-germination sweep: attempted={len(results)}, "
             + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))),
            payload=self.cp._hashes(),
        )
        return {"attempted": len(results), **counts, "results": results}
