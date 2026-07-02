"""
lora_rank_bench.py
------------------
Does the LoRA RANK ladder measure anything real on the actual transformer?
Same protocol as germination_bench.py (every stage, averaged over seeds), but
the expert is a LoRA adapter on MiniLM's own attention query/value projections
(das.platform.lora_expert), the lessons are text, and there are THREE
curricula because the honest answer has three parts:

  * EASY (TopicRiskTextTeacher — topical routine-vs-risk, label carried by
    lexical semantics): the frozen embedding + a head should saturate at
    rank 0, making every adapter parameter pure cost — the regime where
    auto_germinate's parsimony gate refuses to grow.
  * HARD (WordOrderCurriculumTeacher — agent-patient role reversal, both
    classes use identical words): the label lives in word ORDER, which mean
    pooling discards — the regime where an adapter should EARN its rank by
    re-weighting the encoder's own attention.
  * INTERACTION (XorNegationTeacher — negation XOR valence, unigrams
    balanced): a pure interaction term, the milder compositional axis.

The table also answers the gate question directly: given each stage's
accuracy, would GerminationPolicy (floor 0.55, min improvement +0.05 over the
rank-0 seed) promote?

Findings from the 2026-07-02 run (CPU, 3 seeds, n_train=320, steps=400,
disjoint train/eval vocabulary; recorded in das/platform/lora_expert.py too):
easy saturates at rank 0 (1.00 — growth refused); word order is the measured
rung (rank 0 = 0.71 -> rank 1 = 1.00) and OVER-capacity destabilizes (rank 8:
mean 0.83, worst seed 0.47 at the same budget); xor-negation is a second real
rung (0.90 -> 1.00, and also at n_train=64).

Two traps the protocol caught, noted so they aren't re-invented: negation
PARITY measured easy (~1.00 head-only — counting "not" tokens is linear in
the pooled embedding), and at 240 steps an under-trained rank-8 candidate
looked like a memorizer (train 1.0, eval below the seed) — the gate rejects
that candidate either way, which is the protection: promotion is granted on
demonstrated improvement, not on intent.

Needs the [hf] extra + cached MiniLM. Deterministic per seed. ~15 min on CPU.
"""
import time

import numpy as np

from das.platform.germination import GerminationPolicy
from das.platform.lora_expert import (
    RANK_STAGES,
    MiniLMLoRABackbone,
    MiniLMLoRALeaf,
    MiniLMLoRATrainer,
    TopicRiskTextTeacher,
    WordOrderCurriculumTeacher,
    XorNegationTeacher,
)

SEEDS = 3
STEPS = 400
N_TRAIN, N_EVAL = 320, 120
TOPIC = "finance"
POLICY = GerminationPolicy()          # floor 0.55, min improvement +0.05


def run(teacher, backbone):
    trainer = MiniLMLoRATrainer(backbone, steps=STEPS)
    lessons = teacher.generate(TOPIC, n_train=N_TRAIN, n_eval=N_EVAL)
    rows = []
    for stage, rank in RANK_STAGES:
        evals, trains = [], []
        t0 = time.time()
        for s in range(SEEDS):
            leaf = MiniLMLoRALeaf(backbone, rank=rank, out_dim=2, seed=100 + s)
            trainer.fit_leaf(leaf, lessons, seed=s)
            evals.append(leaf.accuracy(lessons.texts_eval, lessons.y_eval))
            trains.append(leaf.accuracy(lessons.texts_train, lessons.y_train))
        rows.append({
            "stage": stage, "rank": rank,
            "params": MiniLMLoRALeaf(backbone, rank=rank).num_params(),
            "eval_mean": float(np.mean(evals)),
            "eval_min": float(np.min(evals)),
            "train_mean": float(np.mean(trains)),
            "secs_per_fit": (time.time() - t0) / SEEDS,
        })
    seed_acc = rows[0]["eval_mean"]
    for r in rows:
        r["gate"] = ("-" if r["rank"] == 0 else
                     "promote" if (r["eval_mean"] >= POLICY.min_accuracy and
                                   r["eval_mean"] - seed_acc >= POLICY.min_delta)
                     else "reject")
    return rows


def show(title, rows):
    print(f"\n{title}")
    print(f"{'stage':<12}{'rank':>5}{'params':>9}{'eval':>8}{'min':>7}"
          f"{'train':>8}{'gate':>10}{'s/fit':>7}")
    for r in rows:
        print(f"{r['stage']:<12}{r['rank']:>5}{r['params']:>9}"
              f"{r['eval_mean']:>8.3f}{r['eval_min']:>7.2f}"
              f"{r['train_mean']:>8.3f}{r['gate']:>10}{r['secs_per_fit']:>7.1f}")


def main():
    backbone = MiniLMLoRABackbone.cached()
    easy = run(TopicRiskTextTeacher(), backbone)
    show("EASY — topical routine-vs-risk (lexical label):", easy)
    hard = run(WordOrderCurriculumTeacher(), backbone)
    show("HARD — agent-patient word order (identical vocabulary):", hard)
    adv = run(XorNegationTeacher(), backbone)
    show("INTERACTION — negation XOR valence (unigrams balanced):", adv)

    print("\nreading:")
    print(f"  easy: rank 0 scores {easy[0]['eval_mean']:.3f} — adapters are "
          f"pure cost; the parsimony gate refuses growth.")
    first, last = hard[1], hard[-1]
    print(f"  hard: rank 0 = {hard[0]['eval_mean']:.3f}, rank "
          f"{first['rank']} = {first['eval_mean']:.3f} — word order is what "
          f"re-weighting the encoder's own attention buys; rank "
          f"{last['rank']} = {last['eval_mean']:.3f} (min "
          f"{last['eval_min']:.2f}) shows over-capacity destabilizing.")
    print(f"  interaction: rank 0 = {adv[0]['eval_mean']:.3f}, rank "
          f"{adv[1]['rank']} = {adv[1]['eval_mean']:.3f} — a second, milder "
          f"compositional rung.")


if __name__ == "__main__":
    main()
