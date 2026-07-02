"""
germination_bench.py
--------------------
Does the germination ladder measure anything real? Two curricula, all five
stages, averaged over seeds:

  * HARD (XOR-of-half-spaces on cluster deviations, NonlinearVectorTeacher):
    the regime where capacity should EARN its parameters — a seed should sit
    near chance and each stage should climb.
  * EASY (linear rule, AlignedVectorTeacher): the regime where a seed should
    saturate immediately — extra capacity buys ~nothing, and auto_germinate's
    parsimony gate should refuse to grow.

What this does and doesn't show:
  ✓ the capacity LADDER is real on hard curricula (measured gradient), and
    unnecessary on easy ones — which is exactly the promote/stay-small policy.
  ✗ the Fibonacci-ness of the widths remains cosmetic (leaf_shapes_bench:
    width schedules within noise of each other) — the ladder is the feature,
    the sequence is the aesthetic.

Pure NumPy, deterministic.
"""
import numpy as np

from das.functional import FibonacciLeaf
from das.platform import SyntheticTrainer
from das.platform.germination import STAGES, leaf_params, stage_dims
from das.platform.teacher_trainer import AlignedVectorTeacher, NonlinearVectorTeacher
from das.training.evaluator import evaluate_leaf, train_leaf

D, OUT = 18, 2
STEPS, LR = 1200, 0.08
SEEDS = 3
TOPIC = "hard-domain"


def run(teacher):
    lessons = teacher.generate(TOPIC, n_train=400, n_eval=300)
    rows = []
    for name, _hidden in STAGES:
        dims = stage_dims(D, OUT, name)
        accs = []
        for s in range(SEEDS):
            leaf = FibonacciLeaf(dims, seed=s)
            train_leaf(leaf, lessons.X_train, lessons.y_train,
                       steps=STEPS, lr=LR, seed=s)
            accs.append(evaluate_leaf(leaf, lessons.X_eval, lessons.y_eval))
        rows.append((name, dims, leaf_params(leaf),
                     float(np.mean(accs)), float(np.std(accs))))
    return rows


def main():
    base = SyntheticTrainer(D, [D, 13, 8, OUT])
    print("=" * 70)
    print(" Germination ladder — does capacity EARN its parameters?")
    print("=" * 70)
    for label, teacher in [
        ("HARD curriculum (XOR — capacity should matter)",
         NonlinearVectorTeacher("bench-hard", base)),
        ("EASY curriculum (linear — a seed should saturate)",
         AlignedVectorTeacher("bench-easy", base)),
    ]:
        print(f"\n  {label}")
        print(f"  {'stage':>12}{'dims':>18}{'params':>8}{'accuracy':>16}")
        print("  " + "-" * 56)
        for name, dims, params, mean, std in run(teacher):
            print(f"  {name:>12}{str(dims):>18}{params:>8}{mean:>10.3f} ± {std:.3f}")
    print("\n  Read honestly: promote only where the hard column climbs; the easy")
    print("  column is why auto_germinate keeps saturated experts as seeds.")


if __name__ == "__main__":
    main()
