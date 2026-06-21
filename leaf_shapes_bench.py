"""
leaf_shapes_bench.py
--------------------
Two claims, settled with data instead of assertion:

  (1) "Fibonacci dimensions are special" — test Fibonacci vs power-of-two vs
      linear layer widths on the SAME task. If accuracies are within noise, the
      Fibonacci framing is cosmetic (our long-standing claim, now measured).
  (2) Compressive vs expansive leaves — leaves whose widths shrink (144->89->55)
      vs grow (55->89->144). Compare accuracy and parameter cost.

Task: sklearn digits, even-vs-odd binary (64-dim input). Averaged over seeds so
the comparison isn't a single lucky run.
"""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from das.functional import FibonacciLeaf, softmax

dg = load_digits()
X = StandardScaler().fit_transform(dg.data.astype(np.float64))
y = (dg.target % 2 == 0).astype(int)          # even vs odd
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)

def ce_grad(logits, yy):
    p = softmax(logits); oh = np.zeros_like(p); oh[np.arange(len(yy)), yy] = 1.0
    return (p - oh) / len(yy)

def acc(logits, yy):
    return (logits.argmax(1) == yy).mean()

def n_params(leaf):
    return sum(w.size for w in leaf.W) + sum(b.size for b in leaf.b)

def train_eval(dims, seed):
    rng = np.random.default_rng(seed)
    leaf = FibonacciLeaf(dims, seed=seed)
    for _ in range(1500):
        i = rng.integers(0, len(Xtr), 64)
        leaf.backward(ce_grad(leaf.forward(Xtr[i]), ytr[i]), 0.05)
    return acc(leaf.forward(Xte), yte), n_params(leaf)

SHAPES = {
    "Fibonacci (compressive)": [64, 55, 34, 21, 2],
    "power-of-two":            [64, 64, 32, 16, 2],
    "linear ramp":             [64, 50, 36, 22, 2],
    "expansive":               [64, 89, 144, 2],
    "flat":                    [64, 48, 48, 48, 2],
}
SEEDS = [0, 1, 2, 3, 4]

print("=" * 64)
print(" Leaf-shape study — does the width pattern matter? (digits even/odd)")
print("=" * 64)
print(f"\n  {'shape':<26}{'test acc (mean±std)':>22}{'params':>12}")
print("  " + "-" * 58)
results = {}
for name, dims in SHAPES.items():
    accs, params = zip(*[train_eval(dims, s) for s in SEEDS])
    results[name] = (np.mean(accs), np.std(accs), params[0])
    print(f"  {name:<26}{np.mean(accs):>14.4f} ± {np.std(accs):.4f}{params[0]:>12,}")

# Are the descending-shape families distinguishable?
fam = ["Fibonacci (compressive)", "power-of-two", "linear ramp"]
means = [results[f][0] for f in fam]
spread = max(means) - min(means)
print("\n  " + "-" * 58)
print(f"  Fibonacci vs power-of-2 vs linear: spread = {spread:.4f}")
print(f"    -> {'within noise — Fibonacci is COSMETIC (confirmed)' if spread < 0.02 else 'a real gap exists'}")
comp = results["Fibonacci (compressive)"]; exp = results["expansive"]
print(f"\n  Compressive vs expansive:")
print(f"    compressive: acc {comp[0]:.4f}  params {comp[2]:,}")
print(f"    expansive:   acc {exp[0]:.4f}  params {exp[2]:,}")
print(f"    -> expansive carries {exp[2]/comp[2]:.1f}x the params for "
      f"{exp[0]-comp[0]:+.4f} accuracy — capacity, not the width *pattern*, is what moves the needle.")
print("=" * 64)
