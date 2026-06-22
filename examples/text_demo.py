"""
text_demo.py
------------
Phase 11: the forest on TEXT via a tokenizer front-end. Synthetic so it's
self-contained (no downloads). Three text domains, each a binary task:
  - math      : "what is 4 plus 7"        -> addition? (plus/sum = 1)
  - sentiment : "i love this"             -> positive? (1)
  - command   : "turn off the light"      -> on? (on = 1, off = 0)

A bag-of-words tokenizer turns strings into vectors; the router learns which
domain a sentence belongs to; each leaf learns its domain's binary task in
isolation. Then we graft a 4th domain (greeting) and prove the first three
leaves stay byte-identical.
"""
import numpy as np
from das.text import Tokenizer
from das.model import DASForest
from das.functional import softmax

rng = np.random.default_rng(0)
N = 400

def gen_math(n):
    ops = [("plus", 1), ("sum of", 1), ("minus", 0), ("times", 0)]
    T, Y = [], []
    for _ in range(n):
        op, lab = ops[rng.integers(len(ops))]
        a, b = rng.integers(1, 10, 2)
        T.append(f"what is {a} {op} {b}"); Y.append(lab)
    return T, Y

def gen_sentiment(n):
    pos = ["love", "great", "wonderful"]; neg = ["hate", "terrible", "awful"]
    T, Y = [], []
    for _ in range(n):
        if rng.random() < 0.5:
            T.append(f"i {pos[rng.integers(3)]} this movie"); Y.append(1)
        else:
            T.append(f"i {neg[rng.integers(3)]} this movie"); Y.append(0)
    return T, Y

def gen_command(n):
    dev = ["light", "fan", "heater"]
    T, Y = [], []
    for _ in range(n):
        if rng.random() < 0.5:
            T.append(f"turn on the {dev[rng.integers(3)]}"); Y.append(1)
        else:
            T.append(f"turn off the {dev[rng.integers(3)]}"); Y.append(0)
    return T, Y

def gen_greeting(n):
    T, Y = [], []
    for _ in range(n):
        if rng.random() < 0.5:
            T.append("hello there friend"); Y.append(1)
        else:
            T.append("goodbye for now friend"); Y.append(0)
    return T, Y

doms = [gen_math(N), gen_sentiment(N), gen_command(N), gen_greeting(N)]
names = ["math", "sentiment", "command", "greeting"]

# Fit the shared tokenizer on ALL domains' text up front (the shared front-end).
tok = Tokenizer().fit([t for (T, _) in doms for t in T])
print("=" * 60)
print(" DAS on TEXT (Phase 11) — bag-of-words front-end")
print("=" * 60)
print(f"\n  vocabulary size: {tok.dim} words")

X = [tok.transform(T) for (T, _) in doms]
Y = [np.array(Yi) for (_, Yi) in doms]
D = tok.dim
LEAF = [D, 16, 8, 2]

def ce_grad(logits, y):
    p = softmax(logits); oh = np.zeros_like(p); oh[np.arange(len(y)), y] = 1.0
    return (p - oh) / len(y)

def acc(logits, y):
    return (logits.argmax(1) == y).mean()

def grow_leaf(forest, lid, Xd, yd, steps=400, lr=0.1):
    leaf = forest.leaves[lid]; leaf.frozen = False
    for _ in range(steps):
        idx = rng.integers(0, len(Xd), 64)
        leaf.backward(ce_grad(leaf.forward(Xd[idx]), yd[idx]), lr)
    leaf.frozen = True
    return acc(leaf.forward(Xd), yd)

# Router + 3 leaves
forest = DASForest(D, LEAF, num_leaves=3, seed=7)
Xr = np.vstack([X[d] for d in range(3)])
dr = np.concatenate([np.full(len(X[d]), d) for d in range(3)])
for _ in range(800):
    idx = rng.integers(0, len(Xr), 64)
    forest.router.train_step(Xr[idx], dr[idx], lr=0.2)
racc = (forest.router.route(Xr)[0] == dr).mean()
print(f"\n  router domain accuracy: {racc:.3f}")
for d in range(3):
    a = grow_leaf(forest, d, X[d], Y[d])
    print(f"  leaf {d} ({names[d]}): task acc {a:.3f}")

# Graft a 4th domain; prove first three unchanged
before = {d: forest.leaves[d].weight_hash() for d in range(3)}
nid = forest.graft(seed=321)
Xr4 = np.vstack([X[d] for d in range(4)])
dr4 = np.concatenate([np.full(len(X[d]), d) for d in range(4)])
for _ in range(400):
    idx = rng.integers(0, len(Xr4), 64)
    forest.router.train_step(Xr4[idx], dr4[idx], lr=0.2)
a4 = grow_leaf(forest, nid, X[3], Y[3])
after = {d: forest.leaves[d].weight_hash() for d in range(3)}
unchanged = all(before[d] == after[d] for d in range(3))
print(f"\n  grafted leaf {nid} ({names[3]}): task acc {a4:.3f}")
print(f"  old leaves byte-identical after graft: {'PASS' if unchanged else 'FAIL'}")

print("\n" + "=" * 60)
print(f"  Router {racc:.2f}  |  forgetting proof: {'PASS' if unchanged else 'FAIL'}")
print("=" * 60)
import sys; sys.exit(0 if unchanged else 1)
