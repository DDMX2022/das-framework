"""
hierarchy_bench.py
------------------
Does the specialty tree actually route better than one flat router? Measured,
not asserted.

Setup: K specialties × M leaves each. Domains are HIERARCHICALLY structured —
each specialty is a well-separated cluster, and its leaves are nearby
sub-clusters inside it (react/hooks looks a lot like react/state, nothing like
math/algebra). That is what real specialty data looks like, and it is exactly
the regime where one flat softmax struggles: it must separate near-identical
siblings AND all other specialties in a single decision, while the tree splits
the job — the top router only tells specialties apart, each sub-router only
tells its own M siblings apart.

Both sides get the SAME total number of router training steps (the tree splits
its budget between the top router and the K sub-routers), the same data, and
the same held-out eval. We report end-to-end routing accuracy (the correct LEAF
chosen) and active router parameters per query (flat d·N vs tree d·(K+M)).

What the numbers actually show (run it):
  = ACCURACY TIES at adequate training budget — the tree does NOT route more
    accurately than a well-trained flat softmax on this data.
  ✗ budget-starved, the tree is WORSE: its budget splits across K+1 routers and
    top×sub errors compound. Don't sell hierarchy as an accuracy win.
  ✓ ACTIVE ROUTING COMPUTE: d·(K+M) vs d·(K·M) — 4× less at 64 leaves, growing.
  ✓ the real (structural) wins are not in this table: ROUTING ISOLATION — a
    graft inside one branch provably leaves every other branch's router
    byte-identical, while the flat design retrains one shared router over
    everyone's data on every graft — plus branch-level prune/policy and
    two-level provenance. See tests/test_hierarchy.py.

Pure NumPy, deterministic.
"""
import numpy as np

from das.hierarchy import HierarchicalDASForest
from das.routing import StemRouter

rng = np.random.default_rng(0)
D = 24
N_TRAIN, N_EVAL = 80, 60          # per leaf
SPECIALTY_SCALE = 4.0             # how far specialties sit apart
LEAF_SCALE = 1.2                  # how far siblings sit apart INSIDE a specialty
TOTAL_STEPS = 4000                # identical budget for both sides


def make_domains(k, m):
    """K specialty centers, each with M nearby leaf centers."""
    centers = []
    for s in range(k):
        cs = rng.normal(0, 1, D); cs = cs / np.linalg.norm(cs) * SPECIALTY_SCALE
        for l in range(m):
            off = rng.normal(0, 1, D); off = off / np.linalg.norm(off) * LEAF_SCALE
            centers.append((s, l, cs + off))
    return centers


def sample(centers, n):
    X, ys, yl, yflat = [], [], [], []
    for flat_id, (s, l, c) in enumerate(centers):
        X.append(c + rng.normal(0, 0.55, (n, D)))
        ys.append(np.full(n, s)); yl.append(np.full(n, l)); yflat.append(np.full(n, flat_id))
    return (np.vstack(X), np.concatenate(ys), np.concatenate(yl), np.concatenate(yflat))


def train_router(router, X, y, steps, lr=0.2, seed=1):
    r = np.random.default_rng(seed)
    for _ in range(int(steps)):
        i = r.integers(0, len(X), 64)
        router.train_step(X[i], y[i], lr=lr)


def run(k, m):
    centers = make_domains(k, m)
    Xtr, ys_tr, yl_tr, yflat_tr = sample(centers, N_TRAIN)
    Xev, ys_ev, yl_ev, yflat_ev = sample(centers, N_EVAL)
    n_leaves = k * m

    # ── flat: one router over all K·M leaves, full budget ────────────
    flat = StemRouter(D, n_leaves, seed=7)
    train_router(flat, Xtr, yflat_tr, TOTAL_STEPS)
    pred, _ = flat.route(Xev)
    flat_acc = float((pred == yflat_ev).mean())
    flat_active = D * n_leaves

    # ── tree: top router over K + one sub-router per specialty ───────
    # identical TOTAL budget, split half to the top, half across the subs
    tree = HierarchicalDASForest(D, seed=7)
    for s in range(k):
        tree.add_branch(f"s{s}", num_leaves=m, seed=100 + s)
    tree.train_router(Xtr, ys_tr, steps=TOTAL_STEPS // 2, lr=0.2, seed=1)
    sub_steps = (TOTAL_STEPS // 2) // k
    for s in range(k):
        mask = ys_tr == s
        train_router(tree.branches[f"s{s}"].router, Xtr[mask], yl_tr[mask],
                     sub_steps, seed=2 + s)
    top_pred, _ = tree.route(Xev)
    correct = 0
    for s in range(k):
        mask = top_pred == s
        if not mask.any():
            continue
        sub_pred, _ = tree.branches[f"s{s}"].router.route(Xev[mask])
        correct += int(((ys_ev[mask] == s) & (sub_pred == yl_ev[mask])).sum())
    tree_acc = correct / len(Xev)
    tree_active = D * (k + m)

    return flat_acc, tree_acc, flat_active, tree_active


def main():
    global TOTAL_STEPS
    print("=" * 74)
    print(" Specialty tree vs flat router — same data, same total training budget")
    print("=" * 74)
    print(f"\n  d_model {D} | {N_TRAIN}/{N_EVAL} train/eval per leaf | "
          f"{TOTAL_STEPS} router steps each side\n")
    print(f"  {'fleet':>10}{'leaves':>8}{'flat acc':>10}{'tree acc':>10}"
          f"{'flat act.params':>17}{'tree act.params':>17}")
    print("  " + "-" * 70)
    for k, m in [(3, 3), (4, 4), (6, 6), (8, 8)]:
        fa, ta, fp, tp = run(k, m)
        print(f"  {k}x{m:>7}{k*m:>8}{fa:>10.3f}{ta:>10.3f}{fp:>17,}{tp:>17,}")

    print("\n  Budget-starved regime (8x8, fewer total router steps):")
    print(f"  {'steps':>10}{'flat acc':>10}{'tree acc':>10}")
    print("  " + "-" * 32)
    full = TOTAL_STEPS
    for budget in [500, 1000, 2000]:
        TOTAL_STEPS = budget
        fa, ta, _, _ = run(8, 8)
        print(f"  {budget:>10}{fa:>10.3f}{ta:>10.3f}")
    TOTAL_STEPS = full

    print("\n  Read honestly: accuracy TIES at adequate budget and the tree LOSES")
    print("  when budget-starved (split budget, compounded errors). The tree's")
    print("  measured win is active routing compute — d*(K+M) vs d*(K*M) — and")
    print("  its structural wins are routing isolation (a graft in one branch")
    print("  provably leaves every other branch's router byte-identical),")
    print("  branch-level prune/policy, and two-level provenance.")


if __name__ == "__main__":
    main()
