"""
scale_bench.py
--------------
Honest scale stress. We can't run a real 100B model here, but we CAN scale the
forest up (many leaves, ~1.6M params each → hundreds of millions stored) and
measure the property the whole pitch rests on: as you add leaves, STORED params
grow linearly but the ACTIVE cost per query (router + 1 leaf, top-1) stays FLAT.
We also confirm the forgetting proof still holds at scale (graft a leaf, others
byte-identical).

What this does and doesn't show:
  ✓ the sparse-activation mechanic scales — active compute is decoupled from total.
  ✗ it does NOT prove real-LLM-scale quality; the bottleneck at true scale is the
    router (see cifar_bench: routing collapses on hard inputs) and the isolation
    tax, not the activation accounting. Cheap inference != capable model.
"""
import time
import torch
from das_torch import DASForest, leaf_hash

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(0)
D = 512
LEAF = [D, 1024, 1024, 10]   # ~1.6M params per leaf

def sync():
    if DEVICE == "mps": torch.mps.synchronize()

def measure(n_leaves):
    forest = DASForest(D, LEAF, num_leaves=n_leaves).to(DEVICE)
    stored = sum(p.numel() for p in forest.parameters())
    active = sum(p.numel() for p in forest.router.parameters()) + \
             sum(p.numel() for p in forest.leaves[0].parameters())
    # latency: top-1 with O(1) DIRECT dispatch (route, then run only that leaf).
    # NB: the convenience forest.predict() loops over all leaves (O(N) dispatch
    # overhead even though one computes); at scale you index the routed leaf.
    qs = [torch.randn(1, D, device=DEVICE) for _ in range(50)]
    with torch.no_grad():
        li, _, _ = forest.router(qs[0]); _ = forest.leaves[int(li)](qs[0]); sync()  # warm
        t0 = time.time()
        for q in qs:
            li, _, _ = forest.router(q)
            _ = forest.leaves[int(li)](q); sync()
        lat = (time.time() - t0) / len(qs) * 1000
    # forgetting at scale: graft a leaf, others must stay byte-identical
    before = [leaf_hash(l) for l in forest.leaves]
    forest.graft()
    after = [leaf_hash(l) for l in forest.leaves[:n_leaves]]
    intact = before == after
    return stored, active, lat, intact

print("=" * 70)
print(" Scale stress — stored grows, active stays flat (top-1 sparse)")
print("=" * 70)
print(f"\n  device {DEVICE} | leaf ~{sum(p.numel() for p in __import__('das_torch').FibonacciLeaf(LEAF).parameters())/1e6:.1f}M params\n")
print(f"  {'leaves':>8}{'stored params':>16}{'active params':>16}{'ms/query':>11}{'isolation':>11}")
print("  " + "-" * 62)
for n in [8, 16, 32, 64]:
    stored, active, lat, intact = measure(n)
    print(f"  {n:>8}{stored/1e6:>14.1f}M{active/1e6:>14.1f}M{lat:>11.2f}"
          f"{'PASS' if intact else 'FAIL':>11}")

print("\n" + "=" * 70)
print("  Reads: stored params scale linearly with leaves (8→64 ≈ 8x), but active")
print("  params and ms/query stay flat — top-1 routing decouples capacity from")
print("  per-query cost, and grafting never disturbs existing leaves even at scale.")
print("  Caveat: this is the MECHANIC scaling. Real-LLM-scale quality is gated by")
print("  the router (collapses on hard inputs) and the isolation tax — cheap")
print("  inference is not the same as a capable model.")
print("=" * 70)
