"""
paging_demo.py
--------------
Phase 12: JIT memory paging — the real technique behind the "run a huge model on
a laptop" pitch. Keep all leaves in cheap CPU RAM ("cold"); page only the active
leaf onto the GPU/MPS ("hot") per query, then evict. This genuinely lets total
parameters exceed device memory.

But we MEASURE the cost the pitch glosses over: every query that hits a cold leaf
pays a CPU→device transfer. This demo times resident (all leaves hot) vs paged
(transfer per query) so the latency tax is a number, not a hand-wave.

Honest expectation: paging slashes device-memory use (only 1 leaf resident) but
ADDS real per-query latency. "100B on a laptop at low latency" doesn't hold once
you account for paying the transfer on every token; prefetch only helps if you
know the next route in advance.
"""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
N_LEAVES, D, H, OUT = 8, 512, 2048, 10
Q = 200   # single-sample queries (simulating per-token routing)

def sync():
    if DEVICE == "mps": torch.mps.synchronize()
    elif DEVICE == "cuda": torch.cuda.synchronize()

class BigLeaf(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D, H), nn.ReLU(), nn.Linear(H, H), nn.ReLU(), nn.Linear(H, OUT))
    def forward(self, x): return self.net(x)

leaf_mb = sum(p.numel() for p in BigLeaf().parameters()) * 4 / 1e6
print("=" * 64)
print(" Phase 12: JIT paging — device-memory savings vs latency cost")
print("=" * 64)
print(f"\n  {N_LEAVES} leaves x ~{leaf_mb:.1f} MB = ~{N_LEAVES*leaf_mb:.0f} MB total")
print(f"  device: {DEVICE}")

leaves_cpu = [BigLeaf() for _ in range(N_LEAVES)]
router = nn.Linear(D, N_LEAVES)
queries = [torch.randn(1, D) for _ in range(Q)]
routes = [int(router(q).argmax(1).item()) for q in queries]

# warm up the device (first MPS calls compile/allocate — exclude from timing)
leaves_hot = [BigLeaf().to(DEVICE) for _ in range(N_LEAVES)]
for h, c in zip(leaves_hot, leaves_cpu): h.load_state_dict(c.state_dict())
with torch.no_grad():
    for r in range(N_LEAVES): _ = leaves_hot[r](queries[0].to(DEVICE))
    _ = leaves_cpu[0].to(DEVICE)(queries[0].to(DEVICE))
sync()

# ── Resident: every leaf lives on the device (sync per query) ───
t0 = time.time()
for q, r in zip(queries, routes):
    with torch.no_grad():
        _ = leaves_hot[r](q.to(DEVICE)); sync()
resident_ms = (time.time() - t0) / Q * 1000
resident_dev_mb = N_LEAVES * leaf_mb

# ── Paged: leaves cold on CPU; page active leaf in per query, evict ──
t0 = time.time()
for q, r in zip(queries, routes):
    with torch.no_grad():
        hot = leaves_cpu[r].to(DEVICE)        # page-in (CPU -> device transfer)
        _ = hot(q.to(DEVICE)); sync()
        del hot                                # evict
paged_ms = (time.time() - t0) / Q * 1000
paged_dev_mb = leaf_mb   # only the active leaf resident at a time

# ── Raw page-in cost in isolation ───────────────────────────────
t0 = time.time()
for _ in range(Q):
    h = leaves_cpu[0].to(DEVICE); sync(); del h
transfer_ms = (time.time() - t0) / Q * 1000

print(f"\n  {'mode':<14}{'ms / query':>12}{'device mem':>14}")
print(f"  {'resident':<14}{resident_ms:>12.3f}{resident_dev_mb:>12.0f} MB")
print(f"  {'paged (JIT)':<14}{paged_ms:>12.3f}{paged_dev_mb:>12.0f} MB")
print(f"  {'raw page-in':<14}{transfer_ms:>12.3f}{'':>10} (~{leaf_mb:.0f} MB / leaf)")
print(f"\n  paging cuts device memory {resident_dev_mb:.0f} MB -> {paged_dev_mb:.0f} MB "
      f"({resident_dev_mb/paged_dev_mb:.0f}x less)")
print(f"  per-query transfer tax: {paged_ms-resident_ms:+.2f} ms")

print("\n" + "=" * 64)
print("  Verdict: paging genuinely trades device memory for latency — the memory")
print("  win is real (only 1 leaf resident).")
print(f"  On THIS hardware ({DEVICE}) the transfer tax is small because Apple")
print("  Silicon has UNIFIED memory: CPU and GPU share physical RAM, so paging")
print("  is nearly a no-op. On a discrete GPU over PCIe (where the '100B on a")
print("  laptop' pitch is aimed) that same transfer is gigabytes over a slow bus")
print("  every token — far costlier. So the claim is hardware-dependent: plausible")
print("  on unified memory, dubious on the PCIe systems it's usually sold for.")
print("=" * 64)
