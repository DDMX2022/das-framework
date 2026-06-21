"""
prefetch_demo.py
----------------
Predictive prefetching for paging. `paging_demo.py` measured the raw page-in
cost; this adds the trick the original pitch relies on: because the router knows
the route a step ahead, you can page the NEXT leaf onto the device WHILE the
current leaf computes, hiding the transfer behind compute.

On Apple Silicon's unified memory the real transfer is ~free, so to show the
mechanism honestly we model a configurable transfer latency (what a discrete GPU
pays over PCIe — gigabytes per token). The compute is a real leaf forward.

Result we expect:
  - transfer ~ 0 (unified memory): prefetch buys almost nothing (matches paging_demo).
  - transfer > 0 (PCIe-like): serial pays transfer+compute every token; prefetch
    pays ~max(transfer, compute) — the transfer disappears IF compute >= transfer.
    If transfer >> compute, prefetch can't save you (you're transfer-bound).
"""
import time
import threading
import torch
import torch.nn as nn

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(0)
D, H, OUT, N_LEAVES, Q = 512, 2048, 10, 8, 100

class BigLeaf(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D, H), nn.ReLU(), nn.Linear(H, H), nn.ReLU(), nn.Linear(H, OUT))
    def forward(self, x): return self.net(x)

def sync():
    if DEVICE == "mps": torch.mps.synchronize()

leaves = [BigLeaf() for _ in range(N_LEAVES)]
router = nn.Linear(D, N_LEAVES)
queries = [torch.randn(1, D) for _ in range(Q)]
routes = [int(router(q).argmax(1)) for q in queries]

def page_in(leaf, transfer_ms):
    hot = leaf.to(DEVICE)
    if transfer_ms: time.sleep(transfer_ms / 1000.0)   # emulate PCIe transfer cost
    return hot

def serial(transfer_ms):
    sync(); t0 = time.time()
    for q, r in zip(queries, routes):
        hot = page_in(leaves[r], transfer_ms)           # page-in (blocks)
        with torch.no_grad(): _ = hot(q.to(DEVICE)); sync()
    return (time.time() - t0) / Q * 1000

def prefetched(transfer_ms):
    """Page-in the NEXT leaf in a worker thread while the current leaf computes."""
    box = {}
    def fetch(i): box[i] = page_in(leaves[routes[i]], transfer_ms)
    sync(); t0 = time.time()
    th = threading.Thread(target=fetch, args=(0,)); th.start(); th.join()  # first must block
    for i, q in enumerate(queries):
        hot = box.pop(i)
        nxt = None
        if i + 1 < Q:
            nxt = threading.Thread(target=fetch, args=(i + 1,)); nxt.start()  # prefetch overlaps compute
        with torch.no_grad(): _ = hot(q.to(DEVICE)); sync()
        if nxt: nxt.join()
    return (time.time() - t0) / Q * 1000

# warm up (discard the first timed loop — MPS compiles/allocates on first use)
_ = leaves[0].to(DEVICE)(queries[0].to(DEVICE)); sync()
serial(0.0)
compute_ms = serial(0.0)   # transfer-free baseline ~= pure compute

print("=" * 64)
print(" Predictive prefetching — hide page-in behind compute")
print("=" * 64)
print(f"\n  device {DEVICE} | leaf compute ~{compute_ms:.2f} ms/query\n")
print(f"  {'sim transfer':>14}{'serial ms':>12}{'prefetch ms':>14}{'hidden':>10}")
for tms in [0.0, 2.0, 10.0]:
    s = serial(tms); p = prefetched(tms)
    print(f"  {tms:>11.0f} ms{s:>12.2f}{p:>14.2f}{(s-p)/max(s,1e-9)*100:>9.0f}%")

print("\n" + "=" * 64)
print("  Reads: at 0 ms transfer (unified memory) prefetch buys ~nothing — matches")
print("  paging_demo. As transfer grows (PCIe), serial pays it every token while")
print("  prefetch hides it behind compute — UNTIL transfer exceeds compute, where")
print("  you're transfer-bound and prefetch can't save you. So 'huge model, low")
print("  latency' needs predictable routes AND compute >= transfer per step.")
print("=" * 64)
