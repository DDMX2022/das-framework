# DAS Framework — Status & Honest Verdict

A single-page summary of what was built, what was measured, and what it all means.
For the full narrative and diagrams see [README.md](README.md).

## What DAS is (one line)
A hard-routed Mixture-of-Experts with provably isolated, hot-swappable experts —
i.e. **per-task LoRA-on-a-frozen-backbone plus a learned router**. Honest core,
toy scale, with a governance angle that's its real defensible value.

## The bottom-line verdict
- The **architecture is real and works**: isolation, grafting, pruning, zero
  forgetting (cryptographically proven), checkpointing, an API, and a full
  baseline suite are all built and tested.
- It is **not "better/cheaper AI."** Phase 14 showed DAS ≈ LoRA + a router; they
  tie on isolation/forgetting/deletion. DAS's one structural edge is the built-in
  task-free router.
- Its **defensible home is governance** — auditable, multi-tenant, deletable
  model fleets — not raw capability.
- The **grand-vision claims do not hold**: the router is the bottleneck on real
  images, paging's cost is hardware-dependent, the mycelial orchestrator's
  economics collapse once the always-on soil is counted, and "beats frontier
  models / 90% cost cut" is unsupported.
- **The last conceptual gap is now built**: unsupervised routing (no domain
  labels) genuinely discovers the hidden domains — but co-training the experts
  to do it **destroys the byte-identical isolation guarantee**. Self-organising
  routing and auditable isolation are mutually exclusive; you pick one. That's
  the deepest honest finding: DAS's value (isolation) and the headline MoE
  capability (learned routing) are in direct tension.

## Built & verified

| Area | What | Result |
|---|---|---|
| Core (NumPy) | router + isolated leaves, grafting, forgetting proof | byte-identical proof PASS |
| PyTorch backend | autograd trainer, checkpoint/restore, ConvLeaf | byte-exact restore PASS |
| Lifecycle | monitor, dormancy prune, redundancy prune, regrow, persistence | full loop PASS |
| Phase 9 | shared frozen backbone + isolated heads | router 0.98 (MNIST) |
| Phase 10 | top-k canopy merge | top-2 ≥ top-1 (graceful) |
| Phase 11 | tokenizer text front-end | 4 text domains @ 100% |
| Continual baselines | EWC, PackNet, fine-tuned, multi-task, Progressive Nets | DAS/PackNet/PNN BWT 0 |
| Governance | multi-tenant isolation + deletion + audit trail | non-interference + unlearning PASS |
| Control plane | RBAC + multi-tenancy + signed audit + save/restore | role/tenant denials + tenant-delete isolation + persistence + tamper-detect PASS |
| REST API | `serve.py` + interactive page | live predictions |

## Key measured findings

| Finding | Number | Source |
|---|---|---|
| Zero forgetting (structural) | BWT 0.000 | demo / cifar / continual |
| EWC: hard vs easy regime | −0.33 (Split) → −0.03 (Permuted) | app.py |
| **Router collapses on raw CIFAR** | 0.42 | `cifar_bench.py` |
| MLP router barely helps on raw CIFAR | 0.40 → 0.45 | `router_bench.py` |
| Shared backbone helps but doesn't fix CIFAR routing | 0.42 → 0.66 (vs 0.98 MNIST) | `backbone_cifar_bench.py` |
| **DAS ≈ LoRA + router** | tie on acc/isolation/deletion | `lora_bench.py` |
| Deletion / unlearning | tenant acc 0.995 → 0.52, others byte-identical | `governance_demo.py` |
| JIT paging | 8× less device mem, +0.17 ms/query (unified mem) | `paging_demo.py` |
| Mycelial soil dominates cost | always-on orchestrator ≈ 100% of active compute | `mycelial_demo.py` |
| Unsupervised routing discovers domains | purity 0.77 (vs 0.33 chance), no labels | `unsupervised_routing.py` |
| Load-balancing can over-balance | forcing even usage dropped purity 0.77 → 0.55 | `unsupervised_routing.py` |
| Fibonacci widths are cosmetic | Fib/pow2/linear within 0.006 (noise) | `leaf_shapes_bench.py` |
| Learned embedding beats BoW on word order | BoW 0.50 → embedding 1.00 | `embedding_demo.py` |
| Prefetch hides page-in (when compute ≥ transfer) | 2 ms transfer: 40% hidden; 10 ms: only 15% (transfer-bound) | `prefetch_demo.py` |
| Sparse activation scales | stored 12.7M→101.5M (8×), active+latency flat (~1 ms) | `scale_bench.py` |
| Pretrained encoder transfers | frozen encoder 1.00 vs from-scratch 0.75 on held-out words | `encoder_demo.py` |
| C++ pager compiles & pages correctly | C++ layer real (CPU/MPS); CUDA async path untestable (no NVIDIA) | `csrc/pager.cpp`, `pager_demo.py` |
| LoRALeaf experts (LoRA on frozen backbone) | isolation byte-identical + checkpoint byte-exact, PASS | `lora_leaf_demo.py` |
| Governance control plane (RBAC + tenancy) | cross-tenant/role denials enforced + logged; tenant-delete leaves others byte-identical | `control_plane_demo.py` |
| Control-plane persistence | save→reload byte-identical; weight-file swap caught by state↔audit binding | `control_plane_demo.py` |

## Not built / out of scope
- "100B on a laptop" at low latency on PCIe GPUs (paging cost is real there).
- Beating frontier models; 90% cost cuts (unsupported by measurements).
- The branding (Fibonacci benefits, "vector torque", "coiled strings") — cosmetic
  over a standard softmax-routed MoE.

## Where it could genuinely go
1. **Governance product** — own the auditable-isolation + deletion + audit-trail
   story (`governance_demo.py` is the seed); prove it on a real compliance need.
2. **Strong learning artifact** — a thorough, honest build-and-evaluate study of
   MoE + continual learning.

Not: the original "new paradigm" framing. The engineering is sound; the claim it
was attached to is not.
