# DAS Framework — Status & Honest Verdict

A single-page summary of what was built, what was measured, and what it all means.
For the full narrative and diagrams see [README.md](../README.md).

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
| Orchestrator integration | governed DAS node under LangGraph (provenance + RBAC) | correct routing + provenance per query; denial surfaced as state + audited PASS |
| Governance API + deploy | `apps/governance_api.py` (NumPy+Flask) + Dockerfile + k8s manifest | boot→predict→RBAC 403→persist→reload byte-identical + chain bound PASS |
| Governance benchmark | Monolith vs Isolated experts vs DAS control plane | DAS ties isolation, wins audit/RBAC/provenance (numbers below) |
| REST API | `apps/serve.py` + interactive page | live predictions |
| Real encoder path (Phase 1) | frozen pretrained MiniLM embeddings of **real text** → router + isolated LoRA experts | routes real sentences; graft/prune byte-identical on real text — `das_text.py`, `examples/hf_governance_demo.py` |

## Key measured findings

| Finding | Number | Source |
|---|---|---|
| Zero forgetting (structural) | BWT 0.000 | demo / cifar / continual |
| EWC: hard vs easy regime | −0.33 (Split) → −0.03 (Permuted) | app.py |
| **Router collapses on raw CIFAR** | 0.42 | `benchmarks/cifar_bench.py` |
| MLP router barely helps on raw CIFAR | 0.40 → 0.45 | `benchmarks/router_bench.py` |
| Shared backbone helps but doesn't fix CIFAR routing | 0.42 → 0.66 (vs 0.98 MNIST) | `benchmarks/backbone_cifar_bench.py` |
| **DAS ≈ LoRA + router** | tie on acc/isolation/deletion | `benchmarks/lora_bench.py` |
| Deletion / unlearning | tenant acc 0.995 → 0.52, others byte-identical | `examples/governance_demo.py` |
| JIT paging | 8× less device mem, +0.17 ms/query (unified mem) | `examples/paging_demo.py` |
| Mycelial soil dominates cost | always-on orchestrator ≈ 100% of active compute | `examples/mycelial_demo.py` |
| Unsupervised routing discovers domains | purity 0.77 (vs 0.33 chance), no labels | `benchmarks/unsupervised_routing.py` |
| Load-balancing can over-balance | forcing even usage dropped purity 0.77 → 0.55 | `benchmarks/unsupervised_routing.py` |
| Fibonacci widths are cosmetic | Fib/pow2/linear within 0.006 (noise) | `benchmarks/leaf_shapes_bench.py` |
| Learned embedding beats BoW on word order | BoW 0.50 → embedding 1.00 | `examples/embedding_demo.py` |
| Prefetch hides page-in (when compute ≥ transfer) | 2 ms transfer: 40% hidden; 10 ms: only 15% (transfer-bound) | `examples/prefetch_demo.py` |
| Sparse activation scales | stored 12.7M→101.5M (8×), active+latency flat (~1 ms) | `benchmarks/scale_bench.py` |
| Pretrained encoder transfers | frozen encoder 1.00 vs from-scratch 0.75 on held-out words | `examples/encoder_demo.py` |
| C++ pager compiles & pages correctly | C++ layer real (CPU/MPS); CUDA async path untestable (no NVIDIA) | `csrc/pager.cpp`, `examples/pager_demo.py` |
| LoRALeaf experts (LoRA on frozen backbone) | isolation byte-identical + checkpoint byte-exact, PASS | `examples/lora_leaf_demo.py` |
| Governance control plane (RBAC + tenancy) | cross-tenant/role denials enforced + logged; tenant-delete leaves others byte-identical | `examples/control_plane_demo.py` |
| Control-plane persistence | save→reload byte-identical; weight-file swap caught by state↔audit binding | `examples/control_plane_demo.py` |
| DAS as a LangGraph node | every routed answer carries tenant/expert/confidence provenance; RBAC denial surfaces as state + is audited | `examples/langgraph_demo.py` |
| Governance API as a deployable unit | boot bootstraps/loads a fleet; /predict provenance, RBAC 403s, save→reload byte-identical + state↔audit bound across restarts | `apps/governance_api.py` |
| Governance benchmark: monolith forgets | BWT −0.467, 0% isolation on add | `benchmarks/governance_benchmark.py` |
| Governance benchmark: DAS = isolation, + governance | isolation/delete 100% (ties LoRA); audit tamper-caught 100%, RBAC denied 100%, router 100% | `benchmarks/governance_benchmark.py` |

## Not built / out of scope
- "100B on a laptop" at low latency on PCIe GPUs (paging cost is real there).
- Beating frontier models; 90% cost cuts (unsupported by measurements).
- The branding (Fibonacci benefits, "vector torque", "coiled strings") — cosmetic
  over a standard softmax-routed MoE.
- **LoRA on the transformer's own weights** + large-real-model scale. The Phase-1
  encoder is a *frozen featurizer*: real pretrained MiniLM → embeddings, with the
  LoRA experts on a small backbone over those embeddings (not PEFT adapters inside
  the encoder's attention). It removes the "synthetic data" objection, not the
  "unproven at scale" one.

## Where it could genuinely go
1. **Governance product** — own the auditable-isolation + deletion + audit-trail
   story (`examples/governance_demo.py` is the seed); prove it on a real compliance need.
2. **Strong learning artifact** — a thorough, honest build-and-evaluate study of
   MoE + continual learning.

Not: the original "new paradigm" framing. The engineering is sound; the claim it
was attached to is not.
