# DAS Framework

A runnable research prototype of a **hard-routed Mixture-of-Experts** — branded "DAS" (a forest of *leaves*, a *stem router*, a *canopy*). Stripped of the branding, it is a clean, honest implementation of an idea worth testing: route each input to exactly **one** expert network, train each expert in **isolation**, and **graft** new experts without touching the old ones.

The one property this design genuinely delivers — and that this repo cryptographically proves — is **zero catastrophic forgetting**: training a new expert leaves every existing expert *byte-identical* (verified by SHA-256).

> This is a learning/benchmarking scaffold, not a language model. See [What is real vs. hype](#what-is-real-vs-hype).

---

## What's in here

```
das-framework/
├── das/                    NumPy core (manual backprop, no autograd)
│   ├── functional.py       FibonacciLeaf  — an expert MLP + frozen flag + weight_hash()
│   ├── routing.py          StemRouter     — MoE gate (linear → softmax → argmax)
│   ├── model.py            DASForest      — assembles router + leaves, graft(), proofs
│   ├── packnet.py          PackNetMLP     — pruning + per-task weight masks (CL baseline)
│   ├── lifecycle.py        ForestLifecycle — usage monitor, prune, regrow loop
│   ├── text.py             Tokenizer — bag-of-words text front-end
│   └── hub.py              Leaf marketplace — publish / list / pull / graft (hash-verified)
├── demo.py                 Full lifecycle on synthetic data + forgetting proof (NumPy)
├── benchmark.py            DAS vs matched-size MLP on sklearn digits (NumPy)
├── leaf_shapes_bench.py    Fibonacci vs pow2 vs linear; compressive vs expansive
├── lifecycle_demo.py       Forest lifecycle: grow → graft → prune → regrow (NumPy)
├── canopy_demo.py          Phase 10: top-k canopy merge (graceful degradation)
├── text_demo.py            Phase 11: forest on text via a tokenizer front-end
├── embedding_demo.py       Learned order-aware embedding vs bag-of-words
├── encoder_demo.py         Pretrained (frozen) encoder front-end — transfer
├── governance_demo.py      Multi-tenant isolation + deletion/unlearning + audit
├── hub_demo.py             Leaf marketplace: publish → pull → graft, hash-verified
├── mycelial_demo.py        Phase 13: orchestrator decomposes + routes to trees
├── das_torch.py            PyTorch backend: trainer, leaf_hash, checkpoint/restore, ConvLeaf
├── demo_torch.py           PyTorch lifecycle on MNIST + forgetting proof (autograd path)
├── checkpoint_demo.py      Per-leaf + whole-forest save/load byte-exact restore proofs
├── conv_demo.py            ConvLeaf (CNN expert) trained, frozen, checkpointed
├── backbone_demo.py        Phase 9: shared frozen backbone + isolated heads (MNIST)
├── backbone_cifar_bench.py Phase 9 on CIFAR: conv backbone, router on features
├── cifar_bench.py          Phase 8: Split-CIFAR — CNN forest vs fine-tuned vs multi-task
├── lora_bench.py           Phase 14: DAS isolated heads vs per-task LoRA adapters
├── pnn_bench.py            Progressive Neural Nets baseline (laterals) vs isolation
├── router_bench.py         Linear vs MLP router on raw pixels (MNIST + CIFAR)
├── paging_demo.py          Phase 12: JIT weight paging — memory win vs latency tax
├── unsupervised_routing.py Router discovers domains with no labels (+ load balance)
├── prefetch_demo.py        Predictive prefetching — hide page-in behind compute
├── scale_bench.py          Scale stress — stored grows, active+latency stay flat
├── csrc/pager.cpp          C++ torch extension for the page-in copy
├── pager_demo.py           Compiles + runs the C++ pager (CUDA path noted, untested)
├── serve.py                REST inference API (loads a saved forest, POST /predict)
├── mnist_stress.py         PyTorch: 10 leaves on real MNIST + 10-way forgetting proof
├── app.py                  Flask server — 7 live, browser-streamed experiments
├── templates/              UI for the web app (SSE + Chart.js)
│   ├── index.html          Forest demo + digits benchmark
│   ├── stress.html         MNIST stress test
│   ├── real_bench.html     Real-world multi-dataset benchmark
│   ├── continual_bench.html Split-MNIST continual learning
│   ├── permuted_bench.html  Permuted-MNIST continual learning
│   └── lifecycle.html       live grow → graft → prune → regrow visualizer
├── checkpoints/            saved leaves/forests (gitignored; written by the demos)
└── data/MNIST/raw/         MNIST IDX files (downloaded by mnist_stress.py)
```

---

## Quick start

The NumPy demo needs only `numpy`. The web app adds `flask`, the digits benchmark adds `scikit-learn`, and the PyTorch scripts add `torch`/`torchvision`.

> **Mac note:** Homebrew Python 3.14 currently ships a broken `libexpat` and `pip` won't run. Use conda or Python 3.13.

### Install as a library

```bash
pip install -e .                 # the NumPy core (`das`) + PyTorch backend module
pip install -e ".[torch]"        # + torch / torchvision
pip install -e ".[all]"          # + flask, scikit-learn, pandas (everything)
```
Then `import das` / `import das_torch` from anywhere.

```bash
# recommended: conda
conda create -n das python=3.11 numpy
conda activate das

# core demo (NumPy only)
python demo.py
```

### The five web experiments

```bash
pip install flask scikit-learn
python app.py
# → http://localhost:5050
```

| Route | Page | What it runs |
|-------|------|--------------|
| `/` | **Forest demo** + **Digits benchmark** | Animated tree growth on synthetic data; DAS vs matched MLP on sklearn digits |
| `/stress` | **MNIST stress test** | 10 leaves × 784-dim, router + 10 isolated leaves vs a 10-class baseline; 45-pairwise forgetting proof |
| `/real` | **Real-world benchmark** | Adult Income, Wine Quality, Credit Default (OpenML), with download progress + heartbeat |
| `/continual` | **Split-MNIST continual learning** | DAS vs EWC vs PackNet vs Fine-tuned vs Multi-task, live accuracy matrices + contamination test |
| `/permuted` | **Permuted-MNIST continual learning** | same five models on the domain-incremental regime |
| `/lifecycle` | **Forest lifecycle** | live grow → graft → prune → regrow with the byte-identical proof |
| `/benchmark` (stream) | digits SSE stream | backing stream for the `/` benchmark tab |

The web app reads MNIST directly from `data/MNIST/raw/*.gz` (stdlib `gzip` + `numpy`, no torchvision). If those files are missing, run `python mnist_stress.py` once to download them.

### PyTorch backend (Apple Silicon)

`das_torch.py` is the real autograd backend (not just a smoke test): isolated training, SHA-256 leaf hashing, per-leaf and whole-forest checkpoint/restore, and a `ConvLeaf` CNN expert. Four runnable demos:

```bash
pip install torch torchvision
python demo_torch.py        # full lifecycle on MNIST + forgetting proof (~6s, CPU)
python checkpoint_demo.py   # byte-exact save/load + graft-from-disk proofs
python conv_demo.py         # a CNN leaf trained, frozen, checkpointed, restored
python mnist_stress.py      # 10-leaf MNIST, ~20s on M-series (auto-selects mps)
```

The proof demos force CPU for bit-reproducible hashes; heavy training auto-selects MPS.

### REST inference API

`serve.py` loads a forest saved by `demo_torch.py` and serves predictions:

```bash
python serve.py                                  # port 5060
curl localhost:5060/health                        # {"leaves":3,"status":"ok"}
curl -X POST localhost:5060/predict \
     -H 'Content-Type: application/json' \
     -d '{"pixels": [ ...784 floats... ]}'         # -> {"leaf":i,"prediction":c,"confidence":p}
```

Each input is routed to exactly one leaf; the response says which leaf fired. Out-of-domain inputs are misrouted (honestly — no leaf was trained on them).

---

## How the architecture works

1. **Stem Router** (`routing.py`) — a single linear layer + softmax. The softmax output is the "vector torque" τ; `argmax(τ)` picks exactly one leaf (hard top-1 routing). Trained supervised to predict each input's domain.
2. **FibonacciLeaf** (`functional.py`) — a standalone MLP with manual forward/backward. A `frozen` flag gates the weight update, so a frozen leaf cannot move even when gradients flow. `weight_hash()` returns a SHA-256 fingerprint used to prove that.
3. **DASForest** (`model.py`) — routes each input to its leaf, collects outputs. `graft()` adds a new leaf **and** a new router slot (the router must learn the new route — see [hype notes](#what-is-real-vs-hype)).

### Inference: one input, one leaf

At prediction time the router commits 100% of the signal to a single leaf. All other leaves stay frozen and are never touched — that hard top-1 path is what bounds compute and gradient flow to one expert.

```mermaid
flowchart LR
    X([Input x]) --> R{{"Stem Router<br/>linear → softmax → argmax"}}
    R -- "τ = [0.02, 0.95, 0.03]<br/>argmax → Leaf 1" --> L1["Leaf 1 (active)<br/>MLP"]
    R -. frozen .-> L0["Leaf 0"]
    R -. frozen .-> L2["Leaf 2"]
    L1 --> OUT([Output logits])
    L0 -.-> X0((idle))
    L2 -.-> X2((idle))

    classDef active fill:#2e7d32,stroke:#1b5e20,color:#fff;
    classDef idle fill:#424242,stroke:#212121,color:#aaa;
    class L1 active;
    class L0,L2 idle;
```

### Training lifecycle + the forgetting proof

Each leaf is trained in isolation (router frozen, all other leaves frozen). Before grafting a new leaf, every existing leaf is fingerprinted with SHA-256; after training the new leaf, the fingerprints are re-checked. They are always byte-identical — that is the proof.

```mermaid
flowchart TD
    A["Phase 1: train Stem Router<br/>(supervised on domain labels)"] --> B["Phase 2: train Leaf 0 in isolation<br/>others frozen"]
    B --> C["train Leaf 1 in isolation<br/>others frozen"]
    C --> D["📸 Snapshot: hash every leaf<br/>(weight_hash)"]
    D --> E["Phase 3: graft Leaf 2<br/>+ add router slot"]
    E --> F["train Leaf 2 in isolation<br/>Leaf 0 &amp; 1 frozen"]
    F --> G["📸 Re-hash all leaves"]
    G --> H{"old hashes == new hashes?"}
    H -- yes --> P["✅ PASS — zero forgetting<br/>(byte-identical)"]
    H -- no --> Q["❌ FAIL — a frozen leaf moved"]

    classDef pass fill:#2e7d32,stroke:#1b5e20,color:#fff;
    classDef fail fill:#b71c1c,stroke:#7f0000,color:#fff;
    class P pass;
    class Q fail;
```

### Continual learning: why DAS doesn't forget

The `/continual` page runs this comparison on Split-MNIST. A fine-tuned MLP overwrites shared weights each task (old-task accuracy decays → negative BWT); DAS adds an isolated leaf per task (old leaves untouched → BWT ≈ 0).

```mermaid
flowchart LR
    subgraph DAS["DAS Forest — graft per task"]
        direction TB
        T1["Task 0/1"] --> Lf0["Leaf 0 🔒"]
        T2["Task 2/3"] --> Lf1["Leaf 1 🔒"]
        T3["Task 4/5"] --> Lf2["Leaf 2 🔒"]
        note1["old leaves frozen → BWT ≈ 0"]
    end
    subgraph FT["Fine-tuned MLP — one shared net"]
        direction TB
        F1["Task 0/1"] --> W["shared weights W"]
        F2["Task 2/3"] --> W
        F3["Task 4/5"] --> W
        note2["each task overwrites W → forgets<br/>BWT ≈ −0.2 to −0.3"]
    end

    classDef good fill:#2e7d32,stroke:#1b5e20,color:#fff;
    classDef bad fill:#b71c1c,stroke:#7f0000,color:#fff;
    class note1 good;
    class note2 bad;
```

### The forest concept

The evolved design (Phase 9): one **shared frozen backbone** extracts features once; the **router routes on those features**; each leaf is a tiny **isolated head**; a **canopy** merges the active head(s) into a prediction. Dormant heads cost nothing; new heads graft on; stale heads are pruned.

```mermaid
flowchart LR
    X([Input]) --> BB["Shared backbone 🧊<br/>learns features once"]
    BB -->|features| R{{"Stem router<br/>routes on features"}}
    R -->|argmax| H0["Head 0 🧊 active"]
    R --> H1["Head 1 🧊 active"]
    R -. dormant .-> H2["Head 2 🧊"]
    R -. dormant .-> H3["Head 3 🧊"]
    H0 --> C[["Canopy<br/>merge → prediction"]]
    H1 --> C
    classDef on fill:#2e7d32,stroke:#1b5e20,color:#fff;
    classDef off fill:#555,stroke:#333,color:#ddd;
    class BB,H0,H1,C on;
    class H2,H3 off;
```

It behaves like a living forest — a continuous **grow → graft → prune → regrow** loop, with every frozen head provably byte-identical across the whole cycle (`lifecycle_demo.py`):

```mermaid
flowchart LR
    Seed["🌱 Seed<br/>router + first leaf"] --> Grow["🌳 Grow<br/>train in isolation"]
    Grow --> Graft["🌿 Graft<br/>add domain, widen gate"]
    Graft --> Route["☀️ Route & serve<br/>log usage"]
    Route --> Monitor["📊 Monitor<br/>usage · accuracy"]
    Monitor --> Prune["✂️ Prune<br/>drop dormant head"]
    Prune -->|regrow| Graft
```

> Roadmap (Phase 13): many such trees linked underneath by a dense LLM — the "mycelial soil" — that decomposes a prompt across trees and synthesises their outputs. Not built yet.

---

## Benchmarks & metrics

- **Digits / MNIST:** DAS specialist leaves are compared against a single MLP of matched parameter count. Each leaf only ever sees its own domain's gradient, so it can't be pulled off-task by unrelated data.
- **Continual-learning baseline suite** — two pages, the honest competitor set: **DAS** vs **EWC** (Elastic Weight Consolidation) vs **PackNet** vs **Fine-tuned MLP** vs **Multi-task MLP** (upper bound).
  - **Split-MNIST** (`/continual`) — class-incremental, single-head: 5 binary tasks (0v1 … 8v9). The known-hard regime for soft methods.
  - **Permuted-MNIST** (`/permuted`) — domain-incremental: same 10-class task, a fixed pixel permutation per task. The regime where EWC is *expected* to work — included precisely so the suite isn't cherry-picked to always favor DAS.
- **Metrics reported:** Backward Transfer (BWT), plasticity (diagonal accuracy), stability (final ÷ first-learned), stored vs. active parameters, inference FLOPs, and wall-clock training time per phase.

Measured BWT (higher = less forgetting):

| Model | Split-MNIST | Permuted-MNIST | How it avoids forgetting |
|---|---|---|---|
| **DAS Forest** | **0.000** | **0.000** | structural — a frozen leaf per task |
| **PackNet** | **0.000** | **0.000** | structural — frozen weight masks in one fixed net |
| EWC MLP | −0.33 | **−0.03** | soft penalty; works in the easy regime, fails in the hard one |
| Fine-tuned MLP | −0.40 | −0.12 | nothing — catastrophic forgetting |

Two honest takeaways the suite is designed to surface: (1) **EWC's BWT improves ~10× from Split→Permuted** — exactly the documented regime sensitivity (van de Ven & Tolias, 2019); a benchmark that only ever favored DAS would be untrustworthy. (2) **PackNet matches DAS on forgetting but not on plasticity**: it shares one fixed-capacity network, so as weights get claimed, later tasks have fewer free weights and new-task accuracy erodes (measured: free weights 41.8k→31.4k→20.9k→10.5k→0 across the 5 tasks). DAS instead grows a new leaf per task — unbounded capacity at the cost of more stored parameters (but the same ~1-leaf inference cost).

- **Cross-domain contamination test** (`/continual`): every trained leaf is run on every task's test set. The diagonal (own domain) stays ~99%; off-diagonal (wrong domain) collapses to ~52% (binary chance). This proves leaves are genuine specialists **and** that the router is doing essential work — without it picking the diagonal, the forest would be near chance.

### Split-CIFAR — the real-image stress test (`cifar_bench.py`)

The first benchmark on real images (CIFAR-10, 5 binary tasks, CNN leaves). It was built specifically to find where the architecture breaks — and it does, exactly where predicted:

| Result | Number | Reading |
|---|---|---|
| **Router accuracy (raw pixels)** | **42%** | **The bottleneck.** A linear gate can't separate visual categories from raw pixels (it routes MNIST near-perfectly). |
| Per-leaf accuracy (task known) | 80–93% | The CNN experts themselves are fine. |
| DAS forgetting (BWT) | **0.000** | Structural — holds on CIFAR too. Checkpoint restore byte-exact. |
| Fine-tuned CNN (BWT) | −0.22 | Forgets, as expected. |
| Multi-task CNN (upper bound) | 79–94% | The ceiling. |

The honest takeaway: on real images the **experts work and the forgetting guarantee holds, but the linear-on-raw-pixels router collapses** — so end-to-end DAS is bottlenecked by routing. Routing on *learned features* via a shared backbone (Phase 9) helps but **does not fully fix it on CIFAR**: a shared conv backbone lifts routing from 0.42 → **0.66** (`backbone_cifar_bench.py`), far short of the ~0.98 it reaches on MNIST. CIFAR task-routing is essentially the hard 10-class problem, so it's capped by how well the backbone separates classes. The experts stay strong (mean head acc 0.91) and forgetting holds — the router remains the real ceiling on hard images.

### DAS vs LoRA — the make-or-break test (`lora_bench.py`)

Per-task LoRA adapters on a frozen backbone are the industry-standard way to "add capability without disturbing the rest." So we tested DAS isolated heads against per-task LoRA on the same frozen backbone:

| Metric | DAS (head) | LoRA (adapter) | Winner |
|---|---|---|---|
| Mean per-task accuracy | 0.993 | 0.997 | ~tie (LoRA +0.004) |
| Params per task | **130** | 11,010 | DAS (lighter) |
| Zero forgetting (isolated) | ✅ | ✅ | tie |
| Deletion (drop the module) | trivial | trivial | tie |
| Task-free routing built in | **yes** (router 95%) | no | DAS |

**The honest verdict:** DAS and per-task LoRA are nearly the same idea. They **tie** on isolation, forgetting, and deletion — LoRA gets those "for free" too. LoRA is marginally more accurate (it can re-tune features) at far more params/task; DAS heads are tiny. DAS's *one* structural edge is the **integrated, task-free router**. If the task is always known, plain LoRA is simpler and equivalent. **DAS earns its keep only where task-free routing + an audit trail genuinely matter** — i.e. the governance niche, not raw capability.

---

## What is real vs. hype

| Branded term | What it actually is |
|---|---|
| Vector torque (τ) | The router's softmax output — routing probabilities. |
| Stem Router | A standard MoE gate (linear + softmax + argmax). |
| Fibonacci leaf | An MLP whose layer widths happen to be Fibonacci numbers. |
| Coiled strings | Embedding vectors / hidden states. |
| Absolute domain isolation | Each expert is a separate net; freezing it freezes it. **Real.** |
| Modular grafting | Add a new expert and train only it. **Real.** |

**Claims to ignore:**
- **Fibonacci dimensions are not magic** — now *measured*, not just asserted (`leaf_shapes_bench.py`): Fibonacci vs power-of-two vs linear widths score within 0.006 of each other (noise). Expansive leaves buy +0.003 accuracy for 3× the parameters — capacity matters, the width *pattern* doesn't.
- **"You never touch the router when adding a domain" is false.** Experts stay isolated, but the router must learn the new route — see `graft()`.
- **Running 100B-param models on a laptop** does not follow from routing alone.
- This classifies small vectors/images. It is a scaffold, not an LLM.

---

## Honest positioning

DAS is not "better AI." It's **modular, auditable AI** for one specific pain: adding new capabilities without disturbing what's already deployed — zero-downtime domain expansion, compliance isolation (the hash proof is an audit trail), and incremental cost. The defensible angle vs. Avalanche / Flower / transformer-MoE is the **auditability + proof-of-isolation** story.

## Next steps

- ✅ **Done (Phase 5):** EWC baseline + cross-domain contamination test on `/continual`.
- ✅ **Done (Phase 6):** PackNet baseline and the Permuted-MNIST regime (`/permuted`).
- ✅ **Done (Phase 7):** PyTorch backend — autograd trainer, per-leaf & whole-forest checkpoint/restore (byte-exact), `ConvLeaf` CNN expert, and a REST inference API (`serve.py`).
- ✅ **Done (Lifecycle):** `ForestLifecycle` — usage monitoring, dormancy-based pruning (with router-gate shrink), regrow, **redundancy pruning** (drop a leaf that duplicates another, by output agreement), and **usage persistence**. The full grow → graft → prune → regrow loop with the forgetting proof holding throughout (`lifecycle_demo.py`).
- ✅ **Done (Phase 10):** top-k **canopy** merge — `DASForest.predict_canopy` blends the top-k leaves by routing weight for graceful degradation under routing uncertainty (top-2 ≥ top-1). Only valid when leaves share an output space; for disjoint-domain experts top-1 is correct (`canopy_demo.py`).
- ✅ **Done (Phase 9):** `BackboneForest` — a shared frozen backbone feeds a router that routes on *learned features* (not raw pixels), with tiny isolated heads (130 params each, ~1672× smaller than the backbone) sharing those features. Forgetting proof holds when grafting a new head (`backbone_demo.py`). Tradeoff: the backbone is a shared trainable component.
- ✅ **Done (Phase 8):** Split-CIFAR — CNN forest vs fine-tuned vs multi-task (`cifar_bench.py`). Finding: experts work, forgetting holds, but the raw-pixel router collapses to 42% — the bottleneck on real images.
- ✅ **Done (Phase 14):** DAS vs LoRA (`lora_bench.py`). Finding: they tie on isolation/forgetting/deletion; DAS's only structural edge is the built-in task-free router. **DAS's value is the governance niche, not raw capability.**

**Where this leaves the project (honest):** the architecture is a competent re-implementation of hard-routed MoE + parameter isolation, equivalent to per-task LoRA plus a router. Its defensible real-world home is **auditable, governed model fleets** — provable non-interference, deletion/unlearning, multi-tenant isolation — not "better/cheaper AI." Sensible directions from here:
1. ✅ Governance scenario proven (`governance_demo.py`): multi-tenant onboarding shows non-interference (prior tenants byte-identical), deletion/unlearning (a tenant's leaf is pruned → their task-acc falls 0.995 → 0.52 while others stay byte-identical), and a hash audit trail. A monolithic model can prove neither — this is DAS's real defensible edge.
2. ✅ Phase 9 on CIFAR (`backbone_cifar_bench.py`): shared conv backbone lifts routing 0.42 → 0.66 (helps, but CIFAR routing stays partly hard — not the ~0.98 MNIST gave).
✅ **Done (Phase 11):** tokenizer text front-end (`das/text.py`, `text_demo.py`) — the forest routes 4 text domains (math / sentiment / command / greeting) at 100%, each binary task solved, forgetting proof holds across grafting. **Front-end upgraded** (`embedding_demo.py`): a learned order-aware embedding beats bag-of-words on a word-order task (BoW stuck at 0.50 — identical word bags, opposite labels — vs embedding 1.00, generalising to unseen tokens). A pretrained LM encoder would slot in the same way.

✅ **Also done:** Progressive Neural Nets baseline (`pnn_bench.py`) — both PNN and DAS-style isolation get BWT 0; PNN's lateral connections add marginal forward transfer at growing parameter cost, DAS stays flat. Same tradeoff family as DAS vs LoRA (isolation is cheap; reuse costs parameters).

✅ **Also done:** expressive router study (`router_bench.py`) — a non-linear MLP router beats a linear gate on raw pixels (MNIST 0.88→0.96, CIFAR 0.40→0.45) but raw CIFAR routing stays poor; routing on *learned features* (Phase 9) helps more (0.42→0.66 on CIFAR, ~0.98 on MNIST) but doesn't fully fix CIFAR.

On the grand-vision pieces from the original framing — now built and measured rather than asserted:
- **JIT paging** (`paging_demo.py`): genuinely cuts device memory ~8× for a small latency tax, but **hardware-dependent** — nearly free on Apple Silicon's unified memory, far costlier on the discrete-GPU/PCIe systems the "100B on a laptop" pitch targets. Real technique, hardware-conditional claim.
- **Mycelial-LLM forest** (`mycelial_demo.py`): the orchestrator-decomposes-and-routes-to-trees pattern *works* (clauses route to the right specialist trees and synthesise). But the honest cost is `soil + k activated trees`, not "one tiny leaf" — a dense orchestrator runs on every query and dominates compute, and multi-domain queries fire multiple trees. The architecture holds; the "run a tiny fraction" economics don't.
- "90% cost cuts" and "beating frontier models" remain unsupported by the evidence.
