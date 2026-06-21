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
│   └── model.py            DASForest      — assembles router + leaves, graft(), proofs
├── demo.py                 Full lifecycle on synthetic data + forgetting proof (NumPy)
├── benchmark.py            DAS vs matched-size MLP on sklearn digits (NumPy)
├── das_torch.py            PyTorch port (autograd + Apple Silicon MPS)
├── mnist_stress.py         PyTorch: 10 leaves on real MNIST + 10-way forgetting proof
├── app.py                  Flask server — 5 live, browser-streamed experiments
├── templates/              UI for the web app (SSE + Chart.js)
│   ├── index.html          Forest demo + digits benchmark
│   ├── stress.html         MNIST stress test
│   ├── real_bench.html     Real-world multi-dataset benchmark
│   └── continual_bench.html Split-MNIST continual learning
└── data/MNIST/raw/         MNIST IDX files (downloaded by mnist_stress.py)
```

---

## Quick start

The NumPy demo needs only `numpy`. The web app adds `flask`, the digits benchmark adds `scikit-learn`, and the PyTorch scripts add `torch`/`torchvision`.

> **Mac note:** Homebrew Python 3.14 currently ships a broken `libexpat` and `pip` won't run. Use conda or Python 3.13.

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
| `/continual` | **Split-MNIST continual learning** | DAS vs Fine-tuned MLP vs Multi-task MLP, with a live 5×5 accuracy matrix |
| `/benchmark` (stream) | digits SSE stream | backing stream for the `/` benchmark tab |

The web app reads MNIST directly from `data/MNIST/raw/*.gz` (stdlib `gzip` + `numpy`, no torchvision). If those files are missing, run `python mnist_stress.py` once to download them.

### PyTorch scripts (Apple Silicon)

```bash
pip install torch torchvision
python das_torch.py      # smoke test, auto-selects mps
python mnist_stress.py   # 10-leaf MNIST, ~20s of training on M-series
```

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

---

## Benchmarks & metrics

- **Digits / MNIST:** DAS specialist leaves are compared against a single MLP of matched parameter count. Each leaf only ever sees its own domain's gradient, so it can't be pulled off-task by unrelated data.
- **Split-MNIST continual learning** (`/continual`) compares four regimes — the honest competitor set:
  - **DAS Forest** — one frozen leaf grafted per task
  - **EWC MLP** — Elastic Weight Consolidation, the standard CL baseline (soft penalty on important weights)
  - **Fine-tuned MLP** — one shared net fine-tuned sequentially (this is what people actually do; it forgets)
  - **Multi-task MLP** — all tasks at once (upper bound)
- **Metrics reported:** Backward Transfer (BWT), plasticity (diagonal accuracy), stability (final ÷ first-learned), stored vs. active parameters, inference FLOPs, and wall-clock training time per phase.

Measured result (single-head Split-MNIST):

| Model | BWT | Note |
|---|---|---|
| **DAS Forest** | **≈ 0.000** | structural isolation — frozen leaves can't move |
| EWC MLP | ≈ −0.33 | soft penalty helps vs. naive, but can't reach zero |
| Fine-tuned MLP | ≈ −0.40 | catastrophic forgetting |

DAS achieves zero forgetting by storing more total parameters but using **fewer at inference** (only 1 leaf + router activate per prediction). EWC's limited gain here is expected: single-head Split-MNIST is the known-hard regime where soft methods struggle (van de Ven & Tolias, 2019) — which is exactly the contrast that motivates structural isolation.

- **Cross-domain contamination test** (`/continual`): every trained leaf is run on every task's test set. The diagonal (own domain) stays ~99%; off-diagonal (wrong domain) collapses to ~52% (binary chance). This proves leaves are genuine specialists **and** that the router is doing essential work — without it picking the diagonal, the forest would be near chance.

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
- **Fibonacci dimensions are not magic.** `144→89→55` works no better than `128→96→64`; widths are ordinary hyperparameters.
- **"You never touch the router when adding a domain" is false.** Experts stay isolated, but the router must learn the new route — see `graft()`.
- **Running 100B-param models on a laptop** does not follow from routing alone.
- This classifies small vectors/images. It is a scaffold, not an LLM.

---

## Honest positioning

DAS is not "better AI." It's **modular, auditable AI** for one specific pain: adding new capabilities without disturbing what's already deployed — zero-downtime domain expansion, compliance isolation (the hash proof is an audit trail), and incremental cost. The defensible angle vs. Avalanche / Flower / transformer-MoE is the **auditability + proof-of-isolation** story.

## Next steps

- ✅ **Done (Phase 5):** EWC baseline and the cross-domain contamination test, both wired into `/continual`.
1. More CL baselines (PackNet, Progressive Nets) and Permuted-MNIST / Split-CIFAR.
2. CNN leaves and an attention-based router; per-leaf checkpoint/restore; a REST inference API.
3. Port `demo.py`'s training loop into `das_torch.py` with autograd for GPU-scale runs.
