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

**The forgetting proof:** snapshot every frozen leaf's hash, train a new leaf, re-hash. If any previously-frozen leaf changed, the proof fails. It never does — that's the point.

---

## Benchmarks & metrics

- **Digits / MNIST:** DAS specialist leaves are compared against a single MLP of matched parameter count. Each leaf only ever sees its own domain's gradient, so it can't be pulled off-task by unrelated data.
- **Split-MNIST continual learning** (`/continual`) compares three regimes — the honest competitor set:
  - **DAS Forest** — one frozen leaf grafted per task
  - **Fine-tuned MLP** — one shared net fine-tuned sequentially (this is what people actually do; it forgets)
  - **Multi-task MLP** — all tasks at once (upper bound)
- **Metrics reported:** Backward Transfer (BWT), plasticity (diagonal accuracy), stability (final ÷ first-learned), stored vs. active parameters, inference FLOPs, and wall-clock training time per phase.

Typical result: DAS BWT ≈ 0.00 while the fine-tuned MLP lands around −0.20 to −0.30 (visible catastrophic forgetting), at the cost of storing more total parameters but using **fewer at inference** (only 1 leaf + router activate per prediction).

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

1. Standard continual-learning baselines (EWC, PackNet, Progressive Nets) on Split/Permuted-MNIST and Split-CIFAR.
2. CNN leaves and an attention-based router; per-leaf checkpoint/restore; a REST inference API.
3. A cross-domain contamination test (run Leaf 0 on Domain 1) to prove the router is doing real work.
