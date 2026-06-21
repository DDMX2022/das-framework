# DAS framework — working prototype

A minimal, runnable implementation of the "DAS" sparse-expert architecture:
a hard-routed Mixture-of-Experts where each expert ("leaf") is a fully
isolated network that can be trained and frozen independently.

## What's in here

```
das-framework/
├── das/
│   ├── functional.py   FibonacciLeaf  — an expert MLP (manual backprop, NumPy)
│   ├── routing.py      StemRouter     — MoE gate (softmax top-1)
│   └── model.py        DASForest      — assembles router + leaves
├── demo.py             Full lifecycle + proof of zero forgetting (NumPy, CPU)
├── das_torch.py        PyTorch version for your Mac (autograd + MPS)
└── checkpoints/        (for saved weights)
```

## Run it

NumPy version (no install needed beyond numpy):

```
python3 demo.py
```

PyTorch version (on your Mac):

```
pip install torch
python das_torch.py
```

On Apple Silicon it auto-selects the `mps` (Metal) device for GPU speedup.

## What is actually real (honest translation)

| Branded term            | What it really is                                            |
|-------------------------|-------------------------------------------------------------|
| Vector torque (τ)       | The softmax output of the router. Routing probabilities.    |
| Stem Router             | A standard MoE gating network (linear + softmax + argmax).  |
| Fibonacci leaf          | An MLP whose layer widths happen to be Fibonacci numbers.   |
| Coiled strings          | Embedding vectors / hidden states.                          |
| Absolute domain isolation | Each expert is a separate network; freezing it freezes it. |
| Modular grafting        | Add a new expert and train only it.                         |

The genuinely useful property — and the one `demo.py` proves — is that
training a newly grafted expert leaves the existing experts **byte-identical**
(zero catastrophic forgetting). That is real and follows directly from the
isolation.

## What is NOT true (claims to ignore)

- **Fibonacci dimensions are not magic.** 144→89→55 works no better than
  128→96→64. Layer widths are ordinary hyperparameters.
- **"You never touch the router when adding a domain" is false.** The experts
  stay isolated, but the router must learn the new route — see `graft()`.
- **Running 100B-parameter models on a laptop** via this design is not something
  the architecture delivers. The hard part (a strong base model, real
  synthesis across experts) is not solved by routing alone.
- This prototype classifies tiny synthetic vectors. It is a learning scaffold,
  not a language model.

## Sensible next steps

1. Swap the synthetic data in `demo.py` for a real small dataset.
2. In `das_torch.py`, port the training loop from `demo.py` using autograd.
3. Add a real tokenizer + embedding instead of raw vectors.
4. Measure: does hard top-1 routing actually beat one shared MLP on YOUR task?
   That comparison is the honest way to know if the approach helps you.
