"""
das_torch.py
------------
PyTorch version of the DAS prototype, for running on your Mac.
Why a separate file? PyTorch gives you automatic differentiation (no manual
backprop), GPU/MPS acceleration, and a clean path to scaling up. The NumPy
version (das/ + demo.py) is for understanding the mechanics; this is for
actually building.
Run on Apple Silicon with Metal acceleration:
    pip install torch
    python das_torch.py
The device line below auto-selects MPS (Apple GPU) if available.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

device = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print(f"Using device: {device}")

class FibonacciLeaf(nn.Module):
    """A specialist expert = a plain MLP with Fibonacci-shaped widths.
    (The Fibonacci sizing is cosmetic; isolation is the real feature.)"""
    def __init__(self, dims):  # e.g. [21, 13, 8, 2]
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad_(False)

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad_(True)

class StemRouter(nn.Module):
    """MoE gate: linear + softmax. argmax = hard top-1 ('vector torque')."""
    def __init__(self, d_model, num_leaves):
        super().__init__()
        self.gate = nn.Linear(d_model, num_leaves)

    def forward(self, h):
        logits = self.gate(h)
        tau = F.softmax(logits, dim=-1)
        leaf = tau.argmax(dim=-1)
        return leaf, tau, logits

class DASForest(nn.Module):
    def __init__(self, d_model, leaf_dims, num_leaves):
        super().__init__()
        self.leaf_dims = leaf_dims
        self.router = StemRouter(d_model, num_leaves)
        self.leaves = nn.ModuleList([FibonacciLeaf(leaf_dims) for _ in range(num_leaves)])

    def predict(self, h):
        leaf_idx, _, _ = self.router(h)
        out = torch.zeros(h.shape[0], self.leaf_dims[-1], device=h.device)
        for i, leaf in enumerate(self.leaves):
            mask = leaf_idx == i
            if mask.any():
                out[mask] = leaf(h[mask])
        return out, leaf_idx

    def graft(self, new_dims=None):
        """Add a new expert. NOTE: you must also give the router a short update
        so it learns the new route — the experts stay isolated, the router does not."""
        self.leaves.append(FibonacciLeaf(new_dims or self.leaf_dims).to(next(self.parameters()).device))
        old = self.router.gate
        new = nn.Linear(old.in_features, old.out_features + 1).to(old.weight.device)
        with torch.no_grad():
            new.weight[:-1] = old.weight
            new.bias[:-1] = old.bias
            new.weight[-1].normal_(0, 0.01)
            new.bias[-1].zero_()
        self.router.gate = new
        return len(self.leaves) - 1

if __name__ == "__main__":
    # quick smoke test on random data
    torch.manual_seed(0)
    forest = DASForest(d_model=21, leaf_dims=[21, 13, 8, 2], num_leaves=2).to(device)
    h = torch.randn(32, 21, device=device)
    out, idx = forest.predict(h)
    print("output shape:", tuple(out.shape), "| routed to leaves:", idx.unique().tolist())
    nid = forest.graft()
    print("grafted new leaf id:", nid, "| total leaves:", len(forest.leaves))
    print("Smoke test passed. Adapt demo.py's training loop here with autograd.")
