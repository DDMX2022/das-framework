"""
das/functional.py
------------------
The "Fibonacci Leaf" — a specialist expert network.
HONEST NOTE: despite the branding, this is a plain multi-layer perceptron (MLP).
The only thing "Fibonacci" about it is that the layer widths can follow a
Fibonacci descent (e.g. 21 -> 13 -> 8 -> output). That sizing is aesthetic;
any descending widths work just as well. What matters is that each leaf is a
*fully separate* network with its own weights, so it can be trained and frozen
in complete isolation.
"""
import hashlib
import numpy as np

def relu(x):
    return np.maximum(0.0, x)

def relu_grad(z):
    return (z > 0).astype(z.dtype)

def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

class FibonacciLeaf:
    """A standalone expert. Forward pass returns class logits.
    Backward applies one gradient step UNLESS the leaf is frozen."""

    def __init__(self, dims, seed=0):
        self.dims = list(dims)
        rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for i in range(len(dims) - 1):
            # He initialisation
            self.W.append(rng.normal(0, np.sqrt(2.0 / dims[i]), (dims[i], dims[i + 1])))
            self.b.append(np.zeros(dims[i + 1]))
        self.frozen = False

    def forward(self, x):
        self.cache = [x]   # activations
        self.pre = []      # pre-activations
        a = x
        for i in range(len(self.W)):
            z = a @ self.W[i] + self.b[i]
            self.pre.append(z)
            a = relu(z) if i < len(self.W) - 1 else z  # last layer = raw logits
            self.cache.append(a)
        return a

    def grads(self, dlogits):
        """Backprop dlogits to parameter gradients WITHOUT applying them.
        Returns (gW, gb) lists. Used both for the normal update and for
        estimating Fisher information (EWC)."""
        gW = [None] * len(self.W)
        gb = [None] * len(self.b)
        d = dlogits
        for i in reversed(range(len(self.W))):
            a_prev = self.cache[i]
            gW[i] = a_prev.T @ d
            gb[i] = d.sum(axis=0)
            if i > 0:
                d = (d @ self.W[i].T) * relu_grad(self.pre[i - 1])
        return gW, gb

    def backward(self, dlogits, lr, ewc_lambda=0.0, ewc_tasks=None):
        """dlogits: gradient of loss w.r.t. output logits, shape (N, out_dim).

        EWC (Elastic Weight Consolidation): when ewc_lambda>0 and ewc_tasks is a
        list of {'fisher','star'} consolidations, add the quadratic penalty
        gradient  λ · Σ_k F_k · (θ − θ*_k)  to each parameter. This pulls weights
        back toward what mattered for earlier tasks — the standard soft way to
        resist forgetting (contrast with DAS, which forbids it structurally)."""
        gW, gb = self.grads(dlogits)
        if ewc_lambda and ewc_tasks:
            for tk in ewc_tasks:
                F, star = tk['fisher'], tk['star']
                for i in range(len(self.W)):
                    gW[i] = gW[i] + ewc_lambda * F['W'][i] * (self.W[i] - star['W'][i])
                    gb[i] = gb[i] + ewc_lambda * F['b'][i] * (self.b[i] - star['b'][i])
        if not self.frozen:                       # <-- the isolation guarantee
            for i in range(len(self.W)):
                self.W[i] -= lr * gW[i]
                self.b[i] -= lr * gb[i]

    def snapshot(self):
        """Deep copy of all parameters (θ*) — the EWC anchor for a finished task."""
        return {'W': [w.copy() for w in self.W], 'b': [b.copy() for b in self.b]}

    def fisher_diagonal(self, X, y, ce_grad_fn, n_batches=24, batch=128, seed=0):
        """Diagonal Fisher information ≈ mean of squared parameter gradients over
        minibatches of (X, y). Tells EWC which weights are important for this task."""
        FW = [np.zeros_like(w) for w in self.W]
        Fb = [np.zeros_like(b) for b in self.b]
        rng = np.random.default_rng(seed)
        n = len(X)
        for _ in range(n_batches):
            idx = rng.integers(0, n, min(batch, n))
            gW, gb = self.grads(ce_grad_fn(self.forward(X[idx]), y[idx]))
            for i in range(len(self.W)):
                FW[i] += gW[i] ** 2
                Fb[i] += gb[i] ** 2
        return {'W': [f / n_batches for f in FW], 'b': [f / n_batches for f in Fb]}

    def weight_hash(self):
        """Fingerprint of every weight. Used to PROVE a frozen leaf never moved."""
        h = hashlib.sha256()
        for w in self.W:
            h.update(np.ascontiguousarray(w).tobytes())
        for b in self.b:
            h.update(np.ascontiguousarray(b).tobytes())
        return h.hexdigest()[:16]
