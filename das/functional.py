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

    def backward(self, dlogits, lr):
        """dlogits: gradient of loss w.r.t. output logits, shape (N, out_dim)."""
        gW = [None] * len(self.W)
        gb = [None] * len(self.b)
        d = dlogits
        for i in reversed(range(len(self.W))):
            a_prev = self.cache[i]
            gW[i] = a_prev.T @ d
            gb[i] = d.sum(axis=0)
            if i > 0:
                d = (d @ self.W[i].T) * relu_grad(self.pre[i - 1])
        if not self.frozen:                       # <-- the isolation guarantee
            for i in range(len(self.W)):
                self.W[i] -= lr * gW[i]
                self.b[i] -= lr * gb[i]

    def weight_hash(self):
        """Fingerprint of every weight. Used to PROVE a frozen leaf never moved."""
        h = hashlib.sha256()
        for w in self.W:
            h.update(np.ascontiguousarray(w).tobytes())
        for b in self.b:
            h.update(np.ascontiguousarray(b).tobytes())
        return h.hexdigest()[:16]
