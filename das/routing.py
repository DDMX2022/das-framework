"""
das/routing.py
--------------
The "Stem Router" — decides which leaf handles each input.
HONEST NOTE: this is a standard Mixture-of-Experts gate. One linear layer,
then a softmax. The "vector torque (tau)" from the marketing IS this softmax
output — a vector of routing probabilities. We then take the argmax (hard
top-1 routing), which is what gives DAS its "send 100% of the signal down one
path" behaviour.
"""
import numpy as np
from .functional import softmax

class StemRouter:
    def __init__(self, d_model, num_leaves, seed=0):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, np.sqrt(2.0 / d_model), (d_model, num_leaves))
        self.b = np.zeros(num_leaves)
        self.num_leaves = num_leaves

    def route(self, h):
        logits = h @ self.W + self.b
        tau = softmax(logits)              # routing weights ("torque")
        leaf = np.argmax(tau, axis=-1)     # hard top-1 selection
        return leaf, tau

    def route_topk(self, h, k=2):
        """Soft top-k routing for the canopy: return the k highest-weighted
        leaves per input and their RENORMALISED weights (sum to 1 over the k).
        HONEST NOTE: merging k leaves only makes sense when the leaves share an
        output space (e.g. all 10-class). For DAS's usual disjoint-domain experts,
        merge is meaningless and top-1 is the correct behaviour — see
        DASForest.predict_canopy."""
        logits = h @ self.W + self.b
        tau = softmax(logits)
        k = min(k, tau.shape[1])
        idx = np.argsort(-tau, axis=1)[:, :k]           # (N, k) top-k leaf ids
        w = np.take_along_axis(tau, idx, axis=1)         # (N, k) their weights
        w = w / w.sum(axis=1, keepdims=True)             # renormalise over the k
        return idx, w

    def train_step(self, h, domain_labels, lr):
        """Supervised: learn to predict the correct domain for each input."""
        logits = h @ self.W + self.b
        tau = softmax(logits)
        N = h.shape[0]
        onehot = np.zeros_like(tau)
        onehot[np.arange(N), domain_labels] = 1.0
        dlogits = (tau - onehot) / N
        self.W -= lr * (h.T @ dlogits)
        self.b -= lr * dlogits.sum(axis=0)
        loss = -np.log(tau[np.arange(N), domain_labels] + 1e-9).mean()
        acc = (np.argmax(tau, axis=-1) == domain_labels).mean()
        return loss, acc

    def expand(self, seed=99):
        """Grafting: add a routing slot for a new leaf (a new output column).
        HONEST NOTE: the marketing claims you NEVER touch the router when adding
        a domain. That is false. The experts stay isolated, but the router must
        learn the new route. This just creates the slot; you still give the
        router a short update so it knows the new domain exists.
        """
        rng = np.random.default_rng(seed)
        new_col = rng.normal(0, 0.01, (self.W.shape[0], 1))
        self.W = np.concatenate([self.W, new_col], axis=1)
        self.b = np.concatenate([self.b, [0.0]])
        self.num_leaves += 1
