"""
das/packnet.py
--------------
PackNet — Mallya & Lazebnik (2018): iterative pruning + a per-task binary
weight mask, all inside ONE shared network. It is the honest "competitor" to
DAS on the zero-forgetting property: once a task claims a weight, that weight
is frozen for every later task, exactly like a frozen DAS leaf. The difference
is capacity. DAS grows a brand new leaf per task (unbounded capacity, at the
cost of more stored parameters). PackNet carves up a FIXED-size network —
weights are a scarce resource that gets divided among tasks as you go. Each
task after the first gets fewer free weights to learn with, so accuracy on
new tasks (plasticity) should erode over time. That tradeoff — zero forgetting
but shrinking capacity — is the whole point of including PackNet here; it is
not a strawman, it is a real and reasonably competitive baseline.

Same forward/backprop math as FibonacciLeaf (plain ReLU MLP, softmax-CE
trained elsewhere). The only new machinery is the ownership mask.
"""
import numpy as np

def relu(x):
    return np.maximum(0.0, x)

def relu_grad(z):
    return (z > 0).astype(z.dtype)


class PackNetMLP:
    """A single shared MLP with a per-weight 'owner' (task id, or -1 = free).
    Once a weight is owned by task t, it is frozen for tasks > t forever."""

    def __init__(self, dims, seed=0):
        self.dims = list(dims)
        rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for i in range(len(dims) - 1):
            self.W.append(rng.normal(0, np.sqrt(2.0 / dims[i]), (dims[i], dims[i + 1])))
            self.b.append(np.zeros(dims[i + 1]))
        # -1 = unclaimed/free. Once set to a task id, that weight is frozen forever.
        self.owner = [np.full(w.shape, -1, dtype=int) for w in self.W]
        # Biases aren't masked/pruned (PackNet's own paper only prunes weights,
        # not biases) — instead we snapshot a per-task bias vector so each task's
        # subnetwork gets the bias values that were in effect when IT finished.
        self.task_bias = {}   # task id -> [b.copy() ...]

    # -- internal: plain forward/backward, identical math to FibonacciLeaf.
    #    `eff_W` lets a caller run the forward/backward through a MASKED copy of
    #    the weights (used by the refinetune pass so it sees exactly the
    #    subnetwork that inference will use — see train_task step 3). --
    def _forward(self, x, eff_W=None):
        Ws = eff_W if eff_W is not None else self.W
        self.cache = [x]
        self.pre = []
        a = x
        for i in range(len(Ws)):
            z = a @ Ws[i] + self.b[i]
            self.pre.append(z)
            a = relu(z) if i < len(Ws) - 1 else z
            self.cache.append(a)
        return a

    def _grads(self, dlogits, eff_W=None):
        Ws = eff_W if eff_W is not None else self.W   # backprop through the same weights forward used
        gW = [None] * len(self.W)
        gb = [None] * len(self.b)
        d = dlogits
        for i in reversed(range(len(self.W))):
            a_prev = self.cache[i]
            gW[i] = a_prev.T @ d
            gb[i] = d.sum(axis=0)
            if i > 0:
                d = (d @ Ws[i].T) * relu_grad(self.pre[i - 1])
        return gW, gb

    def train_task(self, t, X, y, ce_grad, lr, steps, batch, n_tasks, rng=None,
                    refinetune_steps=300):
        """Train task t using ONLY the currently-free weights, then prune: the
        top fraction (by |magnitude|) of what's still free gets permanently
        assigned to task t, the rest stays free for later tasks.

        n_tasks is the total task count — used for the equal-allocation scheme
        keep_count = round(n_free / (n_tasks - t)), so each remaining task
        claims an even slice of whatever capacity is left and the LAST task
        sweeps up everything that's still unclaimed.
        """
        if rng is None:
            rng = np.random.default_rng(1000 + t)
        free_mask = [(own == -1) for own in self.owner]
        n = len(X)

        # 1) Train: gradient updates apply ONLY where free_mask is True.
        for s in range(steps):
            idx = rng.integers(0, n, min(batch, n))
            logits = self._forward(X[idx])
            gW, gb = self._grads(ce_grad(logits, y[idx]))
            for i in range(len(self.W)):
                self.W[i] -= lr * gW[i] * free_mask[i]
                self.b[i] -= lr * gb[i]          # biases are never masked

        # 2) Prune: among weights still free, keep the top |magnitude| chunk
        #    for task t; release the rest back to the free pool for later tasks.
        remaining_tasks = n_tasks - t   # includes this task
        all_free_vals = np.concatenate([
            np.abs(self.W[i][free_mask[i]]) for i in range(len(self.W))
        ]) if any(m.any() for m in free_mask) else np.array([0.0])
        n_free = all_free_vals.size
        keep_count = int(round(n_free / max(remaining_tasks, 1)))
        if keep_count > 0 and n_free > 0:
            # threshold = the keep_count-th largest |weight| among free weights
            thresh = np.partition(all_free_vals, max(n_free - keep_count, 0))[max(n_free - keep_count, 0)]
        else:
            thresh = np.inf
        for i in range(len(self.W)):
            claim = free_mask[i] & (np.abs(self.W[i]) >= thresh)
            self.owner[i][claim] = t
            # zero out anything that did NOT get claimed and is still free —
            # NB: we do not zero it; it just remains free (-1) for next task.

        # 3) Brief re-finetune restricted to task-t-owned positions only, to
        #    recover accuracy lost from pruning. The forward/backward run through
        #    the EXACT subnetwork inference will use for task t — weights owned by
        #    tasks 0..t, with still-free weights masked to zero — so refinetune
        #    optimises the real deployed network, not a larger one it can't keep.
        owned_mask = [(own == t) for own in self.owner]
        eff_mask = [((own != -1) & (own <= t)) for own in self.owner]
        for s in range(refinetune_steps):
            idx = rng.integers(0, n, min(batch, n))
            eff_W = [self.W[i] * eff_mask[i] for i in range(len(self.W))]
            logits = self._forward(X[idx], eff_W=eff_W)
            gW, gb = self._grads(ce_grad(logits, y[idx]), eff_W=eff_W)
            for i in range(len(self.W)):
                self.W[i] -= lr * gW[i] * owned_mask[i]
                self.b[i] -= lr * gb[i]

        # 4) Snapshot biases for this task — used at inference time for task t.
        self.task_bias[t] = [bb.copy() for bb in self.b]

    def free_count(self):
        """How many weights are still unclaimed (diagnostic / plasticity check)."""
        return int(sum((own == -1).sum() for own in self.owner))

    def forward_task(self, x, t):
        """Forward pass using only weights owned by tasks 0..t (mask out
        anything still free / owned by a future task), with task t's bias
        snapshot. This is "the subnetwork as it existed right after task t."""
        eff_mask = [((own != -1) & (own <= t)) for own in self.owner]
        bias = self.task_bias[t]
        a = x
        for i in range(len(self.W)):
            Wi = self.W[i] * eff_mask[i]
            z = a @ Wi + bias[i]
            a = relu(z) if i < len(self.W) - 1 else z
        return a
