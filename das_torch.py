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
import hashlib
import json
import os

device = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print(f"Using device: {device}")

class FibonacciLeaf(nn.Module):
    """A specialist expert = a plain MLP with Fibonacci-shaped widths.
    (The Fibonacci sizing is cosmetic; isolation is the real feature.)"""
    leaf_type = 'mlp'

    def __init__(self, dims):  # e.g. [21, 13, 8, 2]
        super().__init__()
        self.dims = list(dims)
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

class ConvLeaf(nn.Module):
    """A heterogeneous expert: same flat-vector in/out contract as
    FibonacciLeaf (so DASForest doesn't need to know which kind of leaf it's
    holding), but internally reshapes to a square image and runs conv+ReLU+pool
    blocks before a small FC head. This is the proof that DAS's "uniform leaf
    API" isn't just an MLP API in disguise — leaves can be architecturally
    whatever they need to be, as long as forward() maps a flat (N, in_dim)
    tensor to a flat (N, out_dim) tensor.
    dims is kept as [in_dim, out_dim] only, purely for checkpoint metadata
    and to mirror FibonacciLeaf's interface — it does not describe the conv
    stack's internal shapes.
    channels=1 (default) reproduces the original MNIST-only behavior exactly
    (a 2-block 16->32 stack) — that path is untouched so old checkpoints keep
    loading. channels=3 (CIFAR) switches to a deeper 3-block 32->64->128 stack;
    more channels means more raw signal per pixel, so the bigger stack isn't
    cosmetic, it's needed to do anything useful with 32x32x3 input."""
    leaf_type = 'conv'

    def __init__(self, dims, channels=1):  # dims = [in_dim, out_dim], in_dim must be channels * side^2
        super().__init__()
        self.dims = list(dims)
        self.channels = channels
        in_dim, out_dim = dims[0], dims[-1]
        side = int(round((in_dim / channels) ** 0.5))
        assert channels * side * side == in_dim, (
            f"ConvLeaf expects in_dim == channels * side^2, got in_dim={in_dim}, channels={channels}"
        )
        self.side = side
        if channels == 1:
            # Original MNIST-shaped stack, unchanged, so existing checkpoints
            # (saved before multi-channel support existed) still load and run
            # byte-identically.
            self.conv = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # side -> side/2
                nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # side/2 -> side/4
            )
            flat_dim = 32 * (side // 4) * (side // 4)
        else:
            # Deeper stack for real (multi-channel) images, e.g. CIFAR-10's
            # 32x32x3: three conv blocks 32->64->128 with pooling, then an FC head.
            self.conv = nn.Sequential(
                nn.Conv2d(channels, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # side -> side/2
                nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),         # side/2 -> side/4
                nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),        # side/4 -> side/8
            )
            flat_dim = 128 * (side // 8) * (side // 8)
        self.fc = nn.Sequential(
            nn.Linear(flat_dim, 64), nn.ReLU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.side, self.side)
        x = self.conv(x)
        x = x.flatten(1)
        return self.fc(x)

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

    def predict_canopy(self, h, k=2):
        """Top-k canopy: blend the k highest-weighted leaves by routing weight.
        Graceful degradation vs hard top-1; only valid when leaves share an
        output space."""
        _, tau, _ = self.router(h)
        k = min(k, len(self.leaves))
        topw, topi = tau.topk(k, dim=-1)
        topw = topw / topw.sum(-1, keepdim=True)
        out = torch.zeros(h.shape[0], self.leaf_dims[-1], device=h.device)
        for slot in range(k):
            idx = topi[:, slot]
            for i, leaf in enumerate(self.leaves):
                m = idx == i
                if m.any():
                    out[m] += topw[m, slot:slot + 1] * leaf(h[m])
        return out, topi

    def graft(self, new_dims=None):
        """Add a new expert. NOTE: you must also give the router a short update
        so it learns the new route — the experts stay isolated, the router does not."""
        self.leaves.append(FibonacciLeaf(new_dims or self.leaf_dims).to(next(self.parameters()).device))
        self._expand_gate()
        return len(self.leaves) - 1

    def graft_leaf(self, leaf):
        """Like graft(), but for an ALREADY-BUILT leaf (e.g. one restored from
        disk via load_leaf). This is the operational version of grafting: you
        trained a leaf once, somewhere, and now you're attaching it to a forest
        without retraining it. The router still needs a new slot — that part
        is identical to graft()."""
        self.leaves.append(leaf.to(next(self.parameters()).device))
        self._expand_gate()
        return len(self.leaves) - 1

    def _expand_gate(self):
        """Shared plumbing for graft()/graft_leaf(): widen the router's output
        layer by one column, copying over the old weights so existing routes
        don't change, and randomly initialising the new column."""
        old = self.router.gate
        new = nn.Linear(old.in_features, old.out_features + 1).to(old.weight.device)
        with torch.no_grad():
            new.weight[:-1] = old.weight
            new.bias[:-1] = old.bias
            new.weight[-1].normal_(0, 0.01)
            new.bias[-1].zero_()
        self.router.gate = new

def leaf_hash(leaf):
    """SHA-256 over every parameter tensor's raw bytes, truncated to 16 hex
    chars. Same recipe as mnist_stress.py. This is the entire "proof" in DAS:
    if the hash doesn't move, the weights didn't move, full stop — no need to
    trust eval-mode metrics that can look stable while weights still drift."""
    h = hashlib.sha256()
    for p in leaf.parameters():
        h.update(p.data.cpu().numpy().tobytes())
    return h.hexdigest()[:16]

def train_router(forest, X, y_domain, steps=600, lr=1e-3, batch=128, device="cpu"):
    """Supervised router training: cross-entropy on the gate logits against
    the known domain id. This is the ONE part of DAS that is NOT isolated —
    the router is a shared, ordinary classifier and is allowed to keep
    learning as new domains show up (see graft()'s docstring)."""
    X = X.to(device)
    y_domain = y_domain.to(device)
    opt = torch.optim.Adam(forest.router.parameters(), lr=lr)
    n = X.shape[0]
    forest.router.train()
    for s in range(steps):
        idx = torch.randint(0, n, (min(batch, n),), device=device)
        xb, yb = X[idx], y_domain[idx]
        opt.zero_grad()
        _, _, logits = forest.router(xb)
        loss = F.cross_entropy(logits, yb)
        loss.backward()
        opt.step()
    with torch.no_grad():
        _, _, logits = forest.router(X)
        acc = (logits.argmax(1) == y_domain).float().mean().item()
    return acc

def train_leaf_isolated(forest, leaf_id, X, y, steps=600, lr=1e-3, batch=128, device="cpu"):
    """The isolation mechanism, spelled out:
      1. Freeze EVERY leaf in the forest (including leaf_id itself).
      2. Unfreeze only leaf_id.
      3. Build an Adam optimizer over ONLY leaf_id's parameters — even if
         autograd somehow computed a gradient for a frozen leaf, the
         optimizer has no reference to it and could not update it.
      4. Train. Re-freeze leaf_id when done so it's safe to leave lying
         around (the forest's default state is "everything frozen").
    Two independent locks (requires_grad=False AND "not in the optimizer")
    is the honest way to guarantee isolation — either one alone is one bug
    away from leaking."""
    for leaf in forest.leaves:
        leaf.freeze()
    leaf = forest.leaves[leaf_id]
    leaf.unfreeze()

    X = X.to(device)
    y = y.to(device)
    opt = torch.optim.Adam(leaf.parameters(), lr=lr)
    n = X.shape[0]
    leaf.train()
    for s in range(steps):
        idx = torch.randint(0, n, (min(batch, n),), device=device)
        xb, yb = X[idx], y[idx]
        opt.zero_grad()
        loss = F.cross_entropy(leaf(xb), yb)
        loss.backward()
        opt.step()
    leaf.freeze()
    leaf.eval()

    with torch.no_grad():
        acc = (leaf(X).argmax(1) == y).float().mean().item()
    return acc

# ── Checkpointing ──────────────────────────────────────────────────────
# A leaf checkpoint is a plain dict: {'type', 'dims', 'state_dict'}. 'type'
# lets load_leaf() reconstruct the right class (FibonacciLeaf vs ConvLeaf,
# the latter added in Stage 3) instead of assuming everything is an MLP.
# LoRALeaf (Phase 1) is the odd one out: it HOLDS A REFERENCE to a shared
# backbone, so leaf.state_dict() would otherwise include the backbone's
# weights too (nn.Module walks all submodules). We deliberately save/load
# only the adapter+head sub-state-dict — the backbone is persisted exactly
# once by LoRAForest.save(), not duplicated per leaf.

def save_leaf(leaf, path):
    """Freeze a leaf to disk. Just dims + state_dict + a type tag — nothing
    fancy, no optimizer state, because a frozen leaf has no optimizer.
    For ConvLeaf we also persist 'channels' (1 for grayscale MNIST-style,
    3 for RGB CIFAR-style) so load_leaf can rebuild the right conv stack —
    FibonacciLeaf has no such attribute, hence getattr with a default.
    For LoRALeaf we persist 'rank' and ONLY the adapter+head state_dict
    (A1,B1,A2,B2,head) — never the shared backbone's weights, which would
    otherwise get duplicated into every single leaf checkpoint on disk."""
    if getattr(leaf, 'leaf_type', 'mlp') == 'lora':
        ckpt = {
            'type': 'lora',
            'dims': list(leaf.dims),
            'rank': leaf.rank,
            'state_dict': {
                'A1': leaf.A1, 'B1': leaf.B1, 'A2': leaf.A2, 'B2': leaf.B2,
                'head.weight': leaf.head.weight, 'head.bias': leaf.head.bias,
            },
        }
        torch.save(ckpt, path)
        return
    ckpt = {
        'type': getattr(leaf, 'leaf_type', 'mlp'),
        'dims': list(leaf.dims),
        'channels': getattr(leaf, 'channels', 1),
        'state_dict': leaf.state_dict(),
    }
    torch.save(ckpt, path)

def load_leaf(path, device='cpu', backbone=None):
    """Reconstruct a leaf from a checkpoint saved by save_leaf(). Returns it
    frozen, since a leaf loaded from disk is by definition "already trained"
    — if you want to keep training it, call .unfreeze() yourself.
    'channels' defaults to 1 so checkpoints saved before multi-channel
    ConvLeaf support existed (MNIST, grayscale) still load correctly.
    'backbone' is new (Phase 1) and ONLY required for type=='lora' — a
    LoRALeaf can't exist without the shared frozen backbone it adapts, so the
    caller (typically LoRAForest.load) must hand one in. Default None keeps
    this function backward-compatible: existing mlp/conv/head call sites
    that never pass backbone are untouched."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    leaf_type = ckpt.get('type', 'mlp')
    if leaf_type == 'mlp':
        leaf = FibonacciLeaf(ckpt['dims'])
        leaf.load_state_dict(ckpt['state_dict'])
    elif leaf_type == 'conv':
        leaf = ConvLeaf(ckpt['dims'], channels=ckpt.get('channels', 1))
        leaf.load_state_dict(ckpt['state_dict'])
    elif leaf_type == 'lora':
        if backbone is None:
            raise ValueError("load_leaf: type=='lora' checkpoints need backbone= (the shared frozen base)")
        leaf = LoRALeaf(backbone, rank=ckpt.get('rank', 8), out_dim=ckpt['dims'][-1])
        sd = ckpt['state_dict']
        with torch.no_grad():
            leaf.A1.copy_(sd['A1']); leaf.B1.copy_(sd['B1'])
            leaf.A2.copy_(sd['A2']); leaf.B2.copy_(sd['B2'])
            leaf.head.weight.copy_(sd['head.weight']); leaf.head.bias.copy_(sd['head.bias'])
    else:
        raise ValueError(f"unknown leaf type in checkpoint: {leaf_type!r}")
    leaf.to(device)
    leaf.freeze()
    return leaf

def save_forest(forest, dir):
    """Save a whole forest: router state_dict + every leaf (via save_leaf)
    + a manifest.json recording leaf count, dims, and types so load_forest
    can rebuild the exact same shape before loading weights into it."""
    os.makedirs(dir, exist_ok=True)
    torch.save(forest.router.state_dict(), os.path.join(dir, 'router.pt'))
    manifest = {
        'd_model': forest.router.gate.in_features,
        'leaf_dims': list(forest.leaf_dims),
        'num_leaves': len(forest.leaves),
        'leaf_types': [getattr(leaf, 'leaf_type', 'mlp') for leaf in forest.leaves],
    }
    for i, leaf in enumerate(forest.leaves):
        save_leaf(leaf, os.path.join(dir, f'leaf_{i}.pt'))
    with open(os.path.join(dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

def load_forest(dir, device='cpu'):
    """Rebuild a forest exactly as save_forest() left it: same router gate
    shape, same leaves in the same order, all loaded byte-for-byte from
    disk (not re-initialised then maybe-overwritten — reconstructed fresh,
    then load_state_dict'd, so there's nothing for stale init values to
    hide behind)."""
    with open(os.path.join(dir, 'manifest.json')) as f:
        manifest = json.load(f)
    # num_leaves=1 here only to give StemRouter a valid (non-zero-width) gate
    # to build; it gets discarded and replaced below once we know the real
    # width from the saved router state_dict.
    forest = DASForest(manifest['d_model'], manifest['leaf_dims'], 1).to(device)
    forest.leaves = nn.ModuleList()
    router_sd = torch.load(os.path.join(dir, 'router.pt'), map_location=device, weights_only=False)
    # the manifest's num_leaves tells us the router gate's true width, which
    # may differ from leaf_dims-derived defaults if leaves were grafted.
    out_features = router_sd['gate.weight'].shape[0]
    forest.router.gate = nn.Linear(manifest['d_model'], out_features).to(device)
    forest.router.load_state_dict(router_sd)
    for i in range(manifest['num_leaves']):
        leaf = load_leaf(os.path.join(dir, f'leaf_{i}.pt'), device=device)
        forest.leaves.append(leaf)
    return forest

# ── Phase 9: shared frozen backbone + isolated heads ───────────────────
# The original forest makes every leaf a FULL separate network, so each one
# re-learns low-level features from scratch and the router routes on raw input.
# This version factors that out: a SHARED backbone learns features once (then is
# frozen), the router routes on those features, and each leaf shrinks to a tiny
# isolated HEAD on top. Heads still train in isolation and freeze, so
# zero-forgetting holds for them; the tradeoff is that the backbone is a shared
# trainable component (retraining it would shift every head).

class Backbone(nn.Module):
    """Shared feature extractor. Train once, freeze, then everything downstream
    (router + heads) operates on its features instead of the raw input."""
    def __init__(self, in_dim, feat_dim, hidden=256):
        super().__init__()
        self.in_dim, self.feat_dim = in_dim, feat_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, feat_dim), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad_(False)

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad_(True)

class HeadLeaf(nn.Module):
    """A tiny isolated expert: a small head on the shared backbone features.
    dims = [feat_dim, out_dim] or [feat_dim, hidden, out_dim]."""
    leaf_type = 'head'

    def __init__(self, dims):
        super().__init__()
        self.dims = list(dims)
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

class BackboneForest(nn.Module):
    """Shared backbone + router-on-features + isolated heads."""
    def __init__(self, in_dim, feat_dim, head_dims, num_leaves):
        super().__init__()
        self.head_dims = list(head_dims)
        self.backbone = Backbone(in_dim, feat_dim)
        self.router = StemRouter(feat_dim, num_leaves)
        self.heads = nn.ModuleList([HeadLeaf(head_dims) for _ in range(num_leaves)])

    def features(self, x):
        return self.backbone(x)

    def predict(self, x):
        h = self.features(x)
        leaf_idx, _, _ = self.router(h)
        out = torch.zeros(x.shape[0], self.head_dims[-1], device=x.device)
        for i, head in enumerate(self.heads):
            mask = leaf_idx == i
            if mask.any():
                out[mask] = head(h[mask])
        return out, leaf_idx

    def predict_canopy(self, x, k=2):
        """Top-k canopy over the shared-backbone heads (blend by routing weight)."""
        h = self.features(x)
        _, tau, _ = self.router(h)
        k = min(k, len(self.heads))
        topw, topi = tau.topk(k, dim=-1)
        topw = topw / topw.sum(-1, keepdim=True)
        out = torch.zeros(x.shape[0], self.head_dims[-1], device=x.device)
        for slot in range(k):
            idx = topi[:, slot]
            for i, head in enumerate(self.heads):
                m = idx == i
                if m.any():
                    out[m] += topw[m, slot:slot + 1] * head(h[m])
        return out, topi

    def freeze_backbone(self):
        self.backbone.freeze()

    def graft_head(self, dims=None):
        """Add a new isolated head + widen the router gate by one column."""
        self.heads.append(HeadLeaf(dims or self.head_dims).to(next(self.parameters()).device))
        old = self.router.gate
        new = nn.Linear(old.in_features, old.out_features + 1).to(old.weight.device)
        with torch.no_grad():
            new.weight[:-1] = old.weight
            new.bias[:-1] = old.bias
            new.weight[-1].normal_(0, 0.01)
            new.bias[-1].zero_()
        self.router.gate = new
        return len(self.heads) - 1

def train_head_isolated(forest, head_id, feats, y, steps=400, lr=1e-3, batch=128, device="cpu"):
    """Train ONE head on precomputed backbone features, in isolation: backbone
    frozen, all other heads frozen, optimizer scoped to this head only."""
    for head in forest.heads:
        head.freeze()
    head = forest.heads[head_id]
    head.unfreeze()
    feats, y = feats.to(device), y.to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    n = feats.shape[0]
    head.train()
    for _ in range(steps):
        idx = torch.randint(0, n, (min(batch, n),), device=device)
        opt.zero_grad()
        loss = F.cross_entropy(head(feats[idx]), y[idx])
        loss.backward()
        opt.step()
    head.freeze()
    head.eval()
    with torch.no_grad():
        acc = (head(feats).argmax(1) == y).float().mean().item()
    return acc

# ── Phase 1 (PRODUCT_PLAN): LoRA experts on a shared frozen backbone ───
# The project's own benchmark (lora_bench.py) found DAS ~= "LoRA + a router".
# The honest move is to stop competing with LoRA and adopt it: experts become
# real LoRA adapters, not bespoke leaf classes. LoRALeaf below is exactly
# lora_bench.LoRAExpert, promoted from a one-off benchmark class into a
# first-class leaf type with the same freeze/unfreeze/dims/leaf_type contract
# as FibonacciLeaf/ConvLeaf/HeadLeaf, so it drops into the existing
# checkpoint + forest machinery instead of needing its own.
#
# Production hook (NOT done here — no network, no transformer libs in this
# env): swap the local `Backbone` for a frozen HF encoder and use peft's LoRA
# instead of hand-rolled A/B matrices:
#   from transformers import AutoModel
#   base = AutoModel.from_pretrained("...").eval()   # frozen, like Backbone
#   from peft import LoraConfig, get_peft_model
#   base = get_peft_model(base, LoraConfig(r=8, target_modules=[...]))
# The forest API (route on shared features -> one adapter -> logits) does not
# change; only what sits inside "backbone" and "adapter" does.

class LoRALeaf(nn.Module):
    """A LoRA adapter + head on a SHARED FROZEN Backbone — the expert format
    the project decided to standardize on (see PRODUCT_PLAN.md Phase 1).
    Same math as lora_bench.LoRAExpert: low-rank deltas A (rank x in) / B (out
    x rank) added to each of the backbone's two linears, B initialised to zero
    so a fresh adapter starts as a no-op (the leaf begins as the untouched
    backbone + a freshly-initialised head). Only A1,B1,A2,B2,head are ever
    trained; backbone.parameters() are never touched by this class — they are
    a reference to someone else's shared, frozen module."""
    leaf_type = 'lora'

    def __init__(self, backbone, rank=8, out_dim=2):
        super().__init__()
        self.backbone = backbone   # shared, frozen — NOT a copy, NOT trained here
        self.dims = [backbone.in_dim, out_dim]
        self.rank = rank
        l1, l2 = backbone.net[0], backbone.net[2]
        self.A1 = nn.Parameter(torch.randn(rank, l1.in_features) * 0.01)
        self.B1 = nn.Parameter(torch.zeros(l1.out_features, rank))
        self.A2 = nn.Parameter(torch.randn(rank, l2.in_features) * 0.01)
        self.B2 = nn.Parameter(torch.zeros(l2.out_features, rank))
        self.head = nn.Linear(l2.out_features, out_dim)

    def adapter_params(self):
        """The ONLY trainable params for this leaf — used to scope the
        optimizer in train_leaf_isolated so the backbone can't be touched
        even by accident."""
        return [self.A1, self.B1, self.A2, self.B2, *self.head.parameters()]

    def forward(self, x):
        l1, l2 = self.backbone.net[0], self.backbone.net[2]
        h1 = F.relu(F.linear(x, l1.weight, l1.bias) + F.linear(F.linear(x, self.A1), self.B1))
        h2 = F.relu(F.linear(h1, l2.weight, l2.bias) + F.linear(F.linear(h1, self.A2), self.B2))
        return self.head(h2)

    def freeze(self):
        """Freeze the ADAPTER + HEAD only. The backbone is someone else's
        shared module with its own lifecycle (see BackboneForest.freeze_backbone);
        toggling requires_grad on it here would silently affect every other
        leaf sharing the same backbone instance, which is the opposite of
        isolation. adapter_params() is the deliberately narrow surface."""
        for p in self.adapter_params():
            p.requires_grad_(False)

    def unfreeze(self):
        for p in self.adapter_params():
            p.requires_grad_(True)

class LoRAForest(nn.Module):
    """A forest of LoRA experts sharing ONE frozen backbone. Mirrors
    BackboneForest's shape (shared backbone, router-on-features, per-task
    modules) but the per-task module is a LoRALeaf (adapter + head) instead
    of a bare HeadLeaf — i.e. each expert can also re-tune the backbone's
    features via its own low-rank delta, not just add a head on top."""
    def __init__(self, in_dim, feat_dim, out_dim, num_leaves, rank=8):
        super().__init__()
        self.out_dim = out_dim
        self.rank = rank
        self.backbone = Backbone(in_dim, feat_dim)
        self.router = StemRouter(feat_dim, num_leaves)
        self.leaves = nn.ModuleList(
            [LoRALeaf(self.backbone, rank=rank, out_dim=out_dim) for _ in range(num_leaves)]
        )

    def features(self, x):
        return self.backbone(x)

    def freeze_backbone(self):
        self.backbone.freeze()

    def predict(self, x):
        """Route on backbone features (top-1), run the chosen LoRALeaf on the
        RAW input (each adapter re-derives its own features from x through
        backbone + its own delta — that's the point of LoRA vs a plain head)."""
        h = self.features(x)
        leaf_idx, _, _ = self.router(h)
        out = torch.zeros(x.shape[0], self.out_dim, device=x.device)
        for i, leaf in enumerate(self.leaves):
            mask = leaf_idx == i
            if mask.any():
                out[mask] = leaf(x[mask])
        return out, leaf_idx

    def graft_leaf(self):
        """Add a new LoRALeaf on the SAME shared backbone + widen the router
        gate by one column. Mirrors BackboneForest.graft_head."""
        new_leaf = LoRALeaf(self.backbone, rank=self.rank, out_dim=self.out_dim)
        new_leaf.to(next(self.backbone.parameters()).device)
        self.leaves.append(new_leaf)
        old = self.router.gate
        new = nn.Linear(old.in_features, old.out_features + 1).to(old.weight.device)
        with torch.no_grad():
            new.weight[:-1] = old.weight
            new.bias[:-1] = old.bias
            new.weight[-1].normal_(0, 0.01)
            new.bias[-1].zero_()
        self.router.gate = new
        return len(self.leaves) - 1

    def save(self, dir):
        """Save the shared backbone ONCE + every leaf's adapter (via
        save_leaf, which for type=='lora' skips the backbone — see below).
        Manifest records rank/out_dim/num_leaves so load() can rebuild the
        exact same shapes before loading weights in."""
        os.makedirs(dir, exist_ok=True)
        torch.save(self.backbone.state_dict(), os.path.join(dir, 'backbone.pt'))
        torch.save(self.router.state_dict(), os.path.join(dir, 'router.pt'))
        manifest = {
            'in_dim': self.backbone.in_dim,
            'feat_dim': self.backbone.feat_dim,
            'out_dim': self.out_dim,
            'rank': self.rank,
            'num_leaves': len(self.leaves),
        }
        for i, leaf in enumerate(self.leaves):
            save_leaf(leaf, os.path.join(dir, f'leaf_{i}.pt'))
        with open(os.path.join(dir, 'manifest.json'), 'w') as f:
            json.dump(manifest, f, indent=2)

    @staticmethod
    def load(dir, device='cpu'):
        """Rebuild a LoRAForest exactly as save() left it: backbone loaded
        once, then every leaf reconstructed against THAT SAME backbone
        instance (not a fresh one) so leaves truly share it, matching the
        live-object invariant."""
        with open(os.path.join(dir, 'manifest.json')) as f:
            manifest = json.load(f)
        forest = LoRAForest(
            manifest['in_dim'], manifest['feat_dim'], manifest['out_dim'],
            num_leaves=1, rank=manifest['rank'],
        ).to(device)
        backbone_sd = torch.load(os.path.join(dir, 'backbone.pt'), map_location=device, weights_only=False)
        forest.backbone.load_state_dict(backbone_sd)
        forest.backbone.freeze()
        router_sd = torch.load(os.path.join(dir, 'router.pt'), map_location=device, weights_only=False)
        out_features = router_sd['gate.weight'].shape[0]
        forest.router.gate = nn.Linear(manifest['feat_dim'], out_features).to(device)
        forest.router.load_state_dict(router_sd)
        forest.leaves = nn.ModuleList()
        for i in range(manifest['num_leaves']):
            leaf = load_leaf(os.path.join(dir, f'leaf_{i}.pt'), device=device, backbone=forest.backbone)
            forest.leaves.append(leaf)
        return forest

def train_leaf_isolated_lora(forest, leaf_id, X, y, steps=400, lr=1e-3, batch=128, device="cpu"):
    """The LoRAForest analogue of train_leaf_isolated/train_head_isolated:
    freeze every leaf's adapter, unfreeze only leaf_id's, scope the optimizer
    to leaf.adapter_params() (never the backbone, never another leaf's
    adapter). Trains on raw x (not precomputed features) since each adapter
    needs to run its own forward through backbone+delta — that's the cost of
    LoRA vs a plain head, paid once per step, not a correctness issue."""
    for leaf in forest.leaves:
        leaf.freeze()
    leaf = forest.leaves[leaf_id]
    leaf.unfreeze()

    X = X.to(device)
    y = y.to(device)
    opt = torch.optim.Adam(leaf.adapter_params(), lr=lr)
    n = X.shape[0]
    leaf.train()
    for _ in range(steps):
        idx = torch.randint(0, n, (min(batch, n),), device=device)
        xb, yb = X[idx], y[idx]
        opt.zero_grad()
        loss = F.cross_entropy(leaf(xb), yb)
        loss.backward()
        opt.step()
    leaf.freeze()
    leaf.eval()

    with torch.no_grad():
        acc = (leaf(X).argmax(1) == y).float().mean().item()
    return acc

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
