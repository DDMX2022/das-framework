"""
pnn_bench.py
------------
Progressive Neural Networks (Rusu et al., 2016) as a baseline — the closest prior
art to DAS. Like DAS it adds a NEW column per task and freezes old ones (so BWT=0),
but unlike DAS each new column gets LATERAL connections from the frozen hidden
activations of every previous column, so it can reuse old features (forward
transfer). The price: parameters grow super-linearly — column t carries a lateral
matrix to each of the t earlier columns.

This bench compares, on Split-MNIST (5 binary tasks):
  - DAS-style: independent column per task, NO laterals (pure isolation).
  - PNN:       column per task WITH laterals to all previous columns.
Both freeze old columns -> both get BWT=0. The interesting axes are forward
transfer (does PNN's lateral reuse help later tasks?) and parameter growth.

Forced CPU for determinism.
"""
import gzip, struct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cpu"
torch.manual_seed(0); np.random.seed(0)
H = 64
TASKS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]

def read_idx(path):
    with gzip.open(path, 'rb') as f:
        magic = struct.unpack('>I', f.read(4))[0]; nd = magic & 0xFF
        shape = tuple(struct.unpack('>I', f.read(4))[0] for _ in range(nd))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

base = './data/MNIST/raw'
X_tr = read_idx(f'{base}/train-images-idx3-ubyte.gz').reshape(-1, 784).astype(np.float32) / 255.0
y_tr = read_idx(f'{base}/train-labels-idx1-ubyte.gz').astype(np.int64)
X_te = read_idx(f'{base}/t10k-images-idx3-ubyte.gz').reshape(-1, 784).astype(np.float32) / 255.0
y_te = read_idx(f'{base}/t10k-labels-idx1-ubyte.gz').astype(np.int64)

def split(X, y, d0, d1):
    m = (y == d0) | (y == d1)
    return torch.from_numpy(X[m]), torch.from_numpy((y[m] == d1).astype(np.int64))

class Column(nn.Module):
    def __init__(self, in_dim, n_prev, use_laterals):
        super().__init__()
        self.W = nn.Linear(in_dim, H)
        self.laterals = nn.ModuleList(
            [nn.Linear(H, H, bias=False) for _ in range(n_prev)] if use_laterals else [])
        self.V = nn.Linear(H, 2)

    def forward(self, x, prev_hiddens):
        h = self.W(x)
        for lat, ph in zip(self.laterals, prev_hiddens):
            h = h + lat(ph)
        h = F.relu(h)
        return self.V(h), h

class Progressive(nn.Module):
    def __init__(self, use_laterals):
        super().__init__()
        self.use_laterals = use_laterals
        self.columns = nn.ModuleList()

    def add_column(self):
        self.columns.append(Column(784, len(self.columns), self.use_laterals).to(DEVICE))
        return len(self.columns) - 1

    def forward_upto(self, x, t):
        hiddens, outs = [], []
        for k in range(t + 1):
            out, h = self.columns[k](x, [hh.detach() for hh in hiddens])
            hiddens.append(h); outs.append(out)
        return outs[t]

def run(use_laterals):
    net = Progressive(use_laterals)
    matrix = [[None] * len(TASKS) for _ in range(len(TASKS))]
    diag = []
    col_params = []
    for t, (d0, d1) in enumerate(TASKS):
        net.add_column()
        for k in range(t):                                   # freeze previous columns
            for p in net.columns[k].parameters():
                p.requires_grad_(False)
        Xt, yt = split(X_tr, y_tr, d0, d1)
        Xt, yt = Xt.to(DEVICE), yt.to(DEVICE)
        opt = torch.optim.Adam([p for p in net.columns[t].parameters() if p.requires_grad], lr=1e-3)
        net.train()
        for _ in range(300):
            idx = torch.randint(0, len(Xt), (128,), device=DEVICE)
            opt.zero_grad(); F.cross_entropy(net.forward_upto(Xt[idx], t), yt[idx]).backward(); opt.step()
        col_params.append(sum(p.numel() for p in net.columns[t].parameters()))
        net.eval()
        with torch.no_grad():
            for ev in range(t + 1):
                Xe, ye = split(X_te, y_te, *TASKS[ev])
                matrix[t][ev] = (net.forward_upto(Xe.to(DEVICE), ev).argmax(1) == ye.to(DEVICE)).float().mean().item()
        diag.append(matrix[t][t])
    bwt = np.mean([matrix[-1][i] - matrix[i][i] for i in range(len(TASKS) - 1)])
    return diag, bwt, col_params

print("=" * 64)
print(" Progressive Nets vs pure isolation — Split-MNIST")
print("=" * 64)
das_diag, das_bwt, das_params = run(use_laterals=False)
pnn_diag, pnn_bwt, pnn_params = run(use_laterals=True)

print(f"\n  {'task':<8}{'DAS acc':>10}{'PNN acc':>10}{'DAS params':>14}{'PNN params':>14}")
for t in range(len(TASKS)):
    print(f"  {str(TASKS[t]):<8}{das_diag[t]:>10.4f}{pnn_diag[t]:>10.4f}{das_params[t]:>14,}{pnn_params[t]:>14,}")
print(f"\n  mean diagonal (plasticity):  DAS {np.mean(das_diag):.4f}   PNN {np.mean(pnn_diag):.4f}")
print(f"  BWT (forgetting):            DAS {das_bwt:+.4f}   PNN {pnn_bwt:+.4f}   (both ~0 = structural)")
print(f"  params task 0 -> task 4:     DAS {das_params[0]:,} -> {das_params[-1]:,} (flat)")
print(f"                               PNN {pnn_params[0]:,} -> {pnn_params[-1]:,} (grows with laterals)")
print("\n  Takeaway: both freeze old columns so neither forgets. PNN's laterals let")
print("  later tasks reuse earlier features (forward transfer) but cost extra params")
print("  per task; DAS keeps columns independent — cheaper, no transfer. Same tradeoff")
print("  family as DAS vs LoRA: isolation is cheap; reuse costs parameters.")
print("=" * 64)
