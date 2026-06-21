"""
embedding_demo.py
-----------------
Upgrades the text front-end from bag-of-words to a LEARNED, order-aware embedding,
and shows exactly where it matters: word ORDER.

Task: "alpha beats X" (label 1, alpha wins) vs "X beats alpha" (label 0, alpha
loses). For any team X these two sentences have the SAME bag of words
{alpha, beats, X} but OPPOSITE labels — so a BoW front-end literally cannot tell
them apart and is stuck at chance. A learned embedding over token *positions* can.
We also test on UNSEEN teams to check it learns "is position 0 == alpha", not
memorisation.

(Honest scope: this is a learned-from-scratch embedding, not a pretrained LM
encoder — but it's a real, order-sensitive front-end that plugs into the forest
in place of BoW, and it cleanly demonstrates BoW's ceiling.)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0); np.random.seed(0)
DEVICE = "cpu"

teams = [f"team{i}" for i in range(20)]
vocab = ["alpha", "beats"] + teams
w2id = {w: i for i, w in enumerate(vocab)}
V = len(vocab)

def make(team, alpha_first):
    return (["alpha", "beats", team], 1) if alpha_first else ([team, "beats", "alpha"], 0)

train_teams, test_teams = teams[:14], teams[14:]   # held-out teams test generalization
train = [make(t, b) for t in train_teams for b in (True, False)]
test = [make(t, b) for t in test_teams for b in (True, False)]

def bow(tokens):
    v = np.zeros(V, dtype=np.float32)
    for t in tokens: v[w2id[t]] += 1.0
    return v

def seq(tokens):
    return np.array([w2id[t] for t in tokens], dtype=np.int64)

Xb_tr = torch.tensor(np.stack([bow(t) for t, _ in train]))
Xb_te = torch.tensor(np.stack([bow(t) for t, _ in test]))
Xs_tr = torch.tensor(np.stack([seq(t) for t, _ in train]))
Xs_te = torch.tensor(np.stack([seq(t) for t, _ in test]))
y_tr = torch.tensor([l for _, l in train]); y_te = torch.tensor([l for _, l in test])

class BoWNet(nn.Module):
    def __init__(self): super().__init__(); self.net = nn.Sequential(nn.Linear(V, 16), nn.ReLU(), nn.Linear(16, 2))
    def forward(self, x): return self.net(x)

class EmbNet(nn.Module):
    """Learned embedding per token, flattened over positions -> order-aware."""
    def __init__(self, edim=8):
        super().__init__()
        self.emb = nn.Embedding(V, edim)
        self.net = nn.Sequential(nn.Linear(3 * edim, 16), nn.ReLU(), nn.Linear(16, 2))
    def forward(self, ids): return self.net(self.emb(ids).flatten(1))

def train_eval(model, Xtr, Xte):
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    model.train()
    for _ in range(800):
        i = torch.randint(0, len(Xtr), (16,))
        opt.zero_grad(); F.cross_entropy(model(Xtr[i]), y_tr[i]).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        tr = (model(Xtr).argmax(1) == y_tr).float().mean().item()
        te = (model(Xte).argmax(1) == y_te).float().mean().item()
    return tr, te

print("=" * 60)
print(" Text front-end: bag-of-words vs learned embedding")
print(" Task: word ORDER decides the label (alpha beats X / X beats alpha)")
print("=" * 60)
bt, be = train_eval(BoWNet(), Xb_tr, Xb_te)
et, ee = train_eval(EmbNet(), Xs_tr, Xs_te)
print(f"\n  {'front-end':<22}{'train acc':>12}{'test acc':>12}")
print(f"  {'bag-of-words':<22}{bt:>12.3f}{be:>12.3f}")
print(f"  {'learned embedding':<22}{et:>12.3f}{ee:>12.3f}")
print(f"\n  BoW is stuck near chance: identical word bags, opposite labels.")
print(f"  The embedding reads token POSITION and generalises to unseen teams.")
print("=" * 60)
import sys
sys.exit(0 if (ee > 0.9 and be < 0.7) else 1)
