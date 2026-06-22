"""
encoder_demo.py
---------------
A PRETRAINED (frozen) encoder front-end for the forest, and the property that
makes pretraining worth it: TRANSFER to words the downstream task never saw.

Honest scope: the environment has no transformer libs and the network is too slow
to pull a foundation model, so we pretrain a small encoder IN-ENVIRONMENT on a
broad multi-topic corpus, freeze it, and show it transfers. This demonstrates the
*pattern and its value* — not a foundation model. A real LM encoder (BERT,
sentence-transformers) slots in via the exact same interface: see the hook at the
bottom.

Setup:
  - Pretrain an encoder (word embedding + mean-pool) on 4-way TOPIC classification
    over a broad vocabulary, then FREEZE it. Same-topic words end up with similar
    embeddings.
  - Downstream task: "is this about animals?" — trained on SOME animal words, but
    TESTED on DIFFERENT animal words (held out of downstream training, but present
    in pretraining).
  - Compare: frozen pretrained encoder vs an encoder trained FROM SCRATCH on the
    downstream data only. Pretraining should generalise to the held-out words;
    from-scratch can't (it never saw them).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0); np.random.seed(0)
DIM = 24

TOPICS = {
    "animals": ["cat", "dog", "cow", "horse", "sheep", "goat", "pig", "hen"],
    "food":    ["bread", "rice", "apple", "milk", "cheese", "egg", "fish", "meat"],
    "weather": ["rain", "sun", "snow", "wind", "storm", "cloud", "fog", "heat"],
    "sports":  ["run", "jump", "kick", "throw", "swim", "ride", "climb", "row"],
}
vocab = ["the", "and"] + [w for ws in TOPICS.values() for w in ws]
w2id = {w: i for i, w in enumerate(vocab)}
topic_list = list(TOPICS)

def enc_ids(words):
    return [w2id[w] for w in words]

class Encoder(nn.Module):
    """word embedding + mean-pool -> a sentence vector (the front-end)."""
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(len(vocab), DIM)
    def forward(self, batch_ids):
        return torch.stack([self.emb(torch.tensor(ids)).mean(0) for ids in batch_ids])
    def freeze(self):
        for p in self.parameters(): p.requires_grad_(False)

# ── 1. Pretrain the encoder on broad TOPIC classification, then freeze ──
def make_topic_data(n=2000):
    ids, lab = [], []
    for _ in range(n):
        t = np.random.randint(4); ws = TOPICS[topic_list[t]]
        a, b = np.random.choice(ws, 2)
        ids.append(enc_ids(["the", a, "and", b])); lab.append(t)
    return ids, torch.tensor(lab)

enc = Encoder()
head = nn.Linear(DIM, 4)
opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=1e-2)
ids, lab = make_topic_data()
for _ in range(400):
    i = np.random.randint(0, len(ids), 64)
    feat = enc([ids[j] for j in i])
    opt.zero_grad(); F.cross_entropy(head(feat), lab[i]).backward(); opt.step()
enc.freeze()

# ── 2. Downstream "is animals?" — train words != test words ─────
train_animals = ["cat", "dog", "cow", "horse"]
test_animals = ["sheep", "goat", "pig", "hen"]      # held out of downstream training
others = TOPICS["food"] + TOPICS["weather"] + TOPICS["sports"]
np.random.shuffle(others)
neg_train, neg_test = others[:12], others[12:]

def ds(words_pos, words_neg):
    ids = [enc_ids(["the", w]) for w in words_pos] + [enc_ids(["the", w]) for w in words_neg]
    y = torch.tensor([1] * len(words_pos) + [0] * len(words_neg))
    return ids, y

tr_ids, tr_y = ds(train_animals, neg_train)
te_ids, te_y = ds(test_animals, neg_test)

def train_clf(encoder, train_encoder):
    enc2 = encoder
    clf = nn.Linear(DIM, 2)
    params = list(clf.parameters()) + (list(enc2.parameters()) if train_encoder else [])
    if train_encoder:
        for p in enc2.parameters(): p.requires_grad_(True)
    o = torch.optim.Adam(params, lr=1e-2)
    for _ in range(300):
        feat = enc2(tr_ids)
        if not train_encoder: feat = feat.detach()
        o.zero_grad(); F.cross_entropy(clf(feat), tr_y).backward(); o.step()
    with torch.no_grad():
        tr = (clf(enc2(tr_ids)).argmax(1) == tr_y).float().mean().item()
        te = (clf(enc2(te_ids)).argmax(1) == te_y).float().mean().item()
    return tr, te

pt_tr, pt_te = train_clf(enc, train_encoder=False)               # frozen pretrained
scratch = Encoder()
sc_tr, sc_te = train_clf(scratch, train_encoder=True)            # from scratch

print("=" * 60)
print(" Pretrained (frozen) encoder front-end vs from-scratch")
print(" Downstream: 'is animals?' — TEST words held out of training")
print("=" * 60)
print(f"\n  {'encoder':<26}{'train acc':>12}{'test acc':>12}")
print(f"  {'pretrained (frozen)':<26}{pt_tr:>12.3f}{pt_te:>12.3f}")
print(f"  {'from scratch':<26}{sc_tr:>12.3f}{sc_te:>12.3f}")
print(f"\n  Held-out animal words: {test_animals}")
print("  Pretrained features cluster same-topic words, so the classifier")
print("  generalises to animal words it never saw downstream. From-scratch")
print("  never learned those words -> it can't.")

print("\n  --- hook for a REAL LM encoder ---")
print("  Replace Encoder with a frozen sentence-transformer:")
print("    from sentence_transformers import SentenceTransformer")
print("    m = SentenceTransformer('all-MiniLM-L6-v2')   # frozen")
print("    feat = torch.tensor(m.encode(list_of_sentences))  # -> feed the forest")
print("  Same interface (sentence -> vector); not downloaded here (no libs / slow net).")
print("=" * 60)
import sys
sys.exit(0 if pt_te > sc_te else 1)
