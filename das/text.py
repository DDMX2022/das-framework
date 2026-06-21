"""
das/text.py
-----------
A minimal text front-end so the forest can route language, not just raw vectors.
HONEST NOTE: this is a bag-of-words vectorizer, not a learned LLM embedding — it
turns a string into a fixed-length count vector over a fitted vocabulary, which is
enough to demonstrate routing + isolated experts on text. A real system would swap
this for a tokenizer + embedding (or a frozen LM encoder) as the shared front-end;
the forest API downstream (router + leaves on a d_model vector) is unchanged.
"""
import re
import numpy as np

class Tokenizer:
    def __init__(self):
        self.vocab = {}

    def _toks(self, text):
        return re.findall(r"[a-z0-9]+", text.lower())

    def fit(self, texts):
        for t in texts:
            for w in self._toks(t):
                if w not in self.vocab:
                    self.vocab[w] = len(self.vocab)
        return self

    def transform(self, texts):
        X = np.zeros((len(texts), len(self.vocab)), dtype=np.float64)
        for i, t in enumerate(texts):
            for w in self._toks(t):
                j = self.vocab.get(w)
                if j is not None:
                    X[i, j] += 1.0
        return X

    @property
    def dim(self):
        return len(self.vocab)
