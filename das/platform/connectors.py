"""
das/platform/connectors.py
--------------------------
The "last mile" seam. A ``ContextSource`` turns a client's raw query (text, a
record id, a dict) into the fixed-size embedding the DAS router consumes. This is
the ONE place a client's real data integration plugs in — SQL, a legacy REST API,
a vector store — and it is a stable ~50-line contract, not a fork of DAS internals.

The platform ships reference implementations so a deployment runs end-to-end:

  * ``StaticContextSource``   — a fixed mapping (tests, fixtures).
  * ``CallableContextSource`` — wrap any ``query -> vector`` function.
  * ``RestContextSource``     — POST the query to an embedding endpoint (urllib,
                                zero extra deps; matches the repo's http style).
  * ``SpecKeywordConnector``  — keyword-match a text query to the right expert and
                                embed near that expert's center; makes text route
                                sensibly in demos/POCs without a real encoder.

An FDE writes the client-specific source (e.g. ``SqlContextSource``) against the
same ``embed`` contract. Nothing else in the platform changes.
"""
from __future__ import annotations

import json
import urllib.request
from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional

import numpy as np


class ContextSource(ABC):
    """Fixed contract: query -> a (1, d_model) float32 embedding for the router.

    Implementations must return a 2-D array with exactly one row and the
    deployment's ``d_model`` columns. The platform validates shape at the seam so
    a broken connector fails loudly, not silently mis-routes.
    """

    def __init__(self, d_model: int):
        self.d_model = d_model

    @abstractmethod
    def _embed(self, query) -> np.ndarray:
        """Return an embedding for one query. Shape (d_model,) or (1, d_model)."""

    def embed(self, query) -> np.ndarray:
        vec = np.asarray(self._embed(query), dtype=float)
        if vec.ndim == 1:
            vec = vec[None, :]
        if vec.shape != (1, self.d_model):
            raise ValueError(
                f"{type(self).__name__}.embed produced shape {vec.shape}, "
                f"expected (1, {self.d_model})"
            )
        return vec


class StaticContextSource(ContextSource):
    """Serve embeddings from a fixed ``{key: vector}`` mapping. For tests and
    fixtures where the embedding is known ahead of time."""

    def __init__(self, d_model: int, mapping: Dict[object, np.ndarray]):
        super().__init__(d_model)
        self.mapping = {k: np.asarray(v, dtype=float) for k, v in mapping.items()}

    def _embed(self, query) -> np.ndarray:
        if query not in self.mapping:
            raise KeyError(f"StaticContextSource has no embedding for {query!r}")
        return self.mapping[query]


class CallableContextSource(ContextSource):
    """Wrap any ``fn(query) -> vector`` as a ContextSource. The escape hatch for
    integrations that don't warrant their own class."""

    def __init__(self, d_model: int, fn: Callable[[object], np.ndarray]):
        super().__init__(d_model)
        self._fn = fn

    def _embed(self, query) -> np.ndarray:
        return self._fn(query)


class RestContextSource(ContextSource):
    """POST ``{"query": ...}`` to an embedding endpoint and read back
    ``{"embedding": [...]}``. Uses urllib so it adds no dependency. This is the
    reference for wiring a client's own embedding/feature service."""

    def __init__(self, d_model: int, url: str, timeout: float = 10.0,
                 field: str = "embedding", headers: Optional[Dict[str, str]] = None):
        super().__init__(d_model)
        self.url = url
        self.timeout = timeout
        self.field = field
        self.headers = {"Content-Type": "application/json", **(headers or {})}

    def _embed(self, query) -> np.ndarray:
        body = json.dumps({"query": query}).encode("utf-8")
        req = urllib.request.Request(self.url, data=body, headers=self.headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310 (trusted client endpoint)
            payload = json.loads(resp.read().decode("utf-8"))
        if self.field not in payload:
            raise KeyError(f"embedding endpoint response missing '{self.field}' field")
        return np.asarray(payload[self.field], dtype=float)


class SpecKeywordConnector(ContextSource):
    """A demo/POC connector that makes *text* route sensibly without a real
    encoder. It keyword-matches the query against each expert's declared keywords,
    then returns an embedding near that expert's deterministic training center (so
    the router — trained on the same centers — routes there with high confidence).

    This exists so ``dep.route("my card was double charged")`` demonstrably lands
    on the card-dispute specialist in a POC. Production replaces it with a real
    ``ContextSource`` over the client's encoder.
    """

    def __init__(self, spec, trainer, noise: float = 0.15, seed: int = 0):
        super().__init__(trainer.d_model)
        self.trainer = trainer
        self.noise = noise
        self._rng = np.random.default_rng(seed)
        # keyword -> expert name (later keywords for an expert don't override an
        # earlier expert's claim on a shared word; first declaration wins).
        self._kw: Dict[str, str] = {}
        self._expert_names = []
        for _tenant, e in spec.experts:
            self._expert_names.append(e.name)
            for kw in e.keywords:
                self._kw.setdefault(kw.lower(), e.name)

    def match(self, query: str) -> Optional[str]:
        """Return the expert name whose keywords best match the query text."""
        text = str(query).lower()
        best, score = None, 0
        counts: Dict[str, int] = {}
        for kw, name in self._kw.items():
            if kw in text:
                counts[name] = counts.get(name, 0) + 1
        for name, c in counts.items():
            if c > score:
                best, score = name, c
        return best

    def _embed(self, query) -> np.ndarray:
        name = self.match(query)
        if name is None:
            # No keyword hit -> an ambiguous vector far from every center, which
            # the router answers with low confidence (i.e. an escalation case).
            return self._rng.normal(0, 1.0, self.d_model)
        center = self.trainer.center(name)
        return center + self._rng.normal(0, self.noise, self.d_model)
