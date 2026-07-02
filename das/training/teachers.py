"""Teacher adapters for the DAS growing-child loop.

The core growth loop only needs one thing from a teacher: a ``LessonBatch`` of
fixed-size vectors and labels. Local vector teachers generate those directly.
Endpoint-backed LLM teachers ask any chat/JSON-capable model for structured text
lessons, then encode the text into deterministic DAS vectors.
"""

from dataclasses import dataclass
import hashlib
import json
import re
import urllib.error
import urllib.request

import numpy as np


def stable_seed(*parts):
    """Deterministic 32-bit seed from arbitrary labels."""
    text = "::".join(str(p) for p in parts)
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


class TeacherError(RuntimeError):
    """Base error for teacher adapter failures."""


class LLMTeacherError(TeacherError):
    """Raised when an endpoint-backed LLM teacher cannot produce lessons."""


@dataclass
class LessonBatch:
    """A teacher-produced training/evaluation batch for one expert topic."""

    teacher: str
    topic: str
    dataset_version: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_eval: np.ndarray
    y_eval: np.ndarray
    notes: str = ""

    def summary(self):
        return {
            "teacher": self.teacher,
            "topic": self.topic,
            "dataset_version": self.dataset_version,
            "train_examples": int(len(self.X_train)),
            "eval_examples": int(len(self.X_eval)),
            "notes": self.notes,
        }


class HashingTextEncoder:
    """Small deterministic text-to-vector encoder for mobile-friendly lessons."""

    _token_re = re.compile(r"[A-Za-z0-9_+#.-]+")

    def __init__(self, d_model):
        self.d_model = int(d_model)

    def _tokens(self, text):
        return self._token_re.findall(str(text).lower())

    def encode_one(self, text, topic=""):
        v = np.zeros(self.d_model, dtype=float)
        topic_idx = stable_seed("topic", topic) % self.d_model
        v[topic_idx] += 2.5
        tokens = self._tokens(text)
        for token in tokens:
            idx = stable_seed("token", token) % self.d_model
            sign = 1.0 if stable_seed("sign", token) % 2 else -1.0
            v[idx] += sign
        for a, b in zip(tokens, tokens[1:]):
            gram = a + " " + b
            idx = stable_seed("bigram", gram) % self.d_model
            sign = 0.5 if stable_seed("bsign", gram) % 2 else -0.5
            v[idx] += sign
        norm = np.linalg.norm(v)
        if norm == 0:
            v[topic_idx] = 1.0
            norm = 1.0
        return (v / norm) * 6.0

    def encode(self, rows, topic=""):
        return np.vstack([self.encode_one(row["input"], topic=topic) for row in rows]).astype(float)


def _label_to_int(value):
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return int(value)
    text = str(value).strip().lower()
    positives = {"1", "true", "yes", "positive", "relevant", "correct", "belongs"}
    negatives = {"0", "false", "no", "negative", "irrelevant", "incorrect", "other"}
    if text in positives:
        return 1
    if text in negatives:
        return 0
    raise LLMTeacherError(f"lesson label must be 0/1, got {value!r}")


def _coerce_rows(rows, split):
    clean = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        text = (
            row.get("input")
            or row.get("text")
            or row.get("question")
            or row.get("prompt")
            or row.get("example")
        )
        if text is None:
            continue
        clean.append({"input": str(text), "label": _label_to_int(row.get("label", row.get("y", 1)))})
    if not clean:
        raise LLMTeacherError(f"LLM teacher returned no usable {split} lessons")
    return clean


def _json_from_text(text):
    if isinstance(text, (dict, list)):
        return text
    text = str(text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S | re.I)
    if fenced:
        try:
            return json.loads(fenced.group(1).strip())
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start:end + 1])
    raise LLMTeacherError("LLM teacher did not return valid JSON lessons")


def _post_json(url, body, headers=None, timeout=60):
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise LLMTeacherError(f"{url} returned HTTP {e.code}: {detail[:400]}") from e
    except urllib.error.URLError as e:
        raise LLMTeacherError(f"could not reach {url}: {e.reason}") from e
    except json.JSONDecodeError as e:
        raise LLMTeacherError(f"{url} did not return JSON") from e


class EndpointLLMTeacher:
    """LLM teacher that works with OpenAI-compatible, Ollama, or custom JSON APIs."""

    def __init__(
        self,
        name,
        d_model,
        provider="openai-compatible",
        endpoint="",
        model="",
        api_key="",
        label=None,
        temperature=0.2,
        timeout=60,
        max_examples=48,
        encoder=None,
    ):
        self.name = str(name)
        self.label = label or self.name
        self.d_model = int(d_model)
        self.provider = str(provider or "openai-compatible")
        self.endpoint = str(endpoint or "").rstrip("/")
        self.model = str(model or "")
        self.api_key = str(api_key or "")
        self.temperature = float(temperature)
        self.timeout = int(timeout)
        self.max_examples = max(2, int(max_examples))
        # `encoder` swaps the offline word-hashing fallback for a REAL text
        # encoder (contract: encode(rows, topic) -> (n, d_model)); e.g.
        # das.platform.connectors.RealTextLessonEncoder over frozen MiniLM,
        # so lessons live in the same semantic geometry queries route with.
        self.encoder = encoder or HashingTextEncoder(d_model)

    def describe(self):
        return {
            "id": self.name,
            "name": self.label,
            "provider": self.provider,
            "model": self.model,
            "endpoint": self.endpoint,
            "temperature": self.temperature,
            "max_examples": self.max_examples,
            "dynamic": True,
        }

    def _prompt(self, topic, n_train, n_eval):
        return (
            "Create compact JSON lesson data for a tiny DAS expert.\n"
            f"Topic: {topic}\n"
            f"Training examples: {n_train}\n"
            f"Evaluation examples: {n_eval}\n"
            "Return only JSON with this shape:\n"
            "{\"dataset_version\":\"short-id\",\"train\":[{\"input\":\"text\",\"label\":1}],"
            "\"eval\":[{\"input\":\"text\",\"label\":0}],\"notes\":\"short note\"}\n"
            "Use labels 1 for examples that belong to the topic or are correct, "
            "and labels 0 for nearby confusing negatives. Keep each input short."
        )

    def _openai_url(self):
        if self.endpoint.endswith("/chat/completions"):
            return self.endpoint
        return self.endpoint.rstrip("/") + "/chat/completions"

    def _ollama_url(self):
        if self.endpoint.endswith("/api/chat"):
            return self.endpoint
        return self.endpoint.rstrip("/") + "/api/chat"

    def _fetch_payload(self, topic, n_train, n_eval):
        if not self.endpoint:
            raise LLMTeacherError("LLM teacher endpoint is required")
        prompt = self._prompt(topic, n_train, n_eval)
        system = "You generate machine-readable lesson datasets. Return JSON only."
        if self.provider in ("openai-compatible", "openai", "llama.cpp", "vllm"):
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            data = _post_json(
                self._openai_url(),
                {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": self.temperature,
                },
                headers=headers,
                timeout=self.timeout,
            )
            content = data.get("choices", [{}])[0].get("message", {}).get("content")
            if content is None:
                raise LLMTeacherError("OpenAI-compatible response had no message content")
            return _json_from_text(content)
        if self.provider == "ollama":
            data = _post_json(
                self._ollama_url(),
                {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    "stream": False,
                    "options": {"temperature": self.temperature},
                },
                timeout=self.timeout,
            )
            content = data.get("message", {}).get("content") or data.get("response")
            if content is None:
                raise LLMTeacherError("Ollama response had no message content")
            return _json_from_text(content)
        if self.provider == "custom-json":
            data = _post_json(
                self.endpoint,
                {
                    "topic": topic,
                    "n_train": n_train,
                    "n_eval": n_eval,
                    "d_model": self.d_model,
                    "model": self.model,
                    "prompt": prompt,
                },
                headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else None,
                timeout=self.timeout,
            )
            return _json_from_text(data.get("content", data))
        raise LLMTeacherError(f"unknown LLM provider '{self.provider}'")

    def _split_rows(self, payload):
        payload = _json_from_text(payload)
        if isinstance(payload, list):
            train = [r for r in payload if str(r.get("split", "train")).lower() != "eval"]
            eval_rows = [r for r in payload if str(r.get("split", "")).lower() == "eval"]
            return train, eval_rows or train
        if not isinstance(payload, dict):
            raise LLMTeacherError("lesson payload must be a JSON object")
        train = payload.get("train") or payload.get("training") or payload.get("examples")
        eval_rows = payload.get("eval") or payload.get("evaluation") or payload.get("test")
        if eval_rows is None and train is not None:
            eval_rows = train
        return train, eval_rows

    def generate(self, topic, n_train=160, n_eval=120, dataset_version=None):
        requested_train = min(int(n_train), self.max_examples)
        requested_eval = min(int(n_eval), max(2, self.max_examples // 2))
        payload = self._fetch_payload(topic, requested_train, requested_eval)
        train_rows, eval_rows = self._split_rows(payload)
        train_rows = _coerce_rows(train_rows, "train")
        eval_rows = _coerce_rows(eval_rows, "eval")
        X_train = self.encoder.encode(train_rows, topic=topic)
        X_eval = self.encoder.encode(eval_rows, topic=topic)
        y_train = np.asarray([row["label"] for row in train_rows], dtype=int)
        y_eval = np.asarray([row["label"] for row in eval_rows], dtype=int)
        version = dataset_version or (
            f"{self.name}:{topic}:llm:{len(train_rows)}-{len(eval_rows)}:"
            f"{stable_seed(self.name, topic, len(train_rows), len(eval_rows))}"
        )
        notes = ""
        if isinstance(payload, dict):
            notes = str(payload.get("notes") or payload.get("summary") or "")
        return LessonBatch(
            teacher=self.name,
            topic=topic,
            dataset_version=version,
            X_train=X_train,
            y_train=y_train,
            X_eval=X_eval,
            y_eval=y_eval,
            notes=notes or f"{self.provider} LLM teacher '{self.label}'",
        )


class VectorTeacher:
    """Deterministic local teacher for vector-based DAS experts.

    The teacher generates a domain cluster and binary labels from a stable hidden
    rule. Different teacher names vary the sampled examples; the label rule stays
    tied to the topic so multiple teachers can improve the same expert without
    redefining the task.
    """

    def __init__(self, name, d_model, centers=None, noise=0.7, shift=0.0, seed=0,
                 label=None):
        self.name = name
        self.label = label or name
        self.d_model = int(d_model)
        self.centers = centers or {}
        self.noise = float(noise)
        self.shift = float(shift)
        self.seed = int(seed)

    def describe(self):
        return {
            "id": self.name,
            "name": self.label,
            "provider": "local-vector",
            "noise": self.noise,
            "shift": self.shift,
            "dynamic": False,
        }

    def center_for(self, topic):
        dim = self.centers.get(topic)
        if dim is None:
            dim = stable_seed("center", topic) % self.d_model
        c = np.zeros(self.d_model)
        c[int(dim) % self.d_model] = 6.0
        if self.shift:
            aux = stable_seed("shift", self.name, topic) % self.d_model
            c[aux] += self.shift
        return c

    def rule_for(self, topic):
        rng = np.random.default_rng(stable_seed("rule", topic))
        rule = rng.normal(0, 1, self.d_model)
        norm = np.linalg.norm(rule)
        return rule if norm == 0 else rule / norm

    def _sample(self, topic, n, tag):
        rng = np.random.default_rng(stable_seed(self.seed, self.name, topic, tag))
        X = self.center_for(topic) + rng.normal(0, self.noise, (int(n), self.d_model))
        rule = self.rule_for(topic)
        y = (X @ rule > 0).astype(int)
        return X.astype(float), y

    def generate(self, topic, n_train=160, n_eval=120, dataset_version=None):
        version = dataset_version or (
            f"{self.name}:{topic}:n{int(n_train)}-{int(n_eval)}:"
            f"noise{self.noise:.2f}:shift{self.shift:.2f}"
        )
        X_train, y_train = self._sample(topic, n_train, "train:" + version)
        X_eval, y_eval = self._sample(topic, n_eval, "eval:" + version)
        return LessonBatch(
            teacher=self.name,
            topic=topic,
            dataset_version=version,
            X_train=X_train,
            y_train=y_train,
            X_eval=X_eval,
            y_eval=y_eval,
            notes=f"local vector teacher '{self.label}'",
        )


def teacher_from_config(config, d_model, centers=None):
    """Build a runtime teacher from a dashboard/API registration payload."""
    provider = str(config.get("provider") or "local-vector")
    name = str(config.get("id") or config.get("name") or "").strip()
    if not name:
        raise ValueError("teacher id is required")
    label = str(config.get("label") or config.get("display_name") or name).strip()
    if provider in ("local-vector", "vector", "mock"):
        return VectorTeacher(
            name,
            d_model,
            centers=centers,
            noise=float(config.get("noise", 0.72)),
            shift=float(config.get("shift", 0.18)),
            seed=int(config.get("seed", stable_seed("teacher", name))),
            label=label,
        )
    return EndpointLLMTeacher(
        name,
        d_model,
        provider=provider,
        endpoint=config.get("endpoint") or config.get("base_url") or "",
        model=config.get("model") or "",
        api_key=config.get("api_key") or "",
        label=label,
        temperature=float(config.get("temperature", 0.2)),
        timeout=int(config.get("timeout", 60)),
        max_examples=int(config.get("max_examples", 48)),
    )
