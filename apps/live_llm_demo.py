"""
Live LLM comparison demo.

The page compares:
  1. a normal shared LLM prompt that receives all support policies, and
  2. a DAS-routed prompt where the local das package chooses one isolated
     specialist before a real LLM is called.

If OPENAI_API_KEY is present, the backend calls the OpenAI Responses API using
OPENAI_MODEL (default: gpt-4o-mini). If no key is present, it uses an explicit
offline fallback so the UX still works locally and does not pretend to be live.
"""
import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request

import numpy as np
from flask import jsonify, render_template, request

from das.functional import softmax
from das.governance import ControlPlane
from das.integrations import DASExpertNode
from das.model import DASForest


D_MODEL = 18
LEAF_DIMS = [D_MODEL, 13, 8, 2]
N = 150

SPECIALISTS = [
    {
        "tenant": "careplus",
        "actor": "care-agent",
        "name": "medical-claim-llm",
        "title": "Medical claim specialist",
        "center": 0,
        "policy": (
            "You help CarePlus members with medical claim denials. Ask for claim ID, "
            "provider, denial reason, and whether the provider was in network. Never "
            "discuss banking or retail warranty policy."
        ),
        "fallback": "I can help prepare an appeal checklist: claim ID, denial code, provider, and in-network proof.",
        "keywords": ["claim", "clinic", "mri", "provider", "denied", "insurance", "appeal", "medical"],
    },
    {
        "tenant": "fintrust",
        "actor": "bank-agent",
        "name": "card-dispute-llm",
        "title": "Card dispute specialist",
        "center": 6,
        "policy": (
            "You help FinTrust cardholders dispute unknown charges. Ask for merchant, "
            "date, amount, card status, and whether the charge is recurring. Never "
            "discuss medical claims or retail warranties."
        ),
        "fallback": "I can help open a chargeback checklist: merchant, date, amount, and whether the charge is recurring.",
        "keywords": ["charge", "card", "merchant", "bank", "dispute", "recurring", "fraud", "payment"],
    },
    {
        "tenant": "retailops",
        "actor": "retail-agent",
        "name": "warranty-return-llm",
        "title": "Warranty return specialist",
        "center": 12,
        "policy": (
            "You help RetailOps customers with warranty replacements. Ask for purchase "
            "date, serial number, symptom, and proof of warranty. Never discuss medical "
            "claims or card disputes."
        ),
        "fallback": "I can help check replacement eligibility: purchase date, serial number, symptom, and warranty proof.",
        "keywords": ["warranty", "replacement", "phone", "serial", "return", "broken", "repair", "purchase"],
    },
]

SUGGESTIONS = {
    "claim": "My MRI claim was denied even though the clinic was in network. What should I do?",
    "dispute": "I see an unknown recurring card charge from StreamBox. Can I dispute it?",
    "warranty": "My phone stopped working under warranty. Can I get a replacement?",
}


def _seed(text):
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def _ce_grad(logits, y):
    p = softmax(logits)
    oh = np.zeros_like(p)
    oh[np.arange(len(y)), y] = 1.0
    return (p - oh) / len(y)


def _estimate_tokens(text):
    return max(1, int(len(text) / 4))


def _cost_analysis(shared_usage, das_usage, standard_hot, das_hot, context_reduction):
    """Provider-neutral savings model.

    One cost unit means one token-sized unit of billable LLM work in this demo.
    Exact dollars depend on the model/API/GPU price, so the UI presents relative
    savings and a simple 1,000-chat projection.
    """
    standard_input = int(shared_usage.get("inputTokens", 0) or 0)
    standard_output = int(shared_usage.get("outputTokens", 0) or 0)
    das_input = int(das_usage.get("inputTokens", 0) or 0)
    das_output = int(das_usage.get("outputTokens", 0) or 0)
    standard_units = standard_input + standard_output
    das_units = das_input + das_output
    units_saved = max(0, standard_units - das_units)
    input_saved = max(0, standard_input - das_input)
    total_saved_pct = round((units_saved / max(standard_units, 1)) * 100, 1)
    return {
        "unit": "token-cost unit",
        "standardUnits": standard_units,
        "dasUnits": das_units,
        "unitsSaved": units_saved,
        "totalSavedPct": total_saved_pct,
        "inputTokensSaved": input_saved,
        "contextSavedPct": context_reduction,
        "specialistsAvoided": max(0, int(standard_hot) - int(das_hot)),
        "per1000Chats": {
            "standardUnits": standard_units * 1000,
            "dasUnits": das_units * 1000,
            "unitsSaved": units_saved * 1000,
            "inputTokensSaved": input_saved * 1000,
        },
        "note": (
            "DAS local routing/leaf work has no API token bill. External LLM "
            "generation still costs whatever your provider or GPU costs."
        ),
    }


_STOP_WORDS = {
    "about", "after", "also", "because", "before", "being", "could", "every",
    "from", "have", "help", "into", "make", "need", "only", "should", "that",
    "their", "there", "this", "user", "what", "when", "where", "with", "your",
}


def _safe_slug(text, fallback="specialist", limit=34):
    slug = re.sub(r"[^a-z0-9]+", "-", str(text or "").lower()).strip("-")
    return (slug[:limit].strip("-") or fallback)


def _short_text(text, limit=180):
    text = re.sub(r"\s+", " ", str(text or "").strip())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."


def _derive_keywords(*parts, limit=9):
    words = []
    for part in parts:
        for word in re.findall(r"[a-z][a-z0-9]{2,}", str(part or "").lower()):
            if word in _STOP_WORDS or word in words:
                continue
            words.append(word)
            if len(words) >= limit:
                return words
    return words


def _text_anchor(text):
    """Deterministic tiny text embedding for demo-only specialist creation."""
    vec = np.zeros(D_MODEL, dtype=float)
    words = re.findall(r"[a-z0-9]+", str(text or "").lower())
    if not words:
        words = ["custom"]
    for word in words:
        digest = hashlib.sha256(word.encode("utf-8")).digest()
        for i in range(0, min(len(digest), D_MODEL * 2), 2):
            slot = digest[i] % D_MODEL
            sign = 1 if digest[i + 1] % 2 else -1
            vec[slot] += sign * (0.6 + (digest[i] / 255.0))
    norm = np.linalg.norm(vec) or 1.0
    return (vec / norm) * 5.4


class LLMClient:
    def __init__(self):
        self.openai_key = os.environ.get("OPENAI_API_KEY")
        self.openai_model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.openai_url = os.environ.get("OPENAI_RESPONSES_URL", "https://api.openai.com/v1/responses")
        self.google_key = os.environ.get("GOOGLE_API_KEY")
        self.google_model = os.environ.get("GOOGLE_MODEL", "gemini-3.5-flash")
        self.google_url = os.environ.get("GOOGLE_INTERACTIONS_URL", "https://generativelanguage.googleapis.com/v1beta/interactions")
        self.compat_url = os.environ.get("DAS_LLM_URL")
        self.compat_key = os.environ.get("DAS_LLM_KEY")
        self.compat_model = os.environ.get("DAS_LLM_MODEL", self.openai_model)
        self.fallback_error = None
        self.fallback_provider = None

    def mode(self):
        if self.compat_url:
            return {"live": True, "provider": "OpenAI-compatible endpoint", "model": self.compat_model}
        if self.google_key:
            return {"live": True, "provider": "Google Gemini Interactions API", "model": self.google_model}
        if self.openai_key:
            return {"live": True, "provider": "OpenAI Responses API", "model": self.openai_model}
        return {"live": False, "provider": "offline fallback", "model": "local deterministic response"}

    def result_mode(self):
        if not self.fallback_error:
            return self.mode()
        return {
            "live": False,
            "provider": f"offline fallback after {self.fallback_provider} error",
            "model": "local deterministic response",
            "error": self.fallback_error,
        }

    def _fallback_result(self, started, system, user, fallback, provider=None, error=None):
        if error:
            self.fallback_error = error
            self.fallback_provider = provider or "LLM provider"
            prefix = f"Live provider unavailable ({self.fallback_provider}). Using local fallback: "
        else:
            prefix = ""
        latency = int((time.perf_counter() - started) * 1000)
        text = prefix + fallback
        usage = {
            "inputTokens": _estimate_tokens(system + "\n" + user),
            "outputTokens": _estimate_tokens(text),
        }
        mode = self.result_mode() if error else self.mode()
        return {"text": text, "latencyMs": latency, "usage": usage, **mode}

    def complete(self, system, user, fallback):
        started = time.perf_counter()
        if self.compat_url:
            try:
                text, usage = self._call_compatible(system, user)
            except Exception as ex:
                if os.environ.get("DAS_LLM_STRICT") == "1":
                    raise
                return self._fallback_result(started, system, user, fallback, "OpenAI-compatible endpoint", str(ex))
            latency = int((time.perf_counter() - started) * 1000)
            return {"text": text, "latencyMs": latency, "usage": usage, **self.result_mode()}
        if self.google_key:
            try:
                text, usage = self._call_google(system, user)
            except Exception as ex:
                if os.environ.get("DAS_LLM_STRICT") == "1":
                    raise
                return self._fallback_result(started, system, user, fallback, "Google Gemini", str(ex))
            latency = int((time.perf_counter() - started) * 1000)
            return {"text": text, "latencyMs": latency, "usage": usage, **self.result_mode()}
        if self.openai_key:
            try:
                text, usage = self._call_openai(system, user)
            except Exception as ex:
                if os.environ.get("DAS_LLM_STRICT") == "1":
                    raise
                return self._fallback_result(started, system, user, fallback, "OpenAI", str(ex))
            latency = int((time.perf_counter() - started) * 1000)
            return {"text": text, "latencyMs": latency, "usage": usage, **self.result_mode()}

        # Keep the fallback visibly non-live and tiny, so the demo does not fake
        # an external model call when no provider is configured.
        time.sleep(0.05)
        return self._fallback_result(started, system, user, fallback)

    def _call_openai(self, system, user):
        payload = {
            "model": self.openai_model,
            "instructions": system,
            "input": user,
            "max_output_tokens": 220,
        }
        req = urllib.request.Request(
            self.openai_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.openai_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=45) as res:
                data = json.loads(res.read().decode("utf-8"))
        except urllib.error.HTTPError as ex:
            body = ex.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI API error {ex.code}: {body[:240]}") from ex
        text = data.get("output_text")
        if not text:
            chunks = []
            for item in data.get("output", []):
                for content in item.get("content", []):
                    if "text" in content:
                        chunks.append(content["text"])
            text = "\n".join(chunks).strip() or "(no text returned)"
        usage_raw = data.get("usage", {})
        usage = {
            "inputTokens": usage_raw.get("input_tokens", _estimate_tokens(system + "\n" + user)),
            "outputTokens": usage_raw.get("output_tokens", _estimate_tokens(text)),
        }
        return text, usage

    def _call_google(self, system, user):
        payload = {
            "model": self.google_model,
            "system_instruction": system,
            "input": user,
            "generation_config": {
                "temperature": 0.2,
                "max_output_tokens": 220,
            },
        }
        req = urllib.request.Request(
            self.google_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "x-goog-api-key": self.google_key,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=45) as res:
                data = json.loads(res.read().decode("utf-8"))
        except urllib.error.HTTPError as ex:
            body = ex.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Google Gemini API error {ex.code}: {body[:240]}") from ex
        text = data.get("output_text")
        if not text:
            chunks = []
            for step in data.get("steps", []):
                for content in step.get("content", []):
                    if isinstance(content, dict) and content.get("text"):
                        chunks.append(content["text"])
            text = "\n".join(chunks).strip() or "(no text returned)"
        usage_raw = data.get("usage_metadata") or data.get("usage") or {}
        usage = {
            "inputTokens": (
                usage_raw.get("prompt_token_count")
                or usage_raw.get("input_tokens")
                or _estimate_tokens(system + "\n" + user)
            ),
            "outputTokens": (
                usage_raw.get("candidates_token_count")
                or usage_raw.get("output_tokens")
                or _estimate_tokens(text)
            ),
        }
        return text, usage

    def _call_compatible(self, system, user):
        payload = {
            "model": self.compat_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
            "max_tokens": 220,
        }
        headers = {"Content-Type": "application/json"}
        if self.compat_key:
            headers["Authorization"] = f"Bearer {self.compat_key}"
        req = urllib.request.Request(
            self.compat_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=45) as res:
            data = json.loads(res.read().decode("utf-8"))
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "(no text returned)")
        usage_raw = data.get("usage", {})
        usage = {
            "inputTokens": usage_raw.get("prompt_tokens", _estimate_tokens(system + "\n" + user)),
            "outputTokens": usage_raw.get("completion_tokens", _estimate_tokens(text)),
        }
        return text, usage


class LiveLLMLab:
    def __init__(self):
        self.lock = threading.RLock()
        self.rng = np.random.default_rng(22)
        self.specs = {}
        for x in SPECIALISTS:
            spec = dict(x)
            spec.setdefault("version", 1)
            spec.setdefault("learned", [])
            spec.setdefault("userCreated", False)
            self.specs[spec["name"]] = spec
        self.order = [x["name"] for x in SPECIALISTS]
        self.data = {name: self._make_domain(name) for name in self.order}
        self.cp = self._build_control_plane()
        self.node = DASExpertNode(self.cp, default_actor="care-agent")
        self.seq = 0

    def _center_for_spec(self, spec):
        if "anchor" in spec:
            return np.asarray(spec["anchor"], dtype=float)
        center = np.zeros(D_MODEL)
        center[spec["center"]] = 6.0
        center[(spec["center"] + 2) % D_MODEL] = 2.5
        return center

    def _make_domain(self, name):
        spec = self.specs[name]
        center = self._center_for_spec(spec)
        rule_rng = np.random.default_rng(_seed("live-rule:" + name))
        rule = rule_rng.normal(0, 1, D_MODEL)
        sample_rng = np.random.default_rng(_seed("live-sample:" + name))
        x = center + sample_rng.normal(0, 0.52, (N, D_MODEL))
        y = (x @ rule > 0).astype(int)
        return x, y

    def _train_leaf(self, forest, idx, name, steps=230):
        x, y = self.data[name]
        leaf = forest.leaves[idx]
        leaf.frozen = False
        for _ in range(steps):
            batch = self.rng.integers(0, len(x), 32)
            leaf.backward(_ce_grad(leaf.forward(x[batch]), y[batch]), 0.045)
        leaf.frozen = True

    def _train_router(self, forest, names, steps=650):
        x = np.vstack([self.data[name][0] for name in names])
        d = np.concatenate([np.full(len(self.data[name][0]), i) for i, name in enumerate(names)])
        for _ in range(steps):
            batch = self.rng.integers(0, len(x), 64)
            forest.router.train_step(x[batch], d[batch], lr=0.18)

    def _train_fn(self, name):
        def train(forest, idx):
            self._train_leaf(forest, idx, name)
            names = [r["name"] for r in self.cp.experts] + [name]
            self._train_router(forest, names)
        return train

    def _build_control_plane(self):
        seed_name = self.order[0]
        forest = DASForest(D_MODEL, LEAF_DIMS, num_leaves=1, seed=31)
        self._train_leaf(forest, 0, seed_name, steps=300)
        cp = ControlPlane(
            forest,
            seed_tenant=self.specs[seed_name]["tenant"],
            seed_name=seed_name,
            secret="das-live-llm-demo-key",
        )
        for tenant in sorted({x["tenant"] for x in SPECIALISTS}):
            if tenant != self.specs[seed_name]["tenant"]:
                cp.register_tenant("root", tenant)
        for spec in SPECIALISTS:
            cp.add_user("root", spec["actor"], role="operator", tenant=spec["tenant"])
        self.cp = cp
        for name in self.order[1:]:
            spec = self.specs[name]
            cp.graft("root", spec["tenant"], name, self._train_fn(name), seed=_seed("live-leaf:" + name))
        return cp

    def _classify_message(self, message):
        m = message.lower()
        words = set(re.findall(r"[a-z0-9]+", m))
        best_name = None
        best_score = 0
        for name in self.order:
            score = 0
            for kw in self.specs[name].get("keywords", []):
                score += 2 if kw in words else (1 if kw in m else 0)
            if score > best_score:
                best_name = name
                best_score = score
        if best_name:
            return best_name

        anchor = _text_anchor(message)
        best_name = self.order[0]
        best_dot = -float("inf")
        for name in self.order:
            center = self._center_for_spec(self.specs[name])
            dot = float(anchor @ center / ((np.linalg.norm(center) or 1.0) * (np.linalg.norm(anchor) or 1.0)))
            if dot > best_dot:
                best_name = name
                best_dot = dot
        return best_name

    def _embedding(self, message):
        name = self._classify_message(message)
        x, _ = self.data[name]
        self.seq += 1
        row = x[(self.seq * 23 + 7) % len(x)].copy()
        noise_rng = np.random.default_rng(_seed(f"live-msg:{message}:{self.seq}"))
        return row + noise_rng.normal(0, 0.025, D_MODEL)

    def _feedback_embedding(self, message, name):
        center = self._center_for_spec(self.specs[name])
        anchor = _text_anchor(message)
        noise_rng = np.random.default_rng(_seed(f"live-feedback:{name}:{message}"))
        return (0.72 * center) + (0.28 * anchor) + noise_rng.normal(0, 0.035, D_MODEL)

    def _spec_policy(self, spec):
        policy = spec["policy"]
        learned = spec.get("learned") or []
        if learned:
            policy += "\n\nUser-taught specialist notes:\n" + "\n".join(
                f"- {note}" for note in learned[-4:]
            )
        return policy

    def _suggestions(self):
        suggestions = dict(SUGGESTIONS)
        custom = [self.specs[name] for name in self.order if self.specs[name].get("userCreated")]
        for i, spec in enumerate(custom[:2], start=1):
            suggestions[f"custom{i}"] = spec.get("sample") or f"I need help with {spec['title']}."
        return suggestions

    def _expert_rows(self):
        rows = []
        for idx, rec in enumerate(self.cp.experts):
            spec = self.specs[rec["name"]]
            rows.append({
                "eid": rec["eid"],
                "tenant": rec["tenant"],
                "name": rec["name"],
                "title": spec["title"],
                "hash": self.cp.forest.leaves[idx].weight_hash(),
                "version": spec.get("version", 1),
                "learnedCount": len(spec.get("learned") or []),
                "userCreated": bool(spec.get("userCreated")),
                "keywords": spec.get("keywords", []),
            })
        return rows

    def state(self):
        llm = LLMClient().mode()
        ok, broken_idx, reason = self.cp.audit.verify()
        return {
            "llm": llm,
            "experts": self._expert_rows(),
            "suggestions": self._suggestions(),
            "tenants": sorted(self.cp.tenants),
            "audit": {"ok": ok, "brokenIndex": broken_idx, "reason": reason},
            "auditEntries": len(self.cp.audit.entries),
        }

    def _find_expert_by_name(self, name):
        for idx, rec in enumerate(self.cp.experts):
            if rec["name"] == name:
                return idx, rec
        raise KeyError(f"unknown specialist '{name}'")

    def _unique_expert_name(self, title):
        base = _safe_slug(title, "custom-specialist")
        name = f"{base}-llm"
        suffix = 2
        while name in self.specs:
            name = f"{base}-{suffix}-llm"
            suffix += 1
        return name

    def create_specialist(self, title, description, sample=None, tenant=None):
        with self.lock:
            if len(self.order) >= 8:
                raise ValueError("this pocket demo allows up to 8 specialists")
            title = _short_text(title or "", 48)
            description = _short_text(description or "", 260)
            sample = _short_text(sample or "", 180)
            if not title or not description:
                raise ValueError("specialist name and job description are required")

            tenant = _safe_slug(tenant or "creator-lab", "creator-lab", limit=24)
            actor = f"{tenant}-agent"
            name = self._unique_expert_name(title)
            keywords = _derive_keywords(title, description, sample)
            if not keywords:
                keywords = [_safe_slug(title, "custom").replace("-", "")]
            prompt_seed = " ".join([title, description, sample, tenant])
            spec = {
                "tenant": tenant,
                "actor": actor,
                "name": name,
                "title": title,
                "anchor": _text_anchor(prompt_seed).tolist(),
                "policy": (
                    f"You are a private user-created specialist named {title}. "
                    f"Your job: {description}. Stay inside this specialty, ask for "
                    "missing details, and refuse unrelated tenant data."
                ),
                "fallback": f"I am the {title} specialist. I can help with: {description}",
                "keywords": keywords,
                "sample": sample,
                "version": 1,
                "learned": [],
                "userCreated": True,
            }

            self.specs[name] = spec
            self.order.append(name)
            self.data[name] = self._make_domain(name)
            try:
                if tenant not in self.cp.tenants:
                    self.cp.register_tenant("root", tenant)
                if actor not in self.cp.users:
                    self.cp.add_user("root", actor, role="operator", tenant=tenant)
                eid = self.cp.graft("root", tenant, name, self._train_fn(name), seed=_seed("live-custom:" + name))
            except Exception:
                self.specs.pop(name, None)
                self.order = [x for x in self.order if x != name]
                self.data.pop(name, None)
                raise

            self.cp.audit.append(
                "live_create_specialist",
                (
                    f"root created live demo specialist eid={eid} ('{tenant}/{name}') "
                    f"from user description; keywords={keywords}"
                ),
                payload=self.cp._hashes(),
            )
            return {
                "created": {
                    "eid": eid,
                    "tenant": tenant,
                    "name": name,
                    "title": title,
                    "keywords": keywords,
                    "sample": sample,
                },
                "state": self.state(),
            }

    def teach(self, message, expert=None, eid=None, guidance=None):
        with self.lock:
            message = (message or "").strip()
            guidance = _short_text(guidance or "This is a good example for this specialist.", 150)
            if eid is not None:
                idx, rec = self.cp._find(int(eid))
            elif expert:
                idx, rec = self._find_expert_by_name(expert)
            else:
                idx, rec = self._find_expert_by_name(self._classify_message(message or guidance))

            name = rec["name"]
            spec = self.specs[name]
            if not message:
                message = spec.get("sample") or guidance

            h = self._feedback_embedding(message + " " + guidance, name)
            leaf = self.cp.forest.leaves[idx]
            before_hashes = {
                row["eid"]: self.cp.forest.leaves[i].weight_hash()
                for i, row in enumerate(self.cp.experts)
            }
            before_target_hash = before_hashes[rec["eid"]]
            before_route_idx, before_tau = self.cp.forest.router.route(h.reshape(1, -1))
            before_score = float(softmax(leaf.forward(h.reshape(1, -1)))[0, 1])

            learn_rng = np.random.default_rng(_seed(f"live-teach:{name}:{message}:{guidance}"))
            x_aug = h + learn_rng.normal(0, 0.11, (72, D_MODEL))
            y_aug = np.ones(len(x_aug), dtype=int)
            self.data[name] = (
                np.vstack([self.data[name][0], x_aug])[-360:],
                np.concatenate([self.data[name][1], y_aug])[-360:],
            )

            leaf.frozen = False
            for _ in range(110):
                batch = learn_rng.integers(0, len(x_aug), 24)
                leaf.backward(_ce_grad(leaf.forward(x_aug[batch]), y_aug[batch]), 0.055)
            leaf.frozen = True
            self._train_router(self.cp.forest, [row["name"] for row in self.cp.experts], steps=90)

            new_keywords = _derive_keywords(message, guidance, limit=6)
            for kw in new_keywords:
                if kw not in spec["keywords"]:
                    spec["keywords"].append(kw)
            spec["keywords"] = spec["keywords"][:14]
            spec.setdefault("learned", []).append(guidance)
            spec["learned"] = spec["learned"][-5:]
            spec["version"] = int(spec.get("version", 1)) + 1

            after_route_idx, after_tau = self.cp.forest.router.route(h.reshape(1, -1))
            after_score = float(softmax(leaf.forward(h.reshape(1, -1)))[0, 1])
            after_hashes = {
                row["eid"]: self.cp.forest.leaves[i].weight_hash()
                for i, row in enumerate(self.cp.experts)
            }
            after_target_hash = after_hashes[rec["eid"]]
            survivors_intact = all(
                after_hashes[row["eid"]] == before_hashes[row["eid"]]
                for row in self.cp.experts
                if row["eid"] != rec["eid"]
            )
            self.cp.audit.append(
                "live_teach_specialist",
                (
                    f"root taught live demo specialist eid={rec['eid']} ('{rec['name']}'); "
                    f"leaf score {before_score:.3f}->{after_score:.3f}; "
                    f"other leaf hashes unchanged: {survivors_intact}"
                ),
                payload=self.cp._hashes(),
            )
            return {
                "taught": {
                    "eid": rec["eid"],
                    "tenant": rec["tenant"],
                    "name": rec["name"],
                    "title": spec["title"],
                    "version": spec["version"],
                    "guidance": guidance,
                    "hashBefore": before_target_hash,
                    "hashAfter": after_target_hash,
                    "targetChanged": before_target_hash != after_target_hash,
                    "survivorsIntact": survivors_intact,
                    "leafScoreBefore": round(before_score, 4),
                    "leafScoreAfter": round(after_score, 4),
                    "routeBefore": {
                        "expert": self.cp.experts[int(before_route_idx[0])]["name"],
                        "confidence": round(float(before_tau[0, int(before_route_idx[0])]), 4),
                    },
                    "routeAfter": {
                        "expert": self.cp.experts[int(after_route_idx[0])]["name"],
                        "confidence": round(float(after_tau[0, int(after_route_idx[0])]), 4),
                    },
                    "learnedCount": len(spec.get("learned") or []),
                },
                "state": self.state(),
            }

    def chat(self, message, tenant=None):
        with self.lock:
            message = (message or "").strip()
            if not message:
                message = SUGGESTIONS["claim"]
            h = self._embedding(message)
            leaf_idx, tau = self.cp.forest.router.route(h.reshape(1, -1))
            idx = int(leaf_idx[0])
            rec = self.cp.experts[idx]
            spec = self.specs[rec["name"]]
            tenant = tenant or rec["tenant"]
            llm = LLMClient()

            shared_system = (
                "You are a shared customer-support LLM. You have all client policies below. "
                "Answer the user, but do not leak irrelevant policies.\n\n"
                + "\n\n".join(
                    f"{self.specs[name]['tenant']} / {self.specs[name]['title']} policy:\n"
                    f"{self._spec_policy(self.specs[name])}"
                    for name in self.order
                )
            )
            shared_fallback = (
                "I can help, but this shared path carried every client policy into the prompt. "
                "For safety, confirm the correct department before acting."
            )
            shared = llm.complete(shared_system, message, shared_fallback)

            blocked = rec["tenant"] != tenant
            if blocked:
                self.cp.audit.append(
                    "live_llm_block",
                    f"blocked token '{tenant}' before tenant '{rec['tenant']}' specialist LLM could run",
                    payload={"eid": rec["eid"], "token": tenant, "confidence": float(tau[0, idx])},
                )
                das = {
                    "text": "DAS blocked this before the specialist LLM ran because the tenant token did not match the routed specialist.",
                    "latencyMs": 0,
                    "usage": {"inputTokens": 0, "outputTokens": 0},
                    **llm.result_mode(),
                }
            else:
                # Use the package integration node for permission/provenance, then
                # call the selected real LLM specialist with only its policy.
                self.node({"embedding": h.tolist(), "actor": spec["actor"]})
                das_system = (
                    f"You are the {spec['title']} for tenant {spec['tenant']}. "
                    "Use only this specialist policy and answer concisely.\n\n"
                    + self._spec_policy(spec)
                )
                das = llm.complete(das_system, message, spec["fallback"])

            shared_input = shared["usage"]["inputTokens"]
            das_input = das["usage"]["inputTokens"]
            reduction = 100 if shared_input and blocked else (
                round(max(0, 1 - (das_input / max(shared_input, 1))) * 100, 1)
            )
            standard_hot = len(self.order)
            das_hot = 0 if blocked else 1
            cost = _cost_analysis(
                shared["usage"],
                das["usage"],
                standard_hot,
                das_hot,
                reduction,
            )
            return {
                "message": message,
                "tenantToken": tenant,
                "routed": {
                    "tenant": rec["tenant"],
                    "expert": rec["name"],
                    "title": spec["title"],
                    "confidence": round(float(tau[0, idx]), 4),
                },
                "blocked": blocked,
                "standard": {
                    "answer": shared["text"],
                    "latencyMs": shared["latencyMs"],
                    "inputTokens": shared_input,
                    "outputTokens": shared["usage"]["outputTokens"],
                    "policiesSent": len(self.order),
                    "specialistsOpened": standard_hot,
                    "privacyBoundary": "soft prompt instruction",
                },
                "das": {
                    "answer": das["text"],
                    "latencyMs": das["latencyMs"],
                    "inputTokens": das_input,
                    "outputTokens": das["usage"]["outputTokens"],
                    "policiesSent": 0 if blocked else 1,
                    "specialistsOpened": das_hot,
                    "privacyBoundary": "tenant token + isolated specialist",
                    "contextReductionPct": reduction,
                },
                "cost": cost,
                "llm": llm.result_mode(),
                "auditEntries": len(self.cp.audit.entries),
            }


_LAB = None
_LOCK = threading.Lock()


def _lab():
    global _LAB
    with _LOCK:
        if _LAB is None:
            _LAB = LiveLLMLab()
        return _LAB


def register_live_llm_routes(app):
    @app.route("/live-llm")
    def live_llm():
        return render_template("live_llm.html")

    @app.route("/api/live-llm/state")
    def live_llm_state():
        return jsonify(_lab().state())

    @app.route("/api/live-llm/chat", methods=["POST"])
    def live_llm_chat():
        body = request.get_json(silent=True) or {}
        try:
            return jsonify(_lab().chat(body.get("message", ""), tenant=body.get("tenant")))
        except Exception as ex:
            return jsonify({"error": str(ex)}), 502

    @app.route("/api/live-llm/create-specialist", methods=["POST"])
    def live_llm_create_specialist():
        body = request.get_json(silent=True) or {}
        try:
            payload = _lab().create_specialist(
                body.get("title", ""),
                body.get("description", ""),
                sample=body.get("sample", ""),
                tenant=body.get("tenant", ""),
            )
            return jsonify(payload), 201
        except ValueError as ex:
            return jsonify({"error": str(ex)}), 400
        except Exception as ex:
            return jsonify({"error": str(ex)}), 502

    @app.route("/api/live-llm/teach", methods=["POST"])
    def live_llm_teach():
        body = request.get_json(silent=True) or {}
        try:
            return jsonify(_lab().teach(
                body.get("message", ""),
                expert=body.get("expert"),
                eid=body.get("eid"),
                guidance=body.get("guidance"),
            ))
        except (KeyError, ValueError) as ex:
            return jsonify({"error": str(ex)}), 400
        except Exception as ex:
            return jsonify({"error": str(ex)}), 502
