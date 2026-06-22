"""
das_text.py — real frozen pretrained encoder + real text (PRODUCT_PLAN Phase 1)
-------------------------------------------------------------------------------
Until now the shared "backbone" was a tiny MLP pretrained on *synthetic* clustered
vectors — console.LoRAService._pretrain_backbone even calls itself "a stand-in for
a frozen, broadly-pretrained encoder", and the benchmarks routed random noise. That
is the one line a technical reviewer pulls on: "your experts are toy data."

This module removes that objection. `TextEncoder` wraps a REAL frozen, pretrained
sentence-transformer (MiniLM by default): real English text -> a real 384-d semantic
embedding. The forest (router + LoRA leaves in das_torch) then trains and routes on
those embeddings instead of clustered noise, so the isolation / grafting / pruning /
audit guarantees are demonstrated on genuine language features.

Deliberately OPTIONAL and isolated here (not in the das/ core): importing the NumPy
governance control plane must never require transformers, and the deployable
NumPy + Flask API / Docker image stays torch-free. The integration pattern for the
control plane is "encode text on the client -> POST the embedding"; the encoder is a
featurizer that lives outside the torch-free control plane.

    pip install -e ".[hf]"        # torch + sentence-transformers
    python examples/hf_governance_demo.py
"""
# torch / transformers are imported lazily inside the methods below, so that
# `import das_text` (and DEMO_CORPUS) works in the torch-free environment the
# governance core/CI runs in — only embedding actually needs the heavy deps.

# ── The frozen pretrained encoder ────────────────────────────────────────────
_ENCODER_CACHE = {}


class TextEncoder:
    """A frozen, pretrained sentence encoder — the real 'broadly-pretrained'
    stage of the stack. Wraps sentence-transformers (preferred) or falls back to
    a mean-pooled HF AutoModel. Its weights are NEVER trained here; only the
    downstream router + LoRA leaves learn. `embed()` returns L2-normalised
    float32 embeddings of shape [n, dim]. Models are cached per process."""

    DEFAULT = "sentence-transformers/all-MiniLM-L6-v2"  # 384-d, ~90 MB, CPU-fast

    def __init__(self, model_name=DEFAULT, device="cpu"):
        self.model_name = model_name
        self.device = device
        self._st = None   # a SentenceTransformer, if available
        self._hf = None   # (tokenizer, model) fallback
        self._dim = None
        self._load()

    def _load(self):
        if self.model_name in _ENCODER_CACHE:
            self._st, self._hf, self._dim = _ENCODER_CACHE[self.model_name]
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._st = SentenceTransformer(self.model_name, device=self.device)
            # get_embedding_dimension is the current name; fall back for older versions.
            get_dim = (getattr(self._st, "get_embedding_dimension", None)
                       or self._st.get_sentence_embedding_dimension)
            self._dim = int(get_dim())
        except Exception:
            # No sentence-transformers — use raw transformers + mean pooling.
            from transformers import AutoTokenizer, AutoModel
            tok = AutoTokenizer.from_pretrained(self.model_name)
            mdl = AutoModel.from_pretrained(self.model_name).to(self.device).eval()
            for p in mdl.parameters():
                p.requires_grad_(False)
            self._hf = (tok, mdl)
            self._dim = int(mdl.config.hidden_size)
        _ENCODER_CACHE[self.model_name] = (self._st, self._hf, self._dim)

    @property
    def dim(self):
        return self._dim

    def embed(self, texts):
        """list[str] (or str) -> FloatTensor [n, dim], L2-normalised, on device."""
        import torch
        import torch.nn.functional as F
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        with torch.no_grad():
            if self._st is not None:
                emb = self._st.encode(
                    texts, convert_to_tensor=True, normalize_embeddings=True,
                    show_progress_bar=False,
                )
                return emb.to(self.device).float()
            tok, mdl = self._hf
            enc = tok(texts, padding=True, truncation=True, max_length=128,
                      return_tensors="pt").to(self.device)
            out = mdl(**enc).last_hidden_state             # [n, seq, dim]
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)   # mean pool
            return F.normalize(emb, p=2, dim=1).float()


def embed_domains(encoder, domains, corpus=None):
    """Embed every text in `corpus` once, grouped by domain. Returns a dict
    domain -> (emb [M, dim] float32, y [M] long) on the encoder's device. Callers
    (console, demos) sample minibatches from this cache with replacement, exactly
    like the old synthetic _gen sampled noise — but the rows are now real
    sentences run through a real frozen encoder."""
    corpus = corpus or DEMO_CORPUS
    for name in domains:                       # validate before touching torch
        if name not in corpus:
            raise KeyError(f"no demo text for domain '{name}'; have: {sorted(corpus)}")
    import torch
    cache = {}
    for name in domains:
        texts = [t for t, _ in corpus[name]]
        labels = [lab for _, lab in corpus[name]]
        emb = encoder.embed(texts).detach()
        y = torch.tensor(labels, dtype=torch.long, device=emb.device)
        cache[name] = (emb, y)
    return cache


# ── Real demo corpus ─────────────────────────────────────────────────────────
# Small, curated, and clearly a DEMO set — but real English text, not random
# vectors. Each domain's expert is a binary "routine (0) vs needs-review/risk (1)"
# classifier within its domain, a governance-flavoured intra-domain intent split.
# Enough signal for a frozen MiniLM + a tiny trained head to separate cleanly,
# while routing across domains is what the StemRouter learns.
DEMO_CORPUS = {
    "legal": [
        ("The NDA term is two years with standard mutual confidentiality.", 0),
        ("Please countersign the attached engagement letter at your convenience.", 0),
        ("This clause mirrors our standard governing-law provision.", 0),
        ("The vendor accepted our usual limitation-of-liability language.", 0),
        ("Routine renewal of the existing master services agreement.", 0),
        ("Standard mutual non-disclosure, no carve-outs requested.", 0),
        ("The contract uses our approved boilerplate indemnity.", 0),
        ("Filing the quarterly compliance attestation as usual.", 0),
        ("The counterparty added an uncapped indemnification obligation.", 1),
        ("This agreement waives our right to a jury trial entirely.", 1),
        ("A perpetual, irrevocable license to our source code is demanded.", 1),
        ("The non-compete extends five years across every jurisdiction.", 1),
        ("They inserted an unlimited liability clause overriding our cap.", 1),
        ("The data-processing terms conflict with GDPR transfer rules.", 1),
        ("An auto-renewal with no termination-for-convenience right was added.", 1),
        ("The IP assignment grabs all of our pre-existing background IP.", 1),
    ],
    "medical": [
        ("Patient presents for a routine annual wellness visit.", 0),
        ("Blood pressure is within the normal range at 118 over 76.", 0),
        ("Refill request for an existing maintenance medication.", 0),
        ("Stable vitals, follow-up scheduled in six months.", 0),
        ("Routine vaccination administered without complication.", 0),
        ("Mild seasonal allergies, advised over-the-counter antihistamine.", 0),
        ("Annual lab panel returned within expected reference ranges.", 0),
        ("Post-op check shows normal healing, no concerns.", 0),
        ("Patient reports sudden chest pain radiating to the left arm.", 1),
        ("Severe allergic reaction with throat swelling after the dose.", 1),
        ("Acute shortness of breath and oxygen saturation dropping to 84.", 1),
        ("Possible adverse drug interaction flagged for immediate review.", 1),
        ("Sudden onset of slurred speech and facial drooping.", 1),
        ("Uncontrolled bleeding that is not responding to pressure.", 1),
        ("Dangerously high potassium level requires urgent intervention.", 1),
        ("Patient is unresponsive and requires emergency escalation.", 1),
    ],
    "finance": [
        ("Monthly expense report submitted within the approved budget.", 0),
        ("Routine reconciliation of the corporate card statement.", 0),
        ("Standard payroll run processed on schedule.", 0),
        ("Quarterly forecast is in line with prior guidance.", 0),
        ("Vendor invoice matches the purchase order exactly.", 0),
        ("Petty cash float reconciled with no discrepancies.", 0),
        ("Recurring subscription renewed at the contracted rate.", 0),
        ("Travel reimbursement filed under the normal per-diem policy.", 0),
        ("A wire transfer request was redirected to a new offshore account.", 1),
        ("Unusual after-hours transactions drained the operating account.", 1),
        ("The invoice amount is ten times the contracted purchase order.", 1),
        ("A duplicate payment to an unverified supplier was detected.", 1),
        ("Potential insider trading pattern flagged in the trade log.", 1),
        ("Expense claims show signs of fabricated receipts.", 1),
        ("A sudden margin call threatens to liquidate the position.", 1),
        ("Suspected money-laundering structuring across many small deposits.", 1),
    ],
    "code": [
        ("Bumped the linter version and reformatted the imports.", 0),
        ("Added a unit test for the existing date-parsing helper.", 0),
        ("Updated the README with the new install command.", 0),
        ("Refactored a function name for clarity, no behaviour change.", 0),
        ("Pinned a transitive dependency to the current minor version.", 0),
        ("Tidied logging messages and removed a stray print.", 0),
        ("Renamed a variable and added a docstring.", 0),
        ("Minor CSS tweak to align the footer.", 0),
        ("This change disables TLS certificate verification in production.", 1),
        ("Hardcoded AWS secret keys were committed to the repository.", 1),
        ("The query concatenates user input directly into raw SQL.", 1),
        ("A force-push rewrote shared history on the main branch.", 1),
        ("The migration drops the users table without a backup.", 1),
        ("An unbounded recursion can crash the production service.", 1),
        ("eval() is run on untrusted request payloads.", 1),
        ("Authentication was removed from the internal admin endpoint.", 1),
    ],
    "support": [
        ("Customer asked how to reset their password.", 0),
        ("Routine question about changing the account email.", 0),
        ("User wants to know the difference between two plans.", 0),
        ("Request to update the billing address on file.", 0),
        ("How do I export my data to CSV?", 0),
        ("Customer is happy and left positive feedback.", 0),
        ("Simple question about business hours.", 0),
        ("Asking where to find the mobile app download.", 0),
        ("Customer is threatening legal action over a data breach.", 1),
        ("The entire account was locked and production is down.", 1),
        ("User reports their private data is visible to other customers.", 1),
        ("A high-value client is escalating to cancel immediately.", 1),
        ("Reported a security vulnerability in the login flow.", 1),
        ("Payment was charged three times and the customer is furious.", 1),
        ("Outage is affecting every user in the region right now.", 1),
        ("Customer says their account was hacked and funds are missing.", 1),
    ],
    "hr": [
        ("Employee submitted a routine paid-time-off request.", 0),
        ("Standard onboarding checklist for a new hire.", 0),
        ("Annual benefits enrollment reminder went out.", 0),
        ("Manager approved a normal expense for a team lunch.", 0),
        ("Updating an emergency contact on the employee record.", 0),
        ("Scheduling a regular quarterly performance check-in.", 0),
        ("Routine confirmation of a remote-work day.", 0),
        ("New employee completed the standard compliance training.", 0),
        ("A formal harassment complaint was filed against a manager.", 1),
        ("Allegations of discriminatory hiring practices were raised.", 1),
        ("An employee reported a serious workplace safety violation.", 1),
        ("Whistleblower complaint about financial misconduct received.", 1),
        ("A wrongful-termination claim is being threatened.", 1),
        ("Credible report of a hostile work environment.", 1),
        ("Suspected falsification of timesheets across a team.", 1),
        ("An employee disclosed a serious conflict of interest.", 1),
    ],
}
