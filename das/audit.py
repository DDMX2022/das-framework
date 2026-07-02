"""
das/audit.py
------------
A tamper-evident, signed audit log — the governance differentiator made real.

Every action on a forest (graft, prune, route, ...) is appended as an entry that
is (1) HMAC-signed with a secret key and (2) hash-chained to the previous entry.
Any later edit, reorder, insert, or delete breaks the chain or the signature, so
`verify()` detects tampering. Export it as JSON and it's a compliance artifact:
proof, after the fact, of exactly what changed and in what order.

Default signing is HMAC-SHA256 with **no third-party dependencies** (hashlib + hmac
+ json). Optionally, entries can be signed with an **Ed25519 private key** instead
(pass `private_key=`), so an exported log is verifiable with only the *public* key —
no shared secret. That path needs the optional `cryptography` dependency
(`pip install -e ".[crypto]"`); see SECURITY_REVIEW.md F7.
"""
import hashlib
import hmac
import json
import time


class AuditLog:
    def __init__(self, secret="das-dev-key", private_key=None, time_fn=None):
        self.entries = []
        self._priv = None
        self.public_key_hex = None
        # F4: timestamps are UTC by default and ADVISORY (the chain proves order,
        # not wall-clock time). Inject `time_fn` -> str to bind a trusted-time
        # source (e.g. an RFC 3161 / roughtime token) for stronger guarantees.
        self._time_fn = time_fn
        if private_key is not None:
            self._init_ed25519(private_key)
        else:
            self.scheme = "hmac-sha256 + sha256 hash-chain"
            self.secret = secret.encode() if isinstance(secret, str) else secret

    def _init_ed25519(self, private_key):
        """Switch this log to asymmetric Ed25519 signing: entries are signed with
        the private key and verifiable by anyone holding only the PUBLIC key — no
        shared secret. `private_key` may be a PEM (bytes/str) or an
        Ed25519PrivateKey object. Closes SECURITY_REVIEW.md F7. Requires the
        optional `cryptography` dependency (`pip install -e ".[crypto]"`)."""
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        if isinstance(private_key, (bytes, str)):
            pem = private_key.encode() if isinstance(private_key, str) else private_key
            private_key = serialization.load_pem_private_key(pem, password=None)
        if not isinstance(private_key, Ed25519PrivateKey):
            raise TypeError("private_key must be an Ed25519 private key (PEM bytes/str or object)")
        self._priv = private_key
        self.secret = None
        self.scheme = "ed25519 + sha256 hash-chain"
        self.public_key_hex = private_key.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw).hex()

    def _sign(self, body):
        if self._priv is not None:                 # Ed25519 (deterministic)
            return self._priv.sign(body.encode()).hex()
        return hmac.new(self.secret, body.encode(), hashlib.sha256).hexdigest()

    @staticmethod
    def _body(e):
        return f"{e['seq']}|{e['ts']}|{e['event']}|{e['detail']}|{e['payload_hash']}|{e['prev']}"

    def _now(self):
        """Entry timestamp. UTC ISO-8601 (…Z) by default; a `time_fn` can supply a
        trusted-time token instead (F4). Advisory either way — ordering is proven
        by the chain + the freshness anchor (F1), not the clock."""
        if self._time_fn is not None:
            return self._time_fn()
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def append(self, event, detail, payload=None):
        """Record an action. `payload` (e.g. {expert: weight_hash}) is hashed in,
        so the log also fingerprints the forest state at each step. The raw
        payload is retained on the entry so an exported document carries the
        actual fingerprints (re-checkable against the signed `payload_hash`
        without the secret)."""
        seq = len(self.entries)
        ts = self._now()
        payload = payload or {}
        payload_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        prev = self.entries[-1]["sig"] if self.entries else "genesis"
        e = {"seq": seq, "ts": ts, "event": event, "detail": detail,
             "payload": payload, "payload_hash": payload_hash, "prev": prev}
        e["sig"] = self._sign(self._body(e))
        self.entries.append(e)
        return e

    def verify(self):
        """Re-walk the chain + signatures. Returns (ok, broken_index, reason)."""
        prev = "genesis"
        for i, e in enumerate(self.entries):
            if e["prev"] != prev:
                return False, i, "chain broken (entry reordered/inserted/deleted)"
            if "payload" in e:
                ph = hashlib.sha256(json.dumps(e["payload"], sort_keys=True).encode()).hexdigest()
                if ph != e["payload_hash"]:
                    return False, i, "payload does not match its signed fingerprint"
            if not hmac.compare_digest(self._sign(self._body(e)), e["sig"]):
                return False, i, "signature mismatch (entry altered)"
            prev = e["sig"]
        return True, -1, "ok"

    @property
    def head(self):
        """The chain head — the last entry's signature (or 'genesis' if empty)."""
        return self.entries[-1]["sig"] if self.entries else "genesis"

    def sign_blob(self, obj):
        """Sign an arbitrary JSON-serializable object with the log's scheme
        (HMAC or Ed25519) over its canonical form. Used to attest metadata that
        travels WITH an exported document but outside the entry chain — e.g. a
        compliance bundle's manifest — so a keyed verifier catches wrapper
        tampering, not just chain tampering."""
        body = json.dumps(obj, sort_keys=True, separators=(",", ":"))
        return self._sign(body)

    def to_document(self, meta=None):
        """A self-contained, exportable compliance document: the full signed
        chain (with the actual weight fingerprints) plus scheme/provenance
        metadata. The secret is NEVER included. Verify it offline with
        `verify_document` / the `das-verify` CLI."""
        doc = {
            "das_audit_version": 1,
            "scheme": self.scheme,
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "count": len(self.entries),
            "head": self.head,
            "entries": list(self.entries),
        }
        if self.public_key_hex:        # so an Ed25519 doc is self-describing
            doc["public_key"] = self.public_key_hex
        if meta:
            doc["meta"] = meta
        return doc

    def export(self, path, meta=None):
        with open(path, "w") as f:
            json.dump(self.to_document(meta), f, indent=2)

    @classmethod
    def load(cls, path, secret="das-dev-key", private_key=None):
        log = cls(secret=secret, private_key=private_key)
        with open(path) as f:
            log.entries = json.load(f)["entries"]
        return log


def generate_keypair():
    """Generate an Ed25519 audit-signing keypair. Returns (private_pem_bytes,
    public_key_hex). Keep the private PEM on the writer; publish the public hex —
    a regulator verifies an exported log with the public key alone."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    priv = Ed25519PrivateKey.generate()
    pem = priv.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    pub_hex = priv.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw).hex()
    return pem, pub_hex


def _load_ed25519_public(pub):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    if isinstance(pub, str):
        pub = bytes.fromhex(pub.strip())
    return Ed25519PublicKey.from_public_bytes(pub)


def verify_document(doc, secret=None, public_key=None):
    """Independently verify an exported audit document. Returns (ok, issues),
    issues being a list of (entry_index, reason).

    Structural checks need NO key: sequence order, hash-chain continuity, and that
    each entry's recorded fingerprints still match their signed hash. These catch
    reordering, insertion, deletion, and any edit to the recorded weight
    fingerprints.

    Authenticity (the signatures) is checked when a verifier is available:
      * HMAC docs   — pass `secret` (symmetric; the verifier needs the shared key).
      * Ed25519 docs — pass `public_key` (hex or raw bytes), or fall back to the
        key embedded in the document. The Ed25519 path proves authorship with only
        the PUBLIC key — pin `public_key` out-of-band to prove it wasn't re-signed
        with an attacker's key."""
    issues = []
    if isinstance(doc, dict):
        entries = doc.get("entries", [])
        scheme = doc.get("scheme", "")
        embedded_pub = doc.get("public_key")
    else:
        entries, scheme, embedded_pub = list(doc), "", None

    auth = None
    if "ed25519" in scheme:
        pub_hex = public_key or embedded_pub
        if pub_hex is not None:
            pub = _load_ed25519_public(pub_hex)
            def auth(body, sig, pub=pub):
                try:
                    pub.verify(bytes.fromhex(sig), body.encode())
                    return True
                except Exception:
                    return False
    elif secret is not None:
        key = secret.encode() if isinstance(secret, str) else secret
        def auth(body, sig, key=key):
            return hmac.compare_digest(
                hmac.new(key, body.encode(), hashlib.sha256).hexdigest(), sig)

    prev = "genesis"
    for i, e in enumerate(entries):
        if e.get("seq") != i:
            issues.append((i, f"sequence number {e.get('seq')} out of order (expected {i})"))
        if e.get("prev") != prev:
            issues.append((i, "chain broken (entry reordered/inserted/deleted)"))
        if "payload" in e:
            ph = hashlib.sha256(json.dumps(e["payload"], sort_keys=True).encode()).hexdigest()
            if ph != e.get("payload_hash"):
                issues.append((i, "recorded fingerprints do not match their signed hash"))
        if auth is not None and not auth(AuditLog._body(e), e.get("sig", "")):
            issues.append((i, "signature mismatch (entry altered, or wrong key)"))
        prev = e.get("sig")
    if isinstance(doc, dict):
        if doc.get("count") not in (None, len(entries)):
            issues.append((len(entries), f"count {doc.get('count')} != {len(entries)} entries present"))
        if doc.get("head") not in (None, prev):
            issues.append((len(entries), "head does not match the final entry signature"))
        # Wrapper attestation (e.g. a compliance bundle's manifest): the signed
        # fields are listed in the document itself; authenticity requires a key,
        # same as entry signatures — keyless verification stays structural-only.
        if doc.get("bundle_signature") is not None and auth is not None:
            fields = doc.get("bundle_signed_fields") or []
            body = json.dumps({k: doc.get(k) for k in fields},
                              sort_keys=True, separators=(",", ":"))
            if not auth(body, doc["bundle_signature"]):
                issues.append((len(entries),
                               "bundle metadata signature mismatch (manifest/"
                               "wrapper altered, or wrong key)"))
    return (not issues), issues
