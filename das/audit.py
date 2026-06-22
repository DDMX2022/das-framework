"""
das/audit.py
------------
A tamper-evident, signed audit log — the governance differentiator made real.

Every action on a forest (graft, prune, route, ...) is appended as an entry that
is (1) HMAC-signed with a secret key and (2) hash-chained to the previous entry.
Any later edit, reorder, insert, or delete breaks the chain or the signature, so
`verify()` detects tampering. Export it as JSON and it's a compliance artifact:
proof, after the fact, of exactly what changed and in what order.

No third-party dependencies (hashlib + hmac + json).
"""
import hashlib
import hmac
import json
import time


class AuditLog:
    def __init__(self, secret="das-dev-key"):
        self.secret = secret.encode() if isinstance(secret, str) else secret
        self.entries = []

    def _sign(self, body):
        return hmac.new(self.secret, body.encode(), hashlib.sha256).hexdigest()

    @staticmethod
    def _body(e):
        return f"{e['seq']}|{e['ts']}|{e['event']}|{e['detail']}|{e['payload_hash']}|{e['prev']}"

    def append(self, event, detail, payload=None):
        """Record an action. `payload` (e.g. {expert: weight_hash}) is hashed in,
        so the log also fingerprints the forest state at each step. The raw
        payload is retained on the entry so an exported document carries the
        actual fingerprints (re-checkable against the signed `payload_hash`
        without the secret)."""
        seq = len(self.entries)
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
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

    def to_document(self, meta=None):
        """A self-contained, exportable compliance document: the full signed
        chain (with the actual weight fingerprints) plus scheme/provenance
        metadata. The secret is NEVER included. Verify it offline with
        `verify_document` / the `das-verify` CLI."""
        doc = {
            "das_audit_version": 1,
            "scheme": "hmac-sha256 + sha256 hash-chain",
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "count": len(self.entries),
            "head": self.head,
            "entries": list(self.entries),
        }
        if meta:
            doc["meta"] = meta
        return doc

    def export(self, path, meta=None):
        with open(path, "w") as f:
            json.dump(self.to_document(meta), f, indent=2)

    @classmethod
    def load(cls, path, secret="das-dev-key"):
        log = cls(secret)
        with open(path) as f:
            log.entries = json.load(f)["entries"]
        return log


def verify_document(doc, secret=None):
    """Independently verify an exported audit document. Returns (ok, issues),
    issues being a list of (entry_index, reason).

    Structural checks need NO secret: sequence order, hash-chain continuity, and
    that each entry's recorded fingerprints still match their signed hash. These
    catch reordering, insertion, deletion, and any edit to the recorded weight
    fingerprints. Authenticity — proving the log was produced by the holder of
    the key rather than fabricated — is checked only when `secret` is given
    (HMAC). For keyless third-party authenticity, see the asymmetric-signing item
    in SECURITY_REVIEW.md."""
    issues = []
    entries = doc.get("entries", []) if isinstance(doc, dict) else list(doc)
    key = None
    if secret is not None:
        key = secret.encode() if isinstance(secret, str) else secret
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
        if key is not None:
            sig = hmac.new(key, AuditLog._body(e).encode(), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(sig, e.get("sig", "")):
                issues.append((i, "signature mismatch (entry altered, or wrong key)"))
        prev = e.get("sig")
    if isinstance(doc, dict):
        if doc.get("count") not in (None, len(entries)):
            issues.append((len(entries), f"count {doc.get('count')} != {len(entries)} entries present"))
        if doc.get("head") not in (None, prev):
            issues.append((len(entries), "head does not match the final entry signature"))
    return (not issues), issues
