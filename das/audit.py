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
        so the log also fingerprints the forest state at each step."""
        seq = len(self.entries)
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        payload_hash = hashlib.sha256(json.dumps(payload or {}, sort_keys=True).encode()).hexdigest()
        prev = self.entries[-1]["sig"] if self.entries else "genesis"
        e = {"seq": seq, "ts": ts, "event": event, "detail": detail,
             "payload_hash": payload_hash, "prev": prev}
        e["sig"] = self._sign(self._body(e))
        self.entries.append(e)
        return e

    def verify(self):
        """Re-walk the chain + signatures. Returns (ok, broken_index, reason)."""
        prev = "genesis"
        for i, e in enumerate(self.entries):
            if e["prev"] != prev:
                return False, i, "chain broken (entry reordered/inserted/deleted)"
            if not hmac.compare_digest(self._sign(self._body(e)), e["sig"]):
                return False, i, "signature mismatch (entry altered)"
            prev = e["sig"]
        return True, -1, "ok"

    def export(self, path):
        with open(path, "w") as f:
            json.dump({"entries": self.entries}, f, indent=2)

    @classmethod
    def load(cls, path, secret="das-dev-key"):
        log = cls(secret)
        with open(path) as f:
            log.entries = json.load(f)["entries"]
        return log
