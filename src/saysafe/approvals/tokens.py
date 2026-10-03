"""HMAC-signed approval tokens bound to action_hash, single-use.

token = base64url(payload_json) + "." + base64url(hmac_sha256(secret, payload_b64))

verify checks, in order: signature, that the action about to run hashes to the
approved action_hash, expiry, and finally burns the nonce. Burning is atomic, so two
concurrent uses of one token can't both pass. Nonces live in a NonceStore: in memory by
default, SQLite for a file that survives restarts, or your own (Redis SET NX, a DB row
with a unique key).
"""

import base64
import hashlib
import hmac
import json
import secrets
import sqlite3
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from saysafe.approvals.actions import Action, action_hash
from saysafe.config import load_yaml


class Refused(Exception):
    """The executor must not run the action. reason is one of the REASONS."""

    REASONS = ("bad_signature", "expired", "replayed", "action_changed")

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


class NonceStore(Protocol):
    def burn(self, nonce: str, at: float) -> bool:
        """Record a nonce as used. True the first time, False if it was already used.
        Must be atomic: two concurrent burns of one nonce can't both return True."""
        ...


class MemoryNonceStore:
    """Forgets used nonces when the process exits. Tokens expire in 60 s by default, so
    a restart only matters if it happens inside that window."""

    def __init__(self) -> None:
        self._used: dict[str, float] = {}
        self._lock = threading.Lock()

    def burn(self, nonce: str, at: float) -> bool:
        with self._lock:
            if nonce in self._used:
                return False
            self._used[nonce] = at
            return True


class SQLiteNonceStore:
    """Used nonces in a SQLite file, so replays stay blocked across restarts."""

    def __init__(self, path: Path | str):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._db.execute("CREATE TABLE IF NOT EXISTS used (nonce TEXT PRIMARY KEY, at REAL)")
        self._lock = threading.Lock()

    def burn(self, nonce: str, at: float) -> bool:
        with self._lock:
            try:
                self._db.execute("INSERT INTO used (nonce, at) VALUES (?, ?)", (nonce, at))
            except sqlite3.IntegrityError:
                return False
            return True


class TokenService:
    """Issues and verifies approval tokens.

    secret: at least 32 random bytes, kept server-side (e.g. secrets.token_bytes(32)).
            Tokens signed with one secret never verify under another.
    nonces: a NonceStore, or a file path for SQLiteNonceStore. Default: in memory.
    """

    def __init__(
        self,
        secret: bytes,
        nonces: NonceStore | Path | str | None = None,
        clock: Callable[[], float] = time.time,
        ttl_s: float | None = None,
    ):
        if not isinstance(secret, bytes) or len(secret) < 16:
            raise ValueError("secret must be at least 16 random bytes (32 recommended)")
        self._secret = secret
        self.clock = clock
        self.ttl_s = ttl_s if ttl_s is not None else load_yaml("policy")["tokens"]["ttl_s"]
        if nonces is None:
            nonces = MemoryNonceStore()
        elif isinstance(nonces, (str, Path)):
            nonces = SQLiteNonceStore(nonces)
        self.nonces: NonceStore = nonces

    def _sign(self, payload_b64: str) -> str:
        return _b64(hmac.new(self._secret, payload_b64.encode(), hashlib.sha256).digest())

    def issue(self, action: Action, tier: str, method: str, fused_score: float | None) -> str:
        now = self.clock()
        payload = {
            "action_id": action.id,
            "action_hash": action_hash(action),
            "tier": tier,
            "method": method,
            "fused_score": None if fused_score is None else round(fused_score, 4),
            "nonce": secrets.token_hex(16),
            "issued_at": now,
            "expires_at": now + self.ttl_s,
        }
        body = _b64(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
        return f"{body}.{self._sign(body)}"

    def verify(self, token: str, action: Action) -> dict:
        """Return the payload if `action` may run now; otherwise raise Refused."""
        try:
            body, sig = token.split(".")
            if not hmac.compare_digest(sig, self._sign(body)):
                raise Refused("bad_signature")
            payload = json.loads(_unb64(body))
        except (ValueError, json.JSONDecodeError):
            raise Refused("bad_signature") from None
        if not hmac.compare_digest(payload["action_hash"], action_hash(action)):
            raise Refused("action_changed")
        if self.clock() >= payload["expires_at"]:
            raise Refused("expired")
        if not self.nonces.burn(payload["nonce"], self.clock()):
            raise Refused("replayed")
        return payload
