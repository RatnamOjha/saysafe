"""HMAC-signed approval tokens bound to action_hash, single-use.

token = base64url(payload_json) + "." + base64url(hmac_sha256(secret, payload_b64))

verify_token checks, in order: signature, that the action about to run hashes to the
approved action_hash, expiry, and finally burns the nonce in SQLite. The nonce insert
is a single atomic statement, so two concurrent uses of one token can't both pass.
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


class TokenService:
    """Issues and verifies approval tokens.

    secret:  at least 32 random bytes, kept server-side (e.g. secrets.token_bytes(32)).
             Tokens signed with one secret never verify under another.
    db_path: where used nonces are recorded. The default, ":memory:", forgets them when
             the process exits; pass a file path so replays stay blocked across restarts.
    """

    def __init__(
        self,
        secret: bytes,
        db_path: Path | str = ":memory:",
        clock: Callable[[], float] = time.time,
        ttl_s: float | None = None,
    ):
        if not isinstance(secret, bytes) or len(secret) < 16:
            raise ValueError("secret must be at least 16 random bytes (32 recommended)")
        self._secret = secret
        self.clock = clock
        self.ttl_s = ttl_s if ttl_s is not None else load_yaml("policy")["tokens"]["ttl_s"]
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(db_path), check_same_thread=False, isolation_level=None)
        self._db.execute("CREATE TABLE IF NOT EXISTS used (nonce TEXT PRIMARY KEY, at REAL)")
        self._lock = threading.Lock()

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
        with self._lock:
            try:
                self._db.execute(
                    "INSERT INTO used (nonce, at) VALUES (?, ?)", (payload["nonce"], self.clock())
                )
            except sqlite3.IntegrityError:
                raise Refused("replayed") from None
        return payload
