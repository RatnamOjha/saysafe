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
import logging
import secrets
import sqlite3
import threading
import time
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

from saysafe.approvals.actions import Action, action_hash
from saysafe.config import ROOT, env, load_yaml

log = logging.getLogger(__name__)


class Refused(Exception):
    """The executor must not run the action. reason is one of the REASONS."""

    REASONS = ("bad_signature", "expired", "replayed", "action_changed")

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def data_dir() -> Path:
    return Path(env("EARSHOT_DATA_DIR", str(ROOT / "data"))).expanduser()


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


class TokenService:
    def __init__(
        self,
        secret: bytes | None = None,
        db_path: Path | str | None = None,
        clock: Callable[[], float] = time.time,
        ttl_s: float | None = None,
    ):
        if secret is None:
            configured = env("EARSHOT_SECRET")
            if configured:
                secret = configured.encode()
            else:
                # Fail closed: tokens from a random secret die with this process.
                log.warning("tokens: EARSHOT_SECRET not set, using a per-process secret")
                secret = secrets.token_bytes(32)
        self._secret = secret
        self.clock = clock
        self.ttl_s = ttl_s if ttl_s is not None else load_yaml("policy")["tokens"]["ttl_s"]
        if db_path is None:
            data_dir().mkdir(parents=True, exist_ok=True)
            db_path = data_dir() / "nonces.sqlite"
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


@lru_cache
def default_service() -> TokenService:
    """One service per process, shared by the approvals hook and the executor."""
    return TokenService()


def verify_token(token: str, action: Action) -> None:
    default_service().verify(token, action)
