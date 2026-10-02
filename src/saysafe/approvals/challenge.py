"""One-time challenge words: bound to an action, single use, 30 s expiry.

A recording of the owner saying "yes" works for any action. A recording of an old
challenge word doesn't, because each word is fresh, tied to one action, and burned
after one use.
"""

import secrets
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

from saysafe.config import CONFIG_DIR, load_yaml


@lru_cache
def words() -> tuple[str, ...]:
    lines = (CONFIG_DIR / "challenge_words.txt").read_text().splitlines()
    return tuple(w.strip().lower() for w in lines if w.strip() and not w.startswith("#"))


@dataclass
class Challenge:
    word: str
    action_id: str
    issued_at: float
    expires_at: float
    used: bool = False

    def status(self, now: float) -> str:
        """valid | expired | used"""
        if self.used:
            return "used"
        return "expired" if now >= self.expires_at else "valid"


class ChallengeIssuer:
    """Issues words with `secrets`, never repeating one of the last `avoid_recent` words."""

    def __init__(self, clock: Callable[[], float] = time.time, ttl_s: float | None = None):
        cfg = load_yaml("policy").get("challenge", {})
        self.clock = clock
        self.ttl_s = ttl_s if ttl_s is not None else cfg.get("ttl_s", 30)
        self._recent: deque[str] = deque(maxlen=min(cfg.get("avoid_recent", 50), len(words()) // 2))
        self.by_action: dict[str, Challenge] = {}

    def issue(self, action_id: str) -> Challenge:
        pool = [w for w in words() if w not in self._recent]
        word = secrets.choice(pool)
        self._recent.append(word)
        now = self.clock()
        c = Challenge(word, action_id, now, now + self.ttl_s)
        self.by_action[action_id] = c
        return c

    def consume(self, challenge: Challenge) -> None:
        challenge.used = True
