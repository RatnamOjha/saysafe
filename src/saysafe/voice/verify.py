"""Score audio against a profile and band it: accept / uncertain / reject.

Thresholds, band and load_thresholds() live in saysafe.approvals.fusion (no numpy needed).
"""

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from saysafe.approvals.fusion import Band, Thresholds, band, load_thresholds
from saysafe.voice.embed import Embedding, TooShort, cosine, embed
from saysafe.voice.profile_store import Profile

__all__ = ["Band", "Thresholds", "VerifyResult", "band", "load_thresholds", "score", "verify"]


@dataclass
class VerifyResult:
    score: float | None  # None when there wasn't enough speech to score
    band: Band
    speech_seconds: float
    latency_ms: float

    @property
    def too_short(self) -> bool:
        return self.score is None


def score(audio: np.ndarray, profile: Profile) -> float:
    return cosine(embed(audio).vector, profile.mean)


def verify(
    audio: np.ndarray,
    profile: Profile,
    embedder: Callable[[np.ndarray], Embedding] = embed,
    t: Thresholds | None = None,
) -> VerifyResult:
    """Too little speech is `uncertain` with no score: never accept, never hard-reject."""
    start = time.perf_counter()
    t = t or load_thresholds()
    try:
        e = embedder(audio)
    except TooShort as short:
        return VerifyResult(None, "uncertain", short.speech_seconds, _ms(start))
    s = cosine(e.vector, profile.mean)
    return VerifyResult(s, band(s, t), e.speech_seconds, _ms(start))


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000
