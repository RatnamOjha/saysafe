"""Score audio against a profile and band it: accept / uncertain / reject."""

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import yaml

from saysafe.config import CONFIG_DIR, load_yaml
from saysafe.identity.embed import Embedding, TooShort, cosine, embed
from saysafe.identity.profile_store import Profile

Band = Literal["accept", "uncertain", "reject"]


@dataclass(frozen=True)
class Thresholds:
    t_accept: float
    t_reject: float
    source: str


@dataclass
class VerifyResult:
    score: float | None  # None when there wasn't enough speech to score
    band: Band
    speech_seconds: float
    latency_ms: float

    @property
    def too_short(self) -> bool:
        return self.score is None


def thresholds() -> Thresholds:
    """Calibrated thresholds when eval has produced them, else the placeholders."""
    calibrated = CONFIG_DIR / "thresholds.calibrated.yaml"
    if calibrated.exists():
        data = yaml.safe_load(calibrated.read_text())
        return Thresholds(data["t_accept"], data["t_reject"], calibrated.name)
    data = load_yaml("thresholds")
    return Thresholds(data["t_accept"], data["t_reject"], "thresholds.yaml")


def band(score: float, t: Thresholds) -> Band:
    """score >= t_accept -> accept; score < t_reject -> reject; otherwise uncertain."""
    if score >= t.t_accept:
        return "accept"
    if score < t.t_reject:
        return "reject"
    return "uncertain"


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
    t = t or thresholds()
    try:
        e = embedder(audio)
    except TooShort as short:
        return VerifyResult(None, "uncertain", short.speech_seconds, _ms(start))
    s = cosine(e.vector, profile.mean)
    return VerifyResult(s, band(s, t), e.speech_seconds, _ms(start))


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000
