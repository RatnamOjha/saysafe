"""Voice scores -> accept / uncertain / reject. Plain floats, so the core needs no numpy.

A score is a similarity between a voice and the owner's voiceprint (cosine for ECAPA,
from saysafe.voice or your own speaker-verification model). The reply score and the
command score are fused; either one under t_reject blocks an accept.
"""

from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from saysafe.config import load_yaml

Band = Literal["accept", "uncertain", "reject"]


@dataclass(frozen=True)
class Thresholds:
    t_accept: float  # score >= t_accept -> accept
    t_reject: float  # score <  t_reject -> reject; in between -> uncertain
    source: str = "custom"


def load_thresholds(path: str | Path | None = None) -> Thresholds:
    """Thresholds from a YAML file with t_accept and t_reject (e.g. your own calibration),
    or the packaged placeholders. Calibrate on your own mic and users before relying on them."""
    if path is not None:
        data = yaml.safe_load(Path(path).read_text())
        return Thresholds(data["t_accept"], data["t_reject"], Path(path).name)
    data = load_yaml("thresholds")
    return Thresholds(data["t_accept"], data["t_reject"], "default")


def band(score: float, t: Thresholds) -> Band:
    if score >= t.t_accept:
        return "accept"
    if score < t.t_reject:
        return "reject"
    return "uncertain"


def fuse(
    reply: float | None,
    command: float | None,
    t: Thresholds,
    reply_weight: float = 0.6,
    command_weight: float = 0.4,
) -> tuple[float | None, Band]:
    """(fused score, band). No reply score -> uncertain. No command score -> the reply
    alone. Either score under t_reject -> never accept."""
    if reply is None:
        return None, "uncertain"
    fused = reply if command is None else reply_weight * reply + command_weight * command
    b = band(fused, t)
    if b == "accept" and min(s for s in (reply, command) if s is not None) < t.t_reject:
        b = "uncertain"
    return fused, b


def calibrate(owner: Sequence[float], others: Sequence[float], target: float = 0.01) -> Thresholds:
    """Thresholds from your own scores: the owner's held-out clips vs other voices.

    t_accept is the lowest threshold that accepts at most `target` of the other voices;
    t_reject the highest that rejects at most `target` of the owner's clips. When the two
    sets overlap, the band between them is "uncertain" and steps up to the phone. Use a
    few hundred scores of each, recorded on the mic you ship, and keep them separate from
    the clips you enrolled with.
    """
    if not owner or not others:
        raise ValueError("need scores for both the owner and other voices")
    owner_sorted, others_sorted = sorted(owner), sorted(others)
    grid = [round(-0.2 + i / 1000, 3) for i in range(1201)]

    def far(t: float) -> float:  # other voices at or above t
        return (len(others_sorted) - bisect_left(others_sorted, t)) / len(others_sorted)

    def frr(t: float) -> float:  # owner clips below t
        return bisect_left(owner_sorted, t) / len(owner_sorted)

    accept = min(t for t in grid if far(t) <= target)
    reject = max((t for t in grid if frr(t) <= target), default=grid[0])
    return Thresholds(max(accept, reject), min(accept, reject), "calibrated")
