"""In-memory tracker of who else has spoken recently. Nothing written to disk.

The band has no wake word, so its mics already hear the room. For each speech
segment of 1 s or more, the tracker scores it against the owner, keeps only
(time, label, score), and drops the audio. Observations are forgotten after 60 s.

  others_present  another voice in the last 45 s (unclear counts as another voice)
  unknown         the mic has been on for under 45 s, so silence proves nothing
  alone_likely    only the owner, and at least 45 s of listening

A silent person in the room is invisible to audio. That's why "unknown" is cautious.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from saysafe.config import load_yaml

if TYPE_CHECKING:
    import numpy as np

SR = 16000  # audio is 16 kHz mono everywhere in saysafe

Label = Literal["owner", "other", "unclear"]
AudienceLevel = Literal["alone_likely", "unknown", "others_present"]


@dataclass(frozen=True)
class Observation:
    t: float
    label: Label
    score: float | None


@dataclass
class AudienceState:
    level: AudienceLevel
    listening_seconds: float
    seconds_since_other: float | None
    others_count: int  # rough lower bound: 1 if any other voice was heard, else 0
    other_segments: int
    headphones: bool
    discreet_mode: bool
    evidence: list[str] = field(default_factory=list)


class AudienceTracker:
    def __init__(
        self,
        score: Callable[[np.ndarray], float | None] | None = None,
        clock: Callable[[], float] = time.time,
    ):
        """score(audio) -> cosine to the owner, or None if it can't be scored."""
        self.cfg = load_yaml("routing")["audience"]
        self._score = score
        self.clock = clock
        self._obs: deque[Observation] = deque()
        self._mic_on_since: float | None = None
        self._lock = threading.Lock()
        self.headphones = False
        self.discreet_mode = False

    # mic time survives conversation resets

    def mic_on(self) -> None:
        if self._mic_on_since is None:
            self._mic_on_since = self.clock()

    def mic_off(self) -> None:
        self._mic_on_since = None

    def listening_seconds(self) -> float:
        return 0.0 if self._mic_on_since is None else self.clock() - self._mic_on_since

    def reset(self) -> None:
        """New conversation: forget voices, keep mic time and flags."""
        with self._lock:
            self._obs.clear()

    # observations

    def label(self, score: float | None) -> Label:
        if score is None:
            return "unclear"
        if score >= self.cfg["owner_at_least"]:
            return "owner"
        if score < self.cfg["other_below"]:
            return "other"
        return "unclear"

    def observe_segment(self, audio: np.ndarray) -> Observation | None:
        """Score one VAD segment and forget the audio. Short segments are skipped.
        With no owner enrolled every voice is unclear, so it counts as someone else."""
        if len(audio) / SR < self.cfg["min_segment_s"]:
            return None
        return self.observe_score(self._score(audio) if self._score else None)

    def observe_score(self, score: float | None) -> Observation:
        obs = Observation(self.clock(), self.label(score), score)
        with self._lock:
            self._obs.append(obs)
            self._forget()
        return obs

    def _forget(self) -> None:
        cutoff = self.clock() - self.cfg["memory_s"]
        while self._obs and self._obs[0].t < cutoff:
            self._obs.popleft()

    # state

    def state(self) -> AudienceState:
        now = self.clock()
        window = self.cfg["window_s"]
        with self._lock:
            self._forget()
            others = [o for o in self._obs if o.label != "owner" and now - o.t <= window]
            owner_heard = any(o.label == "owner" for o in self._obs)
        listening = self.listening_seconds()
        since_other = round(now - others[-1].t, 1) if others else None

        evidence = []
        if others:
            evidence.append(f"Another voice {since_other:.0f} s ago")
        if self._mic_on_since is None:
            evidence.append("Mic is off")
        else:
            evidence.append(f"Listening for {listening:.0f} s")
        if not others and owner_heard:
            evidence.append(f"Only your voice in the last {window} s")
        if self.headphones:
            evidence.append("Headphones connected")
        if self.discreet_mode:
            evidence.append("Discreet mode on")

        if others:
            level: AudienceLevel = "others_present"
        elif listening < window:
            level = "unknown"
        else:
            level = "alone_likely"
        return AudienceState(
            level=level,
            listening_seconds=round(listening, 1),
            seconds_since_other=since_other,
            others_count=1 if others else 0,
            other_segments=len(others),
            headphones=self.headphones,
            discreet_mode=self.discreet_mode,
            evidence=evidence,
        )


class FixedAudience:
    """A pretend room for text mode and tests: always reports one level."""

    def __init__(self, level: AudienceLevel):
        self.level = level
        self.headphones = False
        self.discreet_mode = False

    def state(self) -> AudienceState:
        evidence = {
            "alone_likely": ["Simulated: alone"],
            "unknown": ["Simulated: room unknown"],
            "others_present": ["Simulated: another voice nearby"],
        }[self.level]
        return AudienceState(
            self.level, 60.0, 5.0 if self.level == "others_present" else None,
            int(self.level == "others_present"), int(self.level == "others_present"),
            self.headphones, self.discreet_mode, list(evidence),
        )  # fmt: skip
