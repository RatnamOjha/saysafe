"""silero-vad: offline segments() and a StreamingVAD for 32 ms frames.

The silero model ships inside the silero-vad package, so nothing is downloaded.
"""

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from earshot.audio.io import SR
from earshot.config import load_yaml

FRAME = 512  # silero needs exactly 512 samples (32 ms) per call at 16 kHz


@dataclass(frozen=True)
class Segment:
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def _cfg() -> dict:
    return load_yaml("audio")["vad"]


@lru_cache
def _model():
    import torch
    from silero_vad import load_silero_vad

    torch.set_num_threads(1)
    return load_silero_vad()


def segments(audio: np.ndarray) -> list[Segment]:
    """Speech regions in a whole clip."""
    import torch
    from silero_vad import get_speech_timestamps

    if len(audio) < FRAME:
        return []
    cfg = _cfg()
    stamps = get_speech_timestamps(
        torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)),
        _model(),
        threshold=cfg["threshold"],
        neg_threshold=cfg["neg_threshold"],
        sampling_rate=SR,
        min_speech_duration_ms=cfg["min_speech_ms"],
        min_silence_duration_ms=cfg["min_silence_ms"],
        speech_pad_ms=cfg["speech_pad_ms"],
    )
    return [Segment(s["start"] / SR, s["end"] / SR) for s in stamps]


def speech_only(audio: np.ndarray) -> np.ndarray:
    """Concatenate just the speech regions of a clip."""
    segs = segments(audio)
    if not segs:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate([audio[int(s.start_s * SR) : int(s.end_s * SR)] for s in segs])


def silero_prob() -> Callable[[np.ndarray], float]:
    """Stateful per-frame speech probability from silero. Use one per stream."""
    import torch

    model = _model()
    model.reset_states()
    return lambda frame: float(model(torch.from_numpy(frame), SR).item())


class StreamingVAD:
    """Feed audio of any length; get back finished speech segments as arrays.

    `prob_fn` maps one 512-sample frame to a speech probability. It defaults to
    silero; tests pass a fake.
    """

    def __init__(self, prob_fn: Callable[[np.ndarray], float] | None = None):
        cfg = _cfg()
        self.threshold = cfg["threshold"]
        self.neg_threshold = cfg["neg_threshold"]
        self.min_speech_frames = _frames(cfg["min_speech_ms"])
        self.min_silence_frames = _frames(cfg["min_silence_ms"])
        self.max_frames = _frames(cfg["max_segment_s"] * 1000)
        self._pad = _frames(cfg["speech_pad_ms"])
        self._prob = prob_fn or silero_prob()
        self._pending = np.zeros(0, dtype=np.float32)
        self._preroll: deque[np.ndarray] = deque(maxlen=self._pad)
        self._speech: list[np.ndarray] = []
        self._voiced = 0
        self._silence = 0
        self.in_speech = False

    def feed(self, audio: np.ndarray) -> list[np.ndarray]:
        self._pending = np.concatenate([self._pending, np.asarray(audio, dtype=np.float32)])
        done = []
        while len(self._pending) >= FRAME:
            frame, self._pending = self._pending[:FRAME], self._pending[FRAME:]
            segment = self._step(frame)
            if segment is not None:
                done.append(segment)
        return done

    def flush(self) -> np.ndarray | None:
        """End of stream: return any speech in progress."""
        segment = self._finish() if self.in_speech else None
        self._pending = np.zeros(0, dtype=np.float32)
        return segment

    def _step(self, frame: np.ndarray) -> np.ndarray | None:
        p = self._prob(frame)
        if not self.in_speech:
            if p >= self.threshold:
                self.in_speech = True
                self._speech = [*self._preroll, frame]
                self._voiced, self._silence = 1, 0
            else:
                self._preroll.append(frame)
            return None

        self._speech.append(frame)
        if p < self.neg_threshold:
            self._silence += 1
        else:
            self._silence = 0
            self._voiced += 1
        if self._silence >= self.min_silence_frames or len(self._speech) >= self.max_frames:
            return self._finish()
        return None

    def _finish(self) -> np.ndarray | None:
        # Trim trailing silence but keep a little tail, like the leading pad.
        keep = len(self._speech) - max(self._silence - self._pad, 0)
        audio = np.concatenate(self._speech[:keep])
        long_enough = self._voiced >= self.min_speech_frames
        self.in_speech = False
        self._speech, self._voiced, self._silence = [], 0, 0
        self._preroll.clear()
        return audio if long_enough else None


def _frames(ms: float) -> int:
    return max(1, round(ms / 1000 * SR / FRAME))
