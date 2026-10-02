"""Speech-to-text: LocalWhisperSTT (default) and GroqWhisperSTT.

Local is the default because cloud STT means the user's audio leaves the machine.
Set EARSHOT_STT=groq to opt in; if Groq fails, it falls back to local.
"""

import io
import logging
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal, Protocol

import numpy as np
import soundfile as sf

from saysafe.audio.io import SR
from saysafe.config import cache_dir, env, load_yaml

log = logging.getLogger(__name__)

Purpose = Literal["command", "reply"]


@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class Transcript:
    text: str
    words: list[Word] = field(default_factory=list)
    latency_ms: float = 0.0
    model: str = ""


class STT(Protocol):
    def transcribe(self, audio: np.ndarray) -> Transcript: ...


class LocalWhisperSTT:
    """faster-whisper, int8 on CPU."""

    def __init__(self, model: str):
        from faster_whisper import WhisperModel

        cfg = load_yaml("audio")["stt"]
        local = cache_dir() / "whisper" / model
        self.name = model
        self.beam_size = cfg["beam_size"]
        self._model = WhisperModel(
            str(local) if local.exists() else model,
            device="cpu",
            compute_type=cfg["compute_type"],
        )

    def transcribe(self, audio: np.ndarray) -> Transcript:
        start = time.perf_counter()
        segments, _ = self._model.transcribe(
            np.asarray(audio, dtype=np.float32),
            language="en",
            beam_size=self.beam_size,
            word_timestamps=True,
            condition_on_previous_text=False,
            vad_filter=False,  # callers already ran VAD
        )
        words = [
            Word(w.word.strip(), w.start, w.end) for seg in segments for w in (seg.words or [])
        ]
        text = " ".join(w.text for w in words).strip()
        return Transcript(text, words, (time.perf_counter() - start) * 1000, self.name)


class GroqWhisperSTT:
    """whisper-large-v3-turbo on the same OpenAI-compatible endpoint as the LLM."""

    def __init__(self, fallback: STT):
        self.name = load_yaml("audio")["stt"]["groq_model"]
        self._fallback = fallback

    def transcribe(self, audio: np.ndarray) -> Transcript:
        from openai import OpenAI

        start = time.perf_counter()
        buf = io.BytesIO()
        sf.write(buf, audio, SR, format="WAV", subtype="PCM_16")
        try:
            client = OpenAI(
                base_url=env("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
                api_key=env("LLM_API_KEY"),
                timeout=5.0,
                max_retries=0,
            )
            resp = client.audio.transcriptions.create(
                model=self.name,
                file=("audio.wav", buf.getvalue(), "audio/wav"),
                language="en",
                response_format="verbose_json",
                timestamp_granularities=["word"],
            )
        except Exception as e:
            log.warning("stt: groq failed (%s), using local", type(e).__name__)
            return self._fallback.transcribe(audio)
        words = [Word(w.word.strip(), w.start, w.end) for w in (getattr(resp, "words", None) or [])]
        latency = (time.perf_counter() - start) * 1000
        return Transcript(resp.text.strip(), words, latency, self.name)


@lru_cache
def get_stt(purpose: Purpose = "command") -> STT:
    """Commands use the larger model; short approval replies use the faster one."""
    cfg = load_yaml("audio")["stt"]
    local = LocalWhisperSTT(cfg[f"{purpose}_model"])
    if env("EARSHOT_STT") == "groq" and env("LLM_API_KEY"):
        return GroqWhisperSTT(fallback=local)
    return local
