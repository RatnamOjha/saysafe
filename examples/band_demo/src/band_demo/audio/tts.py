"""Text-to-speech: PiperTTS and NullTTS."""

import time
from functools import lru_cache
from typing import Protocol

import numpy as np

from saysafe.config import cache_dir, env, load_yaml
from saysafe.voice.io import SR, resample


class TTS(Protocol):
    def synthesize(self, text: str) -> np.ndarray: ...
    def play(self, audio: np.ndarray, volume: float = 1.0) -> None: ...


class PiperTTS:
    def __init__(self, voice: str | None = None):
        from piper import PiperVoice

        voice = voice or load_yaml("audio")["tts"]["piper_voice"]
        matches = sorted((cache_dir() / "piper").rglob(f"{voice}.onnx"))
        if not matches:
            raise FileNotFoundError(f"Piper voice {voice} not found. Run: make models")
        self._voice = PiperVoice.load(matches[0])
        self.last_latency_ms = 0.0

    def synthesize(self, text: str) -> np.ndarray:
        start = time.perf_counter()
        chunks = list(self._voice.synthesize(text))
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        audio = np.concatenate([c.audio_float_array for c in chunks])
        out = resample(audio, chunks[0].sample_rate)
        self.last_latency_ms = (time.perf_counter() - start) * 1000
        return out

    def play(self, audio: np.ndarray, volume: float = 1.0) -> None:
        import sounddevice as sd

        sd.play(np.clip(audio * volume, -1.0, 1.0), SR)
        sd.wait()


class NullTTS:
    """Silent TTS for tests and text mode. Remembers what it was asked to say."""

    def __init__(self):
        self.spoken: list[str] = []
        self.last_latency_ms = 0.0

    def synthesize(self, text: str) -> np.ndarray:
        self.spoken.append(text)
        return np.zeros(0, dtype=np.float32)

    def play(self, audio: np.ndarray, volume: float = 1.0) -> None:
        pass


@lru_cache
def get_tts() -> TTS:
    """Piper unless EARSHOT_TTS=null."""
    return NullTTS() if env("EARSHOT_TTS") == "null" else PiperTTS()
