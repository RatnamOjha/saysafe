"""ECAPA speaker embedding of the speech-only part of a clip."""

import logging
import time
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from saysafe.config import cache_dir, load_yaml
from saysafe.voice import vad
from saysafe.voice.io import SR


class TooShort(Exception):
    def __init__(self, speech_seconds: float, minimum: float):
        self.speech_seconds = speech_seconds
        super().__init__(f"only {speech_seconds:.2f} s of speech, need {minimum:.1f} s")


@dataclass
class Embedding:
    vector: np.ndarray  # unit-length, 192 dims
    speech_seconds: float
    latency_ms: float

    def __repr__(self) -> str:  # never print the vector
        return f"Embedding(speech_seconds={self.speech_seconds:.2f})"


@lru_cache
def _encoder():
    from speechbrain.inference.speaker import EncoderClassifier

    path = cache_dir() / "ecapa"
    if not (path / "embedding_model.ckpt").exists():
        raise FileNotFoundError(f"ECAPA model not found in {path}. Run: make models")
    logging.disable(logging.INFO)  # speechbrain logs "Fetch ..." chatter at INFO while loading
    try:
        return EncoderClassifier.from_hparams(
            source=str(path), savedir=str(path), run_opts={"device": "cpu"}
        )
    finally:
        logging.disable(logging.NOTSET)


def embed(audio: np.ndarray, min_speech_s: float | None = None) -> Embedding:
    """VAD first, then embed only the speech. Raises TooShort under min_speech_s
    (default: enroll.min_speech_s from config/audio.yaml)."""
    import torch

    start = time.perf_counter()
    speech = vad.speech_only(audio)
    speech_seconds = len(speech) / SR
    minimum = (
        min_speech_s if min_speech_s is not None else load_yaml("audio")["enroll"]["min_speech_s"]
    )
    if speech_seconds < minimum:
        raise TooShort(speech_seconds, minimum)
    with torch.inference_mode():
        out = _encoder().encode_batch(torch.from_numpy(speech).unsqueeze(0))
    vector = normalize(out.squeeze().numpy().astype(np.float32))
    return Embedding(vector, speech_seconds, (time.perf_counter() - start) * 1000)


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return (v / n).astype(np.float32) if n > 0 else v.astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(normalize(np.asarray(a)), normalize(np.asarray(b))))
