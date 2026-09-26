"""Guided 8-clip enrollment that builds an owner profile.

The clip logic (outliers, profile building) is pure so it can be tested without a mic.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from earshot.audio import vad
from earshot.audio.io import rms_dbfs
from earshot.config import load_yaml
from earshot.identity.embed import Embedding, TooShort, cosine, embed, normalize
from earshot.identity.profile_store import Profile

PROMPTS = [
    "yes",
    "yeah, do it",
    "confirm",
    "go ahead",
    "Order my usual from DoorDash.",
    "Send fifty dollars to Jake for dinner.",
    "What's on my calendar tomorrow morning?",
    "Remind me to call mom at six tonight.",
]


@dataclass
class Clip:
    prompt: str
    embedding: Embedding
    rms_dbfs: float


def median_embedding(vectors: list[np.ndarray]) -> np.ndarray:
    return normalize(np.median(np.stack(vectors), axis=0))


def outliers(vectors: list[np.ndarray], min_cosine: float) -> list[int]:
    """Indices of clips whose cosine to the median embedding is under min_cosine."""
    med = median_embedding(vectors)
    return [i for i, v in enumerate(vectors) if cosine(v, med) < min_cosine]


def build_profile(name: str, clips: list[Clip], mic_name: str) -> Profile:
    vectors = [c.embedding.vector for c in clips]
    return Profile(
        name=name,
        mean_embedding=normalize(np.mean(np.stack(vectors), axis=0)).tolist(),
        embeddings=[v.tolist() for v in vectors],
        median_rms_dbfs=float(np.median([c.rms_dbfs for c in clips])),
        mic_name=mic_name,
    )


def run_enrollment(
    name: str,
    record: Callable[[float], np.ndarray],
    mic_name: str,
    show: Callable[[str], None] = print,
    embedder: Callable[[np.ndarray], Embedding] = embed,
) -> Profile:
    """Record every prompt, re-record too-short clips and outliers, return the profile."""
    cfg = load_yaml("audio")["enroll"]

    def capture(i: int) -> Clip:
        prompt = PROMPTS[i]
        while True:
            show(f"[{i + 1}/{len(PROMPTS)}] Say: {prompt}")
            audio = record(cfg["clip_seconds"])
            try:
                e = embedder(audio)
            except TooShort as short:
                show(f"  Too short ({short.speech_seconds:.1f} s of speech). Again, a bit slower.")
                continue
            return Clip(prompt, e, rms_dbfs(vad.speech_only(audio)))

    clips = [capture(i) for i in range(len(PROMPTS))]
    for round_ in range(cfg["max_rerecord_rounds"]):
        bad = outliers([c.embedding.vector for c in clips], cfg["min_cosine_to_median"])
        if not bad:
            break
        show(f"{len(bad)} clip(s) don't match the rest. Re-recording (round {round_ + 1}).")
        for i in bad:
            clips[i] = capture(i)
    else:
        bad = outliers([c.embedding.vector for c in clips], cfg["min_cosine_to_median"])
        if bad:
            show(f"Warning: {len(bad)} clip(s) still don't match. Try a quieter room.")
    return build_profile(name, clips, mic_name)
