"""The enrolled owner's voice, for approvals (who's speaking) and audience (who's listening)."""

import logging
from collections.abc import Callable

import numpy as np

from band_demo.config import env, profile_store, thresholds
from saysafe.config import load_yaml
from saysafe.voice import Embedding, Profile, ProfileError, VerifyResult, VoiceID

log = logging.getLogger(__name__)

VOICE = VoiceID()  # models load on first use
COMMAND_MIN_SPEECH_S = load_yaml("voice")["enroll"]["min_speech_s"]


def embed_command(audio: np.ndarray) -> Embedding:
    """The spoken command's embedding (raises TooShort under 0.8 s of speech)."""
    return VOICE.embed(audio, min_speech_s=COMMAND_MIN_SPEECH_S)


class VoiceScorer:
    """Scores audio and embeddings against the owner's profile."""

    def __init__(self, profile: Profile, voice: VoiceID = VOICE):
        self.profile = profile
        self.voice = voice

    def score_audio(self, audio: np.ndarray, min_speech_s: float | None = None) -> VerifyResult:
        minimum = COMMAND_MIN_SPEECH_S if min_speech_s is None else min_speech_s
        return self.voice.verify(audio, self.profile, thresholds(), min_speech_s=minimum)

    def score_embedding(self, embedding: Embedding) -> float:
        return self.voice.score_embedding(embedding, self.profile)


def load_owner_scorer() -> VoiceScorer | None:
    """The enrolled owner's scorer, or None if EARSHOT_OWNER has no usable profile."""
    name = env("EARSHOT_OWNER", "owner")
    try:
        return VoiceScorer(profile_store().load(name))
    except ProfileError as e:
        log.warning("no usable owner profile (%s)", e)
        return None


def owner_score_fn() -> Callable[[np.ndarray], float | None] | None:
    """score(audio) against the enrolled owner, or None if nobody is enrolled."""
    scorer = load_owner_scorer()
    if scorer is None:
        return None
    return lambda audio: scorer.score_audio(audio).score
