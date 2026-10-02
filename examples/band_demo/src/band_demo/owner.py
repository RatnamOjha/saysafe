"""The enrolled owner's voice, for approvals (who's speaking) and audience (who's listening)."""

import logging
from collections.abc import Callable
from functools import partial

import numpy as np

from band_demo.config import env, profile_store
from saysafe.voice.embed import Embedding, cosine, embed
from saysafe.voice.profile_store import Profile, ProfileError
from saysafe.voice.verify import VerifyResult, verify

log = logging.getLogger(__name__)


class VoiceScorer:
    """Scores audio and embeddings against the owner's profile."""

    def __init__(self, profile: Profile):
        self.profile = profile

    def score_audio(self, audio: np.ndarray, min_speech_s: float | None = None) -> VerifyResult:
        return verify(audio, self.profile, embedder=partial(embed, min_speech_s=min_speech_s))

    def score_embedding(self, embedding: Embedding) -> float:
        return cosine(embedding.vector, self.profile.mean)


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
