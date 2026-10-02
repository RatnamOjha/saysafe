"""The enrolled owner's voice, for approvals (who's speaking) and audience (who's listening)."""

import logging
from functools import partial

import numpy as np

from saysafe.config import env
from saysafe.identity.embed import Embedding, cosine, embed
from saysafe.identity.profile_store import Profile, ProfileError, ProfileStore
from saysafe.identity.verify import VerifyResult, verify

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
        return VoiceScorer(ProfileStore().load(name))
    except ProfileError as e:
        log.warning("no usable owner profile (%s)", e)
        return None
