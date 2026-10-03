"""VoiceID: enroll a voice, then score audio against it. Local and on CPU; the only network
use is the first model download (~80 MB, speechbrain/spkrec-ecapa-voxceleb).

    from saysafe.voice import VoiceID

    vid = VoiceID()
    profile = vid.enroll("owner", clips)                 # 5+ clips, 16 kHz mono float32
    score = vid.score(reply_audio, profile)              # cosine to the owner, or None
    step = approvals.reply(step, transcript, voice_score=score)

    tracker = vid.tracker(profile)                       # who's listening
    tracker.mic_on(); tracker.observe_segment(segment)   # every VAD speech segment
    privacy.check(reply, room=tracker.room())

A voice score is evidence, not proof: recordings and voice clones can score like the
owner. That's why money needs a one-time word, and big actions a phone tap.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from saysafe.approvals.fusion import Thresholds, band, load_thresholds
from saysafe.config import load_yaml
from saysafe.privacy.audience import AudienceTracker
from saysafe.voice import vad
from saysafe.voice.embed import DEFAULT_MODEL_DIR, Embedding, TooShort, _encoder, cosine, embed
from saysafe.voice.enroll import Clip, build_profile, outliers
from saysafe.voice.io import rms_dbfs
from saysafe.voice.profile_store import Profile
from saysafe.voice.verify import VerifyResult


class EnrollmentError(ValueError):
    """Not enough usable clips. The message says why and is safe to show."""


class VoiceID:
    """model_dir: where the ECAPA model is kept (downloaded there on first use).
    min_speech_s: speech needed to score a clip; replies like "yes" are short, so the
    default is policy.yaml's min_reply_speech_s (0.35 s). Shorter clips score None."""

    def __init__(
        self, model_dir: Path | str = DEFAULT_MODEL_DIR, *, min_speech_s: float | None = None
    ):
        self.model_dir = Path(model_dir)
        policy_min = load_yaml("policy")["voice"]["min_reply_speech_s"]
        self.min_speech_s = policy_min if min_speech_s is None else min_speech_s
        self._enroll_cfg = load_yaml("voice")["enroll"]

    def warm(self) -> None:
        """Load the models now, so the first real score isn't slow."""
        _encoder(self.model_dir)
        vad.segments(np.zeros(16000, dtype=np.float32))

    def embed(self, audio: np.ndarray, *, min_speech_s: float | None = None) -> Embedding:
        """The speech-only part of `audio` as a unit vector. Raises TooShort."""
        minimum = self.min_speech_s if min_speech_s is None else min_speech_s
        return embed(audio, minimum, self.model_dir)

    def score(
        self, audio: np.ndarray, profile: Profile, *, min_speech_s: float | None = None
    ) -> float | None:
        """Cosine between `audio` and the profile, or None with too little speech."""
        try:
            e = self.embed(audio, min_speech_s=min_speech_s)
        except TooShort:
            return None
        return cosine(e.vector, profile.mean)

    def score_embedding(self, embedding: Embedding, profile: Profile) -> float:
        return cosine(embedding.vector, profile.mean)

    def verify(
        self,
        audio: np.ndarray,
        profile: Profile,
        thresholds: Thresholds | None = None,
        *,
        min_speech_s: float | None = None,
    ) -> VerifyResult:
        """Score plus band (accept / uncertain / reject) and how much speech there was."""
        import time

        start = time.perf_counter()
        t = thresholds or load_thresholds()
        try:
            e = self.embed(audio, min_speech_s=min_speech_s)
        except TooShort as short:
            return VerifyResult(None, "uncertain", short.speech_seconds, _ms(start))
        s = cosine(e.vector, profile.mean)
        return VerifyResult(s, band(s, t), e.speech_seconds, _ms(start))

    def enroll(
        self,
        name: str,
        clips: Sequence[np.ndarray],
        *,
        mic_name: str = "unknown",
        min_clips: int = 3,
    ) -> Profile:
        """A profile from clips of one person speaking (3 s each is plenty; 5-8 clips).

        Clips with under 0.8 s of speech are skipped, and so are clips that don't sound
        like the rest. Raises EnrollmentError if fewer than `min_clips` are left. Save the
        profile with ProfileStore, which encrypts it.
        """
        cfg = self._enroll_cfg
        kept: list[Clip] = []
        for audio in clips:
            try:
                e = embed(audio, cfg["min_speech_s"], self.model_dir)
            except TooShort:
                continue
            kept.append(Clip("", e, rms_dbfs(vad.speech_only(audio))))
        if len(kept) < min_clips:
            raise EnrollmentError(
                f"Only {len(kept)} of {len(clips)} clips had {cfg['min_speech_s']} s of speech; "
                f"need {min_clips}. Record longer clips, closer to the mic."
            )
        bad = set(outliers([c.embedding.vector for c in kept], cfg["min_cosine_to_median"]))
        kept = [c for i, c in enumerate(kept) if i not in bad]
        if len(kept) < min_clips:
            raise EnrollmentError(
                f"Only {len(kept)} clips sound like the same person; need {min_clips}. "
                "Record in a quiet room, one speaker only."
            )
        return build_profile(name, kept, mic_name)

    def tracker(self, profile: Profile, **kwargs) -> AudienceTracker:
        """An AudienceTracker that labels each speech segment owner / other / unclear."""
        minimum = self._enroll_cfg["min_speech_s"]
        score: Callable[[np.ndarray], float | None] = lambda audio: self.score(  # noqa: E731
            audio, profile, min_speech_s=minimum
        )
        return AudienceTracker(score=score, **kwargs)


def _ms(start: float) -> float:
    import time

    return (time.perf_counter() - start) * 1000
