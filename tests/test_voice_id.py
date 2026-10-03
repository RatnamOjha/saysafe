import numpy as np
import pytest

from saysafe import Action, ApprovalGuard, Room, Thresholds
from saysafe.approvals.fusion import calibrate
from saysafe.voice import EnrollmentError, VoiceID
from saysafe.voice import voice_id as voice_id_module
from saysafe.voice.embed import Embedding, TooShort, normalize

SR = 16000
DIM = 192


def voice(speaker: int, seconds: float = 2.0, jitter: int = 0) -> np.ndarray:
    """A fake clip: sample 0 names the speaker, sample 1 a small variation, the length
    stands for the amount of speech."""
    audio = np.zeros(int(SR * seconds), dtype=np.float32)
    audio[0], audio[1] = speaker, jitter
    return audio


def fake_embed(audio, min_speech_s=None, model_dir=None):
    seconds = len(audio) / SR
    if seconds < (min_speech_s or 0.8):
        raise TooShort(seconds, min_speech_s or 0.8)
    rng = np.random.default_rng(int(audio[0]))
    base = rng.standard_normal(DIM)
    noise = np.random.default_rng(1000 + int(audio[1])).standard_normal(DIM) * 0.15
    return Embedding(normalize(base + noise), seconds, 1.0)


@pytest.fixture
def vid(monkeypatch):
    monkeypatch.setattr(voice_id_module, "embed", fake_embed)
    monkeypatch.setattr(voice_id_module, "rms_dbfs", lambda audio: -20.0)
    monkeypatch.setattr(voice_id_module.vad, "speech_only", lambda audio: audio)
    return VoiceID()


@pytest.fixture
def owner(vid):
    return vid.enroll("owner", [voice(1, jitter=j) for j in range(6)])


def test_enroll_then_score(vid, owner):
    assert owner.name == "owner" and len(owner.embeddings) == 6
    assert vid.score(voice(1, jitter=9), owner) > 0.9
    assert vid.score(voice(2), owner) < 0.3


def test_short_clips_score_none(vid, owner):
    assert vid.score(voice(1, seconds=0.2), owner) is None
    assert vid.score(voice(1, seconds=0.5), owner) is not None  # replies need only 0.35 s


def test_enroll_skips_short_clips_and_other_voices(vid):
    clips = [voice(1, jitter=j) for j in range(4)] + [voice(1, seconds=0.3), voice(7)]
    profile = vid.enroll("owner", clips)
    assert len(profile.embeddings) == 4


def test_enroll_needs_enough_clips(vid):
    with pytest.raises(EnrollmentError, match="had 0.8 s of speech"):
        vid.enroll("owner", [voice(1, seconds=0.3)] * 5)
    with pytest.raises(EnrollmentError, match="same person"):
        vid.enroll("owner", [voice(s) for s in range(1, 6)])


def test_verify_bands(vid, owner):
    t = Thresholds(0.45, 0.25)
    assert vid.verify(voice(1, jitter=3), owner, t).band == "accept"
    assert vid.verify(voice(5), owner, t).band == "reject"
    short = vid.verify(voice(1, seconds=0.1), owner, t)
    assert short.too_short and short.band == "uncertain"


def test_tracker_hears_another_voice(vid, owner):
    now = [100.0]
    tracker = vid.tracker(owner, clock=lambda: now[0])
    tracker.mic_on()
    now[0] += 50
    tracker.observe_segment(voice(1, jitter=4))
    assert tracker.room() is Room.ALONE
    tracker.observe_segment(voice(3))
    assert tracker.room() is Room.OTHERS_PRESENT


def test_scores_drive_an_approval(vid, owner):
    guard = ApprovalGuard(b"k" * 32, thresholds=Thresholds(0.45, 0.25))
    a = Action(type="order_food", counterparty="DoorDash", amount=20)
    step = guard.reply(guard.start(a), "yes", voice_score=vid.score(voice(1, jitter=2), owner))
    assert step.status == "approved"
    step = guard.reply(guard.start(a), "yes", voice_score=vid.score(voice(4), owner))
    assert step.status == "awaiting_phone"


# calibration (core, no numpy needed) ----------------------------------------------------


def test_calibrate_separated():
    owner = [0.5 + i / 1000 for i in range(200)]  # 0.50 .. 0.70
    others = [0.0 + i / 1000 for i in range(200)]  # 0.00 .. 0.20
    t = calibrate(owner, others)
    assert t.t_reject <= 0.2 and 0.5 <= t.t_accept <= 0.51  # the gap is the uncertain band
    assert t.source == "calibrated"


def test_calibrate_overlapping_keeps_an_uncertain_band():
    owner = [0.3 + i / 1000 for i in range(400)]  # 0.30 .. 0.70
    others = [0.1 + i / 1000 for i in range(400)]  # 0.10 .. 0.50
    t = calibrate(owner, others)
    far = sum(s >= t.t_accept for s in others) / len(others)
    frr = sum(s < t.t_reject for s in owner) / len(owner)
    assert far <= 0.01 and frr <= 0.01 and t.t_reject < t.t_accept


def test_calibrate_needs_both_sets():
    with pytest.raises(ValueError):
        calibrate([], [0.1])


@pytest.mark.models
def test_real_enroll_and_score(piper_speech):
    vid = VoiceID()
    n = len(piper_speech)
    clips = [piper_speech[i * n // 4 : (i + 1) * n // 4 + SR] for i in range(4)]
    profile = vid.enroll("piper", clips, min_clips=3)
    assert vid.score(piper_speech, profile) > 0.6
    noise = np.random.default_rng(0).standard_normal(SR * 2).astype(np.float32) * 0.01
    assert vid.score(noise, profile) is None  # no speech, no score
