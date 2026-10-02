import numpy as np
import pytest

from saysafe.audio import vad
from saysafe.audio.vad import FRAME, StreamingVAD


def test_silence_has_no_segments():
    assert vad.segments(np.zeros(16000 * 2, dtype=np.float32)) == []


def test_streaming_silence_emits_nothing():
    s = StreamingVAD()
    assert s.feed(np.zeros(16000 * 2, dtype=np.float32)) == []
    assert s.flush() is None


def _fake(pattern: list[int]):
    """StreamingVAD with a fake prob per frame: 1 = speech, 0 = silence."""
    probs = iter(pattern)
    s = StreamingVAD(prob_fn=lambda frame: float(next(probs)))
    return s, np.zeros(FRAME * len(pattern), dtype=np.float32)


def test_streaming_emits_after_min_silence():
    # 20 speech frames (640 ms) then 20 silence frames; min silence is 400 ms (~12 frames).
    s, audio = _fake([1] * 20 + [0] * 20)
    out = s.feed(audio)
    assert len(out) == 1
    assert 20 * FRAME <= len(out[0]) <= 22 * FRAME  # speech plus a short tail


def test_streaming_drops_short_blips():
    s, audio = _fake([1] * 3 + [0] * 20)  # 96 ms < 250 ms minimum
    assert s.feed(audio) == []


def test_streaming_short_pause_stays_one_segment():
    s, audio = _fake([1] * 10 + [0] * 5 + [1] * 10 + [0] * 20)
    assert len(s.feed(audio)) == 1


def test_streaming_accepts_odd_chunk_sizes():
    s, audio = _fake([1] * 20 + [0] * 20)
    out = [seg for chunk in np.array_split(audio, 37) for seg in s.feed(chunk)]
    assert len(out) == 1


def test_flush_returns_speech_in_progress():
    s, audio = _fake([1] * 20)
    assert s.feed(audio) == []
    assert s.flush() is not None


@pytest.mark.models
def test_vad_finds_piper_speech(piper_speech):
    segs = vad.segments(piper_speech)
    assert segs and sum(seg.duration_s for seg in segs) > 2.0
    padded = np.concatenate(
        [np.zeros(16000, np.float32), piper_speech, np.zeros(16000, np.float32)]
    )
    assert len(StreamingVAD().feed(padded)) >= 1


def test_reply_vad_waits_through_a_pause():
    # 10 speech, 18 silence (~576 ms), 10 speech: one segment at 800 ms, two at 400 ms
    pattern = [1] * 10 + [0] * 18 + [1] * 10 + [0] * 30
    probs = iter(pattern)
    long = StreamingVAD(prob_fn=lambda f: float(next(probs)), min_silence_ms=800)
    assert len(long.feed(np.zeros(FRAME * len(pattern), np.float32))) == 1
    probs2 = iter(pattern)
    short = StreamingVAD(prob_fn=lambda f: float(next(probs2)))
    assert len(short.feed(np.zeros(FRAME * len(pattern), np.float32))) == 2
