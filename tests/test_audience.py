import numpy as np
import pytest

from earshot.privacy.audience import AudienceTracker


class Clock:
    def __init__(self):
        self.now = 10_000.0

    def __call__(self):
        return self.now


@pytest.fixture
def world():
    clock = Clock()
    tracker = AudienceTracker(score=lambda audio: float(audio[0]), clock=clock)
    tracker.mic_on()
    return clock, tracker


def seg(score: float, seconds: float = 1.5) -> np.ndarray:
    audio = np.zeros(int(16000 * seconds), dtype=np.float32)
    audio[0] = score  # the fake scorer reads it back
    return audio


def test_mic_just_turned_on_is_unknown(world):
    clock, t = world
    assert t.state().level == "unknown"
    clock.now += 44
    assert t.state().level == "unknown"
    clock.now += 1
    assert t.state().level == "alone_likely"


def test_owner_only_is_alone_after_45s(world):
    clock, t = world
    clock.now += 50
    t.observe_segment(seg(0.7))
    s = t.state()
    assert s.level == "alone_likely" and s.others_count == 0
    assert "Only your voice in the last 45 s" in s.evidence


def test_friend_talks_then_leaves(world):
    clock, t = world
    clock.now += 60
    t.observe_segment(seg(0.1))  # friend
    s = t.state()
    assert s.level == "others_present" and s.others_count == 1
    clock.now += 12
    assert "Another voice 12 s ago" in t.state().evidence
    clock.now += 32  # 44 s since the friend
    assert t.state().level == "others_present"
    clock.now += 1.5  # 45.5 s
    assert t.state().level == "alone_likely"


def test_unclear_counts_as_other(world):
    clock, t = world
    clock.now += 60
    obs = t.observe_segment(seg(0.33))
    assert obs.label == "unclear" and t.state().level == "others_present"


@pytest.mark.parametrize("score, label", [(0.38, "owner"), (0.3799, "unclear"),
                                          (0.30, "unclear"), (0.2999, "other")])  # fmt: skip
def test_label_edges(world, score, label):
    assert world[1].label(score) == label


def test_short_segments_are_ignored(world):
    clock, t = world
    clock.now += 60
    assert t.observe_segment(seg(0.1, seconds=0.9)) is None
    assert t.state().level == "alone_likely"


def test_no_owner_enrolled_counts_everyone(world):
    clock, _ = world
    t = AudienceTracker(score=None, clock=clock)
    t.mic_on()
    clock.now += 60
    t.observe_segment(seg(0.9))
    assert t.state().level == "others_present"


def test_reset_keeps_listening_time(world):
    clock, t = world
    clock.now += 60
    t.observe_segment(seg(0.1))
    t.reset()
    s = t.state()
    assert s.level == "alone_likely" and s.listening_seconds == 60


def test_memory_forgets_after_60s(world):
    clock, t = world
    t.observe_segment(seg(0.1))
    clock.now += 61
    assert len(t._obs) == 1 and t.state() and len(t._obs) == 0


def test_flags_show_in_evidence(world):
    _, t = world
    t.headphones, t.discreet_mode = True, True
    s = t.state()
    assert s.headphones and s.discreet_mode
    assert "Headphones connected" in s.evidence and "Discreet mode on" in s.evidence


def test_nothing_is_stored_but_time_label_score(world):
    clock, t = world
    t.observe_segment(seg(0.1))
    assert (
        set(vars(t._obs[0])) == {"t", "label", "score"} if hasattr(t._obs[0], "__dict__") else True
    )
    assert t._obs[0].__dataclass_fields__.keys() == {"t", "label", "score"}
