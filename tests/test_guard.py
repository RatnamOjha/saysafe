import pytest

from saysafe import PrivacyDecision, PrivacyGuard, Room
from saysafe.privacy.audience import AudienceTracker
from saysafe.privacy.detect import Verdict
from saysafe.privacy.route import PHONE_ONLY_LINE

OTP = "Your Chase code is 482913."
RENT = "Your rent of $2,400 is due Friday."
EMAIL = "Email from Jake: lunch moved to noon."
WEATHER = "It will be sunny and 72 degrees."


@pytest.fixture
def guard():
    return PrivacyGuard()


def test_secret_goes_to_the_phone_when_others_may_hear(guard):
    d = guard.check(OTP, room=Room.OTHERS_PRESENT)
    assert isinstance(d, PrivacyDecision)
    assert d.channel == "phone_only"
    assert d.say == PHONE_ONLY_LINE
    assert d.send_to_phone == OTP
    assert d.withheld
    assert d.level == "secret"
    assert "482913" not in d.say
    assert "Others may hear" in d.reasons


def test_alone_and_headphones_say_everything(guard):
    for kwargs in ({"room": Room.ALONE}, {"room": Room.OTHERS_PRESENT, "headphones": True}):
        d = guard.check(OTP, **kwargs)
        assert d.say == OTP
        assert d.send_to_phone is None
        assert not d.withheld


def test_unknown_room_is_the_cautious_default(guard):
    rent = guard.check(RENT)
    assert rent.channel == "speak_redacted_and_phone"
    assert rent.say == "Your rent is due Friday. Details are on your phone."
    assert rent.send_to_phone == RENT
    assert guard.check(OTP).channel == "phone_only"
    assert guard.check(EMAIL).say == EMAIL  # personal is fine unless others may hear
    assert guard.check(EMAIL, room=Room.OTHERS_PRESENT).withheld


def test_public_replies_are_always_said(guard):
    for room in Room:
        d = guard.check(WEATHER, room=room)
        assert (d.say, d.send_to_phone, d.level) == (WEATHER, None, "public")


def test_room_accepts_its_string_value(guard):
    assert guard.check(OTP, room="others_present").channel == "phone_only"
    with pytest.raises(ValueError):
        guard.check(OTP, room="alone")


def test_sources_raise_the_level_without_any_digits(guard):
    text = "Here it is: four eight two nine one three."
    assert not guard.check(text, room=Room.OTHERS_PRESENT).withheld
    d = guard.check(text, sources={"otp"}, room=Room.OTHERS_PRESENT)
    assert d.channel == "phone_only"
    assert "otp" in d.categories


def test_discreet_and_whisper_keep_personal_things_quiet(guard):
    for kwargs in ({"discreet": True}, {"whisper": True}):
        assert guard.check(EMAIL, room=Room.ALONE, **kwargs).withheld
        assert not guard.check(WEATHER, room=Room.ALONE, **kwargs).withheld


def test_context_finds_a_code_explained_in_an_earlier_sentence(guard):
    alone = guard.check("It's 482913.", room=Room.OTHERS_PRESENT)
    assert not alone.withheld  # on its own it's just a number
    d = guard.check("It's 482913.", room=Room.OTHERS_PRESENT, context="Your Chase code is ready.")
    assert d.channel == "phone_only"
    assert d.spans[0].text == "482913"
    assert (d.spans[0].start, d.spans[0].end) == (5, 11)


def test_context_never_lowers_a_level(guard):
    d = guard.check(OTP, room=Room.OTHERS_PRESENT, context="The weather is nice.")
    assert d.level == "secret"
    assert guard.detect(RENT, context="Sure.").level == guard.detect(RENT).level


def test_classifier_plugs_in(guard):
    text = "The results on the lump came back."

    def classifier(t):
        return Verdict("sensitive", ["health"], [("the lump", "health")])

    assert not guard.check(text, room=Room.OTHERS_PRESENT).withheld
    d = PrivacyGuard(classifier=classifier).check(text, room=Room.OTHERS_PRESENT)
    assert d.withheld
    assert "health" in d.categories


TEXTS = [OTP, RENT, EMAIL, WEATHER, "Your dermatologist appointment is at 3.", "Hi there!"]
FLAGS = [{}, {"headphones": True}, {"discreet": True}, {"whisper": True}]


@pytest.mark.parametrize("text", TEXTS)
@pytest.mark.parametrize("room", list(Room))
@pytest.mark.parametrize("flags", FLAGS)
def test_says_as_is_agrees_with_check(guard, text, room, flags):
    d = guard.check(text, room=room, **flags)
    unchanged = d.say == text and not d.withheld
    assert guard.says_as_is(text, room=room, **flags) == unchanged


def test_speaks_everything(guard):
    assert guard.speaks_everything(room=Room.ALONE)
    assert guard.speaks_everything(room=Room.OTHERS_PRESENT, headphones=True)
    assert not guard.speaks_everything(room=Room.UNKNOWN)
    assert not guard.speaks_everything(room=Room.OTHERS_PRESENT)
    assert not guard.speaks_everything(room=Room.ALONE, discreet=True)


def test_tracker_reports_a_room():
    now = [1000.0]
    tracker = AudienceTracker(clock=lambda: now[0])
    tracker.mic_on()
    assert tracker.room() is Room.UNKNOWN
    now[0] += 50
    assert tracker.room() is Room.ALONE
    tracker.observe_score(0.1)  # someone else
    assert tracker.room() is Room.OTHERS_PRESENT
    d = PrivacyGuard().check(OTP, room=tracker.room())
    assert d.channel == "phone_only"
