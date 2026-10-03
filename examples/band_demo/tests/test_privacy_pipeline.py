"""before_speak through the demo pipeline (the router itself is tested in tests/test_route.py)."""

from band_demo.agent.channels import PhoneChannel
from band_demo.agent.events import EventBus
from band_demo.agent.mock_agent import MockAgent, Reply
from band_demo.agent.pipeline import Pipeline
from band_demo.audio.tts import NullTTS

from saysafe.privacy.audience import FixedAudience
from saysafe.privacy.route import PHONE_ONLY_LINE


def test_pipeline_code_alone_vs_friend():
    events = EventBus()
    log = []
    events.subscribe(log.append)

    def run(level):
        p = Pipeline(
            tts=NullTTS(), agent=MockAgent(fixed_code="482913"), events=events,
            phone=PhoneChannel(events, console_only=True), audience=FixedAudience(level),
        )  # fmt: skip
        p.run_text("what's my verification code")
        return p

    alone = run("alone_likely")
    assert alone.tts.spoken == ["Your Chase verification code is 482913."]
    friend = run("others_present")
    assert friend.tts.spoken == [PHONE_ONLY_LINE]
    assert "482913" in friend.phone.sent[-1].text
    types = {e.type for e in log}
    assert {"detection", "audience_state", "route_decision"} <= types


def test_pipeline_flags_reach_the_router():
    p = Pipeline(
        tts=NullTTS(), agent=MockAgent(fixed_code="482913"),
        phone=PhoneChannel(console_only=True), audience=FixedAudience("others_present"),
    )  # fmt: skip
    p.flags["headphones"] = True
    r = p.run_text("what's my verification code")
    assert r.speak.channel == "headphones_full"
    assert p.headphones.sent[-1].text == "Your Chase verification code is 482913."


def test_reply_with_no_audience_tracker_is_cautious():
    p = Pipeline(tts=NullTTS(), agent=MockAgent(fixed_code="482913"),
                 phone=PhoneChannel(console_only=True))  # fmt: skip
    assert p.run_text("what's my verification code").speak.channel == "phone_only"


def test_reply_model_default_is_public():
    assert Reply(text="hi").source_tags == {"public"}
