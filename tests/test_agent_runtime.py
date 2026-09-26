import json

import httpx
import numpy as np
import pytest

from earshot.agent.channels import PhoneChannel, SpeakerChannel
from earshot.agent.events import EventBus
from earshot.agent.executor import Executor, MissingApproval, Refused
from earshot.agent.mock_agent import MockAgent
from earshot.agent.pipeline import Pipeline
from earshot.approvals import hook as approvals_hook
from earshot.approvals.hook import ApprovalDecision
from earshot.audio.tts import NullTTS
from earshot.identity.embed import Embedding, TooShort
from earshot.privacy import hook as privacy_hook


@pytest.fixture
def events():
    bus = EventBus()
    bus.log = []
    bus.subscribe(bus.log.append)
    return bus


def _approve_all(action, ctx):
    from earshot.approvals.tokens import default_service

    return ApprovalDecision("approve", default_service().issue(action, "test", "test", None))


@pytest.fixture
def pipeline(events, monkeypatch):
    """Pipeline mechanics with an approve-everything hook; the real hook is tested elsewhere."""
    monkeypatch.delenv("EARSHOT_AGENT", raising=False)
    monkeypatch.setattr(approvals_hook, "before_execute", _approve_all)
    return Pipeline(
        tts=NullTTS(), agent=MockAgent(fixed_code="482913"), events=events,
        phone=PhoneChannel(events, console_only=True),
    )  # fmt: skip


# events


def test_bus_step_times_and_survives_broken_subscriber(events):
    events.subscribe(lambda e: 1 / 0)
    with events.step("thing", a=1) as out:
        out["b"] = 2
    e = events.log[-1]
    assert e.type == "thing" and e.data == {"a": 1, "b": 2} and e.latency_ms >= 0


# executor


def test_executor_requires_token():
    a = MockAgent().order_usual()
    with pytest.raises(MissingApproval):
        Executor().run(a, None)
    from earshot.approvals.tokens import default_service

    token = default_service().issue(a, "voice", "voice", 0.9)
    assert "Order placed" in Executor().run(a, token).text


def test_executor_uses_verifier():
    def refuse(token, action):
        raise Refused("action_changed")

    with pytest.raises(Refused):
        Executor(verify_token=refuse).run(MockAgent().order_usual(), "tok")


# channels


def test_phone_posts_ntfy_json_with_approval_buttons(events, monkeypatch):
    monkeypatch.setenv("NTFY_TOPIC", "earshot-test")
    monkeypatch.setenv("EARSHOT_PUBLIC_URL", "http://10.0.0.5:8000")
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"id": "x"})

    phone = PhoneChannel(events, client=httpx.Client(transport=httpx.MockTransport(handler)))
    d = phone.request_approval("abc123", "Jake, fifty dollars")
    body = seen[0]
    assert body["topic"] == "earshot-test" and d.extra["via"] == "ntfy"
    urls = [a["url"] for a in body["actions"]]
    assert urls == ["http://10.0.0.5:8000/approvals/abc123/approve",
                    "http://10.0.0.5:8000/approvals/abc123/deny"]  # fmt: skip
    assert all(a["method"] == "POST" for a in body["actions"])


def test_phone_falls_back_to_console(events, monkeypatch):
    monkeypatch.setenv("NTFY_TOPIC", "earshot-test")

    def down(request):
        raise httpx.ConnectError("offline")

    phone = PhoneChannel(events, client=httpx.Client(transport=httpx.MockTransport(down)))
    assert phone.notify("hello").extra["via"] == "console"
    assert events.log[-1].type == "phone" and events.log[-1].data["text"] == "hello"


def test_speaker_records_what_it_said(events):
    tts = NullTTS()
    SpeakerChannel(tts, events).deliver("Order placed.", volume=0.5)
    assert tts.spoken == ["Order placed."]
    assert any(e.type == "spoken" and e.data["volume"] == 0.5 for e in events.log)


# pipeline, text mode

DEMO = [
    "order my usual", "send fifty dollars to Jake", "send twenty dollars to Priya",
    "cancel my Netflix", "what's my verification code", "what's my bank balance",
    "when is my dermatologist appointment", "read my last email", "what's the weather",
    "set a reminder to call mom at six",
]  # fmt: skip


@pytest.mark.parametrize("text", DEMO)
def test_every_demo_phrase_runs_end_to_end(pipeline, text):
    result = pipeline.run_text(text)
    assert result.speak is not None and result.speak.spoken_text
    if result.action is not None:
        assert result.decision.outcome == "approve"
        assert result.action in pipeline.executor.executed


def test_order_usual_says_order_placed(pipeline, events):
    result = pipeline.run_text("order my usual")
    assert pipeline.tts.spoken == ["Order placed. Arrives at 7:40."]
    types = [e.type for e in events.log]
    assert types.index("agent") < types.index("executed") < types.index("route")
    assert [e.data["state"] for e in events.log if e.type == "led"] == ["working", "done"]
    assert result.action.type == "order_food"


def test_code_is_spoken_with_stub_hook(pipeline):
    pipeline.run_text("what's my verification code")
    assert pipeline.tts.spoken == ["Your Chase verification code is 482913."]


def test_rejected_action_does_not_run(pipeline, monkeypatch):
    monkeypatch.setattr(
        approvals_hook, "before_execute",
        lambda a, ctx: ApprovalDecision("reject", message="Okay, I won't."),
    )  # fmt: skip
    result = pipeline.run_text("order my usual")
    assert pipeline.executor.executed == [] and pipeline.tts.spoken == ["Okay, I won't."]
    assert result.reply.text == "Okay, I won't."


def test_crashing_approval_hook_fails_closed(pipeline, monkeypatch):
    monkeypatch.setattr(approvals_hook, "before_execute", lambda a, ctx: 1 / 0)
    pipeline.run_text("send fifty dollars to Jake")
    assert pipeline.executor.executed == []


def test_crashing_privacy_hook_fails_closed(pipeline, monkeypatch):
    monkeypatch.setattr(privacy_hook, "before_speak", lambda r, ctx: 1 / 0)
    pipeline.run_text("what's my verification code")
    assert "482913" not in " ".join(pipeline.tts.spoken)
    assert "482913" in pipeline.phone.sent[-1].text


def test_hooks_get_speak_listen_and_phone(pipeline, monkeypatch):
    seen = {}

    def hook(action, ctx):
        ctx.speak("DoorDash, forty-three twenty, to home. Say yes.")
        seen["heard"] = ctx.listen(8.0)
        seen["phone"] = ctx.phone
        return ApprovalDecision("step_up", message="Tap on your phone to approve.")

    monkeypatch.setattr(approvals_hook, "before_execute", hook)
    pipeline.run_text("order my usual")
    assert pipeline.tts.spoken[0].startswith("DoorDash") and seen["heard"] is None
    assert seen["phone"] is pipeline.phone


# pipeline, audio mode (fake STT and embedder)


class FakeSTT:
    def __init__(self, text):
        self.text = text

    def transcribe(self, audio):
        from earshot.audio.stt import Transcript

        return Transcript(self.text, [], 1.0, "fake")


def test_run_audio_keeps_command_embedding(pipeline, monkeypatch):
    vec = np.ones(192, dtype=np.float32) / np.sqrt(192)
    pipeline._stt = FakeSTT("order my usual")
    pipeline.embedder = lambda audio: Embedding(vec, 1.7, 3.0)
    seen = {}
    monkeypatch.setattr(
        approvals_hook, "before_execute",
        lambda a, ctx: seen.setdefault("ctx", ctx) and _approve_all(a, ctx),
    )  # fmt: skip
    pipeline.run_audio(np.zeros(16000, dtype=np.float32))
    assert seen["ctx"].command_embedding.vector is vec
    assert seen["ctx"].command_speech_seconds == 1.7


def test_run_audio_too_short_command(pipeline):
    pipeline._stt = FakeSTT("order my usual")

    def short(audio):
        raise TooShort(0.5, 0.8)

    pipeline.embedder = short
    result = pipeline.run_audio(np.zeros(8000, dtype=np.float32))
    assert result.action.type == "order_food"


def test_run_audio_ignores_empty_transcript(pipeline):
    pipeline._stt = FakeSTT("  ")
    assert pipeline.run_audio(np.zeros(8000, dtype=np.float32)) is None


def test_spoken_chatter_is_ignored_but_typed_gets_an_answer(pipeline, events):
    pipeline._stt = FakeSTT("did you watch the game last night")
    pipeline.embedder = lambda audio: Embedding(np.ones(192, np.float32), 1.5, 1.0)
    result = pipeline.run_audio(np.zeros(16000, dtype=np.float32))
    assert result.reply is None and pipeline.tts.spoken == []
    assert any(e.type == "ignored" for e in events.log)
    pipeline.run_text("did you watch the game last night")
    assert "can't help" in pipeline.tts.spoken[-1]


def test_live_mic_reports_every_segment_to_the_observer():
    import queue as q

    from earshot.agent.pipeline import LiveMic
    from earshot.audio.vad import FRAME, StreamingVAD

    class FakeStream:
        def __init__(self, frames):
            self.q = q.Queue()
            self.frames = frames

        def read(self, timeout=None):
            if self.frames:
                return self.frames.pop(0)
            raise q.Empty

    pattern = [1] * 20 + [0] * 30
    frames = [np.full(FRAME, p, np.float32) for p in pattern]
    seen = []
    mic = LiveMic(
        FakeStream(frames),
        vad_factory=lambda **kw: StreamingVAD(prob_fn=lambda f: float(f[0]), **kw),
        on_segment=seen.append,
    )
    mic.drain = lambda: None
    assert mic.listen(1.0) is not None and len(seen) == 1
