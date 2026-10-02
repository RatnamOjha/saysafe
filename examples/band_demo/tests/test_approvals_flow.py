import json
import threading
from decimal import Decimal

import numpy as np
import pytest
from band_demo import approvals_hook
from band_demo.agent.channels import PhoneChannel
from band_demo.agent.events import EventBus
from band_demo.agent.executor import Executor, Refused
from band_demo.agent.mock_agent import MockAgent
from band_demo.agent.pipeline import Pipeline
from band_demo.approvals_hook import Approver, fuse
from band_demo.audio.stt import Transcript
from band_demo.audio.tts import NullTTS
from band_demo.server.app import create_app
from fastapi.testclient import TestClient

from saysafe.approvals.audit import AuditLog
from saysafe.approvals.challenge import ChallengeIssuer
from saysafe.approvals.pending import PendingApprovals
from saysafe.approvals.tokens import TokenService
from saysafe.privacy.audience import FixedAudience
from saysafe.voice.embed import Embedding
from saysafe.voice.verify import Thresholds, VerifyResult, band

T = Thresholds(0.45, 0.25, "test")
WEIGHTS = {"reply_weight": 0.6, "command_weight": 0.4}
OWNER = np.ones(192, dtype=np.float32) / np.sqrt(192)


class Clock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now


class FakeSTT:
    def __init__(self):
        self.text = ""

    def transcribe(self, audio):
        return Transcript(self.text, [], 5.0, "fake")


class FakeScorer:
    """Reply score comes from the test; command score from the embedding's first value."""

    def __init__(self):
        self.reply: float | None = 0.8
        self.speech = 1.2

    def score_audio(self, audio, min_speech_s=None):
        s = self.reply
        return VerifyResult(s, "uncertain" if s is None else band(s, T), self.speech, 3.0)

    def score_embedding(self, embedding):
        return float(embedding.vector[0])


@pytest.fixture
def world(tmp_path):
    """A pipeline wired to a real Approver with fake voice, STT and clock."""
    clock = Clock()
    events = EventBus()
    events.log = []
    events.subscribe(events.log.append)
    tokens = TokenService(b"k" * 32, tmp_path / "n.sqlite", clock=clock, ttl_s=60)
    pending = PendingApprovals(tokens, clock=clock, ttl_s=120)
    stt, scorer = FakeSTT(), FakeScorer()
    approver = Approver(
        stt=stt, scorer=scorer, tokens=tokens, pending=pending,
        issuer=ChallengeIssuer(clock=clock), audit=AuditLog(tmp_path / "audit.jsonl"),
        clock=clock, t=T,
    )  # fmt: skip
    replies: list = []
    pipeline = Pipeline(
        tts=NullTTS(), agent=MockAgent(fixed_code="482913"), events=events,
        phone=PhoneChannel(events, console_only=True),
        executor=Executor(verify_token=tokens.verify),
        listen=lambda timeout: replies.pop(0) if replies else None,
        audience=FixedAudience("alone_likely"),  # these tests are about approvals, not privacy
    )  # fmt: skip
    pipeline.approver = approver
    pipeline.set_command_score = lambda s: setattr(pipeline, "_cmd", s)

    class W:
        pass

    w = W()
    w.__dict__.update(
        clock=clock, events=events, tokens=tokens, pending=pending, stt=stt, scorer=scorer,
        approver=approver, pipeline=pipeline, replies=replies, audit=tmp_path / "audit.jsonl",
    )  # fmt: skip
    return w


@pytest.fixture(autouse=True)
def use_world_approver(monkeypatch, request):
    if "world" in request.fixturenames:
        w = request.getfixturevalue("world")
        monkeypatch.setattr(approvals_hook, "_approver", w.approver)


def say(w, text, reply_voice=0.8, command_voice=0.8, reply_audio=True):
    """Queue a spoken reply and run a command as if spoken (command embedding set)."""
    w.stt.text = text
    w.scorer.reply = reply_voice
    w.replies.append(np.zeros(16000, dtype=np.float32) if reply_audio else text)
    return w


def run(w, command, command_voice=0.8):
    ctx_emb = None if command_voice is None else Embedding(
        np.full(192, command_voice, dtype=np.float32), 1.5, 3.0
    )  # fmt: skip
    orig = w.pipeline._context

    def ctx_with_embedding(text):
        ctx = orig(text)
        ctx.command_embedding = ctx_emb
        return ctx

    w.pipeline._context = ctx_with_embedding
    try:
        return w.pipeline.run_text(command)
    finally:
        w.pipeline._context = orig


def challenge_word(w):
    return next(e.data["challenge_word"] for e in w.events.log if e.type == "readback_spoken")


def audit_lines(w):
    return [json.loads(line) for line in w.audit.read_text().splitlines()]


# fusion ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "reply, command, fused, b",
    [
        (0.8, 0.8, 0.8, "accept"),
        (0.5, None, 0.5, "accept"),          # command too short -> reply alone
        (None, 0.9, None, "uncertain"),      # no reply voice -> never accept
        (0.9, 0.2, 0.62, "uncertain"),       # command under t_reject blocks accept
        (0.2, 0.95, 0.5, "uncertain"),       # reply under t_reject blocks accept
        (0.1, 0.1, 0.1, "reject"),
        (0.4, 0.55, 0.46, "accept"),
    ],
)  # fmt: skip
def test_fuse(reply, command, fused, b):
    f, got = fuse(reply, command, T, WEIGHTS)
    assert got == b and (f is None if fused is None else f == pytest.approx(fused, abs=1e-6))


# the flow ---------------------------------------------------------------------


def test_owner_yes_approves_and_runs(world):
    r = run(say(world, "yeah, do it"), "order my usual")
    assert r.decision.outcome == "approve"
    assert world.pipeline.tts.spoken == [
        "DoorDash, forty-three twenty, to home. Say yes.", "Order placed. Arrives at 7:40.",
    ]  # fmt: skip
    assert r.action in world.pipeline.executor.executed


def test_friend_yes_steps_up_then_phone_approve_completes(world):
    r = run(say(world, "yes", reply_voice=0.12), "order my usual", command_voice=0.1)
    assert r.decision.outcome == "step_up"
    assert world.pipeline.executor.executed == []
    assert "Tap on your phone" in world.pipeline.tts.spoken[-1]
    assert world.pipeline.phone.sent[-1].extra["action_id"] == r.action.id
    assert [e.data["state"] for e in world.events.log if e.type == "led"][-1] == "amber"

    assert world.pending.approve(r.action.id).changed
    assert r.action in world.pipeline.executor.executed
    assert world.pipeline.tts.spoken[-1] == "Order placed. Arrives at 7:40."
    again = world.pending.approve(r.action.id)  # already resolved, no rerun
    assert again.status == "approved" and not again.changed
    assert world.pipeline.executor.executed.count(r.action) == 1


def test_phone_deny_stops(world):
    r = run(say(world, "yes", reply_voice=0.1), "order my usual", command_voice=0.1)
    assert world.pending.deny(r.action.id).changed
    assert world.pending.approve(r.action.id).status == "denied"
    assert world.pipeline.executor.executed == []
    assert world.pipeline.tts.spoken[-1] == "Okay, I won't."


def test_phone_request_expires(world):
    r = run(say(world, "yes", reply_voice=0.1), "order my usual", command_voice=0.1)
    world.clock.now += 121
    assert world.pending.approve(r.action.id).status == "expired"
    assert world.pipeline.executor.executed == []


def test_negation_rejects(world):
    r = run(say(world, "yes, no wait"), "order my usual")
    assert r.decision.outcome == "reject" and world.pipeline.executor.executed == []
    assert world.pipeline.tts.spoken[-1] == "Okay, I won't."


def test_timeout_rejects(world):
    r = run(world, "order my usual")  # nothing queued -> listen returns None
    assert r.decision.outcome == "reject"


def test_low_score_steps_up(world):
    r = run(say(world, "yes", reply_voice=0.35), "order my usual", command_voice=0.35)
    assert r.decision.outcome == "step_up"
    assert "didn't match" in r.decision.reasons[0]


def test_too_short_reply_steps_up(world):
    r = run(say(world, "yes", reply_voice=None), "order my usual")
    assert r.decision.outcome == "step_up" and "Too little speech" in r.decision.reasons[0]


def test_typed_reply_never_approves(world):
    r = run(say(world, "yes", reply_audio=False), "order my usual")
    assert r.decision.outcome == "step_up"


def test_challenge_word_approves_money(world):
    world.stt.text = ""
    world.scorer.reply = 0.8
    # the word is only known after the read-back, so answer from inside listen
    world.pipeline.listen = lambda timeout: (
        setattr(world.stt, "text", challenge_word(world)) or np.zeros(16000, np.float32)
    )  # fmt: skip
    r = run(world, "send fifty dollars to Jake")
    assert r.decision.outcome == "approve"
    assert world.pipeline.tts.spoken[-1] == "Sent fifty dollars to Jake."


def test_replayed_yes_cannot_pass_a_challenge(world):
    r = run(say(world, "yes"), "send fifty dollars to Jake")
    assert r.decision.outcome == "step_up" and world.pipeline.executor.executed == []


def test_old_challenge_word_steps_up(world):
    run(say(world, "nope"), "send fifty dollars to Jake")
    old = challenge_word(world)
    world.events.log.clear()
    r = run(say(world, old), "send fifty dollars to Jake")  # replay the previous word
    assert r.decision.outcome == "step_up"
    assert "Wrong challenge word" in r.decision.reasons[0]


def test_expired_challenge_word_steps_up(world):
    def late_reply(timeout):
        world.stt.text = challenge_word(world)
        world.clock.now += 31
        return np.zeros(16000, np.float32)

    world.pipeline.listen = late_reply
    r = run(world, "send fifty dollars to Jake")
    assert r.decision.outcome == "step_up"


def test_new_payee_goes_straight_to_phone(world):
    r = run(world, "send twenty dollars to Priya")
    assert r.decision.outcome == "step_up"
    assert world.pipeline.tts.spoken == [
        "New payee. Priya, twenty dollars. Check your phone to approve."
    ]
    world.pending.approve(r.action.id)
    assert world.pipeline.tts.spoken[-1] == "Sent twenty dollars to Priya."


def test_from_content_bumps_a_tier(world):
    action = (
        MockAgent().cancel_subscription("Netflix").model_copy(update={"source": "from_content"})
    )
    world.pipeline.agent.handle = lambda text: action
    r = run(say(world, "yes"), "cancel my Netflix")
    assert r.decision.outcome == "step_up"  # a yes isn't enough once bumped to a challenge
    assert "email" in world.pipeline.tts.spoken[0]


def test_reminder_needs_no_approval(world):
    r = run(world, "set a reminder to call mom at six")
    assert r.decision.outcome == "approve" and world.pipeline.tts.spoken == [
        "Reminder set: call mom at six."
    ]


def test_missing_profile_steps_up(world):
    world.approver._scorer = None
    r = run(say(world, "yes"), "order my usual")
    assert r.decision.outcome == "step_up"
    assert r.decision.reasons == ["No owner voice enrolled"]


def test_audit_has_no_audio_embeddings_or_transcripts(world):
    run(say(world, "yeah, do it, my code is 482913"), "order my usual")
    lines = audit_lines(world)
    assert lines and lines[-1]["outcome"] == "approve"
    assert lines[-1]["scores"]["fused"] == pytest.approx(0.8, abs=1e-3)
    raw = world.audit.read_text()
    assert "482913" not in raw and "yeah" not in raw and "embedding" not in raw


def test_audit_rejects_unknown_fields(tmp_path):
    with pytest.raises(ValueError):
        AuditLog(tmp_path / "a.jsonl").record(transcript="yes")


def test_trace_events_in_order(world):
    run(say(world, "yes"), "order my usual")
    types = [e.type for e in world.events.log]
    order = ["risk_assessed", "readback_spoken", "reply_captured", "speaker_scored",
             "reply_matched", "decision", "executed"]  # fmt: skip
    assert [t for t in types if t in order] == order


# tokens -------------------------------------------------------------------------


@pytest.fixture
def tokens(tmp_path):
    return TokenService(b"s" * 32, tmp_path / "n.sqlite", clock=Clock(), ttl_s=60)


def test_token_approved_action_runs(tokens):
    a = MockAgent().order_usual()
    assert tokens.verify(tokens.issue(a, "voice", "voice", 0.8), a)["method"] == "voice"


def test_changed_amount_after_approval_is_refused(tokens):
    a = MockAgent().order_usual()
    token = tokens.issue(a, "voice", "voice", 0.8)
    tampered = a.model_copy(update={"amount": Decimal("430.20")})
    with pytest.raises(Refused, match="action_changed"):
        Executor(verify_token=tokens.verify).run(tampered, token)


@pytest.mark.parametrize("field, value", [("counterparty", "Evil"), ("destination", "work"),
                                          ("id", "other")])  # fmt: skip
def test_any_changed_field_is_refused(tokens, field, value):
    a = MockAgent().order_usual()
    token = tokens.issue(a, "voice", "voice", 0.8)
    with pytest.raises(Refused, match="action_changed"):
        tokens.verify(token, a.model_copy(update={field: value}))


def test_replayed_token_is_refused(tokens):
    a = MockAgent().order_usual()
    token = tokens.issue(a, "voice", "voice", 0.8)
    tokens.verify(token, a)
    with pytest.raises(Refused, match="replayed"):
        tokens.verify(token, a)


def test_expired_token_is_refused(tokens):
    a = MockAgent().order_usual()
    token = tokens.issue(a, "voice", "voice", 0.8)
    tokens.clock.now += 60
    with pytest.raises(Refused, match="expired"):
        tokens.verify(token, a)


@pytest.mark.parametrize("mangle", [
    lambda t: t[:-2] + ("AA" if not t.endswith("AA") else "BB"),  # bad signature
    lambda t: "garbage",
    lambda t: t.split(".")[0],                                    # signature stripped
])  # fmt: skip
def test_forged_token_is_refused(tokens, mangle):
    a = MockAgent().order_usual()
    with pytest.raises(Refused, match="bad_signature"):
        tokens.verify(mangle(tokens.issue(a, "voice", "voice", 0.8)), a)


def test_token_from_another_secret_is_refused(tokens, tmp_path):
    a = MockAgent().order_usual()
    other = TokenService(b"x" * 32, tmp_path / "o.sqlite", clock=Clock())
    with pytest.raises(Refused, match="bad_signature"):
        tokens.verify(other.issue(a, "voice", "voice", 0.8), a)


def test_concurrent_reuse_only_one_wins(tokens):
    a = MockAgent().order_usual()
    token = tokens.issue(a, "voice", "voice", 0.8)
    results = []

    def use():
        try:
            tokens.verify(token, a)
            results.append("ok")
        except Refused as e:
            results.append(e.reason)

    threads = [threading.Thread(target=use) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results.count("ok") == 1 and results.count("replayed") == 15


# routes --------------------------------------------------------------------------


def test_routes_approve_deny_404_409(world):
    client = TestClient(create_app(world.pending))
    r = run(say(world, "yes", reply_voice=0.1), "order my usual", command_voice=0.1)
    assert client.post("/approvals/nope/approve").status_code == 404
    assert client.post(f"/approvals/{r.action.id}/approve").json()["status"] == "approved"
    assert client.post(f"/approvals/{r.action.id}/approve").status_code == 409
    assert r.action in world.pipeline.executor.executed

    r2 = run(say(world, "yes", reply_voice=0.1), "order my usual", command_voice=0.1)
    assert client.post(f"/approvals/{r2.action.id}/deny").json()["status"] == "denied"
    assert r2.action not in world.pipeline.executor.executed


@pytest.mark.parametrize("room", ["others_present", "unknown"])
def test_money_readback_stays_private_when_others_may_hear(world, room):
    world.pipeline.audience = FixedAudience(room)
    r = run(say(world, "yes"), "send fifty dollars to Jake")
    assert r.decision.outcome == "step_up"
    spoken = " ".join(world.pipeline.tts.spoken)
    assert "fifty" not in spoken and "confirm" not in spoken  # no amount, no challenge word
    assert world.pipeline.tts.spoken == [
        "Someone else might be listening. Check your phone to approve."
    ]
    assert "Jake, fifty dollars." in world.pipeline.phone.sent[-1].text
    assert world.pending.approve(r.action.id).changed
    assert r.action in world.pipeline.executor.executed


def test_money_readback_spoken_with_headphones(world):
    world.pipeline.audience = FixedAudience("others_present")
    world.pipeline.flags["headphones"] = True
    run(say(world, "yes"), "send fifty dollars to Jake")
    assert world.pipeline.tts.spoken[0].startswith("Jake, fifty dollars.")


def test_food_readback_still_spoken_with_others(world):
    world.pipeline.audience = FixedAudience("others_present")
    run(say(world, "yes"), "order my usual")
    assert world.pipeline.tts.spoken[0] == "DoorDash, forty-three twenty, to home. Say yes."


def test_new_payee_readback_is_private_with_others(world):
    world.pipeline.audience = FixedAudience("others_present")
    run(world, "send twenty dollars to Priya")
    assert world.pipeline.tts.spoken == [
        "Someone else might be listening. Check your phone to approve."
    ]
    assert "Priya, twenty dollars." in world.pipeline.phone.sent[-1].text
