import json
import threading
from decimal import Decimal

import pytest

from saysafe import (
    Action,
    ApprovalGuard,
    AuditLog,
    Refused,
    Room,
    SQLiteNonceStore,
    Step,
    StepClosed,
    Thresholds,
)
from saysafe.approvals.fusion import fuse
from saysafe.approvals.guard import MemoryStepStore

SECRET = b"k" * 32
T = Thresholds(0.45, 0.25, "test")


class Clock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def events():
    return []


@pytest.fixture
def guard(clock, events, tmp_path):
    return ApprovalGuard(
        SECRET, thresholds=T, clock=clock, audit=AuditLog(tmp_path / "audit.jsonl"),
        on_event=lambda name, data: events.append((name, data)),
    )  # fmt: skip


def food():
    return Action(type="order_food", counterparty="DoorDash", amount=Decimal("43.20"),
                  destination="home")  # fmt: skip


def money(to="Jake", amount=50, new=False):
    return Action(type="send_money", counterparty=to, amount=amount, is_new_counterparty=new)


# fusion -------------------------------------------------------------------------


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
    f, got = fuse(reply, command, T)
    assert got == b and (f is None if fused is None else f == pytest.approx(fused, abs=1e-6))


# voice tier ----------------------------------------------------------------------


def test_owner_yes_approves_and_the_token_runs_once(guard):
    a = food()
    step = guard.start(a, room=Room.ALONE)
    assert isinstance(step, Step) and step.status == "awaiting_reply"
    assert step.say == "DoorDash, forty-three twenty, to home. Say yes."
    step = guard.reply(step, "yeah, do it", voice_score=0.8, command_score=0.8)
    assert (step.status, step.method, step.say) == ("approved", "voice", "")
    assert guard.verify(step.token, a)["method"] == "voice"
    with pytest.raises(Refused, match="replayed"):
        guard.verify(step.token, a)


def test_someone_else_saying_yes_steps_up_to_the_phone(guard):
    step = guard.start(food())
    step = guard.reply(step, "yes", voice_score=0.12, command_score=0.1)
    assert step.status == "awaiting_phone" and step.token is None
    assert step.say == "I couldn't confirm it's you. Check your phone to approve."
    assert step.phone_request == "DoorDash, forty-three twenty, to home."
    assert "didn't match" in step.reasons[0]


def test_middling_voice_steps_up(guard):
    step = guard.reply(guard.start(food()), "yes", voice_score=0.35, command_score=0.35)
    assert step.status == "awaiting_phone"


def test_no_voice_score_steps_up_unless_the_host_vouches(guard, clock):
    step = guard.reply(guard.start(food()), "yes")
    assert step.status == "awaiting_phone" and "no voice score" in step.reasons[0]
    trusting = ApprovalGuard(SECRET, clock=clock, require_voice_score=False)
    assert trusting.reply(trusting.start(food()), "yes").status == "approved"
    # a score the host does pass still counts
    assert trusting.reply(trusting.start(food()), "yes", voice_score=0.1).status != "approved"


def test_negation_wins(guard):
    step = guard.reply(guard.start(food()), "yes, no wait", voice_score=0.9)
    assert (step.status, step.say) == ("rejected", "Okay, I won't.")


def test_no_reply_rejects(guard):
    step = guard.reply(guard.start(food()), None)
    assert step.status == "rejected" and step.reasons == ("No reply",)


def test_unclear_reply_steps_up(guard):
    step = guard.reply(guard.start(food()), "what's the weather", voice_score=0.9)
    assert step.status == "awaiting_phone"
    assert step.say == "I didn't catch that. Check your phone to approve."


def test_late_yes_steps_up(guard, clock):
    step = guard.start(food())
    clock.now += 31
    step = guard.reply(step, "yes", voice_score=0.9)
    assert step.status == "awaiting_phone" and "expired" in step.reasons[0]


# challenge tier ----------------------------------------------------------------------


def test_challenge_word_approves_money(guard):
    a = money()
    step = guard.start(a, room=Room.ALONE)
    assert step.tier == "voice_challenge"
    assert step.say == f"Jake, fifty dollars. Say '{step.challenge_word}' to confirm."
    step = guard.reply(step, step.challenge_word, voice_score=0.8)
    assert step.status == "approved"
    guard.verify(step.token, a)


def test_a_replayed_yes_cannot_pass_a_challenge(guard):
    step = guard.reply(guard.start(money(), room=Room.ALONE), "yes", voice_score=0.9)
    assert step.status == "awaiting_phone"


def test_an_old_challenge_word_steps_up(guard):
    first = guard.start(money(), room=Room.ALONE)
    guard.reply(first, "nope", voice_score=0.9)
    step = guard.reply(guard.start(money(), room=Room.ALONE), first.challenge_word,
                       voice_score=0.9)  # fmt: skip
    assert step.status == "awaiting_phone"
    assert step.reasons[0].startswith("Wrong challenge word")
    assert step.say == "That's not the word I asked for. Check your phone to approve."


def test_an_expired_challenge_word_steps_up(guard, clock):
    step = guard.start(money(), room=Room.ALONE)
    clock.now += 31
    assert guard.reply(step, step.challenge_word, voice_score=0.9).status == "awaiting_phone"


def test_one_reply_per_step(guard):
    step = guard.start(money(), room=Room.ALONE)
    guard.reply(step, "banana", voice_score=0.9)
    with pytest.raises(StepClosed):
        guard.reply(step, step.challenge_word, voice_score=0.9)


# phone tier ------------------------------------------------------------------------


def test_new_payee_goes_straight_to_the_phone(guard):
    a = money("Priya", 20, new=True)
    step = guard.start(a, room=Room.ALONE)
    assert step.status == "awaiting_phone"
    assert step.say == "New payee. Priya, twenty dollars. Check your phone to approve."
    step = guard.phone_approve(step.id)
    assert step.status == "approved" and step.method == "phone_tap"
    assert guard.verify(step.token, a)["method"] == "phone_tap"


def test_phone_approve_works_once(guard):
    step = guard.reply(guard.start(food()), "yes", voice_score=0.1)
    guard.phone_approve(step)
    with pytest.raises(StepClosed) as e:
        guard.phone_approve(step)
    assert e.value.status == "approved"


def test_phone_deny(guard):
    step = guard.reply(guard.start(food()), "yes", voice_score=0.1)
    step = guard.phone_deny(step.id)
    assert (step.status, step.say) == ("rejected", "Okay, I won't.")
    with pytest.raises(StepClosed):
        guard.phone_approve(step.id)


def test_phone_request_expires(guard, clock):
    step = guard.start(money("Priya", 20, new=True))
    clock.now += 121
    assert guard.get(step.id).status == "expired"
    with pytest.raises(StepClosed) as e:
        guard.phone_approve(step.id)
    assert e.value.status == "expired"


def test_unknown_step(guard):
    with pytest.raises(StepClosed) as e:
        guard.phone_approve("nope")
    assert e.value.status is None


def test_concurrent_taps_approve_once(guard):
    step = guard.start(money("Priya", 20, new=True))
    results = []

    def tap():
        try:
            results.append(guard.phone_approve(step.id).status)
        except StepClosed:
            results.append("closed")

    threads = [threading.Thread(target=tap) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results.count("approved") == 1 and results.count("closed") == 15


def test_cancel(guard):
    step = guard.cancel(guard.start(food()))
    assert step.status == "rejected"
    with pytest.raises(StepClosed):
        guard.reply(step, "yes", voice_score=0.9)


# policy -------------------------------------------------------------------------------


def test_low_risk_is_approved_at_once(guard):
    a = Action(type="set_reminder", params={"what": "call mom"})
    step = guard.start(a)
    assert (step.status, step.method, step.say) == ("approved", "none", "")
    guard.verify(step.token, a)


def test_content_triggered_actions_need_one_more_tier(guard):
    a = Action(type="cancel_subscription", counterparty="Netflix", source="from_content")
    step = guard.start(a)
    assert step.tier == "voice_challenge" and "email" in step.say
    assert guard.reply(step, "yes", voice_score=0.9).status == "awaiting_phone"


def test_unknown_action_types_need_a_phone_tap(guard):
    step = guard.start(Action(type="unlock_door", destination="front door"))
    assert step.tier == "phone_tap"
    assert step.say == "Unlock door: front door. Check your phone to approve."


def test_custom_policy(clock, tmp_path):
    policy = {"rules": [
        {"id": "door", "when": {"type": "unlock_door"}, "tier": "voice_challenge",
         "reason": "Opens the house"},
        {"id": "default", "when": {}, "tier": "phone_tap", "reason": "Everything else"},
    ]}  # fmt: skip
    guard = ApprovalGuard(SECRET, policy=policy, clock=clock)
    step = guard.start(Action(type="unlock_door"))
    assert (step.tier, step.reasons) == ("voice_challenge", ("Opens the house",))
    assert guard.policy["tokens"]["ttl_s"] == 60  # sections left out keep their defaults

    path = tmp_path / "policy.yaml"
    path.write_text(json.dumps(policy))  # JSON is YAML
    assert ApprovalGuard(SECRET, policy=path).assess(Action(type="unlock_door")).rule_id == "door"


@pytest.mark.parametrize("rule", [
    {"id": "x", "when": {}, "tier": "maybe"},
    {"id": "x", "when": {"colour": "red"}, "tier": "voice"},
])  # fmt: skip
def test_bad_policy_fails_loudly(rule):
    with pytest.raises(ValueError):
        ApprovalGuard(SECRET, policy={"rules": [rule]})


def test_short_secret_is_refused():
    with pytest.raises(ValueError):
        ApprovalGuard(b"short")


# who's listening meets who's speaking --------------------------------------------------


@pytest.mark.parametrize("room", [Room.OTHERS_PRESENT, Room.UNKNOWN])
def test_money_readback_stays_private_when_others_may_hear(guard, room):
    step = guard.start(money(), room=room)
    assert step.status == "awaiting_phone"
    assert step.say == "Someone else might be listening. Check your phone to approve."
    assert "fifty" not in step.say
    assert step.phone_request == "Jake, fifty dollars."


def test_money_readback_spoken_in_headphones(guard):
    step = guard.start(money(), room=Room.OTHERS_PRESENT, headphones=True)
    assert step.say.startswith("Jake, fifty dollars.")


def test_food_readback_still_spoken_with_others(guard):
    step = guard.start(food(), room=Room.OTHERS_PRESENT)
    assert step.say == "DoorDash, forty-three twenty, to home. Say yes."


def test_screen_wording(clock):
    guard = ApprovalGuard(SECRET, clock=clock, screen="watch")
    assert guard.start(money(), room=Room.OTHERS_PRESENT).say.endswith(
        "Check your watch to approve."
    )
    step = guard.reply(guard.start(food()), "yes", voice_score=0.1)
    assert step.say.endswith("Check your watch to approve.")


# tokens ----------------------------------------------------------------------------------


def test_changing_the_action_after_approval_voids_it(guard):
    a = food()
    step = guard.reply(guard.start(a), "yes", voice_score=0.8)
    with pytest.raises(Refused, match="action_changed"):
        guard.verify(step.token, a.model_copy(update={"amount": Decimal("430.20")}))


def test_replays_stay_blocked_across_restarts(tmp_path):
    a = Action(type="set_reminder")
    first = ApprovalGuard(SECRET, nonces=tmp_path / "nonces.sqlite")
    token = first.start(a).token
    first.verify(token, a)
    restarted = ApprovalGuard(SECRET, nonces=SQLiteNonceStore(tmp_path / "nonces.sqlite"))
    with pytest.raises(Refused, match="replayed"):
        restarted.verify(token, a)


def test_tokens_expire(guard, clock):
    a = Action(type="set_reminder")
    token = guard.start(a).token
    clock.now += 60
    with pytest.raises(Refused, match="expired"):
        guard.verify(token, a)


# events, audit, store ------------------------------------------------------------------------


def test_events_in_order(guard, events):
    guard.reply(guard.start(food()), "yes", voice_score=0.8)
    assert [n for n, _ in events] == [
        "risk_assessed", "readback", "voice_scored", "reply_matched", "decision",
    ]  # fmt: skip
    assert events[-1][1]["outcome"] == "approve"


def test_audit_has_no_transcripts(guard, tmp_path):
    guard.reply(guard.start(food()), "yeah do it, my code is 482913", voice_score=0.8)
    raw = (tmp_path / "audit.jsonl").read_text()
    line = json.loads(raw.splitlines()[-1])
    assert line["outcome"] == "approve" and line["scores"]["fused"] == pytest.approx(0.8)
    assert "482913" not in raw and "yeah" not in raw


def test_memory_store_forgets_the_oldest():
    store = MemoryStepStore(max_steps=2)
    guard = ApprovalGuard(SECRET, steps=store)
    steps = [guard.start(Action(type="unlock_door")) for _ in range(3)]
    assert guard.get(steps[0].id) is None and guard.get(steps[2].id) is not None
