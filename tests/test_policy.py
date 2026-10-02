from decimal import Decimal

import pytest

from saysafe.approvals.actions import Action
from saysafe.approvals.policy import assess, bump


def A(type, amount=None, new=False, source="user_voice", **kw) -> Action:
    return Action(
        type=type, counterparty=kw.pop("counterparty", "X"),
        amount=None if amount is None else Decimal(str(amount)),
        is_new_counterparty=new, source=source, **kw,
    )  # fmt: skip


@pytest.mark.parametrize(
    "action, tier, rule",
    [
        # send money
        (A("send_money", 50, new=True), "phone_tap", "new_payee"),
        (A("send_money", 1, new=True), "phone_tap", "new_payee"),
        (A("send_money", 50), "voice_challenge", "send_money"),
        (A("send_money", 200), "voice_challenge", "send_money"),       # $200 is not over $200
        (A("send_money", "200.01"), "phone_tap", "over_200"),
        (A("send_money", 5000), "phone_tap", "over_200"),
        (A("send_money", None), "voice_challenge", "send_money"),
        # purchases at the $75 edge
        (A("order_food", "43.20"), "voice", "purchase_under_75"),
        (A("order_food", "74.99"), "voice", "purchase_under_75"),
        (A("order_food", 75), "voice_challenge", "purchase_75_plus"),
        (A("order_food", 200), "voice_challenge", "purchase_75_plus"),
        (A("order_food", 201), "phone_tap", "over_200"),
        (A("order_food", None), "voice", "purchase_under_75"),
        (A("book_appointment", None), "voice", "purchase_under_75"),
        (A("book_appointment", 150), "voice_challenge", "purchase_75_plus"),
        (A("cancel_subscription", "15.49"), "voice", "purchase_under_75"),
        (A("cancel_subscription", 99), "voice_challenge", "purchase_75_plus"),
        # email, reminders, default
        (A("send_email"), "voice", "email_known_contact"),
        (A("send_email", new=True), "phone_tap", "default"),
        (A("set_reminder"), "none", "low_risk"),
    ],
)  # fmt: skip
def test_tiers(action, tier, rule):
    r = assess(action)
    assert (r.tier, r.rule_id) == (tier, rule)
    assert not r.bumped and r.reasons


@pytest.mark.parametrize(
    "action, tier, base",
    [
        (A("order_food", "43.20", source="from_content"), "voice_challenge", "voice"),
        (A("send_money", 50, source="agent_initiated"), "phone_tap", "voice_challenge"),
        (A("set_reminder", source="from_content"), "voice", "none"),
        (A("send_money", 50, new=True, source="from_content"), "phone_tap", "phone_tap"),
    ],
)  # fmt: skip
def test_source_bump(action, tier, base):
    r = assess(action)
    assert r.tier == tier and r.base_tier == base and r.bumped
    assert any("not by you" in reason for reason in r.reasons)


def test_reasons_are_plain_english():
    r = assess(A("send_money", 20, new=True, counterparty="Priya"))
    assert r.reasons == ["New payee: you haven't paid Priya before"]
    r = assess(A("cancel_subscription", 15, source="from_content"))
    assert "Triggered by an email, not by you" in r.reasons


def test_bump_caps_at_phone_tap():
    assert [bump(t) for t in ("none", "voice", "voice_challenge", "phone_tap")] == [
        "voice", "voice_challenge", "phone_tap", "phone_tap"
    ]  # fmt: skip


def test_policy_without_default_fails_closed():
    r = assess(A("send_email", new=True), policy={"rules": [], "source_bump": {}})
    assert r.tier == "phone_tap"


def test_unknown_condition_is_an_error():
    with pytest.raises(ValueError, match="unknown policy condition"):
        assess(A("order_food", 5), policy={"rules": [{"id": "x", "when": {"amout": 5},
                                                      "tier": "none", "reason": ""}]})  # fmt: skip
