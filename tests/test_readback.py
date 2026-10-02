from decimal import Decimal

import pytest

from saysafe.approvals.actions import Action
from saysafe.approvals.policy import assess
from saysafe.approvals.readback import MAX_BODY_WORDS, money_words, readback


def demo_actions():
    """The demo's actions, built directly so the library tests don't need the demo."""
    order = Action(type="order_food", counterparty="DoorDash", amount=Decimal("43.20"),
                   destination="home", params={"eta": "7:40"})  # fmt: skip
    jake = Action(type="send_money", counterparty="Jake", amount=Decimal("50"),
                  destination="@jake-morales")  # fmt: skip
    netflix = Action(type="cancel_subscription", counterparty="Netflix",
                     amount=Decimal("15.49"), params={"renews": "October 3"})  # fmt: skip
    return [
        ("order my usual", order),
        ("send fifty dollars to Jake", jake),
        ("send twenty dollars to Priya", Action(type="send_money", counterparty="Priya",
                                                amount=Decimal("20"), is_new_counterparty=True)),
        ("cancel my Netflix", netflix),
        ("set a reminder to call mom at six",
         Action(type="set_reminder", params={"what": "call mom", "when": "six"})),
        ("big order ($80)", order.model_copy(update={"amount": Decimal("80")})),
        ("send $250 to Jake", jake.model_copy(update={"amount": Decimal("250")})),
        ("email says: cancel Netflix", netflix.model_copy(update={"source": "from_content"})),
    ]  # fmt: skip


@pytest.mark.parametrize(
    "amount, words",
    [("43.20", "forty-three twenty"), ("50", "fifty dollars"), ("1", "one dollar"),
     ("0.75", "seventy-five cents"), ("43.05", "forty-three oh five"),
     ("15.49", "fifteen forty-nine"), ("250", "two hundred fifty dollars"),
     ("1200", "one thousand two hundred dollars")],
)  # fmt: skip
def test_money_words(amount, words):
    assert money_words(Decimal(amount)) == words


@pytest.mark.parametrize("said, action", demo_actions(), ids=[s for s, _ in demo_actions()])
def test_every_demo_readback_is_short(said, action):
    risk = assess(action)
    rb = readback(action, risk, "maple" if risk.tier == "voice_challenge" else None)
    assert rb.body_words <= MAX_BODY_WORDS
    assert rb.body_words / 2.6 < 4.0, rb.body  # the body alone is under 4 s
    assert rb.est_seconds < 4.5, rb.text  # with the instruction, about 4 s at most


def test_target_styles():
    by_said = dict(demo_actions())
    rb = lambda said, word=None: readback(by_said[said], assess(by_said[said]), word).text  # noqa: E731
    assert rb("order my usual") == "DoorDash, forty-three twenty, to home. Say yes."
    assert (
        rb("send fifty dollars to Jake", "copper")
        == "Jake, fifty dollars. Say 'copper' to confirm."
    )
    assert (
        rb("email says: cancel Netflix", "maple")
        == "This came from an email. Cancel Netflix? Say 'maple' to confirm."
    )
    assert rb("send twenty dollars to Priya").startswith("New payee. Priya, twenty dollars.")


def test_challenge_tier_needs_a_word():
    action = dict(demo_actions())["send fifty dollars to Jake"]
    with pytest.raises(ValueError):
        readback(action, assess(action))


def test_long_body_is_capped():
    action = dict(demo_actions())["set a reminder to call mom at six"]
    long = action.model_copy(update={"params": {"what": " ".join(["word"] * 30), "when": "six"}})
    assert readback(long, assess(long)).body_words <= MAX_BODY_WORDS
