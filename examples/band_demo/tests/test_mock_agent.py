from decimal import Decimal

import pytest
from band_demo.agent.mock_agent import MockAgent, Reply, parse_amount, words_to_number

from saysafe.approvals.actions import Action


@pytest.fixture
def agent(monkeypatch):
    monkeypatch.delenv("EARSHOT_AGENT", raising=False)
    return MockAgent(clock=lambda: 1_000_000.0)


# Each demo phrase, typed and as Whisper actually writes it.
@pytest.mark.parametrize("text", ["order my usual", "Order my usual."])
def test_order_usual(agent, text):
    a = agent.handle(text)
    assert isinstance(a, Action) and a.type == "order_food"
    assert a.counterparty == "DoorDash" and a.amount == Decimal("43.20")
    assert a.destination == "home" and a.source == "user_voice"


@pytest.mark.parametrize(
    "text", ["send fifty dollars to Jake", "Send $50 to Jake.", "send 50 dollars to jake"]
)
def test_send_jake(agent, text):
    a = agent.handle(text)
    assert a.type == "send_money" and a.counterparty == "Jake"
    assert a.amount == Decimal("50") and a.is_new_counterparty is False


@pytest.mark.parametrize("text", ["send twenty dollars to Priya", "Send $20 to Priya."])
def test_send_priya_is_new(agent, text):
    a = agent.handle(text)
    assert a.counterparty == "Priya" and a.amount == Decimal("20")
    assert a.is_new_counterparty is True


def test_send_unknown_person_is_new(agent):
    assert agent.handle("send $5 to Marcus").is_new_counterparty is True


def test_send_without_amount_asks(agent):
    r = agent.handle("send money to Jake")
    assert isinstance(r, Reply) and "How much" in r.text


@pytest.mark.parametrize("text", ["cancel my Netflix", "Cancel my Netflix."])
def test_cancel_netflix(agent, text):
    a = agent.handle(text)
    assert a.type == "cancel_subscription" and a.counterparty == "Netflix"
    assert a.amount == Decimal("15.49")


@pytest.mark.parametrize(
    "text, tags, needle",
    [
        ("what's my verification code", {"otp"}, "verification code is"),
        ("What's my verification code?", {"otp"}, "verification code is"),
        ("what's my bank balance", {"bank"}, "$2,847.16"),
        ("What's my balance?", {"bank"}, "$2,847.16"),
        ("when is my dermatologist appointment", {"health", "calendar"}, "Dr. Mehta"),
        ("read my last email", {"email"}, "read this code out loud"),
        ("what's the weather", {"public"}, "sunny"),
    ],
)
def test_replies(agent, text, tags, needle):
    r = agent.handle(text)
    assert isinstance(r, Reply) and r.source_tags == tags and needle in r.text


@pytest.mark.parametrize(
    "text", ["set a reminder to call mom at six", "Set a reminder to call mom at 6."]
)
def test_reminder(agent, text):
    a = agent.handle(text)
    assert a.type == "set_reminder" and a.params["what"] == "call mom"
    assert a.params["when"] in ("six", "6")


def test_unknown(agent):
    assert "can't help" in agent.handle("sing me a song").text


def test_code_rotates_every_60s():
    t = {"now": 600.0}
    agent = MockAgent(clock=lambda: t["now"])
    first = agent.current_code()
    t["now"] = 659.0
    assert agent.current_code() == first
    t["now"] = 660.0
    assert agent.current_code() != first
    assert len(first) == 6 and first.isdigit()
    assert agent.current_code() in agent.handle("what's my verification code").text


def test_fixed_code_for_demo():
    assert MockAgent(fixed_code="482913").current_code() == "482913"


def test_email_embeds_current_code(agent):
    assert agent.current_code() in agent.handle("read my last email").text


@pytest.mark.parametrize(
    "words, n",
    [("fifty", 50), ("twenty five", 25), ("a hundred", 100), ("two hundred and five", 205),
     ("one thousand two hundred", 1200), ("banana", None)],
)  # fmt: skip
def test_words_to_number(words, n):
    assert words_to_number(words.split()) == n


@pytest.mark.parametrize(
    "text, amount",
    [("$50", "50"), ("$50.00", "50.00"), ("$1,200", "1200"), ("20 bucks", "20"),
     ("send a hundred dollars", "100"), ("twenty-five dollars", "25"), ("no money", None)],
)  # fmt: skip
def test_parse_amount(text, amount):
    assert parse_amount(text) == (Decimal(amount) if amount else None)


@pytest.mark.parametrize(
    "answer, amount",
    [("$50.", "50"), ("fifty", "50"), ("50 dollars", "50"), ("5 million.", "5000000"),
     ("five million", "5000000")],
)  # fmt: skip
def test_follow_up_amount(agent, answer, amount):
    assert "How much" in agent.handle("send money to Drake").text
    a = agent.handle(answer)
    assert a.type == "send_money" and a.counterparty == "Drake" and a.amount == Decimal(amount)


def test_follow_up_only_once(agent):
    agent.handle("send money to Jake")
    assert isinstance(agent.handle("what's the weather"), Reply)  # not an amount: drop the question
    assert isinstance(agent.handle("$50"), Reply)
