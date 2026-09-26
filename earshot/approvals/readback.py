"""Short spoken read-back that stands in for a confirmation screen.

Who, how much, where, anything unusual, then what to say. The body is capped at
14 words and should take under 4 s at 2.6 words per second.
"""

from dataclasses import dataclass
from decimal import Decimal

from num2words import num2words

from earshot.approvals.actions import Action
from earshot.approvals.policy import RiskAssessment

MAX_BODY_WORDS = 14
WORDS_PER_SECOND = 2.6

_SOURCE_PREFIX = {
    "from_content": "This came from an email.",
    "agent_initiated": "I suggested this.",
}


@dataclass(frozen=True)
class Readback:
    body: str
    instruction: str

    @property
    def text(self) -> str:
        return f"{self.body} {self.instruction}".strip()

    @property
    def body_words(self) -> int:
        return len(self.body.split())

    @property
    def est_seconds(self) -> float:
        return len(self.text.split()) / WORDS_PER_SECOND


def _n(n: int) -> str:
    return num2words(n).replace(",", "").replace(" and ", " ")


def money_words(amount: Decimal) -> str:
    """43.20 -> 'forty-three twenty', 50 -> 'fifty dollars', 0.75 -> 'seventy-five cents',
    43.05 -> 'forty-three oh five'."""
    dollars = int(amount)
    cents = int((amount - dollars) * 100)
    if cents == 0:
        return f"{_n(dollars)} dollar{'s' if dollars != 1 else ''}"
    if dollars == 0:
        return f"{_n(cents)} cents"
    return f"{_n(dollars)} {'oh ' + _n(cents) if cents < 10 else _n(cents)}"


def _body(a: Action) -> list[str]:
    money = money_words(a.amount) if a.amount is not None else None
    who = a.counterparty or ""
    if a.type == "order_food":
        parts = [who, money, f"to {a.destination}" if a.destination else None]
        return [", ".join(p for p in parts if p) + "."]
    if a.type == "send_money":
        new = ["New payee."] if a.is_new_counterparty else []
        return [*new, f"{who}, {money}." if money else f"{who}."]
    if a.type == "cancel_subscription":
        return [f"Cancel {who}?"]
    if a.type == "book_appointment":
        when = a.params.get("when")
        return [f"Book {who}{', ' + when if when else ''}."]
    if a.type == "send_email":
        return [f"Email {who}."]
    if a.type == "set_reminder":
        when = a.params.get("when")
        return [f"Reminder: {a.params.get('what', '')}{' at ' + when if when else ''}."]
    return [f"{a.type.replace('_', ' ').capitalize()}."]


def _instruction(tier: str, challenge_word: str | None) -> str:
    if tier == "voice":
        return "Say yes."
    if tier == "voice_challenge":
        if not challenge_word:
            raise ValueError("voice_challenge read-back needs a challenge word")
        return f"Say '{challenge_word}' to confirm."
    if tier == "phone_tap":
        return "Check your phone to approve."
    return ""


def readback(action: Action, risk: RiskAssessment, challenge_word: str | None = None) -> Readback:
    parts = [_SOURCE_PREFIX[action.source]] if action.source in _SOURCE_PREFIX else []
    parts += _body(action)
    body = " ".join(parts)
    words = body.split()
    if len(words) > MAX_BODY_WORDS:  # long names or params: keep the start, it has who and how much
        body = " ".join(words[:MAX_BODY_WORDS]).rstrip(",.") + "."
    return Readback(body, _instruction(risk.tier, challenge_word))
