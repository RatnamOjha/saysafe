"""Short spoken read-back that stands in for a confirmation screen.

Who, how much, where, anything unusual, then what to say. The body is capped at
14 words and should take under 4 s at 2.6 words per second.
"""

import json
from dataclasses import dataclass
from decimal import Decimal

from saysafe.approvals.actions import Action
from saysafe.approvals.numwords import number_words
from saysafe.approvals.policy import RiskAssessment

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
    return number_words(n)


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
        parts = [who, money, f"to {a.destination}" if a.destination else None]
        return [*new, ", ".join(p for p in parts if p) + "."]
    if a.type == "cancel_subscription":
        return [f"Cancel {who}{', ' + money if money else ''}?"]
    if a.type == "book_appointment":
        when = a.params.get("when")
        return [f"Book {who}{', ' + when if when else ''}."]
    if a.type == "send_email":
        return [f"Email {who}."]
    if a.type == "set_reminder":
        when = a.params.get("when")
        return [f"Reminder: {a.params.get('what', '')}{' at ' + when if when else ''}."]
    # any other action type: "Unlock door: front door." / "Refund: Jake, fifty dollars."
    what = a.type.replace("_", " ").capitalize()
    rendered = ", ".join(p for p in (who, money, a.destination) if p)
    return [f"{what}: {rendered}." if rendered else f"{what}."]


def _instruction(tier: str, challenge_word: str | None, screen: str = "phone") -> str:
    if tier == "voice":
        return "Say yes."
    if tier == "voice_challenge":
        if not challenge_word:
            raise ValueError("voice_challenge read-back needs a challenge word")
        return f"Say '{challenge_word}' to confirm."
    if tier == "phone_tap":
        return f"Check your {screen} to approve."
    return ""


def summary(action: Action) -> str:
    """Who, how much, where, anything unusual: the read-back without the instruction."""
    parts = [_SOURCE_PREFIX[action.source]] if action.source in _SOURCE_PREFIX else []
    parts += _body(action)
    body = " ".join(parts)
    words = body.split()
    if len(words) > MAX_BODY_WORDS:  # long names or params: keep the start, it has who and how much
        body = " ".join(words[:MAX_BODY_WORDS]).rstrip(",.") + "."
    return body


def needs_phone_review(action: Action, risk: RiskAssessment, screen: str = "phone") -> bool:
    """Whether the spoken read-back leaves out action details or is too long.

    Voice approval is allowed only when this package knows how to say every
    user-relevant field. Unknown parameters go to the phone for a complete review.
    """
    body = " ".join([*_SOURCE_PREFIX.get(action.source, "").split(), *_body(action)])
    if len(body.split()) > MAX_BODY_WORDS:
        return True
    sample_word = "approval" if risk.tier == "voice_challenge" else None
    if readback(action, risk, sample_word, screen).est_seconds >= 4.5:
        return True

    handled_params = {
        "book_appointment": {"when"},
        "set_reminder": {"what", "when"},
    }.get(action.type, set())
    if set(action.params) - handled_params:
        return True

    # These fields have meaning but aren't part of the corresponding spoken summary.
    if action.type == "book_appointment" and (action.amount is not None or action.destination):
        return True
    if action.type == "set_reminder" and (
        action.counterparty or action.amount is not None or action.destination
    ):
        return True
    if action.type == "cancel_subscription" and action.destination:
        return True
    if action.type == "send_money" and action.is_new_counterparty:
        return False  # explicitly spoken above
    if action.type != "send_money" and action.is_new_counterparty:
        return True
    return False


def phone_details(action: Action) -> str:
    """Full, structured details for the host app to render on its approval screen."""
    payload = action.model_dump(mode="json", exclude={"id", "created_at"})
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)


def readback(
    action: Action,
    risk: RiskAssessment,
    challenge_word: str | None = None,
    screen: str = "phone",
) -> Readback:
    """screen: what the user approves on ("phone", "watch", "app")."""
    return Readback(summary(action), _instruction(risk.tier, challenge_word, screen))
