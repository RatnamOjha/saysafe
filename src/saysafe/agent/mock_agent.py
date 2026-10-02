"""Rule-based intents over fake_user.json -> Action or Reply.

Deterministic by default. EARSHOT_AGENT=llm routes the text through tool calling
(smart model) and falls back to the rules if the LLM returns nothing.
"""

import hashlib
import json
import re
import time
from collections.abc import Callable
from decimal import Decimal
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field

from saysafe import llm
from saysafe.approvals.actions import Action
from saysafe.config import env

SourceTag = str  # bank | health | otp | email | calendar | public

CODE_PERIOD_S = 60


class Reply(BaseModel):
    text: str
    source_tags: set[SourceTag] = Field(default_factory=lambda: {"public"})
    understood: bool = True  # False: not a request the agent recognized


@lru_cache
def fake_user() -> dict:
    return json.loads((Path(__file__).parent / "fake_user.json").read_text())


# numbers ---------------------------------------------------------------------

_UNITS = {
    w: i
    for i, w in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen "
        "fourteen fifteen sixteen seventeen eighteen nineteen".split()
    )
}
_SCALES = {"thousand": 1_000, "million": 1_000_000}
_TENS = {
    w: 10 * i
    for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split(), start=2)
}


def words_to_number(words: list[str]) -> int | None:
    """'fifty' -> 50, 'two hundred and five' -> 205, 'a hundred' -> 100."""
    total, current, seen = 0, 0, False
    for w in words:
        if w in ("a", "and"):
            continue  # "a hundred" works because hundred treats 0 as 1
        if w in _UNITS:
            current += _UNITS[w]
        elif w in _TENS:
            current += _TENS[w]
        elif w == "hundred":
            current = max(current, 1) * 100
        elif w in _SCALES:
            total += max(current, 1) * _SCALES[w]
            current = 0
        else:
            return None
        seen = True
    return total + current if seen else None


def parse_amount(text: str) -> Decimal | None:
    """'$50', '$50.00', '50 dollars', 'fifty dollars', 'twenty five bucks'."""
    t = text.lower().replace(",", "")
    if m := re.search(r"\$\s?(\d+(?:\.\d{1,2})?)", t):
        return Decimal(m.group(1))
    if m := re.search(r"(\d+(?:\.\d{1,2})?)\s*(?:dollars?|bucks|usd)\b", t):
        return Decimal(m.group(1))
    if m := re.search(r"((?:[a-z]+[\s-]+){1,6}?)(?:dollars?|bucks)\b", t):
        words = m.group(1).replace("-", " ").split()
        # take the longest numeric tail: "send fifty" -> "fifty"
        for i in range(len(words)):
            n = words_to_number(words[i:])
            if n is not None:
                return Decimal(n)
    return None


# the agent -------------------------------------------------------------------


class MockAgent:
    def __init__(self, clock: Callable[[], float] = time.time, fixed_code: str | None = None):
        self.clock = clock
        self.fixed_code = fixed_code
        self.user = fake_user()
        self._awaiting_amount_for: str | None = None  # contact we asked "how much?" about

    def current_code(self) -> str:
        """Six-digit code that rotates every 60 s (fixed in demo mode)."""
        if self.fixed_code:
            return self.fixed_code
        window = int(self.clock() // CODE_PERIOD_S)
        digest = hashlib.sha256(f"alex-otp:{window}".encode()).hexdigest()
        return f"{int(digest, 16) % 1_000_000:06d}"

    def _answer_amount(self, text: str) -> Decimal | None:
        """A bare amount as the answer to "How much should I send?"."""
        t = text.lower().replace(",", "").strip(" .!?")
        if m := re.fullmatch(r"\$?(\d+(?:\.\d{1,2})?)\s*(million|thousand)(?: dollars)?", t):
            return Decimal(m.group(1)) * _SCALES[m.group(2)]  # "5 million": digits then a scale
        return parse_amount(t) or parse_amount(f"{t} dollars")

    def handle(self, text: str) -> Action | Reply:
        if self._awaiting_amount_for:
            name, self._awaiting_amount_for = self._awaiting_amount_for, None
            amount = self._answer_amount(text)
            if amount is not None:
                return self.send_money(name, amount)

        if env("EARSHOT_AGENT") == "llm":
            result = self._handle_llm(text)
            if result is not None:
                return result
        return self._handle_rules(text)

    # rules

    def _handle_rules(self, text: str) -> Action | Reply:
        t = re.sub(r"[^\w\s$.']|\.(?!\d)", " ", text.lower())  # keep "$50.00", drop "6."
        t = re.sub(r"\s+", " ", t).strip()

        if re.search(r"\border\b.*\busual\b", t):
            return self.order_usual()
        if m := re.search(r"\b(?:send|pay|venmo|transfer)\b.*?\bto\s+([a-z]+)", t):
            return self.send_money(m.group(1), parse_amount(t))
        if m := re.search(r"\bcancel\b(?:\s+my)?\s+([a-z+]+)", t):
            return self.cancel_subscription(m.group(1))
        if re.search(r"\b(?:verification|one time|security|login)\s+code\b|\bmy code\b|\botp\b", t):
            return self.verification_code()
        if re.search(r"\bbalance\b|how much (?:money )?(?:do i have|is in)", t):
            return self.bank_balance()
        if re.search(r"\bdermatologist\b|\bappointment\b|\bdoctor\b", t):
            return self.appointment()
        if re.search(r"\bemails?\b", t):
            return self.last_email()
        if re.search(r"\bweather\b|\bforecast\b", t):
            return self.weather()
        if m := re.search(r"\bremind(?:er)?\b(?: me)?(?: to)? (.+?)(?: at (.+))?$", t):
            what = re.sub(r"^(?:a )?(?:reminder )?to ", "", m.group(1)).strip()
            return self.set_reminder(what, (m.group(2) or "").strip() or None)
        return Reply(text="Sorry, I can't help with that yet.", understood=False)

    # intents (also the LLM tools)

    def order_usual(self) -> Action:
        o = self.user["usual_order"]
        return Action(
            type="order_food",
            counterparty=o["service"],
            amount=Decimal(o["total"]),
            destination=o["destination"],
            params={"restaurant": o["restaurant"], "items": o["items"], "eta": o["eta"]},
        )

    def send_money(self, name: str, amount: Decimal | None) -> Action | Reply:
        contact = next(
            (c for c in self.user["contacts"] if c["name"].lower() == name.lower()), None
        )
        display = contact["name"] if contact else name.capitalize()
        if amount is None:
            self._awaiting_amount_for = display
            return Reply(text=f"How much should I send to {display}?")
        return Action(
            type="send_money",
            counterparty=display,
            amount=amount,
            destination=contact["handle"] if contact else None,
            is_new_counterparty=not (contact and contact["paid_before"]),
        )

    def cancel_subscription(self, name: str) -> Action | Reply:
        sub = next(
            (s for s in self.user["subscriptions"] if s["name"].lower() == name.lower()), None
        )
        if sub is None:
            return Reply(text=f"I don't see a {name} subscription.")
        return Action(
            type="cancel_subscription",
            counterparty=sub["name"],
            amount=Decimal(sub["price"]),
            params={"renews": sub["renews"]},
        )

    def verification_code(self) -> Reply:
        bank = self.user["bank"]["institution"]
        return Reply(
            text=f"Your {bank} verification code is {self.current_code()}.",
            source_tags={"otp"},
        )

    def bank_balance(self) -> Reply:
        b = self.user["bank"]
        return Reply(
            text=f"Your {b['institution']} {b['account']} balance is ${Decimal(b['balance']):,}.",
            source_tags={"bank"},
        )

    def appointment(self) -> Reply:
        a = self.user["appointments"][0]
        return Reply(
            text=f"Your {a['title'].lower()} appointment with {a['with']} is {a['when']}.",
            source_tags={"health", "calendar"},
        )

    def last_email(self) -> Reply:
        e = self.user["emails"][-1]
        body = e["body"].format(code=self.current_code())
        return Reply(text=f"Email from {e['from']}: {body}", source_tags={"email"})

    def weather(self) -> Reply:
        return Reply(text=f"It's 64 and sunny in {self.user['city']}, windy this afternoon.")

    def set_reminder(self, what: str, when: str | None) -> Action:
        return Action(type="set_reminder", params={"what": what, "when": when})

    # optional LLM mode

    def _handle_llm(self, text: str) -> Action | Reply | None:
        result = llm.complete(
            [
                {"role": "system", "content": "You are a voice assistant. Call exactly one tool."},
                {"role": "user", "content": text},
            ],
            model="smart",
            timeout_s=3.0,
            tools=_TOOLS,
        )
        if result is None or not result.tool_calls:
            return None
        call = result.tool_calls[0]
        try:
            args = json.loads(call.function.arguments or "{}")
            if call.function.name == "send_money":
                return self.send_money(args["name"], parse_amount(f"${args['amount']}"))
            return getattr(self, call.function.name)(**args)
        except (AttributeError, KeyError, TypeError, ValueError):
            return None


def _tool(name: str, description: str, /, **props: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {k: {"type": v} for k, v in props.items()},
                "required": list(props),
            },
        },
    }


_TOOLS = [
    _tool("order_usual", "Reorder the user's usual food delivery."),
    _tool("send_money", "Send money to a contact.", name="string", amount="number"),
    _tool("cancel_subscription", "Cancel a subscription by name.", name="string"),
    _tool("verification_code", "Read the user's latest one-time verification code."),
    _tool("bank_balance", "Tell the user their bank balance."),
    _tool("appointment", "Tell the user about their next appointment."),
    _tool("last_email", "Read the user's most recent email."),
    _tool("weather", "Tell the user the weather."),
    _tool("set_reminder", "Set a reminder.", what="string", when="string"),
]
