"""Fake executors. Refuse to run without a valid approval token.

Refused is re-exported here for callers that only know the executor.
"""

from collections.abc import Callable
from decimal import Decimal

from num2words import num2words

from band_demo.agent.mock_agent import Reply
from saysafe.approvals.actions import Action
from saysafe.approvals.tokens import Refused, verify_token


class MissingApproval(Exception):
    pass


TokenVerifier = Callable[[str, Action], None]  # raises Refused

__all__ = ["Executor", "MissingApproval", "Refused", "TokenVerifier"]


class Executor:
    def __init__(self, verify_token: TokenVerifier = verify_token):
        self.verify_token = verify_token
        self.executed: list[Action] = []

    def run(self, action: Action, approval_token: str | None) -> Reply:
        """Raises MissingApproval or Refused instead of running an unapproved action."""
        if not approval_token:
            raise MissingApproval(f"{action.type} {action.id} has no approval token")
        self.verify_token(approval_token, action)
        self.executed.append(action)
        return _CONFIRM[action.type](action)


def _money(amount: Decimal | None) -> str:
    if amount is None:
        return "it"
    dollars = int(amount)
    words = num2words(dollars)
    return f"{words} dollars" if amount == dollars else f"{words} {int(amount * 100) % 100:02d}"


def _order(a: Action) -> Reply:
    return Reply(text=f"Order placed. Arrives at {a.params.get('eta', 'soon')}.")


def _send(a: Action) -> Reply:
    return Reply(text=f"Sent {_money(a.amount)} to {a.counterparty}.", source_tags={"bank"})


def _cancel(a: Action) -> Reply:
    renews = a.params.get("renews")
    tail = f" You have access until {renews}." if renews else ""
    return Reply(text=f"{a.counterparty} is cancelled.{tail}")


def _book(a: Action) -> Reply:
    return Reply(text=f"Booked with {a.counterparty}.", source_tags={"calendar"})


def _email(a: Action) -> Reply:
    return Reply(text=f"Email sent to {a.counterparty}.", source_tags={"email"})


def _reminder(a: Action) -> Reply:
    when = a.params.get("when")
    return Reply(text=f"Reminder set: {a.params.get('what')}" + (f" at {when}." if when else "."))


_CONFIRM: dict[str, Callable[[Action], Reply]] = {
    "order_food": _order,
    "send_money": _send,
    "cancel_subscription": _cancel,
    "book_appointment": _book,
    "send_email": _email,
    "set_reminder": _reminder,
}
