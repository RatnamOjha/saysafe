"""before_execute(action): the approval decision flow.

STUB (A3): approves everything with an unsigned token so the pipeline runs end to
end. The approvals track (B2) replaces the body of before_execute; the signature
and ApprovalDecision are the contract pipeline.py relies on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from earshot.approvals.actions import Action

if TYPE_CHECKING:
    from earshot.agent.pipeline import TurnContext

Outcome = Literal["approve", "step_up", "reject"]


@dataclass
class ApprovalDecision:
    outcome: Outcome
    token: str | None = None  # present only when outcome == "approve"
    message: str | None = None  # what to say when not approved ("Okay, I won't.")


def before_execute(action: Action, ctx: TurnContext) -> ApprovalDecision:
    """Decide whether `action` may run.

    ctx gives the hook what it needs without touching pipeline.py:
      ctx.speak(text)           say something through the speaker (read-backs)
      ctx.listen(timeout_s)     next speech segment from the mic, or None
      ctx.command_embedding     speaker embedding of the command, or None if too short
      ctx.phone                 PhoneChannel (request_approval)
      ctx.events                EventBus for trace events and LED state
    """
    return ApprovalDecision(outcome="approve", token="UNSIGNED-STUB")
