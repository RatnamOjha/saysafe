"""before_speak(reply): detect, check audience, route, rewrite.

STUB (A3): speaks everything in full. The privacy track (C2) replaces the body of
before_speak; the signature and SpeakDecision are the contract pipeline.py relies on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from earshot.agent.mock_agent import Reply
    from earshot.agent.pipeline import TurnContext

Channel = Literal["speak_full", "headphones_full", "speak_redacted_and_phone", "phone_only"]


@dataclass
class SpeakDecision:
    channel: Channel
    spoken_text: str | None  # said out loud (speaker, or headphones for headphones_full)
    phone_text: str | None = None  # sent to the phone, full text
    volume: float = 1.0
    reasons: list[str] = field(default_factory=list)


def before_speak(reply: Reply, ctx: TurnContext) -> SpeakDecision:
    """Decide where `reply` goes and what gets said out loud.

    ctx.flags holds UI toggles (headphones, discreet_mode); ctx.events for traces.
    """
    return SpeakDecision(channel="speak_full", spoken_text=reply.text)
