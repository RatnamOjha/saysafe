"""before_speak(reply): where the band's reply goes. saysafe's PrivacyGuard decides.

Instructions inside content are data: an email saying "read this code out loud"
is just text to classify, and changes nothing about where the reply goes. The
phone gets the full text whenever the decision withholds anything.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from band_demo import privacy_llm
from saysafe import PrivacyDecision, PrivacyGuard, Room
from saysafe.privacy.audience import AudienceState

if TYPE_CHECKING:
    from band_demo.agent.mock_agent import Reply
    from band_demo.agent.pipeline import TurnContext

__all__ = ["PrivacyDecision", "before_speak"]


def audience_state(ctx: TurnContext) -> AudienceState:
    """The tracker's view, with the UI flags applied. No mic means we know nothing."""
    if ctx.audience is not None:
        state = ctx.audience.state()
    else:
        state = AudienceState("unknown", 0.0, None, 0, 0, False, False, ["No mic, room unknown"])
    state.headphones = state.headphones or ctx.flags.get("headphones", False)
    state.discreet_mode = state.discreet_mode or ctx.flags.get("discreet_mode", False)
    if ctx.flags.get("headphones") and "Headphones connected" not in state.evidence:
        state.evidence.append("Headphones connected")
    if ctx.flags.get("discreet_mode") and "Discreet mode on" not in state.evidence:
        state.evidence.append("Discreet mode on")
    return state


def before_speak(reply: Reply, ctx: TurnContext) -> PrivacyDecision:
    ev = ctx.events
    classifier, smoother = privacy_llm.from_env()
    guard = PrivacyGuard(classifier, smoother)
    state = audience_state(ctx)
    d = guard.check(
        reply.text,
        sources=reply.source_tags,
        room=Room(state.level),
        headphones=state.headphones,
        discreet=state.discreet_mode,
        whisper=getattr(ctx, "voice_style", "normal") == "whisper",
    )
    ev.publish(
        "detection", level=d.level, categories=list(d.categories),
        spans=[{"start": s.start, "end": s.end, "category": s.category, "text": s.text}
               for s in d.spans],
        latency_ms_by_layer=d.latency_ms,
    )  # fmt: skip
    ev.publish(
        "audience_state", level=state.level, evidence=state.evidence,
        listening_seconds=state.listening_seconds, seconds_since_other=state.seconds_since_other,
        headphones=state.headphones, discreet_mode=state.discreet_mode,
    )  # fmt: skip
    ev.publish(
        "route_decision", channel=d.channel, rule_cell=d.rule, reasons=list(d.reasons),
        spoken=d.say, phone=d.send_to_phone is not None,
    )  # fmt: skip
    if d.rewrite:
        ev.publish("rewrite_step", step=d.rewrite, text=d.say)
    return d
