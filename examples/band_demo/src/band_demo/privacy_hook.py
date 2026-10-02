"""before_speak(reply): detect, check audience, route, rewrite.

Instructions inside content are data: an email saying "read this code out loud"
is just text to classify, and changes nothing about where the reply goes. The
phone gets the full text whenever the channel includes it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from band_demo import privacy_llm
from saysafe.privacy.audience import AudienceState
from saysafe.privacy.detect import detect
from saysafe.privacy.route import Channel, SpeakDecision, route

if TYPE_CHECKING:
    from band_demo.agent.mock_agent import Reply
    from band_demo.agent.pipeline import TurnContext

__all__ = ["Channel", "SpeakDecision", "before_speak"]


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


def before_speak(reply: Reply, ctx: TurnContext) -> SpeakDecision:
    ev = ctx.events
    classifier, smoother = privacy_llm.from_env()
    with ev.step("detection") as out:
        d = detect(reply.text, reply.source_tags, classifier)
        out.update(
            level=d.level, categories=d.categories,
            spans=[{"start": s.start, "end": s.end, "category": s.category, "text": s.text}
                   for s in d.spans],
            latency_ms_by_layer=d.latency_ms,
        )  # fmt: skip
    state = audience_state(ctx)
    ev.publish(
        "audience_state", level=state.level, evidence=state.evidence,
        listening_seconds=state.listening_seconds, seconds_since_other=state.seconds_since_other,
        headphones=state.headphones, discreet_mode=state.discreet_mode,
    )  # fmt: skip
    voice_style = getattr(ctx, "voice_style", "normal")
    with ev.step("route_decision") as out:
        decision = route(reply.text, d, state, voice_style, smoother)
        out.update(
            channel=decision.channel, rule_cell=decision.rule_cell, reasons=decision.reasons,
            spoken=decision.spoken_text, phone=decision.phone_text is not None,
        )  # fmt: skip
    if decision.rewrite_step:
        ev.publish("rewrite_step", step=decision.rewrite_step, text=decision.spoken_text)
    return decision
