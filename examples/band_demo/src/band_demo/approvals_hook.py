"""before_execute(action): the band's side of an approval. saysafe decides; this does the IO.

1. ApprovalGuard.start: the tier, and what to say (a read-back, maybe a one-time word,
   or "check your phone" when others may hear a money read-back).
2. Voice tiers: speak it, listen for the reply, transcribe it, score the reply voice
   (and the command's) against the owner, and hand all that to ApprovalGuard.reply.
3. Whatever needs the phone goes out as an ntfy request; its Approve / Deny buttons hit
   the server, which calls Approver.resolve.

saysafe's events are forwarded to the trace UI under the demo's event names.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from band_demo import config as demo_config
from band_demo.owner import VoiceScorer, load_owner_scorer
from saysafe import Action, ApprovalGuard, Room, Step

if TYPE_CHECKING:
    from band_demo.agent.events import EventBus
    from band_demo.agent.pipeline import TurnContext
    from band_demo.audio.stt import STT

log = logging.getLogger(__name__)

Outcome = Literal["approve", "step_up", "reject"]

_EVENT_NAMES = {"readback": "readback_spoken", "voice_scored": "speaker_scored"}
_LED = {"approve": "done", "step_up": "amber", "reject": "off"}


@dataclass
class ApprovalDecision:
    outcome: Outcome
    token: str | None = None  # present only when outcome == "approve"
    message: str | None = None  # what to say after the decision, if anything
    reasons: list[str] = field(default_factory=list)


class Approver:
    """before_execute with every dependency injectable (STT, scorer, the guard)."""

    def __init__(
        self,
        *,
        stt: STT | None = None,
        scorer: VoiceScorer | None | Callable[[], VoiceScorer | None] = load_owner_scorer,
        guard: ApprovalGuard | None = None,
    ):
        self._stt = stt
        self._scorer = scorer
        self.guard = guard or demo_config.approvals()
        self.guard.on_event = self._forward
        self.cfg = self.guard.policy["voice"]
        self._events: EventBus | None = None
        self._turn: dict[str, Any] = {}  # extras for the trace of the current turn
        self._waiting: dict[str, TurnContext] = {}  # step id -> the turn that asked the phone

    @property
    def stt(self) -> STT:
        if self._stt is None:
            from band_demo.audio.stt import get_stt

            self._stt = get_stt("reply")
        return self._stt

    @property
    def scorer(self) -> VoiceScorer | None:
        if callable(self._scorer) and not isinstance(self._scorer, VoiceScorer):
            self._scorer = self._scorer()  # load the profile once, on first use
        return self._scorer

    # the flow

    def before_execute(self, action: Action, ctx: TurnContext) -> ApprovalDecision:
        self._events = ctx.events
        self._turn = {"latency": {}, "start": time.perf_counter()}
        try:
            return self._before_execute(action, ctx)
        finally:
            self._turn = {}  # a later phone tap isn't part of this turn's trace

    def _before_execute(self, action: Action, ctx: TurnContext) -> ApprovalDecision:
        latency = self._turn["latency"]
        room, headphones = _room(ctx)
        step = self.guard.start(action, room=room, headphones=headphones)
        if step.status == "approved":
            return ApprovalDecision("approve", step.token, reasons=list(step.reasons))

        t0 = time.perf_counter()
        ctx.speak(step.say)
        latency["readback"] = _ms(t0)
        if step.status == "awaiting_phone":
            return self._to_phone(step, ctx)

        ctx.events.led("listening")
        t0 = time.perf_counter()
        heard = ctx.listen(self.cfg["reply_timeout_s"])
        latency["listen"] = _ms(t0)
        if heard is None:
            ctx.events.publish("reply_captured", text=None, timed_out=True)
            step = self.guard.reply(step, None)
            ctx.speak(step.say)
            return ApprovalDecision("reject", reasons=list(step.reasons))

        # transcript + voice scores
        detail = None
        if isinstance(heard, str):  # text mode: no audio, so no voice evidence
            transcript, reply_result, detail = heard, None, "No voice to check (text reply)"
        else:
            t0 = time.perf_counter()
            transcript = self.stt.transcribe(heard).text
            latency["stt"] = _ms(t0)
            t0 = time.perf_counter()
            minimum = self.cfg["min_reply_speech_s"]  # replies are short; see policy.yaml
            reply_result = self.scorer.score_audio(heard, minimum) if self.scorer else None
            latency["speaker"] = _ms(t0)
            if self.scorer is None:
                detail = "No owner voice enrolled"
            elif reply_result.too_short:
                detail = "Too little speech to check the voice"
        speech = reply_result.speech_seconds if reply_result else 0.0
        ctx.events.publish("reply_captured", text=transcript, timed_out=False,
                           speech_seconds=round(speech, 2) if reply_result else None)  # fmt: skip
        command_score = (
            self.scorer.score_embedding(ctx.command_embedding)
            if self.scorer and ctx.command_embedding is not None
            else None
        )
        self._turn.update(speech_seconds=round(speech, 2), owner_enrolled=self.scorer is not None)
        t0 = time.perf_counter()
        step = self.guard.reply(
            step,
            transcript,
            voice_score=reply_result.score if reply_result else None,
            command_score=command_score,
        )
        latency["decide"] = _ms(t0)
        if step.say:
            ctx.speak(step.say)
        reasons = [*([detail] if detail and step.status != "approved" else []), *step.reasons]
        if step.status == "approved":
            return ApprovalDecision("approve", step.token, reasons=reasons)
        if step.status == "awaiting_phone":
            return self._to_phone(step, ctx, reasons)
        return ApprovalDecision("reject", reasons=reasons)

    def resolve(self, step_id: str, approve: bool) -> Step:
        """A tap on the phone. Raises saysafe.StepClosed if it's unknown or already settled."""
        step = self.guard.phone_approve(step_id) if approve else self.guard.phone_deny(step_id)
        ctx = self._waiting.pop(step_id, None)
        if ctx is not None:
            if approve:
                ctx.complete_approved(step.action, step.token)
            else:
                ctx.speak(step.say)
        return step

    def _to_phone(
        self, step: Step, ctx: TurnContext, reasons: list[str] | None = None
    ) -> ApprovalDecision:
        self._waiting[step.id] = ctx
        ctx.phone.request_approval(step.id, step.phone_request)
        return ApprovalDecision("step_up", reasons=reasons or list(step.reasons))

    def _forward(self, name: str, data: dict[str, Any]) -> None:
        """saysafe events -> the trace UI."""
        events = self._events
        if events is None:
            return
        if name == "voice_scored":
            data = {**data, "owner_enrolled": self._turn.get("owner_enrolled", False),
                    "speech_seconds": self._turn.get("speech_seconds", 0.0)}  # fmt: skip
        if name == "decision" and self._turn.get("latency"):
            latency = dict(self._turn["latency"])
            latency["total"] = _ms(self._turn["start"])
            data = {**data, "latency_ms_by_step": latency}
        events.publish(_EVENT_NAMES.get(name, name), **data)
        if name == "decision":
            events.led(_LED[data["outcome"]])


def _room(ctx: TurnContext) -> tuple[Room, bool]:
    from band_demo.privacy_hook import audience_state

    state = audience_state(ctx)
    return Room(state.level), state.headphones


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)


_approver: Approver | None = None


def get_approver() -> Approver:
    global _approver
    if _approver is None:
        _approver = Approver()
    return _approver


def before_execute(action: Action, ctx: TurnContext) -> ApprovalDecision:
    """Decide whether `action` may run. See the module docstring for the flow.

    ctx provides: speak(text), listen(timeout_s) -> audio | typed str | None,
    command_embedding, phone, events, complete_approved(action, token).
    """
    return get_approver().before_execute(action, ctx)
