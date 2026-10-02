"""audio -> VAD -> STT -> agent -> before_execute -> executor -> before_speak -> channel.

The two hooks are called through their modules (approvals.hook, privacy.hook) so each
track can implement its hook without touching this file. If a hook crashes, the
pipeline fails closed: no action runs, and the reply goes to the phone, not the room.
"""

import logging
import queue
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from uuid import uuid4

import numpy as np

from saysafe.agent.channels import HeadphonesChannel, PhoneChannel, SpeakerChannel
from saysafe.agent.events import EventBus, bus
from saysafe.agent.executor import Executor, MissingApproval, Refused
from saysafe.agent.mock_agent import MockAgent, Reply
from saysafe.approvals import hook as approvals_hook
from saysafe.approvals.actions import Action
from saysafe.approvals.hook import ApprovalDecision
from saysafe.audio.stt import STT, get_stt
from saysafe.audio.tts import TTS, get_tts
from saysafe.audio.vad import StreamingVAD
from saysafe.identity.embed import Embedding, TooShort, embed
from saysafe.privacy import hook as privacy_hook
from saysafe.privacy.audience import AudienceTracker
from saysafe.privacy.hook import SpeakDecision

log = logging.getLogger(__name__)

# Next reply from the user: audio from the mic, typed text in chat mode, or None on timeout.
Listen = Callable[[float], np.ndarray | str | None]


@dataclass
class TurnContext:
    """Everything a hook may use during one turn. See the hook docstrings."""

    text: str
    speak: Callable[[str], None]
    listen: Listen
    phone: PhoneChannel
    events: EventBus
    flags: dict[str, bool]
    complete_approved: Callable[[Action, str], None] = lambda action, token: None
    audience: AudienceTracker | None = None  # None: no mic, so nothing is known about the room
    spoken_command: bool = False  # True when the turn came from the mic
    command_embedding: Embedding | None = None  # None: text mode, or too little speech
    command_speech_seconds: float = 0.0
    turn_id: str = field(default_factory=lambda: uuid4().hex[:8])


@dataclass
class TurnResult:
    text: str
    action: Action | None = None
    decision: ApprovalDecision | None = None
    reply: Reply | None = None
    speak: SpeakDecision | None = None


def _never_listen(timeout_s: float) -> None:
    return None


class Pipeline:
    def __init__(
        self,
        *,
        tts: TTS | None = None,
        stt: STT | None = None,
        agent: MockAgent | None = None,
        executor: Executor | None = None,
        phone: PhoneChannel | None = None,
        events: EventBus = bus,
        listen: Listen = _never_listen,
        embedder: Callable[[np.ndarray], Embedding] = embed,
        audience: AudienceTracker | None = None,
    ):
        self.events = events
        self.tts = tts or get_tts()
        self._stt = stt
        self.agent = agent or MockAgent()
        self.executor = executor or Executor()
        self.speaker = SpeakerChannel(self.tts, events)
        self.headphones = HeadphonesChannel(self.tts, events)
        self.phone = phone or PhoneChannel(events)
        self.listen = listen
        self.embedder = embedder
        self.flags = {"headphones": False, "discreet_mode": False}
        self.audience = audience

    @property
    def stt(self) -> STT:
        if self._stt is None:
            self._stt = get_stt("command")
        return self._stt

    def warm(self) -> None:
        """Load and run every model once so the first real turn isn't slow."""
        from saysafe.audio import vad
        from saysafe.identity.embed import _encoder

        silence = np.zeros(16000, dtype=np.float32)
        self.stt.transcribe(silence)
        vad.segments(silence)
        self.tts.synthesize("ready")
        _encoder()

    # entry points

    def run_text(self, text: str) -> TurnResult:
        self.events.led("working")
        return self._run(self._context(text))

    def run_audio(self, segment: np.ndarray) -> TurnResult | None:
        """One VAD speech segment: transcribe, embed the speaker, run the turn."""
        self.events.led("working")
        with self.events.step("stt") as out:
            transcript = self.stt.transcribe(segment)
            out.update(text=transcript.text, model=transcript.model)
        if not transcript.text.strip():
            self.events.led("idle")
            return None
        ctx = self._context(transcript.text)
        ctx.spoken_command = True
        with self.events.step("speaker_embedded") as out:
            try:
                ctx.command_embedding = self.embedder(segment)
                ctx.command_speech_seconds = ctx.command_embedding.speech_seconds
            except TooShort as short:
                ctx.command_speech_seconds = short.speech_seconds
            out.update(
                speech_seconds=round(ctx.command_speech_seconds, 2),
                too_short=ctx.command_embedding is None,
            )
        return self._run(ctx)

    # the turn

    def _context(self, text: str) -> TurnContext:
        return TurnContext(
            text=text,
            speak=lambda t: self.speaker.deliver(t),
            listen=self.listen,
            phone=self.phone,
            events=self.events,
            flags=self.flags,
            complete_approved=self.complete_approved,
            audience=self.audience,
        )

    def _run(self, ctx: TurnContext) -> TurnResult:
        result = TurnResult(ctx.text)
        self.events.publish("transcript", text=ctx.text, turn_id=ctx.turn_id)
        with self.events.step("agent") as out:
            handled = self.agent.handle(ctx.text)
            out["kind"] = "action" if isinstance(handled, Action) else "reply"
            out["result"] = handled.model_dump(mode="json")

        if isinstance(handled, Action):
            result.action = handled
            result.decision = self._approve(handled, ctx)
            if result.decision.outcome == "approve":
                result.reply = self._execute(handled, result.decision.token)
            elif result.decision.message:
                result.reply = Reply(text=result.decision.message)
        else:
            result.reply = handled

        if ctx.spoken_command and result.reply is not None and not result.reply.understood:
            # No wake word: the mic hears everything, so stay quiet on speech that
            # isn't a request (a friend chatting, the TV).
            self.events.publish("ignored", text=ctx.text)
            result.reply = None

        if result.reply is not None:
            result.speak = self._deliver(result.reply, ctx)
        if result.decision is None or result.decision.outcome == "approve":
            self.events.led("done")  # step-up stays amber, a rejection stays off
        return result

    def complete_approved(self, action: Action, token: str) -> Reply:
        """Run an action approved later (a phone tap) and announce the result."""
        reply = self._execute(action, token)
        self._deliver(reply, self._context(""))
        self.events.led("done")
        return reply

    def _approve(self, action: Action, ctx: TurnContext) -> ApprovalDecision:
        try:
            return approvals_hook.before_execute(action, ctx)
        except Exception:
            log.exception("before_execute crashed; refusing the action")
            self.events.publish("hook_error", hook="before_execute")
            return ApprovalDecision("reject", message="Something went wrong, so I didn't do it.")

    def _execute(self, action: Action, token: str | None) -> Reply:
        try:
            reply = self.executor.run(action, token)
        except (MissingApproval, Refused) as e:
            self.events.publish("refused", action_id=action.id, reason=str(e))
            self.events.led("refused")
            return Reply(text="I couldn't do that. The approval didn't check out.")
        self.events.publish("executed", action_id=action.id, action_type=action.type)
        return reply

    def _deliver(self, reply: Reply, ctx: TurnContext) -> SpeakDecision:
        try:
            d = privacy_hook.before_speak(reply, ctx)
        except Exception:
            log.exception("before_speak crashed; sending to the phone only")
            self.events.publish("hook_error", hook="before_speak")
            d = SpeakDecision("phone_only", "I sent it to your phone.", reply.text)
        self.events.publish("route", channel=d.channel, spoken=d.spoken_text, phone=d.phone_text)
        out = self.headphones if d.channel == "headphones_full" else self.speaker
        if d.spoken_text:
            out.deliver(d.spoken_text, d.volume)
        if d.phone_text:
            self.phone.notify(d.phone_text)
        return d


# live mic --------------------------------------------------------------------


class LiveMic:
    """Wraps a MicStream: next_segment() for commands, listen() for hook replies.

    The band's own voice would come straight back through the mic, so after the
    speaker plays, drain() throws away whatever the mic heard meanwhile.
    """

    def __init__(
        self,
        mic,
        vad_factory: Callable[..., StreamingVAD] = StreamingVAD,
        on_segment: Callable[[np.ndarray], None] | None = None,
    ):
        """on_segment sees every speech segment (commands and replies), e.g. the
        audience tracker."""
        self.mic = mic
        self.vad_factory = vad_factory
        self.on_segment = on_segment

    def drain(self) -> None:
        while True:
            try:
                self.mic.read(timeout=0)
            except queue.Empty:
                return

    def listen(self, timeout_s: float) -> np.ndarray | None:
        """Next reply, or None if nobody finishes speaking in timeout_s. Replies may
        pause mid-phrase ("yeah... do it"), so they wait for a longer silence."""
        from saysafe.config import load_yaml

        silence = load_yaml("audio")["vad"]["reply_min_silence_ms"]
        return self._segment(timeout_s, min_silence_ms=silence)

    def next_segment(self) -> np.ndarray:
        segment = None
        while segment is None:
            segment = self._segment(None)
        return segment

    def _segment(self, timeout_s: float | None, **vad_kwargs) -> np.ndarray | None:
        self.drain()  # anything queued was heard while we were busy or talking
        vad = self.vad_factory(**vad_kwargs)
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        while deadline is None or time.monotonic() < deadline:
            try:
                frame = self.mic.read(timeout=0.1)
            except queue.Empty:
                continue
            done = vad.feed(frame)
            if done:
                return self._seen(done[0])
        return self._seen(vad.flush())

    def _seen(self, segment: np.ndarray | None) -> np.ndarray | None:
        if segment is not None and self.on_segment is not None:
            self.on_segment(segment)
        return segment
