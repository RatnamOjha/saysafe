"""Pipecat: keep private replies off the speaker.

    from saysafe import Room
    from saysafe.integrations.pipecat import PrivacyFilter

    privacy = PrivacyFilter(
        send_to_phone=push_to_phone,                    # your push notification / app
        tool_sources={"get_login_code": {"otp"}, "get_balance": {"bank"}},
    )
    pipeline = Pipeline([transport.input(), stt, user_aggregator, llm, privacy, tts,
                         transport.output(), assistant_aggregator])

    privacy.room = Room.OTHERS_PRESENT   # whenever your app reports it
    privacy.headphones = True            # earbuds connected: everything is said

Put it right after the LLM. Sentences pass straight through while this room may hear
them. A sentence that may not, or that has a number in it (the words that make a number
secret can come after it), is held; the rest of the reply is then decided in one go:
spoken as is, spoken redacted with the details sent to the phone, or sent to the phone
with one short line spoken. Alone or in headphones nothing is held, so nothing is slower.

The LLM context records what was actually said ("I sent it to your phone."). Function
results, codes included, stay in the context as usual.
"""

from __future__ import annotations

import inspect
import logging
import re
from collections.abc import Awaitable, Callable, Iterable, Mapping

try:
    from pipecat.frames.frames import (
        Frame,
        FunctionCallResultFrame,
        InterruptionFrame,
        LLMFullResponseEndFrame,
        LLMFullResponseStartFrame,
        LLMTextFrame,
        TTSSpeakFrame,
    )
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
    from pipecat.utils.string import match_endofsentence
except ImportError as e:  # pragma: no cover - depends on the environment
    raise ImportError(
        "saysafe.integrations.pipecat needs Pipecat (Python 3.11+). "
        "Install it with: pip install 'saysafe[pipecat]'"
    ) from e

from saysafe.privacy.audience import Room
from saysafe.privacy.guard import PrivacyDecision, PrivacyGuard

log = logging.getLogger("saysafe")

# Called with the full reply; may be async. Raise if it didn't reach the phone.
PhoneSender = Callable[[str], "Awaitable[None] | None"]

PHONE_FAILED_LINE = "I couldn't send that to your phone, and I can't say it out loud here."


def failed_line(screen: str = "phone") -> str:
    return f"I couldn't send that to your {screen}, and I can't say it out loud here."


_DIGIT = re.compile(r"\d")


class PrivacyFilter(FrameProcessor):
    """Checks LLM replies and TTSSpeakFrames before TTS.

    send_to_phone: delivers withheld replies privately (to guard.screen). Required, because
        the spoken line says "I sent it to your phone"; if it raises, failed_line() is said.
    guard: a PrivacyGuard, e.g. with an LLM classifier; rules only by default.
    room / headphones: values, or zero-argument callables read for every sentence
        (e.g. room=tracker.room). Set the attributes any time.
    tool_sources: function name -> source tags, applied to the reply after its result.

    Event "on_decision"(processor, decision): every reply that was checked as a whole.
    """

    def __init__(
        self,
        *,
        send_to_phone: PhoneSender,
        guard: PrivacyGuard | None = None,
        room: Room | Callable[[], Room] = Room.UNKNOWN,
        headphones: bool | Callable[[], bool] = False,
        tool_sources: Mapping[str, Iterable[str]] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.guard = guard or PrivacyGuard()
        self.room = room
        self.headphones = headphones
        self._send = send_to_phone
        self._tool_sources = {name: frozenset(tags) for name, tags in (tool_sources or {}).items()}
        self._sources: set[str] = set()  # from function results, for the next reply
        self._register_event_handler("on_decision")
        self._new_reply()

    def _new_reply(self) -> None:
        self._pending = ""  # text after the last sentence end
        self._spoken = ""  # this reply's text already let through
        self._held = ""  # held until the reply ends
        self._had_text = False

    def _room(self) -> Room:
        return Room(self.room() if callable(self.room) else self.room)

    def _headphones(self) -> bool:
        return bool(self.headphones() if callable(self.headphones) else self.headphones)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if direction != FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
        elif isinstance(frame, LLMTextFrame) and not frame.skip_tts:
            self._had_text = True
            self._pending += frame.text
            while end := match_endofsentence(self._pending):
                sentence, self._pending = self._pending[:end], self._pending[end:]
                await self._sentence(sentence)
        elif isinstance(frame, LLMFullResponseEndFrame):
            await self._end_reply()
            await self.push_frame(frame, direction)
        elif isinstance(frame, LLMFullResponseStartFrame):
            self._new_reply()
            await self.push_frame(frame, direction)
        elif isinstance(frame, TTSSpeakFrame):
            await self._speak_frame(frame)
        elif isinstance(frame, FunctionCallResultFrame):
            self._sources |= self._tool_sources.get(frame.function_name, frozenset())
            await self.push_frame(frame, direction)
        elif isinstance(frame, InterruptionFrame):
            self._new_reply()  # the reply was cut off: nothing held gets said
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)

    async def _sentence(self, sentence: str) -> None:
        if not self._held and self._streamable(sentence):
            self._spoken += sentence
            await self.push_frame(LLMTextFrame(text=sentence))
        else:
            self._held += sentence  # once holding, hold to the end of the reply

    def _streamable(self, sentence: str) -> bool:
        room, headphones = self._room(), self._headphones()
        if self.guard.speaks_everything(room=room, headphones=headphones):
            return True
        if _DIGIT.search(sentence):
            return False
        return self.guard.says_as_is(
            sentence, sources=self._sources, room=room, context=self._spoken
        )

    async def _end_reply(self) -> None:
        if self._pending.strip():
            await self._sentence(self._pending)
        elif self._pending and not self._held:
            await self.push_frame(LLMTextFrame(text=self._pending))
        if self._held.strip():
            d = self.guard.check(
                self._held.strip(),
                sources=self._sources,
                room=self._room(),
                headphones=self._headphones(),
                context=self._spoken,
            )
            say = await self._deliver(d, (self._spoken + self._held).strip())
            if say:
                space = " " if self._spoken and not self._spoken[-1].isspace() else ""
                await self.push_frame(LLMTextFrame(text=space + say))
            await self._call_event_handler("on_decision", d)
        if self._had_text:
            self._sources.clear()
        self._new_reply()

    async def _speak_frame(self, frame: TTSSpeakFrame) -> None:
        d = self.guard.check(frame.text, room=self._room(), headphones=self._headphones())
        if not d.withheld:
            await self.push_frame(frame)
        else:
            say = await self._deliver(d, frame.text)
            await self.push_frame(
                TTSSpeakFrame(text=say, append_to_context=frame.append_to_context)
            )
        await self._call_event_handler("on_decision", d)

    async def _deliver(self, decision: PrivacyDecision, full_text: str) -> str:
        """Send the withheld reply to the phone; return what may be said."""
        if not decision.withheld:
            return decision.say
        try:
            result = self._send(full_text)
            if inspect.isawaitable(result):
                await result
        except Exception as e:  # anything failing means nothing reached the phone
            log.warning("send_to_phone failed (%s); the reply stays private", type(e).__name__)
            return failed_line(self.guard.screen)
        return decision.say
