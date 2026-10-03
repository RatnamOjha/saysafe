"""PrivacyGuard: decide what a voice agent may say out loud, in one call.

    guard = PrivacyGuard()
    d = guard.check("Your Chase code is 482913.", room=Room.OTHERS_PRESENT)
    d.say            # "I sent it to your phone."
    d.send_to_phone  # "Your Chase code is 482913."

The host owns the speaker and the phone: speak `say`, and deliver `send_to_phone`
privately (push notification, companion app) whenever it isn't None.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from saysafe.config import load_yaml
from saysafe.privacy.audience import AudienceState, Room
from saysafe.privacy.detect import (
    Classifier,
    Detection,
    Level,
    Span,
    detect,
    highest,
    rank,
    regex_spans,
)
from saysafe.privacy.rewrite import Smoother
from saysafe.privacy.route import Channel, route, table_cell

_ROOM_EVIDENCE = {
    Room.ALONE: "Only you can hear",
    Room.UNKNOWN: "Not sure who can hear",
    Room.OTHERS_PRESENT: "Others may hear",
}


@dataclass(frozen=True)
class PrivacyDecision:
    say: str  # speak this: to the room, or into the headphones
    send_to_phone: str | None  # deliver this privately, or None
    channel: Channel
    level: Level
    categories: tuple[str, ...] = ()
    spans: tuple[Span, ...] = ()  # what made it sensitive, as positions in the text
    reasons: tuple[str, ...] = ()
    rule: str = ""  # which rule or routing table cell fired
    rewrite: str | None = None  # span | llm | fallback, when the reply was rewritten
    latency_ms: dict[str, float] = field(default_factory=dict)
    volume: float = 1.0  # lower when the user whispered

    @property
    def withheld(self) -> bool:
        """True when some of the reply was kept off the speaker."""
        return self.send_to_phone is not None


class PrivacyGuard:
    """Checks replies before they're spoken. Rules only by default: no models, no network.

    classifier: optional extra detection layer, e.g. an LLM (see detect.CLASSIFIER_PROMPT).
    smoother: optional rewriter that makes redacted replies sound natural
        (see rewrite.smoother_prompt).
    screen: where private replies go, as the user would say it: "phone", "watch", "app".
        The host delivers send_to_phone there; the spoken lines name it.
    """

    def __init__(
        self,
        classifier: Classifier | None = None,
        smoother: Smoother | None = None,
        *,
        screen: str = "phone",
    ):
        self.classifier = classifier
        self.smoother = smoother
        self.screen = screen

    def check(
        self,
        text: str,
        *,
        sources: Iterable[str] = (),
        room: Room | str = Room.UNKNOWN,
        headphones: bool = False,
        discreet: bool = False,
        whisper: bool = False,
        context: str = "",
    ) -> PrivacyDecision:
        """Where `text` may go.

        sources: where the reply's data came from (otp, bank, health, email, calendar).
        room: who may hear the speaker; UNKNOWN is cautious.
        headphones: only the owner hears it, so everything is said.
        discreet / whisper: say nothing personal or above out loud.
        context: earlier text of the same reply that was already spoken. It's only used to
            find what makes `text` sensitive ("Your code is ready." then "It's 482913.").
        """
        room = Room(room)
        detection = self.detect(text, sources=sources, context=context)
        evidence = [_ROOM_EVIDENCE[room]]
        if headphones:
            evidence.append("Headphones connected")
        if discreet:
            evidence.append("Discreet mode on")
        state = AudienceState(
            level=room.value,
            listening_seconds=0.0,
            seconds_since_other=None,
            others_count=int(room is Room.OTHERS_PRESENT),
            other_segments=0,
            headphones=headphones,
            discreet_mode=discreet,
            evidence=evidence,
        )
        style = "whisper" if whisper else "normal"
        r = route(text, detection, state, style, self.smoother, self.screen)
        return PrivacyDecision(
            say=r.spoken_text or "",
            send_to_phone=r.phone_text,
            channel=r.channel,
            level=detection.level,
            categories=tuple(detection.categories),
            spans=tuple(detection.spans),
            reasons=tuple(r.reasons),
            rule=r.rule_cell,
            rewrite=r.rewrite_step,
            latency_ms=dict(detection.latency_ms),
            volume=r.volume,
        )

    def says_as_is(
        self,
        text: str,
        *,
        sources: Iterable[str] = (),
        room: Room | str = Room.UNKNOWN,
        headphones: bool = False,
        discreet: bool = False,
        whisper: bool = False,
        context: str = "",
    ) -> bool:
        """True if check() would speak `text` unchanged. Never calls the smoother, and
        skips detection when this room may hear anything."""
        room = Room(room)
        if self.speaks_everything(room=room, headphones=headphones, discreet=discreet,
                                  whisper=whisper):  # fmt: skip
            return True
        level = self.detect(text, sources=sources, context=context).level
        if discreet or whisper:
            return rank(level) < rank("personal")
        return table_cell(level, room.value) == "speak"

    def speaks_everything(
        self,
        *,
        room: Room | str = Room.UNKNOWN,
        headphones: bool = False,
        discreet: bool = False,
        whisper: bool = False,
    ) -> bool:
        """True if any reply, however sensitive, may be said here (alone, or headphones)."""
        if headphones:
            return True
        quiet = discreet or whisper
        return not quiet and table_cell("secret", Room(room).value) == "speak"

    def detect(self, text: str, *, sources: Iterable[str] = (), context: str = "") -> Detection:
        """How sensitive `text` is. Context can only add to what `text` alone shows."""
        tags = frozenset(sources)
        d = detect(text, tags, self.classifier)
        if not context.strip():
            return d
        joined = f"{context.rstrip()} {text}"
        offset = len(joined) - len(text)
        new: list[Span] = []
        for s in regex_spans(joined, tags):
            if s.end <= offset:
                continue  # entirely in the context
            start, end = max(s.start, offset) - offset, s.end - offset
            span = Span(start, end, s.category, s.source, text[start:end])
            if not any(span.start < k.end and k.start < span.end for k in d.spans):
                new.append(span)
        if not new:
            return d
        rules = load_yaml("sensitivity")["rules"]
        return Detection(
            level=highest(d.level, *(rules[s.category] for s in new)),
            categories=sorted(set(d.categories) | {s.category for s in new}),
            spans=sorted([*d.spans, *new], key=lambda s: s.start),
            latency_ms=d.latency_ms,
            source_hint_level=d.source_hint_level,
        )
