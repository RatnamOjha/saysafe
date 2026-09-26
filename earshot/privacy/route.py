"""Pick the output channel from sensitivity x audience.

Checked in order:
  1. headphones connected      -> headphones_full (only the owner hears it)
  2. discreet mode or whisper  -> anything personal or above: speak_redacted_and_phone
  3. otherwise the table in config/routing.yaml:
                 alone_likely   unknown          others_present
     public      speak          speak            speak
     personal    speak          speak            redacted+phone
     sensitive   speak          redacted+phone   redacted+phone
     secret      speak          phone            phone
"""

from dataclasses import dataclass, field
from typing import Literal

from earshot.config import load_yaml
from earshot.privacy.audience import AudienceState
from earshot.privacy.detect import Detection, Level, rank
from earshot.privacy.rewrite import rewrite

Channel = Literal["speak_full", "headphones_full", "speak_redacted_and_phone", "phone_only"]
VoiceStyle = Literal["normal", "whisper"]

PHONE_ONLY_LINE = "I sent it to your phone."
WHISPER_VOLUME = 0.35

_CELL_TO_CHANNEL: dict[str, Channel] = {
    "speak": "speak_full",
    "redacted_and_phone": "speak_redacted_and_phone",
    "phone": "phone_only",
}


@dataclass
class SpeakDecision:
    channel: Channel
    spoken_text: str | None  # said out loud (speaker, or headphones for headphones_full)
    phone_text: str | None = None  # sent to the phone, full text
    volume: float = 1.0
    reasons: list[str] = field(default_factory=list)
    rule_cell: str = ""  # which rule or table cell fired
    rewrite_step: str | None = None  # span | llm | fallback, when a rewrite ran


def table_cell(level: Level, audience_level: str) -> str:
    return load_yaml("routing")["table"][level][audience_level]


def route(
    text: str,
    detection: Detection,
    audience: AudienceState,
    voice_style: VoiceStyle = "normal",
) -> SpeakDecision:
    volume = WHISPER_VOLUME if voice_style == "whisper" else 1.0
    level = detection.level

    if audience.headphones:
        return SpeakDecision(
            "headphones_full", text, None, volume, ["Headphones connected"], "headphones"
        )

    if audience.discreet_mode or voice_style == "whisper":
        why = "Discreet mode on" if audience.discreet_mode else "You whispered"
        if rank(level) < rank("personal"):
            return SpeakDecision("speak_full", text, None, volume, [why, "Nothing private"],
                                 "discreet:public")  # fmt: skip
        # nothing personal or above is spoken, so only public rewrites pass
        return _redacted(text, detection, volume, [why], "discreet", lambda lv: lv == "public")

    cell = table_cell(level, audience.level)
    rule_cell = f"{level} x {audience.level}"
    reasons = [f"{level.capitalize()} reply", *audience.evidence[:1]]
    channel = _CELL_TO_CHANNEL[cell]
    if channel == "speak_full":
        return SpeakDecision(channel, text, None, volume, reasons, rule_cell)
    if channel == "phone_only":
        return SpeakDecision(channel, PHONE_ONLY_LINE, text, volume, reasons, rule_cell)

    def speakable(lv: Level) -> bool:
        return table_cell(lv, audience.level) == "speak"

    return _redacted(text, detection, volume, reasons, rule_cell, speakable)


def _redacted(text, detection, volume, reasons, rule_cell, speakable) -> SpeakDecision:
    r = rewrite(text, detection, speakable)
    return SpeakDecision(
        "speak_redacted_and_phone", r.text, text, volume, [*reasons, *r.reasons], rule_cell,
        rewrite_step=r.step,
    )  # fmt: skip
