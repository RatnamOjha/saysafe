"""Redact flagged spans so the useful part can still be spoken.

1. Span rewrite: each flagged span becomes a placeholder ("your code", "the amount",
   "the address"); after "is" it becomes "on your phone". Health words are dropped,
   and "with Dr. X" becomes "appointment".
2. Optional LLM smoothing (EARSHOT_REWRITE_LLM=1), told which spans must not appear.
3. Check: run the detector on the result. If it's still too sensitive for this
   audience, any original span text survived, or the grammar broke, fall back to
   "Got it. I sent the details to your phone."
"""

import re
from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel

from earshot import llm
from earshot.config import env
from earshot.privacy.detect import Detection, Level, Span, detect

FALLBACK = "Got it. I sent the details to your phone."
DETAILS = "Details are on your phone."

_PLACEHOLDER = {
    "otp_code": "your code",
    "password": "your password",
    "card_number": "your card",
    "ssn": "your number",
    "account_number": "your account number",
    "money_personal": "the amount",
    "street_address": "the address",
    "email_sender": "someone",
    "legal": "",
    "health": "",
}
# After deleting "dermatologist" from "your dermatologist appointment", these nouns
# still read fine; anything else means the sentence broke.
_NOUNS_AFTER_DELETE = {
    "appointment", "appointments", "results", "result", "visit", "test", "prescription",
    "refill", "hearing", "case", "meeting", "call", "checkup", "check", "consultation",
}  # fmt: skip
_UNFIXABLE = "\x00"

_BROKEN = re.compile(
    r"\b(?:your|the|a|an|my|with|take|of|about|to|for)\s*[.,:;?!]|^\s*[.,:;]|\s{2,}", re.I
)


@dataclass
class Rewrite:
    text: str
    step: str  # span | llm | fallback
    reasons: list[str]


def span_rewrite(text: str, spans: list[Span]) -> str:
    out = text
    for s in sorted(spans, key=lambda s: s.start, reverse=True):
        before, after = out[: s.start], out[s.end :]
        if s.category == "health" and s.text.lower().startswith("dr"):
            m = re.search(r"\bwith\s+$", before, re.I)
            if m:
                before = before[: m.start()]
                repl = "" if re.search(r"\bappointment\b", out, re.I) else "appointment "
                out = before + repl + after.lstrip()
                continue
        if s.category == "money_personal" and (m := re.search(r"\s+of\s+$", before)):
            out = before[: m.start()] + after  # "your rent of $2,400 is due" -> "your rent is due"
            continue
        repl = _PLACEHOLDER.get(s.category, "that")
        if repl and re.search(r"\b(?:is|are|was|:)\s*$", before, re.I):
            repl = "on your phone"
        if not repl:
            after = after.lstrip()
            next_word = re.match(r"[a-z]+", after.lower())
            if not next_word or next_word.group(0) not in _NOUNS_AFTER_DELETE:
                # Deleting a word only reads cleanly before a safe noun ("your [dermatologist]
                # appointment"). Anything else ("your lawyer called", "take 40 mg") gives up.
                return _UNFIXABLE
        out = before + repl + after
    return _tidy(out)


def _tidy(text: str) -> str:
    text = re.sub(r"\s+([.,:;?!])", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text).strip()
    return text[:1].upper() + text[1:]


def with_details(text: str) -> str:
    if "on your phone" in text.lower():
        return text
    return f"{text.rstrip()} {DETAILS}"


class _Smoothed(BaseModel):
    text: str


def llm_smooth(redacted: str, forbidden: list[str]) -> str | None:
    result = llm.complete(
        [
            {
                "role": "system",
                "content": (
                    "Rewrite this assistant reply so it sounds natural when spoken, keeping "
                    "its meaning. It has been redacted; the full details are on the user's "
                    "phone. Never include any of these strings or their values: "
                    f"{forbidden}. Keep it under 20 words."
                ),
            },
            {"role": "user", "content": redacted},
        ],
        model="fast",
        timeout_s=1.0,
        json_schema=_Smoothed,
    )
    return result.parsed.text if result else None


def rewrite(
    text: str,
    detection: Detection,
    speakable: Callable[[Level], bool],
) -> Rewrite:
    """speakable(level) says whether a reply of that level may be spoken to this audience."""
    reasons: list[str] = []
    forbidden = [s.text for s in detection.spans]
    if not detection.spans:
        return Rewrite(FALLBACK, "fallback", ["Nothing specific to redact"])

    redacted = span_rewrite(text, detection.spans)
    if redacted == _UNFIXABLE:
        return Rewrite(FALLBACK, "fallback", ["Couldn't redact without breaking the sentence"])
    candidate, step = with_details(redacted), "span"
    if env("EARSHOT_REWRITE_LLM") == "1":
        smoothed = llm_smooth(candidate, forbidden)
        if smoothed:
            candidate, step = smoothed, "llm"
        else:
            reasons.append("LLM smoothing unavailable, kept span rewrite")

    lowered = candidate.lower()
    if any(f.lower() in lowered for f in forbidden if f):
        return Rewrite(FALLBACK, "fallback", [*reasons, "A flagged span survived the rewrite"])
    recheck = detect(candidate)  # no source tags: judge only the words being spoken
    if not speakable(recheck.level):
        return Rewrite(FALLBACK, "fallback", [*reasons, f"Rewrite still {recheck.level}"])
    if _BROKEN.search(candidate.replace(DETAILS, "")):
        return Rewrite(FALLBACK, "fallback", [*reasons, "Rewrite didn't read cleanly"])
    return Rewrite(candidate, step, reasons)
