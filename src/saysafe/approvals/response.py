"""Classify a reply: affirm / negate / challenge_ok / challenge_wrong / unclear.

Negation always wins ("yes, no wait" is a no). Challenge words match fuzzily, since
STT mangles rare words: within one edit, or the same metaphone code. Saying a
different word from the list (an old challenge, a replay) is challenge_wrong.
"""

import re
import time
from dataclasses import dataclass
from typing import Literal

import jellyfish
from rapidfuzz.distance import Levenshtein

from saysafe.approvals.challenge import Challenge, words

Kind = Literal["affirm", "negate", "challenge_ok", "challenge_wrong", "unclear"]

_FILLERS = re.compile(r"\b(?:okay so|ok so|uh+|um+|er+|hmm+|so|well|okay|ok|oh)\b")
_NEGATE = re.compile(
    r"\b(?:no|nope|nah|not|don't|dont|do not|cancel|stop|wait|hold on|never ?mind|abort|"
    r"negative|wrong)\b"
)
_AFFIRM = re.compile(
    r"\b(?:yes|yeah|yep|yup|yea|sure|confirm|confirmed|do it|go ahead|sounds good|please do|"
    r"correct|affirmative)\b"
)


@dataclass(frozen=True)
class Match:
    kind: Kind
    detail: str = ""


def normalize(transcript: str) -> str:
    t = transcript.lower().replace("’", "'")
    t = re.sub(r"[^a-z' ]+", " ", t)
    t = _FILLERS.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def _similar(token: str, word: str) -> str | None:
    """How token matches word, or None. Short tokens only match exactly."""
    if token == word:
        return "exact"
    if len(token) < 4:
        return None
    if Levenshtein.distance(token, word) <= 1:
        return "edit"
    if jellyfish.metaphone(token) == jellyfish.metaphone(word):
        return "metaphone"
    return None


def _candidates(t: str) -> list[str]:
    """Tokens plus adjacent pairs joined ('sun flower' -> 'sunflower')."""
    toks = t.split()
    return toks + [a + b for a, b in zip(toks, toks[1:], strict=False)]


def match_response(
    transcript: str,
    tier: str,
    challenge: Challenge | None = None,
    now: float | None = None,
) -> Match:
    t = normalize(transcript)
    if not t:
        return Match("unclear", "nothing said")
    if m := _NEGATE.search(t):
        return Match("negate", f"heard {m.group(0)!r}")

    if tier == "voice_challenge" and challenge is not None:
        now = time.time() if now is None else now
        for tok in _candidates(t):
            how = _similar(tok, challenge.word)
            if how:
                status = challenge.status(now)
                if status != "valid":
                    return Match("challenge_wrong", f"right word but {status}")
                return Match("challenge_ok", f"{tok!r} ~ {challenge.word!r} ({how})")
        for tok in _candidates(t):
            for w in words():
                if w != challenge.word and _similar(tok, w):
                    return Match("challenge_wrong", f"said another list word {w!r}")

    if m := _AFFIRM.search(t):
        return Match("affirm", f"heard {m.group(0)!r}")
    return Match("unclear", "no yes, no, or challenge word")  # never echo the transcript
