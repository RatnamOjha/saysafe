"""Plug the demo's LLM (Groq by default, see llm.py) into saysafe's optional privacy hooks.

EARSHOT_DETECT_LLM=1   adds an LLM classifier layer to detection
EARSHOT_REWRITE_LLM=1  smooths redacted replies with an LLM
"""

from pydantic import BaseModel

from band_demo import llm
from band_demo.config import env
from saysafe.privacy.detect import CLASSIFIER_PROMPT, Classifier, Level, Verdict
from saysafe.privacy.rewrite import Smoother, smoother_prompt

CLASSIFIER_TIMEOUT_S = 1.2
SMOOTHER_TIMEOUT_S = 1.0


class _Span(BaseModel):
    text: str
    category: str


class _Verdict(BaseModel):
    level: Level
    categories: list[str] = []
    spans: list[_Span] = []


class _Smoothed(BaseModel):
    text: str


def classify(text: str) -> Verdict | None:
    result = llm.complete(
        [{"role": "system", "content": CLASSIFIER_PROMPT}, {"role": "user", "content": text}],
        model="fast",
        timeout_s=CLASSIFIER_TIMEOUT_S,
        json_schema=_Verdict,
    )
    if result is None:
        return None
    v: _Verdict = result.parsed
    return Verdict(v.level, list(v.categories), [(s.text, s.category) for s in v.spans])


def smooth(redacted: str, forbidden: list[str]) -> str | None:
    result = llm.complete(
        [{"role": "system", "content": smoother_prompt(forbidden)},
         {"role": "user", "content": redacted}],
        model="fast",
        timeout_s=SMOOTHER_TIMEOUT_S,
        json_schema=_Smoothed,
    )  # fmt: skip
    return result.parsed.text if result else None


def from_env() -> tuple[Classifier | None, Smoother | None]:
    return (
        classify if env("EARSHOT_DETECT_LLM") == "1" else None,
        smooth if env("EARSHOT_REWRITE_LLM") == "1" else None,
    )
