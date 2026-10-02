from pathlib import Path

import pytest
import yaml

from saysafe.privacy.detect import detect, luhn_ok

CASES = yaml.safe_load((Path(__file__).parent / "data" / "sensitivity_cases.yaml").read_text())


@pytest.fixture(autouse=True)
def rules_only(monkeypatch):
    monkeypatch.delenv("EARSHOT_DETECT_LLM", raising=False)


def test_there_are_enough_cases():
    assert len(CASES) >= 50
    assert sum(c["level"] == "public" for c in CASES) >= 15


@pytest.mark.parametrize("case", CASES, ids=[c["text"][:40] for c in CASES])
def test_case(case):
    d = detect(case["text"], set(case.get("tags", [])))
    assert d.level == case["level"], [(s.category, s.text) for s in d.spans]
    flagged = {s.text for s in d.spans}
    for want in case.get("spans", []):
        assert want in flagged, f"{want!r} not in {flagged}"
    for s in d.spans:
        assert case["text"][s.start : s.end] == s.text


def test_latency_per_layer():
    d = detect("Your Chase code is 482913", {"otp"})
    assert set(d.latency_ms) == {"regex", "source"}


def test_luhn():
    assert luhn_ok("4111111111111111") and not luhn_ok("4111111111111112")
    assert not luhn_ok("1234")


def test_llm_failure_with_nothing_else_is_personal(monkeypatch):
    from saysafe import llm

    monkeypatch.setenv("EARSHOT_DETECT_LLM", "1")
    monkeypatch.setattr(llm, "complete", lambda *a, **k: None)
    assert detect("The meeting moved.").level == "personal"
    assert detect("It's sunny.", {"public"}).level == "personal"
    assert detect("Your code is 1234.").level == "secret"  # regex fired, no fallback needed


def test_llm_adds_spans(monkeypatch):
    from types import SimpleNamespace

    from saysafe import llm
    from saysafe.privacy import detect as d

    def fake(messages, model, timeout_s, json_schema):
        parsed = json_schema(level="sensitive", categories=["health"],
                             spans=[{"text": "the rash", "category": "health"}])  # fmt: skip
        return SimpleNamespace(parsed=parsed)

    monkeypatch.setenv("EARSHOT_DETECT_LLM", "1")
    monkeypatch.setattr(llm, "complete", fake)
    got = d.detect("Ask about the rash on Monday.")
    assert got.level == "sensitive" and "llm" in got.latency_ms
