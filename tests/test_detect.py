from pathlib import Path

import pytest
import yaml

from saysafe.privacy.detect import Verdict, detect, luhn_ok

CASES = yaml.safe_load((Path(__file__).parent / "data" / "sensitivity_cases.yaml").read_text())


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


def test_classifier_failure_with_nothing_else_is_personal():
    failing = lambda text: None  # noqa: E731
    assert detect("The meeting moved.", classifier=failing).level == "personal"
    assert detect("It's sunny.", {"public"}, classifier=failing).level == "personal"
    assert detect("Your code is 1234.", classifier=failing).level == "secret"  # regex fired


def test_classifier_adds_level_and_spans():
    def classifier(text):
        return Verdict("sensitive", ["health"], [("the lump", "health"), ("not in text", "x")])

    assert detect("Ask the nurse about the lump on Monday.").level == "public"  # rules miss it
    got = detect("Ask the nurse about the lump on Monday.", classifier=classifier)
    assert got.level == "sensitive" and "llm" in got.latency_ms
    assert [(s.text, s.source) for s in got.spans] == [("the lump", "llm")]


def test_no_classifier_means_no_llm_layer():
    assert "llm" not in detect("Ask about the rash on Monday.").latency_ms
