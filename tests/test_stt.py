import pytest
from rapidfuzz import fuzz

from tests.conftest import FIXTURE_TEXT


@pytest.mark.models
@pytest.mark.parametrize("purpose", ["command", "reply"])
def test_transcribes_piper_speech(piper_speech, purpose):
    from earshot.audio.stt import get_stt

    t = get_stt(purpose).transcribe(piper_speech)
    assert fuzz.ratio(t.text.lower(), FIXTURE_TEXT.lower()) > 85, t.text
    assert t.words and t.latency_ms > 0
