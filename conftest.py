"""Fixtures shared by the library tests (tests/) and the demo tests (examples/band_demo/tests/)."""

import importlib.util
import sys
import types

import pytest

from tests.helpers import FIXTURE_TEXT

HAS_DEMO = importlib.util.find_spec("band_demo") is not None


def _drop_unloadable_lazy_modules() -> None:
    """speechbrain registers lazy stand-ins for optional packages (k2, ...) that raise
    ImportError on any attribute access. Hypothesis reads __file__ from every loaded
    module and crashes on them, so remove the ones that can't load."""
    for name, module in list(sys.modules.items()):
        if type(module) is types.ModuleType:
            continue  # real modules are fine; only stand-in objects misbehave
        try:
            getattr(module, "__file__", None)
        except ImportError:
            del sys.modules[name]


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Keep tests away from real state: audit log, nonce DB, voice profiles, keychain."""
    from cryptography.fernet import Fernet

    monkeypatch.setenv("EARSHOT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("EARSHOT_PROFILES_DIR", str(tmp_path / "profiles"))
    monkeypatch.setenv("EARSHOT_OWNER", "nobody")
    monkeypatch.setenv("EARSHOT_SECRET", "test-secret-test-secret-test-secret")
    monkeypatch.setenv("EARSHOT_PROFILE_KEY", Fernet.generate_key().decode())  # never the keychain
    monkeypatch.delenv("NTFY_TOPIC", raising=False)
    monkeypatch.delenv("EARSHOT_DETECT_LLM", raising=False)
    monkeypatch.delenv("EARSHOT_REWRITE_LLM", raising=False)
    if HAS_DEMO:
        from band_demo import approvals_hook, config

        config.token_service.cache_clear()
        config.pending.cache_clear()
        monkeypatch.setattr(approvals_hook, "_approver", None)
    _drop_unloadable_lazy_modules()
    yield
    if HAS_DEMO:
        config.token_service.cache_clear()
        config.pending.cache_clear()


@pytest.fixture(scope="session")
def piper_speech():
    """Real synthesized speech (16 kHz) for VAD/STT/embedding tests. Needs models + the demo."""
    tts = pytest.importorskip("band_demo.audio.tts")
    return tts.PiperTTS().synthesize(FIXTURE_TEXT)
