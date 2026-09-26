import sys
import types

import numpy as np
import pytest

SR = 16000
FIXTURE_TEXT = "Please order my usual from DoorDash and send it to my home address."


def sine(freq: float = 440.0, seconds: float = 1.0, sr: int = SR, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


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
    """Keep tests away from real state: audit log, nonce DB, voice profiles."""
    from earshot.approvals import hook, tokens

    monkeypatch.setenv("EARSHOT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("EARSHOT_PROFILES_DIR", str(tmp_path / "profiles"))
    monkeypatch.setenv("EARSHOT_OWNER", "nobody")
    monkeypatch.setenv("EARSHOT_SECRET", "test-secret")
    from cryptography.fernet import Fernet

    monkeypatch.setenv(
        "EARSHOT_PROFILE_KEY", Fernet.generate_key().decode()
    )  # never the real keychain
    monkeypatch.delenv("NTFY_TOPIC", raising=False)
    tokens.default_service.cache_clear()
    monkeypatch.setattr(hook, "_approver", None)
    _drop_unloadable_lazy_modules()
    yield
    tokens.default_service.cache_clear()


@pytest.fixture(scope="session")
def piper_speech() -> np.ndarray:
    """Real synthesized speech (16 kHz) for VAD/STT/embedding tests. Needs models."""
    from earshot.audio.tts import PiperTTS

    return PiperTTS().synthesize(FIXTURE_TEXT)
