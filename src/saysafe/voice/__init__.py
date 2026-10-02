"""Voice ID: speaker embeddings, enrollment, verification and VAD.

Needs the optional extra: pip install "saysafe[voice]". Audio is 16 kHz mono float32 numpy.
"""

from importlib.util import find_spec

_NEEDS = ("numpy", "scipy", "soundfile", "torch", "speechbrain", "silero_vad", "cryptography")
_missing = [name for name in _NEEDS if find_spec(name) is None]
if _missing:
    raise ImportError(
        f"saysafe.voice needs the voice extra (missing: {', '.join(_missing)}). "
        "Install it with: pip install 'saysafe[voice]'"
    )
