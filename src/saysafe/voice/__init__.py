"""Voice ID: enroll a voice, score audio against it, find speech, track who's around.

Needs the optional extra: pip install "saysafe[voice]". Audio is 16 kHz mono float32 numpy.

    from saysafe.voice import VoiceID, ProfileStore
"""

from importlib.util import find_spec

_NEEDS = ("numpy", "scipy", "soundfile", "torch", "speechbrain", "silero_vad", "cryptography")
_missing = [name for name in _NEEDS if find_spec(name) is None]
if _missing:
    raise ImportError(
        f"saysafe.voice needs the voice extra (missing: {', '.join(_missing)}). "
        "Install it with: pip install 'saysafe[voice]'"
    )

from saysafe.voice.embed import Embedding, TooShort  # noqa: E402
from saysafe.voice.enroll import PROMPTS  # noqa: E402
from saysafe.voice.io import SR, load_audio, save_wav  # noqa: E402
from saysafe.voice.profile_store import (  # noqa: E402
    Profile,
    ProfileError,
    ProfileStore,
    keyring_key,
)
from saysafe.voice.vad import StreamingVAD  # noqa: E402
from saysafe.voice.verify import VerifyResult  # noqa: E402
from saysafe.voice.voice_id import EnrollmentError, VoiceID  # noqa: E402

__all__ = [
    "PROMPTS",
    "SR",
    "Embedding",
    "EnrollmentError",
    "Profile",
    "ProfileError",
    "ProfileStore",
    "StreamingVAD",
    "TooShort",
    "VerifyResult",
    "VoiceID",
    "keyring_key",
    "load_audio",
    "save_wav",
]
