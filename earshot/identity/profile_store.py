"""Fernet-encrypted profile storage in EARSHOT_PROFILES_DIR.

The key comes from EARSHOT_PROFILE_KEY, or is generated once and kept in the OS
keyring (macOS Keychain). Embeddings are never printed: they're excluded from repr.
"""

import re
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from cryptography.fernet import Fernet, InvalidToken
from pydantic import BaseModel, Field

from earshot.config import ROOT, env

_KEYRING_SERVICE = "earshot"
_KEYRING_USER = "profile_key"
_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")


class ProfileError(Exception):
    """A user-facing profile problem. The message is safe to print as is."""


class Profile(BaseModel):
    name: str
    mean_embedding: list[float] = Field(repr=False)
    embeddings: list[list[float]] = Field(repr=False)
    median_rms_dbfs: float
    mic_name: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def mean(self) -> np.ndarray:
        return np.asarray(self.mean_embedding, dtype=np.float32)

    def __str__(self) -> str:
        return repr(self)


def default_dir() -> Path:
    return Path(env("EARSHOT_PROFILES_DIR", str(ROOT / "profiles"))).expanduser()


def resolve_key() -> bytes:
    """EARSHOT_PROFILE_KEY, else the keyring, else a new key saved to the keyring."""
    key = env("EARSHOT_PROFILE_KEY")
    if key:
        return key.encode()
    import keyring

    key = keyring.get_password(_KEYRING_SERVICE, _KEYRING_USER)
    if not key:
        key = Fernet.generate_key().decode()
        keyring.set_password(_KEYRING_SERVICE, _KEYRING_USER, key)
    return key.encode()


class ProfileStore:
    def __init__(self, directory: Path | None = None, key: bytes | None = None):
        self.dir = Path(directory) if directory else default_dir()
        try:
            self._fernet = Fernet(key or resolve_key())
        except ValueError as e:
            raise ProfileError("EARSHOT_PROFILE_KEY isn't a valid Fernet key.") from e

    def _path(self, name: str) -> Path:
        if not _NAME.match(name):
            raise ProfileError(f"Bad profile name {name!r}: use lowercase letters, digits, - or _.")
        return self.dir / f"{name}.profile"

    def save(self, profile: Profile) -> Path:
        path = self._path(profile.name)
        self.dir.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self._fernet.encrypt(profile.model_dump_json().encode()))
        path.chmod(0o600)
        return path

    def load(self, name: str) -> Profile:
        path = self._path(name)
        if not path.exists():
            raise ProfileError(f"No profile named {name!r} in {self.dir}. Run: earshot enroll")
        try:
            data = self._fernet.decrypt(path.read_bytes())
        except InvalidToken:
            raise ProfileError(
                f"Can't decrypt profile {name!r}: wrong EARSHOT_PROFILE_KEY or a corrupted file."
            ) from None
        return Profile.model_validate_json(data)

    def exists(self, name: str) -> bool:
        return self._path(name).exists()

    def delete(self, name: str) -> None:
        self._path(name).unlink(missing_ok=True)
