"""Demo settings: the repo's .env (loaded in band_demo/__init__.py), demo.yaml, and the
shared objects the library leaves to the app (token secret, nonce DB, profiles, audit log).

The library (saysafe) never reads any of this; the demo passes it in explicitly.
"""

import os
from functools import lru_cache
from pathlib import Path

import yaml

from saysafe import Action, ApprovalGuard, AuditLog, Thresholds, load_thresholds

ROOT = Path(__file__).resolve().parents[4]  # the repo
CONFIG_DIR = ROOT / "config"  # repo-level overrides, e.g. eval's thresholds.calibrated.yaml
CALIBRATED = CONFIG_DIR / "thresholds.calibrated.yaml"


def env(name: str, default: str | None = None) -> str | None:
    """Env var, treating empty strings (as in .env.example) as unset."""
    value = os.environ.get(name, "").strip()
    return value or default


def cache_dir() -> Path:
    """Where downloaded models live (ECAPA, Whisper, Piper)."""
    return Path(env("EARSHOT_CACHE_DIR", str(Path.home() / ".cache" / "saysafe"))).expanduser()


def data_dir() -> Path:
    """Nonce DB and audit log."""
    return Path(env("EARSHOT_DATA_DIR", str(ROOT / "data"))).expanduser()


@lru_cache
def demo_config() -> dict:
    return yaml.safe_load((Path(__file__).parent / "demo.yaml").read_text())


def thresholds() -> Thresholds:
    """Calibrated thresholds when the eval has produced them, else saysafe's placeholders."""
    return load_thresholds(CALIBRATED if CALIBRATED.exists() else None)


def profile_store():
    """Voice profiles: EARSHOT_PROFILES_DIR (default ./profiles), key from EARSHOT_PROFILE_KEY
    or the OS keychain (service "earshot", so profiles enrolled before the split still open)."""
    from saysafe.voice.profile_store import ProfileStore, keyring_key

    key = env("EARSHOT_PROFILE_KEY")
    directory = Path(env("EARSHOT_PROFILES_DIR", str(ROOT / "profiles"))).expanduser()
    return ProfileStore(directory, key.encode() if key else keyring_key(service="earshot"))


@lru_cache
def approvals() -> ApprovalGuard:
    """One per process, shared by the approvals hook, the executor and the phone routes."""
    secret = env("EARSHOT_SECRET")
    if not secret:
        # Fail closed: tokens from a random secret die with this process.
        import logging
        import secrets

        logging.getLogger(__name__).warning("EARSHOT_SECRET not set, using a per-process secret")
        key = secrets.token_bytes(32)
    else:
        key = secret.encode()
    return ApprovalGuard(
        key,
        nonces=data_dir() / "nonces.sqlite",
        audit=AuditLog(data_dir() / "audit.jsonl"),
        thresholds=thresholds(),
    )


def verify_token(token: str, action: Action) -> None:
    approvals().verify(token, action)
