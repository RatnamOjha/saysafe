"""Loads config/*.yaml and .env into typed settings. Single place tunable numbers are read."""

import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"

# Real environment wins over .env, so tests and CI can override anything.
load_dotenv(ROOT / ".env", override=False)


def env(name: str, default: str | None = None) -> str | None:
    """Env var, treating empty strings (as in .env.example) as unset."""
    value = os.environ.get(name, "").strip()
    return value or default


def cache_dir() -> Path:
    """Where downloaded models live. Shared by every worktree."""
    return Path(env("EARSHOT_CACHE_DIR", str(Path.home() / ".cache" / "earshot"))).expanduser()


@lru_cache
def load_yaml(name: str) -> dict:
    """config/<name>.yaml as a dict. An empty file gives {}."""
    with open(CONFIG_DIR / f"{name}.yaml") as f:
        return yaml.safe_load(f) or {}
