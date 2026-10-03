"""Small helpers for tests (fixtures live in the repo-root conftest.py)."""

import numpy as np

SR = 16000
FIXTURE_TEXT = "Please order my usual from DoorDash and send it to my home address."


def sine(freq: float = 440.0, seconds: float = 1.0, sr: int = SR, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
