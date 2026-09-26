"""load_audio / save_wav / rms_dbfs. ffmpeg for non-wav formats.

Every array in earshot is 16 kHz mono float32 in [-1, 1].
"""

import subprocess
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SR = 16000
_SOUNDFILE_EXTS = {".wav", ".flac", ".ogg"}


def to_mono(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    return audio.mean(axis=1, dtype=np.float32) if audio.ndim == 2 else audio


def resample(audio: np.ndarray, sr_from: int, sr_to: int = SR) -> np.ndarray:
    if sr_from == sr_to:
        return np.asarray(audio, dtype=np.float32)
    g = gcd(sr_from, sr_to)
    return resample_poly(audio, sr_to // g, sr_from // g).astype(np.float32)


def load_audio(path: str | Path) -> np.ndarray:
    """Any wav/flac/mp3/m4a/webm file as 16 kHz mono float32."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in _SOUNDFILE_EXTS:
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
        return resample(to_mono(audio), sr)
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path)]
    cmd += ["-f", "f32le", "-ac", "1", "-ar", str(SR), "-"]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise ValueError(f"ffmpeg couldn't decode {path}: {proc.stderr.decode().strip()}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def save_wav(path: str | Path, audio: np.ndarray, sr: int = SR) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.clip(audio, -1.0, 1.0), sr, subtype="PCM_16")
    return path


def rms_dbfs(audio: np.ndarray, floor: float = -120.0) -> float:
    """RMS level in dB relative to full scale. Silence returns `floor`."""
    audio = np.asarray(audio, dtype=np.float64)
    if audio.size == 0:
        return floor
    rms = np.sqrt(np.mean(audio**2))
    return float(max(20 * np.log10(rms), floor)) if rms > 0 else floor


def seconds(audio: np.ndarray, sr: int = SR) -> float:
    return len(audio) / sr
