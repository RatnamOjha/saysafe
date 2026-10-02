import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf

from saysafe.voice import io
from tests.helpers import sine


def test_wav_roundtrip(tmp_path):
    audio = sine(seconds=0.5)
    path = io.save_wav(tmp_path / "a.wav", audio)
    back = io.load_audio(path)
    assert back.dtype == np.float32 and len(back) == len(audio)
    assert np.max(np.abs(back - audio)) < 1e-3  # PCM_16 quantization only


def test_resample_and_downmix(tmp_path):
    stereo = np.stack([sine(sr=44100), sine(sr=44100)], axis=1)
    sf.write(tmp_path / "s.wav", stereo, 44100)
    audio = io.load_audio(tmp_path / "s.wav")
    assert audio.ndim == 1
    assert abs(len(audio) - io.SR) <= 1
    peak_hz = np.argmax(np.abs(np.fft.rfft(audio))) * io.SR / len(audio)
    assert abs(peak_hz - 440) < 2


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
@pytest.mark.parametrize("ext", ["mp3", "m4a", "webm"])
def test_ffmpeg_formats(tmp_path, ext):
    src = io.save_wav(tmp_path / "a.wav", sine(seconds=1.0))
    out = tmp_path / f"a.{ext}"
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(src), str(out)], check=True)
    audio = io.load_audio(out)
    assert audio.dtype == np.float32
    assert 0.9 < io.seconds(audio) < 1.2  # codecs add a little padding


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        io.load_audio("nope.wav")


def test_rms_dbfs():
    assert io.rms_dbfs(sine(amp=1.0)) == pytest.approx(-3.01, abs=0.05)
    assert io.rms_dbfs(np.zeros(100)) == -120.0
    assert io.rms_dbfs(np.zeros(0)) == -120.0
