"""Fetch every model earshot uses into ~/.cache/earshot (or EARSHOT_CACHE_DIR).

Safe to rerun: anything already present is skipped. Each model is independent,
so one failure doesn't stop the others.

    uv run python scripts/download_models.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from earshot.config import cache_dir  # noqa: E402

CACHE = cache_dir()
PIPER_VOICE = "en_US-lessac-medium"


def ecapa() -> None:
    from speechbrain.inference.speaker import EncoderClassifier

    EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(CACHE / "ecapa"),
        run_opts={"device": "cpu"},
    )


def whisper(size: str) -> None:
    from faster_whisper import download_model

    download_model(size, output_dir=str(CACHE / "whisper" / size))


def silero() -> None:
    # The silero-vad package ships the model weights; loading it proves they work.
    from silero_vad import load_silero_vad

    load_silero_vad()


def piper() -> None:
    from huggingface_hub import hf_hub_download

    base = f"en/en_US/lessac/medium/{PIPER_VOICE}"
    for suffix in (".onnx", ".onnx.json"):
        hf_hub_download("rhasspy/piper-voices", f"{base}{suffix}", local_dir=str(CACHE / "piper"))


STEPS = [
    ("ECAPA speaker model", ecapa),
    ("faster-whisper base.en", lambda: whisper("base.en")),
    ("faster-whisper small.en", lambda: whisper("small.en")),
    ("silero VAD", silero),
    (f"Piper voice {PIPER_VOICE}", piper),
]


def main() -> int:
    CACHE.mkdir(parents=True, exist_ok=True)
    print(f"Cache: {CACHE}", flush=True)
    failed = []
    for name, step in STEPS:
        start = time.perf_counter()
        try:
            step()
            print(f"  ok    {name} ({time.perf_counter() - start:.1f} s)", flush=True)
        except Exception as e:
            failed.append(name)
            print(f"  FAIL  {name}: {type(e).__name__}: {e}", flush=True)
    print("All models ready." if not failed else f"Failed: {', '.join(failed)}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
