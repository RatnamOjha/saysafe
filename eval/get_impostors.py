"""Download LibriSpeech test-clean and keep 20 speakers x 5 utterances as impostors.

    uv run python eval/get_impostors.py

Utterances are cropped to their first 4 s so they're comparable to spoken commands.
Saved as 16 kHz wav in eval/data/librispeech/{speaker}/{utterance}.wav (gitignored).
"""

import io
import random
import sys
import tarfile
from collections import defaultdict

import httpx
import numpy as np
import soundfile as sf
from _common import DATA, LIBRISPEECH

from earshot.audio.io import save_wav

URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
SPEAKERS, PER_SPEAKER, MAX_S = 20, 5, 4.0


def download(dest):
    if dest.exists() and dest.stat().st_size > 300_000_000:
        print(f"already downloaded: {dest}")
        return
    tmp = dest.with_suffix(".part")
    with httpx.stream("GET", URL, follow_redirects=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done / 1e6:.0f} / {total / 1e6:.0f} MB", end="", flush=True)
    print()
    tmp.rename(dest)


def main() -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    archive = DATA / "test-clean.tar.gz"
    print(f"Downloading {URL}")
    download(archive)

    by_speaker: dict[str, list[str]] = defaultdict(list)
    with tarfile.open(archive) as tar:
        names = [m.name for m in tar.getmembers() if m.name.endswith(".flac")]
    for name in names:
        by_speaker[name.split("/")[-3]].append(name)
    rng = random.Random(7)
    speakers = sorted(rng.sample(sorted(by_speaker), SPEAKERS))
    wanted = {n for s in speakers for n in sorted(rng.sample(by_speaker[s], PER_SPEAKER))}

    kept = 0
    with tarfile.open(archive) as tar:
        for member in tar:
            if member.name not in wanted:
                continue
            audio, sr = sf.read(io.BytesIO(tar.extractfile(member).read()), dtype="float32")
            assert sr == 16000
            audio = audio[: int(MAX_S * sr)].astype(np.float32)
            speaker = member.name.split("/")[-3]
            save_wav(LIBRISPEECH / speaker / (member.name.split("/")[-1][:-5] + ".wav"), audio)
            kept += 1
    print(f"Saved {kept} clips from {len(speakers)} speakers to {LIBRISPEECH}")
    return 0 if kept == SPEAKERS * PER_SPEAKER else 1


if __name__ == "__main__":
    sys.exit(main())
