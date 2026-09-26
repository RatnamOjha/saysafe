"""Shared paths and helpers for the eval scripts. Recorded data never leaves eval/data/."""

import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Overridable so the eval scripts can be smoke-tested on synthetic data.
DATA = Path(os.environ.get("EARSHOT_EVAL_DATA", ROOT / "eval" / "data"))
VOICES = DATA / "voices"
LIBRISPEECH = DATA / "librispeech"
NOISE = DATA / "noise"
REPORTS = Path(os.environ.get("EARSHOT_EVAL_REPORTS", ROOT / "eval" / "reports"))
LATEST = REPORTS / "latest"
MANIFEST = VOICES / "manifest.csv"
FIELDS = ["pid", "imitator", "condition", "n", "text", "kind", "duration_s", "speech_s",
          "rms_dbfs", "peak", "mic", "file"]  # fmt: skip


def read_manifest(path: Path = MANIFEST) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_manifest(rows: list[dict], path: Path = MANIFEST) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
