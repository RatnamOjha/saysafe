"""Replay attack recordings: play the owner's close clips from a speaker while the
laptop mic re-records them (condition "replay").

    uv run python eval/replay.py                    # plays through the default output
    uv run python eval/replay.py --list-devices
    uv run python eval/replay.py --output 3         # e.g. a Bluetooth speaker

Setup (printed again when you run it):
  1. Record p00 first (eval/collect.py --pid p00).
  2. Best: a Bluetooth speaker or a second laptop/phone as the output, 30-50 cm from the
     laptop mic, pointing at it. That's an attacker holding a phone up to the band.
     Using the laptop's own speaker also works but is a weaker attack.
  3. Quiet room, same place you recorded p00. Volume at a normal speaking level.
"""

import argparse

import numpy as np
from _common import VOICES, read_manifest, write_manifest
from collect import describe

from saysafe.audio import capture
from saysafe.audio.io import SR, load_audio, save_wav


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--output", help="output device index or name")
    ap.add_argument("--list-devices", action="store_true")
    args = ap.parse_args()

    import sounddevice as sd

    if args.list_devices:
        print(sd.query_devices())
        return
    rows = read_manifest()
    sources = [r for r in rows if r["pid"] == "p00" and r["condition"] == "close"]
    if not sources:
        raise SystemExit("Record p00 first: uv run python eval/collect.py --pid p00")
    print(__doc__.split("Setup")[1])
    output = int(args.output) if args.output and args.output.isdigit() else args.output
    out = sd.query_devices(output, kind="output")["name"]
    mic = capture.input_device_name()
    input(f"Playing {len(sources)} clips on '{out}', recording on '{mic}'. Enter to start.")

    rows = [r for r in rows if not (r["pid"] == "p00" and r["condition"] == "replay")]
    for i, src in enumerate(sources):
        audio = load_audio(VOICES / src["file"])
        padded = np.concatenate([audio, np.zeros(int(0.5 * SR), np.float32)])
        rec = sd.playrec(padded, SR, channels=1, dtype="float32", device=(None, output))
        sd.wait()
        rec = rec[:, 0]
        path = VOICES / "p00" / "replay" / f"{i:03d}.wav"
        save_wav(path, rec)
        rows.append({
            **{k: src[k] for k in ("pid", "imitator", "text", "kind")},
            "condition": "replay", "n": i, "mic": mic, "file": str(path.relative_to(VOICES)),
            **describe(rec),
        })  # fmt: skip
        print(f"  [{i + 1}/{len(sources)}] {src['text']}")
        write_manifest(rows)
    print("Done. Check it: uv run python eval/collect.py --report p00")


if __name__ == "__main__":
    main()
