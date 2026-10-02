"""Guided voice collection for the eval. About 5 minutes per person.

    uv run python eval/collect.py --pid p00            # you (the owner): extra held-out clips
    uv run python eval/collect.py --pid p01            # a friend
    uv run python eval/collect.py --pid p02 --imitator # a friend trying to sound like you
    uv run python eval/collect.py --delete p01         # remove someone completely
    uv run python eval/collect.py --report p01         # re-print the validation report

Clips go to eval/data/voices/{pid}/{condition}/{n}.wav with a row each in
eval/data/voices/manifest.csv. Nothing here is committed (eval/data/ is gitignored).
"""

import argparse
import re
import secrets
import shutil
import sys

import numpy as np
from _common import VOICES, read_manifest, write_manifest
from band_demo.audio import capture

from saysafe.approvals.challenge import words as challenge_pool
from saysafe.voice import vad
from saysafe.voice.io import SR, rms_dbfs, save_wav

CONSENT = (
    "This records about 30 short clips of your voice to test a voice-approval prototype.\n"
    "The audio stays on this laptop, is used only to compute totals, and gets deleted\n"
    "afterwards. Only totals (like '0 of 60 attempts approved') are published."
)
AFFIRMATIVES = [
    "yes", "yeah do it", "confirm", "go ahead", "yes please", "sure", "yep", "do it",
    "sounds good", "okay yes",
]  # fmt: skip
COMMANDS = [
    "order my usual", "send fifty dollars to Jake", "what's my verification code",
    "cancel my Netflix", "what's my bank balance", "read my last email",
    "set a reminder to call mom at six", "when is my dermatologist appointment",
    "what's the weather", "send twenty dollars to Priya",
]  # fmt: skip
CONDITIONS = {
    "close": "Sit normally, about 30 cm (one forearm) from the laptop.",
    "far": "Stand 2 to 3 m away from the laptop, facing it. Speak at a normal volume.",
}
CLIP_S = 3.0
MIN_SPEECH_S = 0.25  # below this we assume nobody spoke and record again


def challenge_words(k: int) -> list[str]:
    pool = list(challenge_pool())
    return [pool.pop(secrets.randbelow(len(pool))) for _ in range(k)]


def script(pid: str) -> list[tuple[str, str, str]]:
    """(condition, kind, text) for every clip, in recording order."""
    words = challenge_words(5)
    base = [("affirm", t) for t in AFFIRMATIVES] + [("challenge", w) for w in words]
    clips = [(c, kind, text) for c in CONDITIONS for kind, text in base]
    if pid == "p00":  # held-out clips for calibration and testing
        extra_close = [("affirm", t) for t in AFFIRMATIVES] + [("command", t) for t in COMMANDS]
        extra_far = [("affirm", t) for t in AFFIRMATIVES]
        clips += [("close", k, t) for k, t in extra_close] + [("far", k, t) for k, t in extra_far]
    return clips


def describe(audio: np.ndarray) -> dict:
    speech = vad.speech_only(audio)
    return {
        "duration_s": round(len(audio) / SR, 2),
        "speech_s": round(len(speech) / SR, 2),
        "rms_dbfs": round(rms_dbfs(speech if len(speech) else audio), 1),
        "peak": round(float(np.max(np.abs(audio))) if len(audio) else 0.0, 3),
    }


def collect(pid: str, imitator: bool) -> None:
    rows = [r for r in read_manifest() if r["pid"] != pid]
    if (VOICES / pid).exists():
        if input(f"{pid} already has clips. Record again from scratch? [y/N] ").lower() != "y":
            return
        shutil.rmtree(VOICES / pid)

    print(CONSENT)
    if input("\nDo you agree? Type yes: ").strip().lower() != "yes":
        print("No problem. Nothing was recorded.")
        return
    mic = capture.input_device_name()
    clips = script(pid)
    print(f"\n{len(clips)} clips of {CLIP_S:.0f} s from '{mic}'. Press Enter, then say the words.")
    if imitator:
        print("You're an imitator: try to sound as much like the owner as you can.")

    condition = None
    for n, (cond, kind, text) in enumerate(clips):
        if cond != condition:
            condition = cond
            input(f"\n== {cond.upper()}: {CONDITIONS[cond]}\nPress Enter when you're in place.")
        while True:
            input(f'[{n + 1}/{len(clips)}] Say: "{text}"  (Enter)')
            print("   🎙  recording...")
            audio = capture.record(CLIP_S)
            info = describe(audio)
            if info["speech_s"] >= MIN_SPEECH_S:
                break
            print(
                f"   Didn't hear anything (level {rms_dbfs(audio):.0f} dBFS). Once more, out loud."
            )
        path = VOICES / pid / cond / f"{n:03d}.wav"
        save_wav(path, audio)
        rows.append({
            "pid": pid, "imitator": int(imitator), "condition": cond, "n": n, "text": text,
            "kind": kind, "mic": mic, "file": str(path.relative_to(VOICES)), **info,
        })  # fmt: skip
        write_manifest(rows)  # save as we go, so a crash loses nothing
    print("\nDone. Thank you!\n")
    report(pid)


def report(pid: str) -> bool:
    rows = [r for r in read_manifest() if r["pid"] == pid]
    if not rows:
        print(f"No clips for {pid}.")
        return False
    print(f"Validation report for {pid} ({len(rows)} clips, imitator={rows[0]['imitator']})")
    ok = True
    for cond in sorted({r["condition"] for r in rows}):
        sub = [r for r in rows if r["condition"] == cond]
        print(f"  {cond:8} {len(sub):3} clips   median speech "
              f"{np.median([float(r['speech_s']) for r in sub]):.2f} s   median level "
              f"{np.median([float(r['rms_dbfs']) for r in sub]):.0f} dBFS")  # fmt: skip
    checks = [
        ("under 0.8 s of speech (can't be scored as a command)",
         [r for r in rows if float(r["speech_s"]) < 0.8 and r["kind"] == "command"]),
        ("under 0.35 s of speech (can't be scored at all)",
         [r for r in rows if float(r["speech_s"]) < 0.35]),
        ("clipping (peak >= 0.99)", [r for r in rows if float(r["peak"]) >= 0.99]),
        ("very quiet (speech under -45 dBFS)", [r for r in rows if float(r["rms_dbfs"]) < -45]),
    ]  # fmt: skip
    for label, bad in checks:
        if bad:
            ok = False
            files = ", ".join(r["file"] for r in bad[:6]) + (" ..." if len(bad) > 6 else "")
            print(f"  ! {len(bad)} {label}: {files}")
    print("  Clean." if ok else "  Consider re-recording the flagged clips (or the person).")
    return ok


def delete(pid: str) -> None:
    if not re.fullmatch(r"p\d{2}", pid):
        sys.exit("pid must look like p01")
    shutil.rmtree(VOICES / pid, ignore_errors=True)
    write_manifest([r for r in read_manifest() if r["pid"] != pid])
    print(f"Deleted every clip and manifest row for {pid}.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--pid", help="p00 is the owner; p01, p02, ... for everyone else")
    ap.add_argument("--imitator", action="store_true", help="this person imitates the owner")
    ap.add_argument("--delete", metavar="PID", help="remove someone's clips completely")
    ap.add_argument("--report", metavar="PID", help="print the validation report")
    args = ap.parse_args()
    if args.delete:
        delete(args.delete)
    elif args.report:
        report(args.report)
    elif args.pid and re.fullmatch(r"p\d{2}", args.pid):
        collect(args.pid, args.imitator)
    else:
        ap.error("give --pid p00 (owner) or p01, p02 ...")


if __name__ == "__main__":
    main()
