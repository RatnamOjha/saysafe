"""Voice ID on public data: LibriSpeech test-clean (40 speakers, read audiobooks, CC BY 4.0).

    uv run python bench/voice_librispeech.py            # downloads 346 MB once

Each speaker is enrolled from 3 utterances (3 s of speech each, via VoiceID.enroll). Eight
other utterances per speaker are cut to 0.5, 1, 2 and 3 s of speech: a reply like "yes" is
about 0.5 s, a command 2-3 s. Every cut is scored against every profile.

Then end to end through ApprovalGuard (voice tier, a reply that says "yes"): the owner,
and a stranger whose own voice says both the command and the reply. Thresholds are the
packaged ones and ones calibrated with saysafe.calibrate on half the speakers; the
end-to-end numbers come from the other half.

Clean read speech in a quiet room is close to the best case. A wearable mic in a noisy
room will do worse: calibrate on your own recordings.
"""

from __future__ import annotations

import argparse
import io
import json
import platform
import random
import sys
import tarfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
from _common import machine

from saysafe import Action, ApprovalGuard, Thresholds, calibrate, load_thresholds
from saysafe.voice import TooShort, VoiceID, vad

URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
CACHE = Path.home() / ".cache" / "saysafe" / "bench"
SR = 16000
SEED = 7
ENROLL, TEST = 3, 8
CUTS = (0.5, 1.0, 2.0, 3.0)
COMMAND_CUT = 3.0


def tarball(path: Path | None) -> Path:
    if path is not None:
        return path
    dest = CACHE / "test-clean.tar.gz"
    if dest.exists() and dest.stat().st_size > 300_000_000:
        return dest
    import httpx

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    print(f"Downloading {URL} (346 MB)...", flush=True)
    with httpx.stream("GET", URL, follow_redirects=True, timeout=60) as r, open(tmp, "wb") as f:
        r.raise_for_status()
        for chunk in r.iter_bytes(1 << 20):
            f.write(chunk)
    tmp.rename(dest)
    return dest


def load_speech(tar_path: Path) -> dict[str, list[np.ndarray]]:
    """speaker -> speech-only audio of ENROLL + TEST utterances with >= 3 s of speech."""
    by_speaker: dict[str, list[tuple[str, bytes]]] = defaultdict(list)
    with tarfile.open(tar_path, "r:gz") as tar:
        for m in tar:
            if m.isfile() and m.name.endswith(".flac"):
                speaker = m.name.split("/")[-3]
                by_speaker[speaker].append((m.name, tar.extractfile(m).read()))
    rng = random.Random(SEED)
    out: dict[str, list[np.ndarray]] = {}
    for speaker in sorted(by_speaker):
        files = sorted(by_speaker[speaker])
        rng.shuffle(files)
        kept = []
        for _, data in files:
            audio, sr = sf.read(io.BytesIO(data), dtype="float32")
            assert sr == SR
            speech = vad.speech_only(audio)
            if len(speech) >= COMMAND_CUT * SR:
                kept.append(speech)
            if len(kept) == ENROLL + TEST:
                break
        out[speaker] = kept
    return out


def eer(genuine: list[float], impostor: list[float]) -> float:
    g, i = np.asarray(genuine), np.asarray(impostor)
    best = (2.0, 0.0)
    for t in np.unique(np.concatenate([g, i])):
        far, frr = float(np.mean(i >= t)), float(np.mean(g < t))
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return round(best[1], 4)


def rate(count: int, n: int) -> dict:
    return {"count": count, "n": n, "rate": round(count / n, 4) if n else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", type=Path, help="an existing test-clean.tar.gz")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "results" / "voice.json")
    args = ap.parse_args()

    t_start = time.perf_counter()
    speech = load_speech(tarball(args.tar))
    speakers = [s for s, clips in speech.items() if len(clips) == ENROLL + TEST]
    print(f"{len(speakers)} speakers with {ENROLL + TEST} usable utterances", flush=True)

    vid = VoiceID()
    vid.warm()
    profiles = {}
    for s in speakers:
        profiles[s] = vid.enroll(s, [c[: int(3 * SR)] for c in speech[s][:ENROLL]], min_clips=3)

    # embeddings of every test cut: emb[speaker][cut] -> list of vectors (None = too short)
    emb: dict[str, dict[float, list]] = {s: {} for s in speakers}
    score_ms: dict[float, list[float]] = defaultdict(list)
    for s in speakers:
        for cut in CUTS:
            vectors = []
            for clip in speech[s][ENROLL:]:
                t0 = time.perf_counter()
                try:
                    vectors.append(vid.embed(clip[: int(cut * SR)], min_speech_s=0.2).vector)
                except TooShort:  # the second VAD pass can trim a short cut further
                    vectors.append(None)
                score_ms[cut].append((time.perf_counter() - t0) * 1000)
            emb[s][cut] = vectors
    print("embedded", flush=True)

    def cos(v, s: str) -> float | None:
        return None if v is None else float(np.dot(v, profiles[s].mean))

    per_cut = []
    packaged = load_thresholds()
    for cut in CUTS:
        genuine = [cos(v, s) for s in speakers for v in emb[s][cut]]
        impostor = [cos(v, s) for s in speakers for o in speakers if o != s for v in emb[o][cut]]
        g = [x for x in genuine if x is not None]
        i = [x for x in impostor if x is not None]
        per_cut.append({
            "speech_s": cut,
            "genuine_trials": len(genuine), "impostor_trials": len(impostor),
            "no_score": rate(len(genuine) + len(impostor) - len(g) - len(i),
                             len(genuine) + len(impostor)),
            "eer": eer(g, i),
            "owner_accepted_at_packaged": rate(sum(x >= packaged.t_accept for x in g), len(genuine)),
            "stranger_accepted_at_packaged": rate(sum(x >= packaged.t_accept for x in i), len(impostor)),
            "score_ms_p50": round(float(np.median(score_ms[cut])), 1),
        })  # fmt: skip

    # calibrate on half the speakers, test end to end on the other half
    dev, test = speakers[0::2], speakers[1::2]
    reply_cut = 1.0
    dev_g = [x for s in dev for v in emb[s][reply_cut] if (x := cos(v, s)) is not None]
    dev_i = [x for s in dev for o in dev if o != s for v in emb[o][reply_cut]
             if (x := cos(v, s)) is not None]  # fmt: skip
    calibrated = calibrate(dev_g, dev_i)

    def end_to_end(t: Thresholds, reply: float) -> dict:
        guard = ApprovalGuard(b"bench" * 8, thresholds=t)
        action = Action(type="order_food", counterparty="DoorDash", amount=20)
        owner = stranger = n_owner = n_stranger = 0
        for s in test:
            for k in range(TEST):
                r = cos(emb[s][reply][k], s)
                c = cos(emb[s][COMMAND_CUT][(k + 1) % TEST], s)  # another of their utterances
                step = guard.reply(guard.start(action), "yes", voice_score=r, command_score=c)
                owner += step.status == "approved"
                n_owner += 1
            for o in test:
                if o == s:
                    continue
                for k in range(2):  # two attempts per stranger per profile
                    r = cos(emb[o][reply][k], s)
                    c = cos(emb[o][COMMAND_CUT][(k + 1) % TEST], s)
                    step = guard.reply(guard.start(action), "yes", voice_score=r, command_score=c)
                    stranger += step.status == "approved"
                    n_stranger += 1
        return {"owner_approved": rate(owner, n_owner),
                "stranger_approved": rate(stranger, n_stranger)}  # fmt: skip

    e2e = {}
    for name, t in (("packaged", packaged), ("calibrated", calibrated)):
        for reply in (0.5, 1.0):
            e2e[f"{name}, reply {reply} s"] = {
                "t_accept": t.t_accept, "t_reject": t.t_reject, **end_to_end(t, reply),
            }  # fmt: skip

    result = {
        "generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
        "data": "LibriSpeech test-clean (CC BY 4.0), "
        f"{len(speakers)} speakers, {ENROLL} enrollment + {TEST} test utterances each, seed {SEED}",
        "model": "speechbrain/spkrec-ecapa-voxceleb + Silero VAD, CPU",
        "machine": machine(),
        "python": platform.python_version(),
        "per_cut": per_cut,
        "calibrated_on": f"{len(dev)} dev speakers, {reply_cut} s cuts "
        f"({len(dev_g)} owner, {len(dev_i)} stranger scores)",
        "calibrated": {"t_accept": calibrated.t_accept, "t_reject": calibrated.t_reject},
        "end_to_end_on": f"{len(test)} other speakers, voice tier, reply 'yes', "
        f"command {COMMAND_CUT} s, both from the same person",
        "end_to_end": e2e,
        "runtime_s": round(time.perf_counter() - t_start, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
