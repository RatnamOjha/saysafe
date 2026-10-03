"""Voice eval: calibrate thresholds on held-out owner clips vs LibriSpeech, then test on
clips never used for calibration, against friends, imitators and replays.

    uv run python eval/speaker_eval.py            # writes eval/reports/latest/speaker.json + PNGs
    uv run python eval/speaker_eval.py --no-write-thresholds

1. Enroll the owner (p00) from 6 close clips. Split the rest in half: dev and test.
2. Calibrate on dev only (owner dev clips vs LibriSpeech):
     t_accept = lowest threshold with FAR <= 1%   (few strangers get accepted)
     t_reject = highest threshold with FRR <= 1%  (the owner is almost never rejected)
   If the classes separate on dev, those two cross; the band then spans the gap:
   accept above both, reject below both, uncertain (phone tap) in between.
   Written to config/thresholds.calibrated.yaml, which the app then uses.
3. Test: owner test clips (close, far, far + cafe noise at 10 dB SNR) vs friends,
   imitators and replays. Friends are never used for calibration.
4. End to end: approval attempts through the real approvals hook with real STT.

Reports raw numbers with sample sizes, even when they look bad.
"""

import argparse
import json
import math
import os
import random
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from _common import LATEST, LIBRISPEECH, NOISE, VOICES, read_manifest
from band_demo.config import CONFIG_DIR

from saysafe.config import load_yaml
from saysafe.voice.embed import TooShort, cosine, embed, normalize
from saysafe.voice.io import SR, load_audio
from saysafe.voice.verify import Thresholds

SEED = 13
MIN_SPEECH = None  # set from config (the reply minimum) in main


@dataclass(eq=False)  # clips compare by identity (they hold numpy arrays)
class Clip:
    group: str  # owner | friend | imitator | replay | librispeech
    pid: str
    condition: str  # close | far | far_noise | replay | librispeech
    text: str
    kind: str  # affirm | challenge | command | read
    audio: np.ndarray = field(repr=False)
    vector: np.ndarray | None = field(default=None, repr=False)
    speech_s: float = 0.0
    score: float | None = None


# loading ------------------------------------------------------------------------------


def load_clips(manifest: list[dict]) -> list[Clip]:
    clips = []
    for r in manifest:
        if r["pid"] == "p00":
            group = "replay" if r["condition"] == "replay" else "owner"
        else:
            group = "imitator" if r["imitator"] in ("1", "True", "true") else "friend"
        audio = load_audio(VOICES / r["file"])
        clips.append(Clip(group, r["pid"], r["condition"], r["text"], r["kind"], audio))
    for path in sorted(LIBRISPEECH.glob("*/*.wav")):
        clips.append(Clip("librispeech", f"ls{path.parent.name}", "librispeech", "", "read",
                          load_audio(path)))  # fmt: skip
    return clips


def add_noise(
    audio: np.ndarray, noise: np.ndarray, snr_db: float, rng: random.Random
) -> np.ndarray:
    start = rng.randrange(0, max(1, len(noise) - len(audio)))
    n = np.resize(noise[start : start + len(audio)], len(audio))
    p_s, p_n = np.mean(audio**2) + 1e-12, np.mean(n**2) + 1e-12
    return (audio + n * math.sqrt(p_s / (p_n * 10 ** (snr_db / 10)))).astype(np.float32)


def embed_all(clips: list[Clip]) -> None:
    for c in clips:
        try:
            e = embed(c.audio, min_speech_s=MIN_SPEECH)
            c.vector, c.speech_s = e.vector, e.speech_seconds
        except TooShort as short:
            c.vector, c.speech_s = None, short.speech_seconds


# calibration ----------------------------------------------------------------------------


def far(impostor: np.ndarray, t: float) -> float:
    return float(np.mean(impostor >= t)) if len(impostor) else float("nan")


def frr(genuine: np.ndarray, t: float) -> float:
    return float(np.mean(genuine < t)) if len(genuine) else float("nan")


def calibrate(genuine: np.ndarray, impostor: np.ndarray, target: float = 0.01) -> dict:
    grid = np.round(np.arange(-0.2, 1.0001, 0.001), 3)
    ok_accept = [t for t in grid if far(impostor, t) <= target]
    ok_reject = [t for t in grid if frr(genuine, t) <= target]
    raw_accept, raw_reject = float(min(ok_accept)), float(max(ok_reject))
    separated = raw_reject > raw_accept
    t_accept, t_reject = max(raw_accept, raw_reject), min(raw_accept, raw_reject)
    return {
        "t_accept": round(t_accept, 3), "t_reject": round(t_reject, 3),
        "raw_accept_far1": round(raw_accept, 3), "raw_reject_frr1": round(raw_reject, 3),
        "separated_on_dev": separated,
        "dev_far_at_accept": round(far(impostor, t_accept), 4),
        "dev_frr_at_reject": round(frr(genuine, t_reject), 4),
    }  # fmt: skip


def eer(genuine: np.ndarray, impostor: np.ndarray) -> float | None:
    if not len(genuine) or not len(impostor):
        return None
    best = None
    for t in np.unique(np.concatenate([genuine, impostor])):
        a, b = far(impostor, t), frr(genuine, t)
        if best is None or abs(a - b) < best[0]:
            best = (abs(a - b), (a + b) / 2)
    return round(best[1], 4)


# charts ---------------------------------------------------------------------------------


def charts(owner_by_cond: dict, others: dict, t: Thresholds, out: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"close": "#2f9e6e", "far": "#3b6fd6", "far_noise": "#8a5cd6",
              "friend": "#d65a4a", "imitator": "#e0a030", "replay": "#6b6b6b"}  # fmt: skip
    files = []
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(-0.2, 1.0, 49)
    for name, scores in [*owner_by_cond.items(), *others.items()]:
        if len(scores):
            ax.hist(scores, bins=bins, alpha=0.55, label=f"{name} (n={len(scores)})",
                    color=colors.get(name))  # fmt: skip
    for x, label in [(t.t_reject, "t_reject"), (t.t_accept, "t_accept")]:
        ax.axvline(x, color="black", linestyle="--", linewidth=1)
        ax.text(x, ax.get_ylim()[1] * 0.95, f" {label} {x:.2f}", fontsize=9)
    ax.set_xlabel("cosine to the owner's voiceprint")
    ax.set_ylabel("clips")
    ax.set_title("Owner vs other voices (test clips only)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "score_histogram.png", dpi=150)
    files.append("score_histogram.png")
    plt.close(fig)

    impostor = np.concatenate([s for k, s in others.items() if k in ("friend", "imitator") and len(s)]
                              or [np.array([])])  # fmt: skip
    fig, ax = plt.subplots(figsize=(6, 6))
    for cond, genuine in owner_by_cond.items():
        if not len(genuine) or not len(impostor):
            continue
        ts = np.unique(np.concatenate([genuine, impostor, [1.1]]))
        ax.plot([100 * far(impostor, x) for x in ts], [100 * frr(genuine, x) for x in ts],
                label=f"owner {cond}", color=colors.get(cond))  # fmt: skip
    ax.set_xlabel("False accepts: other people accepted (%)")
    ax.set_ylabel("False rejects: owner not accepted (%)")
    ax.set_title("DET: owner vs friends + imitators (test)")
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "det_curve.png", dpi=150)
    files.append("det_curve.png")
    plt.close(fig)
    return files


# end to end through the approvals hook ------------------------------------------------------


class _Scorer:
    def __init__(self, mean: np.ndarray):
        self.mean = mean

    def score_audio(self, audio, min_speech_s=None):
        from saysafe.voice.verify import verify

        profile = type("P", (), {"mean": self.mean})()
        from functools import partial

        return verify(audio, profile, embedder=partial(embed, min_speech_s=min_speech_s))

    def score_embedding(self, e):
        return cosine(e.vector, self.mean)


class _ForcedIssuer:
    """Issues a chosen word, so a recorded clip can be the 'right' or an 'old' word."""

    def __init__(self, clock):
        from saysafe.approvals.challenge import ChallengeIssuer

        self._inner = ChallengeIssuer(clock=clock)
        self.word = "zebra"

    def issue(self, action_id):
        c = self._inner.issue(action_id)
        c.word = self.word
        return c

    def consume(self, c):
        self._inner.consume(c)


class _TextSTT:
    """For LibriSpeech: their audio, with the transcript forced to 'yes'."""

    def __init__(self, real, forced: str | None):
        self.real, self.forced = real, forced

    def transcribe(self, audio):
        from band_demo.audio.stt import Transcript

        return (
            Transcript(self.forced, [], 0.0, "forced")
            if self.forced
            else self.real.transcribe(audio)
        )


def run_attempts(
    attempts: list[dict], profile_mean: np.ndarray, t: Thresholds
) -> tuple[list, list]:
    """Each attempt: {group, condition, tier, reply: Clip, command: Clip|None, word, forced_text}."""
    from band_demo.agent.channels import PhoneChannel
    from band_demo.agent.events import EventBus
    from band_demo.agent.mock_agent import MockAgent
    from band_demo.approvals_hook import Approver
    from band_demo.audio.stt import get_stt

    from saysafe import ApprovalGuard, AuditLog
    from saysafe.privacy.audience import FixedAudience
    from saysafe.voice.embed import Embedding

    tmp = Path(tempfile.mkdtemp())
    guard = ApprovalGuard(b"eval" * 8, nonces=tmp / "n.sqlite", thresholds=t,
                          audit=AuditLog(tmp / "audit.jsonl"))  # fmt: skip
    issuer = guard.issuer = _ForcedIssuer(time.time)
    stt = _TextSTT(get_stt("reply"), None)
    approver = Approver(stt=stt, scorer=_Scorer(profile_mean), guard=guard)
    agent = MockAgent()
    events = EventBus()
    phone = PhoneChannel(events, console_only=True)
    results, latencies = [], []

    for a in attempts:
        action = agent.order_usual() if a["tier"] == "voice" else agent.send_money("Jake", 50)
        issuer.word = a.get("word") or "zebra"
        stt.forced = a.get("forced_text")
        cmd = a.get("command")
        emb = None
        if cmd is not None and cmd.vector is not None:
            emb = Embedding(cmd.vector, cmd.speech_s, 0.0)
        ctx = type("Ctx", (), {})()
        ctx.speak = lambda text: None
        ctx.listen = lambda timeout, audio=a["reply"].audio: audio
        ctx.command_embedding = emb
        ctx.phone, ctx.events, ctx.flags = phone, events, {}
        ctx.complete_approved = lambda action, token: None
        ctx.audience = FixedAudience(
            "alone_likely"
        )  # read-backs are spoken; privacy is eval'd elsewhere
        t0 = time.perf_counter()
        d = approver.before_execute(action, ctx)
        latencies.append((time.perf_counter() - t0) * 1000)
        results.append({"group": a["group"], "condition": a["condition"], "tier": a["tier"],
                        "case": a.get("case", ""), "outcome": d.outcome})  # fmt: skip
    return results, latencies


def tally(results: list[dict], **where) -> dict:
    rows = [r for r in results if all(r[k] == v for k, v in where.items())]
    out = {"n": len(rows)}
    for o in ("approve", "step_up", "reject"):
        out[o] = sum(r["outcome"] == o for r in rows)
    return out


# main ------------------------------------------------------------------------------------


def main() -> int:
    global MIN_SPEECH
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-write-thresholds", action="store_true")
    ap.add_argument(
        "--e2e-per-group", type=int, default=40, help="cap end-to-end attempts per group"
    )
    args = ap.parse_args()
    MIN_SPEECH = load_yaml("policy")["voice"]["min_reply_speech_s"]
    rng = random.Random(SEED)

    manifest = read_manifest()
    if not any(r["pid"] == "p00" for r in manifest):
        raise SystemExit("No owner clips. Record: uv run python eval/collect.py --pid p00")
    if not list(LIBRISPEECH.glob("*/*.wav")):
        raise SystemExit("No LibriSpeech impostors. Run: uv run python eval/get_impostors.py")

    print("Loading and embedding clips...", flush=True)
    clips = load_clips(manifest)
    owner_all = [c for c in clips if c.group == "owner"]

    # 1. enrollment and split
    enroll_pool = [c for c in owner_all if c.condition == "close"]
    enroll = enroll_pool[:6]
    rest = [c for c in owner_all if c not in enroll]
    dev, test = [], []
    for cond in sorted({c.condition for c in rest}):
        group = [c for c in rest if c.condition == cond]
        rng.shuffle(group)
        dev += group[: len(group) // 2]
        test += group[len(group) // 2 :]

    # far + cafe noise copies of the far test clips
    noise_path = NOISE / "cafe.wav"
    if noise_path.exists():
        noise = load_audio(noise_path)
        for c in [c for c in test if c.condition == "far"]:
            test.append(Clip("owner", c.pid, "far_noise", c.text, c.kind,
                             add_noise(c.audio, noise, 10.0, rng)))  # fmt: skip

    embed_all(enroll + dev + test + [c for c in clips if c.group != "owner"])
    enroll_vectors = [c.vector for c in enroll if c.vector is not None]
    if len(enroll_vectors) < 4:
        raise SystemExit("Fewer than 4 usable enrollment clips; check p00's recordings.")
    mean = normalize(np.mean(np.stack(enroll_vectors), axis=0))
    for c in clips + test + dev:
        if c.vector is not None:
            c.score = cosine(c.vector, mean)

    scored = lambda cs: np.array([c.score for c in cs if c.score is not None])  # noqa: E731

    # 2. calibration on dev
    libri = [c for c in clips if c.group == "librispeech"]
    cal = calibrate(scored(dev), scored(libri))
    t = Thresholds(cal["t_accept"], cal["t_reject"], "calibrated")
    cal.update(n_dev_owner=len(scored(dev)), n_impostor=len(scored(libri)),
               n_enroll=len(enroll_vectors),
               n_librispeech_speakers=len({c.pid for c in libri}))  # fmt: skip
    print(f"Calibrated: t_accept {t.t_accept}, t_reject {t.t_reject}"
          + ("  (dev scores separated; band spans the gap)" if cal["separated_on_dev"] else ""))  # fmt: skip
    # Only real eval data may change the app's thresholds; smoke tests write next to the report.
    target = CONFIG_DIR if "EARSHOT_EVAL_DATA" not in os.environ else LATEST
    LATEST.mkdir(parents=True, exist_ok=True)
    if not args.no_write_thresholds:
        target.mkdir(parents=True, exist_ok=True)
        (target / "thresholds.calibrated.yaml").write_text(
            "# Written by eval/speaker_eval.py. Do not edit by hand; rerun make eval.\n"
            + yaml.safe_dump(
                {
                    "t_accept": t.t_accept,
                    "t_reject": t.t_reject,
                    "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                    "n_dev_owner_clips": cal["n_dev_owner"],
                    "n_impostor_clips": cal["n_impostor"],
                    "n_librispeech_speakers": cal["n_librispeech_speakers"],
                    "separated_on_dev": cal["separated_on_dev"],
                },
                sort_keys=False,
            )  # fmt: skip
        )

    # 3. test metrics
    others = {g: scored([c for c in clips if c.group == g])
              for g in ("friend", "imitator", "replay")}  # fmt: skip
    impostors = np.concatenate([others["friend"], others["imitator"]])
    owner_by_cond = {cond: scored([c for c in test if c.condition == cond])
                     for cond in ("close", "far", "far_noise")}  # fmt: skip
    too_short = defaultdict(lambda: [0, 0])
    for c in test:
        too_short[c.condition][1] += 1
        too_short[c.condition][0] += c.score is None
    per_condition = {}
    for cond, g in owner_by_cond.items():
        n_all = too_short[cond][1]
        if not n_all:
            continue
        per_condition[cond] = {
            "n_owner": n_all, "n_too_short": too_short[cond][0],
            "eer": eer(g, impostors),
            "frr_at_accept": round(1 - np.sum(g >= t.t_accept) / n_all, 4),  # not accepted on voice
            "frr_at_reject": round(np.sum(g < t.t_reject) / n_all, 4),  # rejected outright
            "step_up_rate": round(np.sum((g >= t.t_reject) & (g < t.t_accept)) / n_all
                                  + too_short[cond][0] / n_all, 4),
        }  # fmt: skip
    impostor_groups = {}
    for g, s in {**others, "librispeech_calibration": scored(libri)}.items():
        impostor_groups[g] = {
            "n": len(s), "n_people": len({c.pid for c in clips if c.group == g.split("_")[0]}),
            "far_at_accept": round(far(s, t.t_accept), 4) if len(s) else None,
            "far_at_reject": round(far(s, t.t_reject), 4) if len(s) else None,
            "mean_score": round(float(np.mean(s)), 3) if len(s) else None,
            "max_score": round(float(np.max(s)), 3) if len(s) else None,
        }  # fmt: skip
    LATEST.mkdir(parents=True, exist_ok=True)
    chart_files = charts(owner_by_cond, others, t, LATEST)

    # 4. end to end
    print("End-to-end approval attempts (real STT)...", flush=True)
    cap = args.e2e_per_group
    attempts = []

    def partner(c: Clip, pool: list[Clip]) -> Clip | None:
        """Another clip by the same person, standing in for their spoken command."""
        same = [p for p in pool if p.pid == c.pid and p is not c and p.condition == c.condition]
        return rng.choice(same) if same else None

    owner_test_by_cond = defaultdict(list)
    for c in test:
        owner_test_by_cond[c.condition].append(c)
    for cond, cs in owner_test_by_cond.items():
        for c in [c for c in cs if c.kind == "affirm"][:cap]:
            attempts.append({"group": "owner", "condition": cond, "tier": "voice", "reply": c,
                             "command": partner(c, cs)})  # fmt: skip
        for c in [c for c in cs if c.kind == "challenge"][:cap]:
            attempts.append({"group": "owner", "condition": cond, "tier": "voice_challenge",
                             "reply": c, "command": partner(c, cs), "word": c.text})  # fmt: skip
    for g in ("friend", "imitator"):
        cs = [c for c in clips if c.group == g]
        for c in [c for c in cs if c.kind == "affirm"][:cap]:
            attempts.append({"group": g, "condition": c.condition, "tier": "voice", "reply": c,
                             "command": partner(c, cs)})  # fmt: skip
        for c in [c for c in cs if c.kind == "challenge"][:cap]:  # they say the right word
            attempts.append({"group": g, "condition": c.condition, "tier": "voice_challenge",
                             "reply": c, "command": partner(c, cs), "word": c.text})  # fmt: skip
    for c in libri[:cap]:  # a stranger's voice saying "yes"
        attempts.append({"group": "librispeech", "condition": "librispeech", "tier": "voice",
                         "reply": c, "command": partner(c, libri), "forced_text": "yes"})  # fmt: skip
    replays = [c for c in clips if c.group == "replay"]
    for c in [c for c in replays if c.kind == "affirm"][:cap]:
        attempts.append({"group": "replay", "condition": "replay", "tier": "voice",
                         "case": "replayed yes", "reply": c, "command": partner(c, replays)})  # fmt: skip
        attempts.append({"group": "replay", "condition": "replay", "tier": "voice_challenge",
                         "case": "replayed yes vs challenge", "reply": c,
                         "command": partner(c, replays), "word": "zebra"})  # fmt: skip
    for c in [c for c in replays if c.kind == "challenge"][:cap]:
        attempts.append({"group": "replay", "condition": "replay", "tier": "voice_challenge",
                         "case": "old challenge word", "reply": c, "command": partner(c, replays),
                         "word": "zebra" if c.text != "zebra" else "maple"})  # fmt: skip
    results, decision_ms = run_attempts(attempts, mean, t)

    e2e = {
        "owner": {cond: {tier: tally(results, group="owner", condition=cond, tier=tier)
                         for tier in ("voice", "voice_challenge")}
                  for cond in owner_test_by_cond},
        "impostors": {g: {tier: tally(results, group=g, tier=tier)
                          for tier in ("voice", "voice_challenge")}
                      for g in ("friend", "imitator", "librispeech")},
        "replay": {case: tally(results, group="replay", case=case)
                   for case in ("replayed yes", "replayed yes vs challenge", "old challenge word")},
    }  # fmt: skip
    impostor_attempts = [r for r in results if r["group"] in ("friend", "imitator", "librispeech")]
    replay_challenge = [
        r for r in results if r["group"] == "replay" and r["tier"] == "voice_challenge"
    ]

    # latency
    lat_embed, lat_stt = [], []
    from band_demo.audio.stt import get_stt

    reply_stt = get_stt("reply")
    for c in [c for c in test if c.condition == "close"][:30]:
        two = c.audio[: 2 * SR]
        t0 = time.perf_counter()
        try:
            embed(two, min_speech_s=MIN_SPEECH)
        except TooShort:
            pass
        lat_embed.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        reply_stt.transcribe(c.audio)
        lat_stt.append((time.perf_counter() - t0) * 1000)

    def pctl(xs):
        return {"p50": round(float(np.percentile(xs, 50)), 1),
                "p95": round(float(np.percentile(xs, 95)), 1), "n": len(xs)} if xs else None  # fmt: skip

    report = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "people": {
            "owner": 1,
            "friends": len({c.pid for c in clips if c.group == "friend"}),
            "imitators": len({c.pid for c in clips if c.group == "imitator"}),
            "librispeech_speakers": cal["n_librispeech_speakers"],
        },
        "calibration": cal,
        "test": {
            "owner": per_condition,
            "others": impostor_groups,
            "noise_condition": noise_path.exists(),
        },  # fmt: skip
        "end_to_end": e2e,
        "headlines": {
            "impostor_approvals": {
                "count": sum(r["outcome"] == "approve" for r in impostor_attempts),
                "n": len(impostor_attempts),
            },
            "owner_no_tap": {
                cond: {
                    "count": v["voice"]["approve"] + v["voice_challenge"]["approve"],
                    "n": v["voice"]["n"] + v["voice_challenge"]["n"],
                }
                for cond, v in e2e["owner"].items()
            },
            "replay_challenge_approvals": {
                "count": sum(r["outcome"] == "approve" for r in replay_challenge),
                "n": len(replay_challenge),
            },
        },  # fmt: skip
        "latency_ms": {
            "embed_2s_clip": pctl(lat_embed),
            "stt_short_reply": pctl(lat_stt),
            "decision_after_reply": pctl(decision_ms),
        },  # fmt: skip
        "charts": chart_files,
    }
    (LATEST / "speaker.json").write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps(report["headlines"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
