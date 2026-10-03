"""How long the core checks take, on this machine. No models: rules and HMAC only.

    uv run python bench/latency.py

PrivacyGuard.check on every reply in tests/data/sensitivity_cases.yaml (each reply, both
rooms where something can be withheld, 20 rounds), and an ApprovalGuard start + reply
on a voice-tier action. Warm: the first call compiles the rules and isn't counted.
"""

from __future__ import annotations

import argparse
import json
import platform
import secrets
import statistics
import sys
import time
from pathlib import Path

import yaml
from _common import machine

from saysafe import Action, ApprovalGuard, PrivacyGuard, Room

ROOT = Path(__file__).resolve().parent.parent
ROUNDS = 20


def percentiles(ms: list[float]) -> dict:
    ms = sorted(ms)
    pick = lambda q: round(ms[min(len(ms) - 1, int(q * len(ms)))], 3)  # noqa: E731
    return {"p50_ms": pick(0.5), "p95_ms": pick(0.95), "p99_ms": pick(0.99), "n": len(ms)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "bench" / "results" / "latency.json")
    args = ap.parse_args()

    cases = yaml.safe_load((ROOT / "tests" / "data" / "sensitivity_cases.yaml").read_text())
    texts = [c["text"] for c in cases]
    guard = PrivacyGuard()
    for t in texts:
        guard.check(t, room=Room.OTHERS_PRESENT)
    privacy_ms = []
    for _ in range(ROUNDS):
        for t in texts:
            for room in (Room.UNKNOWN, Room.OTHERS_PRESENT):
                t0 = time.perf_counter()
                guard.check(t, room=room)
                privacy_ms.append((time.perf_counter() - t0) * 1000)

    approvals = ApprovalGuard(secrets.token_bytes(32))
    action = Action(type="order_food", counterparty="DoorDash", amount=20)
    approvals.reply(approvals.start(action), "yes", voice_score=0.8)
    approval_ms = []
    for _ in range(ROUNDS * 50):
        t0 = time.perf_counter()
        step = approvals.reply(approvals.start(action), "yes", voice_score=0.8)
        approval_ms.append((time.perf_counter() - t0) * 1000)
        assert step.status == "approved"

    result = {
        "generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
        "machine": machine(),
        "python": platform.python_version(),
        "privacy_check": {
            **percentiles(privacy_ms),
            "replies": len(texts),
            "mean_words": round(statistics.mean(len(t.split()) for t in texts), 1),
        },
        "approval_start_and_reply": percentiles(approval_ms),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
