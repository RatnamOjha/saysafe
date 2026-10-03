"""Private replies on a reviewed set: does PrivacyGuard keep sensitive replies off the speaker?

    uv run python bench/privacy_replies.py

Reads bench/data/replies.yaml (or eval/data/replies.yaml while it's being reviewed), and
uses only items a person has reviewed (`reviewed: true`). With none reviewed it reports
nothing: unreviewed labels are an LLM's guesses, and numbers on them would be too.

The rules were frozen before this set was generated, and nobody tuned them on it.
Replies are judged on their text alone (no source hints), which is stricter than a real
app, where the agent tags data where it's fetched.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

from saysafe import PrivacyGuard, Room
from saysafe.privacy.detect import rank

ROOT = Path(__file__).resolve().parent.parent
CANDIDATES = [ROOT / "bench" / "data" / "replies.yaml", ROOT / "eval" / "data" / "replies.yaml"]
PRIVACY = {"speak_full": 0, "headphones_full": 0, "speak_redacted_and_phone": 1, "phone_only": 2}


def positive(level: str) -> bool:
    return rank(level) >= rank("sensitive")


def rate(k: int, n: int) -> dict:
    return {"count": k, "n": n, "rate": round(k / n, 4) if n else None}


def evaluate(items: list[dict]) -> dict:
    guard = PrivacyGuard()
    plain = [i for i in items if not i.get("injection")]
    injected = [i for i in items if i.get("injection")]
    tp = fp = fn = 0
    leaks, rewrite_leaks, over = [], [], []
    n_sensitive = n_public = 0
    for item in plain:
        gold, text = item["level"], item["text"]
        d = guard.check(text, room=Room.OTHERS_PRESENT)
        tp += positive(gold) and positive(d.level)
        fp += positive(d.level) and not positive(gold)
        fn += positive(gold) and not positive(d.level)
        if positive(gold):
            n_sensitive += 1
            if d.channel == "speak_full":
                leaks.append(item["id"])
            elif any(s and s in d.say for s in item.get("sensitive") or []):
                rewrite_leaks.append(item["id"])
        if gold == "public":
            n_public += 1
            if d.withheld:
                over.append(item["id"])

    inj_fail = []
    for item in injected:
        same = all(
            PRIVACY[guard.check(item["text"], room=r).channel]
            >= PRIVACY[guard.check(item["base"], room=r).channel]
            for r in Room
        )
        said = guard.check(item["text"], room=Room.OTHERS_PRESENT).say
        if not same or any(s and s in said for s in item.get("sensitive") or []):
            inj_fail.append(item["id"])

    return {
        "items": len(plain),
        "injection_items": len(injected),
        "sensitive_spoken_in_full_with_others": rate(len(leaks), n_sensitive),
        "sensitive_value_left_in_rewrite": rate(len(rewrite_leaks), n_sensitive),
        "public_withheld_with_others": rate(len(over), n_public),
        "sensitive_or_above_precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "sensitive_or_above_recall": round(tp / (tp + fn), 4) if tp + fn else None,
        "injections_that_changed_routing": rate(len(inj_fail), len(injected)),
        "leaked_ids": leaks,
        "over_redacted_ids": over,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "bench" / "results" / "privacy.json")
    args = ap.parse_args()
    path = next((p for p in CANDIDATES if p.exists()), None)
    if path is None:
        result = {"status": "no dataset"}
    else:
        items = yaml.safe_load(path.read_text())
        reviewed = [i for i in items if i.get("reviewed")]
        result = {"status": "waiting for reviewed labels"} if not reviewed else {"status": "ok"}
        result.update(dataset=str(path.relative_to(ROOT)), reviewed=len(reviewed),
                      total=len(items))  # fmt: skip
        if reviewed:
            result.update(evaluate(reviewed))
    result["generated"] = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
