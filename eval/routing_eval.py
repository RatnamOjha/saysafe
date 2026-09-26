"""Private-reply eval: does the detector + router keep sensitive replies off the speaker?

    uv run python eval/routing_eval.py                 # rules only, reviewed items
    uv run python eval/routing_eval.py --llm           # also rules + LLM (paced for rate limits)
    uv run python eval/routing_eval.py --unreviewed    # dry run on unreviewed labels

Writes eval/reports/routing.json and routing.md. Every number comes with its sample size.
Replies are judged on their text alone (no source tags), which is stricter than the app,
where the agent tags data where it's fetched.
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml
from _common import DATA, REPORTS

from earshot.privacy.audience import AudienceState
from earshot.privacy.detect import detect, rank
from earshot.privacy.route import route

PATH = DATA / "replies.yaml"
AUDIENCES = ("alone_likely", "unknown", "others_present")
PRIVACY = {"speak_full": 0, "headphones_full": 0, "speak_redacted_and_phone": 1, "phone_only": 2}


def room(level: str) -> AudienceState:
    return AudienceState(level, 60.0, None, 0, 0, False, False, [])


def positive(level: str) -> bool:
    return rank(level) >= rank("sensitive")


def rate(k: int, n: int) -> dict:
    return {"count": k, "n": n, "rate": round(k / n, 4) if n else None}


def evaluate(items: list[dict], use_llm: bool, pace_s: float) -> dict:
    if use_llm:
        os.environ["EARSHOT_DETECT_LLM"] = "1"
    else:
        os.environ.pop("EARSHOT_DETECT_LLM", None)

    plain = [i for i in items if not i.get("injection")]
    inj = [i for i in items if i.get("injection")]
    latencies, llm_failures = [], 0
    tp = fp = fn = 0
    per_cat: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "n": 0})
    leaks, rewrite_leaks, over_redacted = [], [], []
    n_sensitive = n_public = 0

    def run_detect(text: str):
        nonlocal llm_failures
        t0 = time.perf_counter()
        d = detect(text)
        latencies.append((time.perf_counter() - t0) * 1000)
        if use_llm:
            if "llm_unavailable" in d.categories or "llm" not in d.latency_ms:
                llm_failures += 1
            time.sleep(pace_s)
        return d

    for item in plain:
        gold, text = item["level"], item["text"]
        d = run_detect(text)
        cat = per_cat[item["category"]]
        cat["n"] += 1
        if positive(gold) and positive(d.level):
            tp += 1
            cat["tp"] += 1
        elif positive(d.level):
            fp += 1
            cat["fp"] += 1
        elif positive(gold):
            fn += 1
            cat["fn"] += 1

        r = route(text, d, room("others_present"))
        spoken = r.spoken_text or ""
        if positive(gold):
            n_sensitive += 1
            if r.channel == "speak_full":
                leaks.append({"id": item["id"], "text": text, "detected": d.level})
            elif any(s and s in spoken for s in item.get("sensitive") or []):
                rewrite_leaks.append({"id": item["id"], "text": text, "spoken": spoken})
        if gold == "public":
            n_public += 1
            if r.channel != "speak_full":
                over_redacted.append({"id": item["id"], "text": text, "detected": d.level,
                                      "channel": r.channel})  # fmt: skip

    inj_pass, inj_fail = 0, []
    for item in inj:
        base_d, inj_d = run_detect(item["base"]), run_detect(item["text"])
        # Pass if the instruction never made routing less private (same or more private is fine).
        same = all(
            PRIVACY[route(item["text"], inj_d, room(a)).channel]
            >= PRIVACY[route(item["base"], base_d, room(a)).channel]
            for a in AUDIENCES
        )
        spoken = route(item["text"], inj_d, room("others_present")).spoken_text or ""
        no_leak = not any(s and s in spoken for s in item.get("sensitive") or [])
        if same and no_leak:
            inj_pass += 1
        else:
            inj_fail.append({"id": item["id"], "text": item["text"], "same_route": same,
                             "no_leak": no_leak})  # fmt: skip

    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {
        "mode": "rules+llm" if use_llm else "rules",
        "n_items": len(plain),
        "sensitive_or_above": {
            "precision": _r(precision), "recall": _r(recall), "tp": tp, "fp": fp, "fn": fn,
            "per_category": {
                c: {**v, "recall": _r(v["tp"] / (v["tp"] + v["fn"])) if v["tp"] + v["fn"] else None}
                for c, v in sorted(per_cat.items())
            },
        },  # fmt: skip
        "leak_rate": rate(len(leaks), n_sensitive),
        "rewrite_leaks": rate(len(rewrite_leaks), n_sensitive),
        "over_redaction": rate(len(over_redacted), n_public),
        "injection_pass": rate(inj_pass, len(inj)),
        "detect_latency_ms": {
            "p50": _r(float(np.percentile(latencies, 50)), 2) if latencies else None,
            "p95": _r(float(np.percentile(latencies, 95)), 2) if latencies else None,
            "n": len(latencies),
        },
        "llm_failures": llm_failures if use_llm else None,
        "leaks": leaks, "rewrite_leak_items": rewrite_leaks,
        "over_redacted_items": over_redacted, "injection_failures": inj_fail,
    }  # fmt: skip


def _r(x, d: int = 4):
    return None if x is None else round(x, d)


def pct(r: dict) -> str:
    return "n/a" if r["rate"] is None else f"{r['count']} of {r['n']} ({100 * r['rate']:.1f}%)"


DRY_RUN = "  \n**DRY RUN on unreviewed labels. Not for publishing.**"


def markdown(report: dict) -> str:
    lines = [
        "# Private-reply routing eval", "",
        f"Generated {report['generated']} from {report['n_reviewed']} reviewed replies "
        f"({report['n_injection']} injection cases)." + DRY_RUN * report["dry_run"],
        "", "Replies are judged on their text alone, with another voice in the room.", "",
        "| Metric | " + " | ".join(m["mode"] for m in report["modes"]) + " |",
        "|---|" + "---|" * len(report["modes"]),
    ]  # fmt: skip
    rows = [
        ("Leak rate: sensitive or secret spoken in full", lambda m: pct(m["leak_rate"])),
        ("Rewrite leaks: flagged detail still in spoken text", lambda m: pct(m["rewrite_leaks"])),
        ("Over-redaction: public reply not spoken in full", lambda m: pct(m["over_redaction"])),
        ("Injection pass: instruction never made routing less private",
         lambda m: pct(m["injection_pass"])),
        ("Precision, sensitive or above", lambda m: _num(m["sensitive_or_above"]["precision"])),
        ("Recall, sensitive or above", lambda m: _num(m["sensitive_or_above"]["recall"])),
        ("Detect latency p50 / p95", lambda m: f"{m['detect_latency_ms']['p50']} / "
                                               f"{m['detect_latency_ms']['p95']} ms"),
        ("LLM calls that failed (fell back)", lambda m: "-" if m["llm_failures"] is None
                                                        else str(m["llm_failures"])),
    ]  # fmt: skip
    for label, fn in rows:
        lines.append(f"| {label} | " + " | ".join(fn(m) for m in report["modes"]) + " |")
    for m in report["modes"]:
        lines += ["", f"## {m['mode']}: per category (recall of sensitive or above)", "",
                  "| Category | n | recall | false positives |", "|---|---|---|---|"]  # fmt: skip
        for c, v in m["sensitive_or_above"]["per_category"].items():
            lines.append(f"| {c} | {v['n']} | {_num(v['recall'])} | {v['fp']} |")
        for title, key, field in LISTS:
            if m[key]:
                lines += ["", f"### {title} ({m['mode']})", ""]
                lines += [f"- `{x['id']}` {x[field]}" for x in m[key]]
    return "\n".join(lines) + "\n"


LISTS = [
    ("Leaks", "leaks", "text"),
    ("Rewrite leaks", "rewrite_leak_items", "spoken"),
    ("Over-redacted public replies", "over_redacted_items", "text"),
    ("Injection failures", "injection_failures", "text"),
]


def _num(x) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="also evaluate rules + LLM")
    ap.add_argument("--pace", type=float, default=2.5, help="seconds between LLM calls")
    ap.add_argument("--unreviewed", action="store_true", help="dry run on unreviewed labels")
    ap.add_argument("--data", default=str(PATH), help="dataset (replies_dev.yaml for tuning)")
    args = ap.parse_args()
    path = Path(args.data)
    if not path.exists():
        raise SystemExit("No dataset: uv run python eval/gen_replies.py")
    items = yaml.safe_load(path.read_text())
    use = items if args.unreviewed else [i for i in items if i.get("reviewed")]
    if not use:
        raise SystemExit("No reviewed items yet: uv run python eval/review.py")

    modes = [evaluate(use, False, 0.0)]
    if args.llm:
        print("rules + LLM (paced for the free tier's rate limit)...", flush=True)
        modes.append(evaluate(use, True, args.pace))
    report = {
        "generated": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "dry_run": args.unreviewed,
        "n_reviewed": len(use),
        "n_injection": sum(bool(i.get("injection")) for i in use),
        "modes": modes,
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "routing.json").write_text(json.dumps(report, indent=2))
    (REPORTS / "routing.md").write_text(markdown(report))
    print(markdown(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
