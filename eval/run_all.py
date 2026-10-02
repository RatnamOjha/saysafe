"""Rebuild every number: voice eval + routing eval -> results.json and RESULTS.md.

    make eval                                  # = uv run python eval/run_all.py
    uv run python eval/run_all.py --no-llm     # skip the (slow, rate-limited) LLM routing mode
    uv run python eval/run_all.py --skip-voice # reuse the last speaker.json

Writes eval/reports/latest/results.json (headline numbers with units and sample sizes)
and RESULTS.md (headline table, details, charts, and "What doesn't work yet", which is
generated from the numbers). Numbers are never rounded into "near zero" or similar.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone

import yaml
from _common import DATA, LATEST, REPORTS, ROOT

from saysafe.config import env

PY = [sys.executable]


def run(script: str, *args: str) -> None:
    print(f"\n== {script} {' '.join(args)}", flush=True)
    subprocess.run([*PY, str(ROOT / "eval" / script), *args], check=True, cwd=ROOT)


def frac(count: int, n: int) -> str:
    return f"{count} of {n}" if n else "no data"


def pct(count: int, n: int) -> str:
    return f"{100 * count / n:.1f}%" if n else "n/a"


def build(speaker: dict, routing: dict | None) -> tuple[dict, str]:
    h = speaker["headlines"]
    people = speaker["people"]
    who = (f"1 owner, {people['friends']} friend(s), {people['imitators']} imitator(s), "
           f"{people['librispeech_speakers']} LibriSpeech speakers")  # fmt: skip
    rules = routing["modes"][0] if routing else None
    routing_ok = routing is not None and not routing["dry_run"]

    headlines = [
        {"name": "Impostor approvals (end to end)", "count": h["impostor_approvals"]["count"],
         "n": h["impostor_approvals"]["n"], "unit": "attempts",
         "note": "friends, imitators and LibriSpeech voices saying yes or the right word"},
    ]  # fmt: skip
    for cond, label in (("close", "30 cm"), ("far", "2-3 m")):
        if cond in h["owner_no_tap"]:
            v = h["owner_no_tap"][cond]
            headlines.append({"name": f"Owner approved without a phone tap ({label})",
                              "count": v["count"], "n": v["n"], "unit": "attempts",
                              "note": "held-out clips, never used for calibration"})  # fmt: skip
    rc = h["replay_challenge_approvals"]
    headlines.append({"name": "Replays that passed a one-time word", "count": rc["count"],
                      "n": rc["n"], "unit": "attempts",
                      "note": "replayed yes or an old word against a challenge"})  # fmt: skip
    if routing_ok:
        lk = rules["leak_rate"]
        headlines.append({"name": "Sensitive replies spoken in full with others present",
                          "count": lk["count"], "n": lk["n"], "unit": "replies",
                          "note": "reviewed labels, rules only, text alone"})  # fmt: skip
    lat = speaker["latency_ms"]["decision_after_reply"]
    if lat:
        headlines.append({"name": "Decision time after the reply (p50)", "value": lat["p50"],
                          "unit": "ms", "n": lat["n"],
                          "note": f"p95 {lat['p95']} ms; STT + voice score + match, CPU laptop"})  # fmt: skip
    headlines = headlines[:6]

    weak = []
    for cond, v in speaker["test"]["owner"].items():
        if v["frr_at_accept"] > 0.10:
            weak.append(f"Owner at {cond.replace('_', ' + ')}: {100 * v['frr_at_accept']:.0f}% of "
                        f"{v['n_owner']} clips weren't accepted on voice alone, so they'd need a "
                        "phone tap.")  # fmt: skip
    rep = speaker["test"]["others"].get("replay", {})
    if rep.get("n") and (rep.get("far_at_accept") or 0) > 0.2:
        ry = speaker["end_to_end"]["replay"]["replayed yes"]
        weak.append(f"Replays of the owner's voice score like the owner: {100 * rep['far_at_accept']:.0f}% "
                    f"of {rep['n']} replayed clips pass the voice check alone, and "
                    f"{frac(ry['approve'], ry['n'])} replayed yes approved a voice-tier action. "
                    "That's why money needs a one-time word and plain-yes approvals are capped "
                    "under $75.")  # fmt: skip
    imi = speaker["test"]["others"].get("imitator", {})
    if imi.get("n") and (imi.get("far_at_reject") or 0) > 0:
        weak.append(f"Imitators: {100 * imi['far_at_reject']:.0f}% of {imi['n']} clips weren't "
                    "rejected outright (they stepped up to a phone tap instead).")  # fmt: skip
    if h["impostor_approvals"]["count"]:
        weak.append(f"{frac(h['impostor_approvals']['count'], h['impostor_approvals']['n'])} "
                    "impostor attempts were approved end to end.")  # fmt: skip
    if routing_ok:
        if rules["leaks"]:
            weak.append("Leaks (sensitive replies spoken in full): "
                        + "; ".join(f"\"{x['text']}\"" for x in rules["leaks"]))  # fmt: skip
        o = rules["over_redaction"]
        if o["n"] and o["count"] / o["n"] > 0.10:
            weak.append(f"Over-redaction: {frac(o['count'], o['n'])} public replies got redacted.")
        if rules["injection_failures"]:
            weak.append(f"{len(rules['injection_failures'])} injection cases changed the routing.")
    else:
        weak.append("The private-reply eval hasn't run on reviewed labels yet.")
    weak.append("Live voice clones aren't handled; an anti-spoofing model is the next step.")
    weak.append("A silent person in the room is invisible to audio, which is why an unknown "
                "room is treated cautiously.")  # fmt: skip
    weak.append(f"Small sample: {who}, one room, one mic.")

    results = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "people": people,
        "thresholds": {k: speaker["calibration"][k] for k in ("t_accept", "t_reject")},
        "headlines": headlines,
        "weaknesses": weak,
    }

    def cell(x: dict) -> str:
        return f"{x['value']} {x['unit']}" if "value" in x else frac(x["count"], x["n"])

    md = [
        "# Results", "",
        f"Generated {results['generated']} by `make eval`. Tested on {who}.", "",
        "| Result | Value | Notes |", "|---|---|---|",
        *[f"| {x['name']} | {cell(x)} | {x['note']} |" for x in headlines],
        "", "## What doesn't work yet", "", *[f"- {w}" for w in weak],
        "", "## Voice", "",
        f"Thresholds calibrated on held-out owner clips (n={speaker['calibration']['n_dev_owner']}) "
        f"vs LibriSpeech (n={speaker['calibration']['n_impostor']}): accept at or above "
        f"**{speaker['calibration']['t_accept']}**, reject below **{speaker['calibration']['t_reject']}**."
        + (" The two sets didn't overlap on calibration data, so the uncertain band spans the gap "
           "between them." if speaker["calibration"]["separated_on_dev"] else ""),
        "", "| Owner condition | clips | EER | not accepted on voice | rejected outright | step-up |",
        "|---|---|---|---|---|---|",
    ]  # fmt: skip
    for cond, v in speaker["test"]["owner"].items():
        md.append(f"| {cond.replace('_', ' + ')} | {v['n_owner']} | {v['eer']} | "
                  f"{pct(round(v['frr_at_accept'] * v['n_owner']), v['n_owner'])} | "
                  f"{pct(round(v['frr_at_reject'] * v['n_owner']), v['n_owner'])} | "
                  f"{pct(round(v['step_up_rate'] * v['n_owner']), v['n_owner'])} |")  # fmt: skip
    md += ["", "| Other voices | clips | accepted on voice alone | not rejected | mean score |",
           "|---|---|---|---|---|"]  # fmt: skip
    for g, v in speaker["test"]["others"].items():
        if v["n"]:
            md.append(f"| {g.replace('_', ' ')} | {v['n']} | {100 * v['far_at_accept']:.1f}% | "
                      f"{100 * v['far_at_reject']:.1f}% | {v['mean_score']} |")  # fmt: skip
    md += ["", "End to end through the approvals hook (real STT on the reply):", "",
           "| Who | tier | approved | stepped up | rejected |", "|---|---|---|---|---|"]  # fmt: skip
    for cond, tiers in speaker["end_to_end"]["owner"].items():
        for tier, v in tiers.items():
            if v["n"]:
                md.append(f"| owner ({cond}) | {tier} | {frac(v['approve'], v['n'])} | "
                          f"{v['step_up']} | {v['reject']} |")  # fmt: skip
    for g, tiers in speaker["end_to_end"]["impostors"].items():
        for tier, v in tiers.items():
            if v["n"]:
                md.append(f"| {g} | {tier} | {frac(v['approve'], v['n'])} | {v['step_up']} | "
                          f"{v['reject']} |")  # fmt: skip
    for case, v in speaker["end_to_end"]["replay"].items():
        if v["n"]:
            md.append(f"| replay: {case} | | {frac(v['approve'], v['n'])} | {v['step_up']} | "
                      f"{v['reject']} |")  # fmt: skip
    md += ["", "LibriSpeech attempts use the stranger's real audio with the transcript forced to "
           "\"yes\", to test the voice check on its own. Friends' \"commands\" are another of "
           "their clips standing in for the spoken command.", ""]  # fmt: skip
    lat = speaker["latency_ms"]
    md += ["| Latency | p50 | p95 | n |", "|---|---|---|---|"]
    for k, v in lat.items():
        if v:
            md.append(f"| {k.replace('_', ' ')} | {v['p50']} ms | {v['p95']} ms | {v['n']} |")
    md += ["", "![Scores](score_histogram.png)", "", "![DET curve](det_curve.png)", ""]
    md += ["## Private replies", ""]
    if routing:
        md.append((REPORTS / "routing.md").read_text().split("\n", 1)[1])
    else:
        md.append("Not run yet.")
    return results, "\n".join(md) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--skip-voice", action="store_true")
    args = ap.parse_args()

    if not args.skip_voice:
        run("speaker_eval.py")
    replies = DATA / "replies.yaml"
    routing = None
    if replies.exists():
        reviewed = sum(i.get("reviewed", False) for i in yaml.safe_load(replies.read_text()))
        flags = [] if reviewed else ["--unreviewed"]
        if env("LLM_API_KEY") and not args.no_llm:
            flags.append("--llm")
        run("routing_eval.py", *flags)
        routing = json.loads((REPORTS / "routing.json").read_text())

    speaker = json.loads((LATEST / "speaker.json").read_text())
    results, md = build(speaker, routing)
    (LATEST / "results.json").write_text(json.dumps(results, indent=2))
    (LATEST / "RESULTS.md").write_text(md)
    print(f"\nWrote {LATEST / 'results.json'} and RESULTS.md")
    print("\n".join(f"  {x['name']}: {x.get('value', '')}{x.get('count', '')} "
                    f"{'of ' + str(x['n']) if 'count' in x else x['unit']}" for x in results["headlines"]))  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
