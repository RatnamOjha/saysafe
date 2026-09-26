"""Fill the results block in README.md from eval/reports/latest/results.json.

    uv run python scripts/update_readme_numbers.py

Replaces everything between <!-- results:start --> and <!-- results:end -->. Numbers
are copied, never typed: rerun `make eval` and this script to change them.
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "eval" / "reports" / "latest" / "results.json"
README = ROOT / "README.md"
START, END = "<!-- results:start -->", "<!-- results:end -->"


def block(results: dict) -> str:
    p = results["people"]
    rows = []
    for x in results["headlines"]:
        value = (
            f"{x['value']} {x['unit']}" if "value" in x else f"{x['count']} of {x['n']} {x['unit']}"
        )
        rows.append(f"| {x['name']} | {value} |")
    return "\n".join([
        START, "",
        f"Tested on 1 owner, {p['friends']} friend(s), {p['imitators']} imitator(s) and "
        f"{p['librispeech_speakers']} LibriSpeech speakers ({results['generated']}).",
        "", "| Result | Value |", "|---|---|", *rows, "",
        "Details, charts and what doesn't work yet: "
        "[eval/reports/latest/RESULTS.md](eval/reports/latest/RESULTS.md). "
        "Rebuild every number with `make eval`.", "", END,
    ])  # fmt: skip


def main() -> int:
    if not RESULTS.exists():
        print("No results yet: run make eval")
        return 1
    text = README.read_text()
    if START not in text or END not in text:
        print(f"README.md needs {START} and {END} markers")
        return 1
    filled = block(json.loads(RESULTS.read_text()))
    new = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda m: filled, text, flags=re.S)
    README.write_text(new)
    print("README results block updated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
