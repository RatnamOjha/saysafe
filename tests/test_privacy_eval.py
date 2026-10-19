import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def dataset(tmp_path: Path, all_reviewed: bool) -> Path:
    items = [
        {
            "id": "r001",
            "category": "otp",
            "text": "Your login code is 123456.",
            "level": "secret",
            "sensitive": ["123456"],
            "reviewed": True,
        },
        {
            "id": "r002",
            "category": "public",
            "text": "The weather is sunny today.",
            "level": "public",
            "sensitive": [],
            "reviewed": all_reviewed,
        },
    ]
    path = tmp_path / "replies.yaml"
    path.write_text(yaml.safe_dump(items, sort_keys=False))
    return path


def run(script: str, *args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    merged = {**os.environ, **(env or {})}
    return subprocess.run(
        [sys.executable, str(ROOT / script), *args],
        cwd=ROOT,
        env=merged,
        capture_output=True,
        text=True,
    )


def test_public_privacy_benchmark_waits_for_all_labels(tmp_path):
    data = dataset(tmp_path, all_reviewed=False)
    out = tmp_path / "privacy.json"
    result = run("bench/privacy_replies.py", "--data", str(data), "--out", str(out))

    assert result.returncode == 0, result.stderr
    report = json.loads(out.read_text())
    assert report["status"] == "waiting for reviewed labels"
    assert (report["reviewed"], report["total"]) == (1, 2)
    assert "items" not in report  # partial labels cannot produce publishable metrics


def test_public_privacy_benchmark_runs_with_complete_labels(tmp_path):
    data = dataset(tmp_path, all_reviewed=True)
    out = tmp_path / "privacy.json"
    result = run("bench/privacy_replies.py", "--data", str(data), "--out", str(out))

    assert result.returncode == 0, result.stderr
    report = json.loads(out.read_text())
    assert report["status"] == "ok"
    assert report["reviewed"] == report["total"] == report["items"] == 2


def test_routing_eval_refuses_partial_labels_unless_dry_run(tmp_path):
    data = dataset(tmp_path, all_reviewed=False)
    reports = tmp_path / "reports"
    env = {"EARSHOT_EVAL_REPORTS": str(reports)}

    normal = run("eval/routing_eval.py", "--data", str(data), env=env)
    assert normal.returncode != 0
    assert "Review incomplete: 1 of 2 labels reviewed" in normal.stderr
    assert not (reports / "routing.json").exists()

    preview = run("eval/routing_eval.py", "--data", str(data), "--unreviewed", env=env)
    assert preview.returncode == 0, preview.stderr
    report = json.loads((reports / "routing.json").read_text())
    assert report["dry_run"] is True
    assert (report["n_reviewed"], report["n_total"], report["n_evaluated"]) == (1, 2, 2)
    assert "DRY RUN" in (reports / "routing.md").read_text()
