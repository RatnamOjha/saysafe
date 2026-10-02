"""Append-only decision log. No audio, embeddings or transcripts.

Records are built from an allow-list of fields, so nothing sensitive gets in by
accident: adding a field here is a deliberate choice.
"""

import json
import threading
import time
from pathlib import Path

from saysafe.approvals.tokens import data_dir

ALLOWED = {
    "time", "action_id", "action_hash", "type", "source", "tier", "rule_id", "reasons",
    "scores", "speech_seconds", "match", "outcome", "method", "latency_ms",
}  # fmt: skip


class AuditLog:
    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else data_dir() / "audit.jsonl"
        self._lock = threading.Lock()

    def record(self, **fields) -> dict:
        unknown = set(fields) - ALLOWED
        if unknown:
            raise ValueError(f"audit fields not allowed: {sorted(unknown)}")
        entry = {"time": time.time(), **fields}
        line = json.dumps(entry, sort_keys=True, default=str)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a") as f:
                f.write(line + "\n")
        return entry
