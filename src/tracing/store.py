"""Persist and retrieve traces as daily JSONL files."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from ..config import Config
from .tracer import Trace


class TraceStore:
    """Writes trace records to daily JSONL files (one line per trace)."""

    def __init__(self, trace_dir: str | None = None) -> None:
        self.trace_dir = Path(trace_dir or Config.TRACE_DIR)

    def _daily_path(self, dt: datetime | None = None) -> Path:
        """Return the JSONL file path for a given date."""
        day = (dt or datetime.now(timezone.utc)).strftime("%Y-%m-%d")
        return self.trace_dir / f"{day}.jsonl"

    def save(self, trace: Trace) -> str | None:
        if not Config.TRACE_ENABLED or trace is None:
            return None
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        path = self._daily_path()
        line = json.dumps(asdict(trace), default=str, separators=(",", ":"))
        with open(path, "a") as f:
            f.write(line + "\n")
        return str(path)

    def load(self, trace_id: str) -> dict | None:
        """Search JSONL files for a trace by ID."""
        if not self.trace_dir.exists():
            return None
        for path in sorted(self.trace_dir.glob("*.jsonl"), reverse=True):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if data.get("trace_id") == trace_id:
                            return data
                    except json.JSONDecodeError:
                        continue
        # Backwards compat: check old single-file traces
        old_path = self.trace_dir / f"{trace_id}.json"
        if old_path.exists():
            with open(old_path) as f:
                return json.load(f)
        return None

    def list_recent(self, limit: int = 20) -> list[str]:
        if not self.trace_dir.exists():
            return []
        trace_ids: list[str] = []
        # Read from newest JSONL files first
        for path in sorted(self.trace_dir.glob("*.jsonl"), reverse=True):
            with open(path) as f:
                lines = f.readlines()
            for line in reversed(lines):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    tid = data.get("trace_id", "")
                    if tid:
                        trace_ids.append(tid)
                        if len(trace_ids) >= limit:
                            return trace_ids
                except json.JSONDecodeError:
                    continue
        return trace_ids
