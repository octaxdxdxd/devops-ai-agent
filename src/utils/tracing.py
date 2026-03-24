"""Lightweight JSONL tracing for agent runs.

Goals:
- Debuggability: understand latency, tool calls, and model behavior per turn.
- Safety: avoid logging raw log contents or secrets by default.

Enable with env var TRACE_ENABLED=1.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_REDACT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # OpenAI/OpenRouter style keys
    (re.compile(r"sk-or-v1-[A-Za-z0-9\-_]{10,}"), "[REDACTED_OPENROUTER_KEY]"),
    (re.compile(r"sk-[A-Za-z0-9\-_]{10,}"), "[REDACTED_API_KEY]"),
    # Generic bearer tokens
    (re.compile(r"(?i)(authorization\s*[:=]\s*bearer\s+)([^\s\"']{8,})"), r"\1[REDACTED]"),
    # Generic api key assignments
    (re.compile(r"(?i)(api[_-]?key\s*[:=]\s*)([^\s\"']{8,})"), r"\1[REDACTED]"),
    # Common password/token assignments in commands or logs
    (re.compile(r"(?i)(password\s*[=:]\s*)([^\s\"']{3,})"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(token\s*[=:]\s*)([^\s\"']{8,})"), r"\1[REDACTED]"),
    # Long base64-like blobs often indicate secret material
    (re.compile(r"\b[A-Za-z0-9+/]{40,}={0,2}\b"), "[REDACTED_BLOB]"),
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _truncate(value: Any, max_chars: int) -> Any:
    if max_chars <= 0:
        return value
    if isinstance(value, str) and len(value) > max_chars:
        return value[:max_chars] + "…"
    return value


def _redact_text(text: str) -> str:
    redacted = text
    for pat, repl in _REDACT_PATTERNS:
        redacted = pat.sub(repl, redacted)
    return redacted


def _safe_jsonable(value: Any, *, max_chars: int, redact: bool) -> Any:
    """Best-effort conversion to JSONable values, with truncation and redaction."""
    try:
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            s = _redact_text(value) if redact else value
            return _truncate(s, max_chars)
        if isinstance(value, (list, tuple)):
            return [_safe_jsonable(v, max_chars=max_chars, redact=redact) for v in value]
        if isinstance(value, dict):
            out: dict[str, Any] = {}
            for k, v in value.items():
                out[str(k)] = _safe_jsonable(v, max_chars=max_chars, redact=redact)
            return out
        # Common LangChain objects
        if hasattr(value, "dict") and callable(getattr(value, "dict")):
            return _safe_jsonable(value.dict(), max_chars=max_chars, redact=redact)
        if hasattr(value, "model_dump") and callable(getattr(value, "model_dump")):
            return _safe_jsonable(value.model_dump(), max_chars=max_chars, redact=redact)
        # Fallback
        s = _redact_text(str(value)) if redact else str(value)
        return _truncate(s, max_chars)
    except Exception:
        return "[unserializable]"


@dataclass
class TraceConfig:
    enabled: bool
    dir_path: Path
    max_field_chars: int = 2000
    redact: bool = True


def trace_config_from_env(default_dir: str = "logs/traces") -> TraceConfig:
    enabled = os.getenv("TRACE_ENABLED", "0").strip().lower() in {"1", "true", "yes", "y", "on"}
    dir_path = Path(os.getenv("TRACE_DIR", default_dir))
    max_field_chars = int(os.getenv("TRACE_MAX_FIELD_CHARS", "2000"))
    redact = os.getenv("TRACE_REDACT", "1").strip().lower() in {"1", "true", "yes", "y", "on"}
    return TraceConfig(enabled=enabled, dir_path=dir_path, max_field_chars=max_field_chars, redact=redact)


class JsonlTraceWriter:
    """Append-only JSONL writer."""

    def __init__(self, cfg: TraceConfig):
        self.cfg = cfg
        self.cfg.dir_path.mkdir(parents=True, exist_ok=True)
        date = datetime.now(timezone.utc).strftime("%Y%m%d")
        self.path = self.cfg.dir_path / f"trace-{date}.jsonl"

    def emit(self, event: dict[str, Any]) -> None:
        if not self.cfg.enabled:
            return

        safe_event = _safe_jsonable(event, max_chars=self.cfg.max_field_chars, redact=self.cfg.redact)
        safe_event.setdefault("ts", _utc_now_iso())
        safe_event.setdefault("schema", "aiops.trace.v1")

        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(safe_event, ensure_ascii=False) + "\n")


class TraceSpan:
    """Context helper for timing."""

    def __init__(self, writer: JsonlTraceWriter, trace_id: str, name: str, fields: dict[str, Any] | None = None):
        self.writer = writer
        self.trace_id = trace_id
        self.name = name
        self.fields = fields or {}
        self._t0 = None

    def __enter__(self):
        self._t0 = time.perf_counter()
        self.writer.emit({"trace_id": self.trace_id, "event": f"{self.name}.start", **self.fields})
        return self

    def __exit__(self, exc_type, exc, tb):
        dt_ms = None
        if self._t0 is not None:
            dt_ms = (time.perf_counter() - self._t0) * 1000.0
        payload: dict[str, Any] = {"trace_id": self.trace_id, "event": f"{self.name}.end", "duration_ms": dt_ms}
        if exc is not None:
            payload["error"] = str(exc)
        self.writer.emit({**payload, **self.fields})
        return False


def new_trace_id() -> str:
    return uuid.uuid4().hex
