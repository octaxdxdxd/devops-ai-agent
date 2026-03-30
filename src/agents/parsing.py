"""Prompt/JSON parsing helpers for the rebuilt agent."""

from __future__ import annotations

import json
import re
from typing import Any


_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{[\s\S]*\}|\[[\s\S]*\])\s*```", re.IGNORECASE)


def truncate_text(text: str, *, max_chars: int = 4000) -> str:
    clean = str(text or "").strip()
    if max_chars <= 0 or len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 3].rstrip() + "..."


def extract_json_payload(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        raise ValueError("empty response")

    fenced = _JSON_BLOCK_RE.search(raw)
    if fenced:
        return fenced.group(1).strip()

    for opener, closer in (("{", "}"), ("[", "]")):
        start = raw.find(opener)
        end = raw.rfind(closer)
        if start >= 0 and end > start:
            candidate = raw[start : end + 1].strip()
            if candidate:
                return candidate

    return raw


def parse_json_response(text: str, *, default: Any | None = None) -> Any:
    try:
        return json.loads(extract_json_payload(text))
    except Exception:
        if default is not None:
            return default
        raise


def safe_json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, default=str)
