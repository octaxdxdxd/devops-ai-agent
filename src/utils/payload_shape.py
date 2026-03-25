"""Helpers for detecting structured tool payloads that should preserve rows/fields."""

from __future__ import annotations

import json
import re


_WHITESPACE_TABLE_SPLIT_RE = re.compile(r"\s{2,}")


def extract_primary_payload(text: str) -> str:
    """Return the likely payload section from wrapped tool output."""
    raw = str(text or "").strip()
    if not raw:
        return ""

    marker = "\nOutput:\n"
    idx = raw.find(marker)
    if idx >= 0:
        candidate = raw[idx + len(marker) :].strip()
        if candidate:
            return _strip_code_fence(candidate)

    if raw.startswith("Output:\n"):
        candidate = raw[len("Output:\n") :].strip()
        if candidate:
            return _strip_code_fence(candidate)

    return _strip_code_fence(raw)


def _strip_code_fence(text: str) -> str:
    stripped = str(text or "").strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if len(lines) < 3:
        return stripped
    if not lines[-1].strip().startswith("```"):
        return stripped
    return "\n".join(lines[1:-1]).strip()


def looks_like_json_payload(text: str) -> bool:
    """Heuristically detect JSON payloads where preserving fields is important."""
    payload = extract_primary_payload(text)
    if not payload or payload[0] not in "[{":
        return False
    try:
        json.loads(payload)
    except Exception:
        return False
    return True


def looks_like_tabular_payload(text: str) -> bool:
    """Detect TSV/columnar outputs so we avoid cropping rows too aggressively."""
    payload = extract_primary_payload(text)
    lines = [line.rstrip() for line in payload.splitlines() if line.strip()]
    if len(lines) < 2:
        return False

    sample = lines[: min(20, len(lines))]

    tab_counts = [line.count("\t") for line in sample if "\t" in line]
    if len(tab_counts) >= 2 and max(tab_counts) >= 2 and max(tab_counts) - min(tab_counts) <= 1:
        return True

    column_counts: list[int] = []
    for line in sample:
        parts = [part for part in _WHITESPACE_TABLE_SPLIT_RE.split(line.strip()) if part]
        if len(parts) >= 3:
            column_counts.append(len(parts))
    if len(column_counts) >= 3 and max(column_counts) - min(column_counts) <= 1:
        return True
    return False


def looks_like_structured_payload(text: str) -> bool:
    """Return True when row/field loss is more harmful than token savings."""
    return looks_like_json_payload(text) or looks_like_tabular_payload(text)
