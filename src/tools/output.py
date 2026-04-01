"""Output compression utilities to keep tool results token-efficient.

Strategy: preserve ALL items/records but trim verbose fields within each item.
Never silently drop records — that causes the LLM to report wrong counts.
"""

from __future__ import annotations

import json as _json


def compress_output(
    output: str,
    max_lines: int = 200,
    max_chars: int = 12000,
) -> str:
    """Trim tool output to fit LLM context.  Preserves head+tail so no data is silently lost."""
    text = str(output or "").strip()
    if not text:
        return "(empty output)"
    if len(text) <= max_chars and text.count("\n") + 1 <= max_lines:
        return text

    # Try JSON-aware compression first — much better than blind char truncation
    try:
        data = _json.loads(text)
        compressed = _compress_json_value(data, depth=0)
        result = _json.dumps(compressed, indent=2, default=str)
        if len(result) <= max_chars:
            return result
        # Still too big — fall through to line-based truncation
        text = result
    except (ValueError, TypeError):
        pass

    lines = text.splitlines()
    if len(lines) <= max_lines and len(text) <= max_chars:
        return text

    # Line-based truncation preserving head + tail
    if len(lines) > max_lines:
        half = max_lines // 2
        head = lines[:half]
        tail = lines[-half:]
        omitted = len(lines) - max_lines
        text = (
            "\n".join(head)
            + f"\n\n... [{omitted} lines omitted] ...\n\n"
            + "\n".join(tail)
        )

    if len(text) > max_chars:
        # Keep first 80% and last 20% of the char budget
        head_budget = int(max_chars * 0.8)
        tail_budget = max_chars - head_budget - 60
        text = (
            text[:head_budget]
            + "\n\n... [output truncated] ...\n\n"
            + text[-tail_budget:]
        )
    return text


# ── JSON-aware compression ───────────────────────────────────────────────

# Keys whose values are large and rarely useful for the LLM summary
_VERBOSE_KEYS = frozenset({
    "ResponseMetadata", "NextToken", "NextMarker",
    "IpPermissionsEgress",  # egress rules are rarely queried
})

# Keys whose string values should be truncated if very long
_TRIM_STRING_KEYS = frozenset({
    "UserData", "PolicyDocument", "AssumeRolePolicyDocument",
    "LaunchConfigurationARN", "NotificationConfigurations",
})


def _compress_json_value(obj, depth: int = 0):
    """Recursively slim down a parsed JSON value while preserving all items."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in _VERBOSE_KEYS:
                continue
            if k in _TRIM_STRING_KEYS and isinstance(v, str) and len(v) > 200:
                out[k] = v[:200] + "...[trimmed]"
                continue
            out[k] = _compress_json_value(v, depth + 1)
        return out
    if isinstance(obj, list):
        return [_compress_json_value(item, depth + 1) for item in obj]
    if isinstance(obj, str) and len(obj) > 500 and depth > 2:
        return obj[:500] + "...[trimmed]"
    return obj


def compress_json_output(output: str, max_items: int = 50) -> str:
    """For JSON array output, keep items but note if list was very long."""
    text = str(output or "").strip()
    try:
        data = _json.loads(text)
    except (_json.JSONDecodeError, ValueError):
        return compress_output(text)

    if isinstance(data, list):
        if len(data) > max_items:
            truncated = data[:max_items]
            truncated.append({"_note": f"{len(data) - max_items} more items not shown (total: {len(data)})"})
            data = truncated
        compressed = [_compress_json_value(item) for item in data]
        return _json.dumps(compressed, indent=2, default=str)

    if isinstance(data, dict):
        compressed = _compress_json_value(data)
        return _json.dumps(compressed, indent=2, default=str)

    return compress_output(text)
