"""LLM invocation helpers with lightweight rate-limit retry/backoff."""

from __future__ import annotations

import time
from typing import Any

from ..config import Config


def _is_rate_limited_error(exc: Exception) -> bool:
    text = str(exc or "").lower()
    return (
        " 429" in text
        or "code: 429" in text
        or "rate limit" in text
        or "rate-limited" in text
        or "too many requests" in text
    )


def _emit(trace_writer: Any, trace_id: str | None, payload: dict) -> None:
    if trace_writer and trace_id:
        trace_writer.emit({"trace_id": trace_id, **payload})


def invoke_with_retries(
    llm: Any,
    messages: Any,
    *,
    trace_writer: Any = None,
    trace_id: str | None = None,
    event: str = "llm.invoke",
) -> Any:
    """Invoke LLM with bounded retries for transient rate-limits."""
    max_attempts = max(1, int(getattr(Config, "LLM_RETRY_MAX_ATTEMPTS", 3)))
    retry_enabled = bool(getattr(Config, "LLM_RETRY_ON_RATE_LIMIT", True))
    base_delay = max(0.0, float(getattr(Config, "LLM_RETRY_BASE_DELAY_SEC", 1.0)))
    max_delay = max(base_delay, float(getattr(Config, "LLM_RETRY_MAX_DELAY_SEC", 8.0)))

    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        t0 = time.perf_counter()
        try:
            response = llm.invoke(messages)
            _emit(
                trace_writer,
                trace_id,
                {
                    "event": event,
                    "attempt": attempt,
                    "duration_ms": (time.perf_counter() - t0) * 1000.0,
                    "has_tool_calls": bool(getattr(response, "tool_calls", None)),
                    "usage": getattr(response, "usage_metadata", None) or getattr(response, "response_metadata", None),
                },
            )
            return response
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            rate_limited = _is_rate_limited_error(exc)
            _emit(
                trace_writer,
                trace_id,
                {
                    "event": f"{event}.error",
                    "attempt": attempt,
                    "duration_ms": (time.perf_counter() - t0) * 1000.0,
                    "rate_limited": rate_limited,
                    "error": str(exc),
                },
            )

            should_retry = retry_enabled and rate_limited and attempt < max_attempts
            if not should_retry:
                raise

            sleep_s = min(max_delay, base_delay * (2 ** (attempt - 1))) if base_delay > 0 else 0.0
            if sleep_s > 0:
                _emit(
                    trace_writer,
                    trace_id,
                    {"event": f"{event}.retry", "attempt": attempt, "sleep_s": sleep_s},
                )
                time.sleep(sleep_s)

    raise last_exc or RuntimeError("LLM invocation failed")

