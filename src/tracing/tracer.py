"""Trace data structures and collector."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4


@dataclass
class TraceStep:
    timestamp: str = ""
    step_type: str = ""       # intent | selector | tool_call | llm_call | approval | verify | error | checkpoint
    handler: str = ""         # lookup | diagnose | action | explain | orchestrator | health_scan
    input_summary: str = ""
    output_summary: str = ""
    tool_name: str | None = None
    tool_args: dict | None = None
    tool_result_preview: str | None = None
    llm_model: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    duration_ms: int = 0
    error: str | None = None


@dataclass
class Trace:
    trace_id: str = ""
    query: str = ""
    intent: str = ""
    steps: list[TraceStep] = field(default_factory=list)
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_duration_ms: int = 0
    outcome: str = ""          # answered | action_proposed | action_executed | action_failed | escalated | error
    started_at: str = ""
    completed_at: str = ""


class Tracer:
    """Collects trace steps during a single agent run."""

    def __init__(self) -> None:
        self.current_trace: Trace | None = None
        self._start_time: float = 0.0

    @property
    def trace_id(self) -> str | None:
        return self.current_trace.trace_id if self.current_trace else None

    def start(self, query: str) -> str:
        trace_id = uuid4().hex[:12]
        self.current_trace = Trace(
            trace_id=trace_id,
            query=query,
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        self._start_time = time.monotonic()
        return trace_id

    def step(
        self,
        step_type: str,
        handler: str,
        *,
        input_summary: str = "",
        output_summary: str = "",
        tool_name: str | None = None,
        tool_args: dict | None = None,
        tool_result_preview: str | None = None,
        llm_model: str | None = None,
        tokens_in: int = 0,
        tokens_out: int = 0,
        duration_ms: int = 0,
        error: str | None = None,
    ) -> None:
        if not self.current_trace:
            return
        s = TraceStep(
            timestamp=datetime.now(timezone.utc).isoformat(),
            step_type=step_type,
            handler=handler,
            input_summary=input_summary[:500],
            output_summary=output_summary[:500],
            tool_name=tool_name,
            tool_args=tool_args,
            tool_result_preview=tool_result_preview[:300] if tool_result_preview else None,
            llm_model=llm_model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            duration_ms=duration_ms,
            error=error,
        )
        self.current_trace.steps.append(s)
        self.current_trace.total_tokens_in += tokens_in
        self.current_trace.total_tokens_out += tokens_out

    def finish(self, outcome: str) -> Trace | None:
        if not self.current_trace:
            return None
        trace = self.current_trace
        trace.outcome = outcome
        trace.completed_at = datetime.now(timezone.utc).isoformat()
        trace.total_duration_ms = int((time.monotonic() - self._start_time) * 1000)
        self.current_trace = None
        return trace
