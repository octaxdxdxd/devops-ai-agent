"""UI-compatible agent wrapper around the rebuilt case runner."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable

from ..config import Config
from ..models import get_model
from ..tools import build_connectors
from ..utils.tracing import JsonlTraceWriter, new_trace_id, trace_config_from_env
from .runner import InvestigationRunner
from .state import CaseState, OperatorIntentState, TurnContext


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso(value: str | None) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


class LogAnalyzerAgent:
    """Long-running evidence-driven infrastructure investigator."""

    def __init__(self, model_provider: str | None = None, model_name: str | None = None):
        self.model = get_model(provider=model_provider, model_name=model_name)
        self.llm = self.model.get_llm()
        self.connectors = build_connectors()
        self.operator_intent_state = OperatorIntentState()
        self.active_case: CaseState | None = None
        self.last_trace_id: str | None = None
        self.last_autonomous_scan: dict | None = None

        cfg = trace_config_from_env(default_dir=Config.TRACE_DIR)
        self._trace_writer = JsonlTraceWriter(cfg) if cfg.enabled else None
        self._runner = InvestigationRunner(
            llm=self.llm,
            connectors=self.connectors,
            system_prompt=Config.get_system_prompt(),
            trace_writer=self._trace_writer,
        )

    def clear_history(self) -> None:
        self.active_case = None
        self.operator_intent_state.clear()
        self.last_trace_id = None

    def process_query(
        self,
        user_input: str,
        chat_history: list | None = None,
        status_callback: Callable[[str], None] | None = None,
    ) -> str:
        chat_history = list(chat_history or [])
        trace_id = new_trace_id()
        self.last_trace_id = trace_id
        if self._trace_writer:
            self._trace_writer.emit(
                {
                    "trace_id": trace_id,
                    "event": "turn.start",
                    "provider": Config.LLM_PROVIDER,
                    "model": Config.get_active_model_name(),
                    "chat_history_len": len(chat_history),
                    "user_input": user_input,
                }
            )
        result = self._runner.run_turn(
            context=TurnContext(user_input=user_input, chat_history=chat_history, trace_id=trace_id),
            case=self.active_case,
            operator_state=self.operator_intent_state,
            status_callback=status_callback,
        )
        self.active_case = result.case_state
        self.operator_intent_state = result.operator_intent_state
        if self._trace_writer:
            self._trace_writer.emit(
                {
                    "trace_id": trace_id,
                    "event": "turn.end",
                    "case_id": getattr(self.active_case, "case_id", None),
                    "status": getattr(self.active_case, "status", None),
                    "response_len": len(result.response_text or ""),
                }
            )
        return result.response_text

    def _cache_autonomous_scan(self, scan: dict | None) -> None:
        self.last_autonomous_scan = dict(scan or {})

    def get_cached_autonomous_scan(self, max_age_sec: int = 600) -> dict | None:
        payload = dict(self.last_autonomous_scan or {})
        completed_at = _parse_iso(str(payload.get("completed_at") or ""))
        if completed_at is None:
            return None
        age_sec = (_utc_now() - completed_at).total_seconds()
        if age_sec > max(0, int(max_age_sec)):
            return None
        return payload

    def capture_autonomous_scan(self, namespace: str | None = None, send_notifications: bool = True) -> dict:
        del namespace, send_notifications
        trace_id = new_trace_id()
        if self._trace_writer:
            self._trace_writer.emit({"trace_id": trace_id, "event": "health_scan.start"})
        scan, case = self._runner.run_health_scan(trace_id=trace_id)
        self.last_trace_id = trace_id
        self.last_autonomous_scan = scan
        self.active_case = case
        if self._trace_writer:
            self._trace_writer.emit({"trace_id": trace_id, "event": "health_scan.end", "summary": scan.get("incident", {}).get("issue_summary")})
        return scan

    def run_autonomous_scan(self, send_notifications: bool = True) -> dict:
        return self.capture_autonomous_scan(namespace=None, send_notifications=send_notifications)

    @staticmethod
    def format_autonomous_scan(scan: dict | None) -> str:
        payload = dict(scan or {})
        incident = payload.get("incident") if isinstance(payload.get("incident"), dict) else {}
        summary = str(incident.get("issue_summary") or "No scan summary available.").strip()
        severity = str(incident.get("severity") or "info").strip()
        confidence = incident.get("confidence_score") or 0
        details = str(incident.get("details_markdown") or "").strip()
        completed_at = str(payload.get("completed_at") or "").strip()
        lines = [
            f"**Bottom Line:** {summary}",
            "",
            f"**Severity:** {severity}",
            f"**Confidence:** {confidence}/100",
        ]
        if completed_at:
            lines.extend(["", f"**Completed At:** {completed_at}"])
        if details:
            lines.extend(["", details])
        return "\n".join(lines).strip()
