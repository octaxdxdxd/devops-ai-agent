"""Kubernetes diagnostics agent with controlled remediation actions."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from ..autonomy import SimpleAutonomyEngine
from ..config import Config
from ..models import get_model
from ..tools import get_all_tools
from ..utils.llm_retry import invoke_with_retries
from ..utils.response import extract_response_text
from ..utils.tracing import JsonlTraceWriter, TraceSpan, new_trace_id, trace_config_from_env
from .approval import ApprovalCoordinator, commands_code_block, format_command_preview
from .autonomy_helpers import format_autonomous_scan, notification_was_sent
from .tool_loop import handle_tool_calls


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class LogAnalyzerAgent:
    """AI Ops orchestration layer for model + tools + approval workflow."""

    def __init__(self, model_provider: str | None = None, model_name: str | None = None):
        self.model = get_model(provider=model_provider, model_name=model_name)
        self.llm = self.model.get_llm()

        self.tools = get_all_tools()
        self.tools_by_name = {tool.name: tool for tool in self.tools}
        self.llm_with_tools = self.model.get_llm_with_tools(self.tools)

        self._approval = ApprovalCoordinator()
        self._last_announced_incident_fingerprint: str | None = None

        self._autonomy = SimpleAutonomyEngine() if Config.AUTONOMY_ENABLED else None
        self.last_autonomous_scan: dict | None = None

        self.last_trace_id: str | None = None
        self._trace_writer = None
        try:
            cfg = trace_config_from_env(default_dir=Config.TRACE_DIR)
            if cfg.enabled:
                self._trace_writer = JsonlTraceWriter(cfg)
        except Exception:
            self._trace_writer = None

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", Config.get_system_prompt()),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),
            ]
        )

    def process_query(self, user_input: str, chat_history: list | None = None) -> str:
        """Process one user query and return the assistant response."""
        if chat_history is None:
            chat_history = []
        prompt_input = self._prepare_prompt_input(user_input=user_input, chat_history=chat_history)

        trace_id = new_trace_id()
        self.last_trace_id = trace_id
        tw = self._trace_writer

        if tw:
            tw.emit(
                {
                    "trace_id": trace_id,
                    "event": "turn.start",
                    "provider": Config.LLM_PROVIDER,
                    "model": Config.get_active_model_name(),
                    "chat_history_len": len(chat_history),
                    "user_input": user_input,
                }
            )

        try:
            alert_prefix = self._build_alert_prefix()

            approval_response = self._handle_pending_approval(user_input, alert_prefix, trace_id)
            if approval_response is not None:
                return approval_response

            messages = self.prompt.format_messages(chat_history=chat_history, input=prompt_input)

            response = invoke_with_retries(
                self.llm_with_tools,
                messages,
                trace_writer=tw,
                trace_id=trace_id,
                event="llm.invoke",
            )

            if hasattr(response, "tool_calls") and response.tool_calls:
                final = handle_tool_calls(
                    response=response,
                    user_input=prompt_input,
                    chat_history=chat_history,
                    prompt=self.prompt,
                    llm=self.llm,
                    llm_with_tools=self.llm_with_tools,
                    tools=self.tools,
                    tools_by_name=self.tools_by_name,
                    approval=self._approval,
                    trace_writer=tw,
                    trace_id=trace_id,
                )
                if tw:
                    tw.emit({"trace_id": trace_id, "event": "turn.end", "final_len": len(final)})
                if not (final or "").strip():
                    trace_hint = f" Trace ID: {trace_id}" if (tw and trace_id) else ""
                    return (
                        "I got an empty response from the model after tool execution. "
                        f"Provider={Config.LLM_PROVIDER}, Model={Config.get_active_model_name()}."
                        + trace_hint
                    )
                return alert_prefix + final

            final = extract_response_text(response)
            if tw:
                tw.emit({"trace_id": trace_id, "event": "turn.end", "final_len": len(final)})
            if not (final or "").strip():
                trace_hint = f" Trace ID: {trace_id}" if (tw and trace_id) else ""
                return (
                    "I got an empty response from the model (no text content). "
                    f"Provider={Config.LLM_PROVIDER}, Model={Config.get_active_model_name()}."
                    + trace_hint
                )
            return alert_prefix + final
        except Exception as exc:
            if tw:
                tw.emit({"trace_id": trace_id, "event": "turn.error", "error": str(exc)})
            return f"Error processing query: {exc}"

    @staticmethod
    def _prepare_prompt_input(*, user_input: str, chat_history: list) -> str:
        """Augment prompt input with execution behavior hints when useful."""
        directives: list[str] = []
        if Config.DEEP_INITIAL_INVESTIGATION and not chat_history:
            directives.append(
                "Perform a comprehensive read-only investigation before answering. "
                "Use multiple relevant diagnostic tools in this turn, avoid shallow interim summaries, "
                "and provide one consolidated diagnosis with clear remediation steps."
            )

        decision = (user_input or "").strip().lower()
        approval_words = {"yes", "y", "approve", "proceed", "do it", "run it", "ok"}
        if decision in approval_words or decision.startswith("approve "):
            last_ai_text = ""
            for msg in reversed(chat_history):
                msg_type = str(getattr(msg, "type", "")).lower()
                if msg_type != "ai":
                    continue
                content = getattr(msg, "content", "")
                if isinstance(content, str):
                    last_ai_text = content
                elif isinstance(content, list):
                    last_ai_text = " ".join(str(x) for x in content)
                else:
                    last_ai_text = str(content or "")
                break

            marker_text = last_ai_text.lower()
            if (
                "would you like me to proceed" in marker_text
                or "please confirm: yes/no" in marker_text
                or "reply `yes` to approve" in marker_text
                or "reply `yes` to proceed" in marker_text
            ):
                directives.append(
                    "User approved the previously proposed remediation. Execute the full approved plan now. "
                    "If multiple write steps are required, emit all required write tool calls in this single response."
                )

        if not directives:
            return user_input
        return f"{user_input}\n\n[Agent directive: {' '.join(directives)}]"

    def _build_alert_prefix(self) -> str:
        if self._autonomy is None or not Config.AUTONOMY_SCAN_ON_USER_TURN or self._approval.has_pending():
            return ""

        scan = self.run_autonomous_scan(send_notifications=True)
        if not scan.get("ok") or not scan.get("incident", {}).get("should_alert"):
            self._last_announced_incident_fingerprint = None
            return ""

        incident = scan.get("incident", {})
        notifications = scan.get("notifications", {})
        fingerprint = str(incident.get("fingerprint") or "")

        should_show_banner = notification_was_sent(notifications)
        if should_show_banner and fingerprint and fingerprint != self._last_announced_incident_fingerprint:
            prefix = (
                "🚨 Autonomous alert monitor detected an incident before handling your request.\n"
                f"Severity: {incident.get('severity', 'unknown')} | "
                f"Confidence: {incident.get('confidence_score', 0)}/100 | "
                f"Impact: {incident.get('impact_score', 0)}/100\n"
                f"Summary: {incident.get('issue_summary', '')}\n"
                f"Notification result: {notifications}\n\n"
            )
        else:
            prefix = ""

        if fingerprint:
            self._last_announced_incident_fingerprint = fingerprint
        return prefix

    def _handle_pending_approval(self, user_input: str, alert_prefix: str, trace_id: str) -> str | None:
        pending_batch = list(self._approval.pending_actions)
        pending = self._approval.pending_action
        if not pending_batch and pending is None:
            return None

        decision = user_input.strip().lower()
        tw = self._trace_writer

        if pending_batch:
            preview_lines: list[str] = []
            for action in pending_batch:
                preview = format_command_preview(action.tool.name, action.args)
                if preview:
                    preview_lines.extend(preview.splitlines())
            cmd_preview = "\n".join(preview_lines) if preview_lines else "- command preview unavailable"
            cmd_block = commands_code_block(cmd_preview)

            if decision in {"yes", "y", "approve", "proceed", "do it", "run it", "ok"} or decision.startswith("approve "):
                actions = list(pending_batch)
                self._approval.pending_actions = []
                sections: list[str] = []
                failures = 0
                for idx, action in enumerate(actions, start=1):
                    tool = action.tool
                    args = action.args
                    try:
                        if tw:
                            tw.emit({"trace_id": trace_id, "event": "approval.accept", "tool": tool.name, "args": args, "batch_index": idx, "batch_size": len(actions)})
                        with TraceSpan(tw, trace_id, "tool.invoke", {"tool": getattr(tool, "name", "<unknown>"), "args": args, "batch_index": idx, "batch_size": len(actions)}) if tw else _NullContext():
                            result = tool.invoke(args)
                        sections.append(f"[{idx}/{len(actions)}] `{tool.name}`\n{result}")
                    except Exception as exc:
                        failures += 1
                        if tw:
                            tw.emit({"trace_id": trace_id, "event": "approval.exec_error", "tool": tool.name, "error": str(exc), "batch_index": idx, "batch_size": len(actions)})
                        sections.append(f"[{idx}/{len(actions)}] `{tool.name}`\n❌ Execution failed: {exc}")

                status = "Approved. Executed all planned write actions." if failures == 0 else (
                    f"Approved. Executed planned write actions with {failures} failure(s)."
                )
                return (
                    alert_prefix
                    + "Write command(s) requested:\n"
                    + f"{cmd_block}\n\n"
                    + status
                    + "\n\nResult:\n"
                    + "\n\n".join(sections)
                )

            if decision in {"no", "n", "cancel", "stop"}:
                self._approval.pending_actions = []
                if tw:
                    tw.emit({"trace_id": trace_id, "event": "approval.deny", "tool": "<batch>", "batch_size": len(pending_batch)})
                return alert_prefix + "Cancelled. I will not run the planned write actions."

            return (
                alert_prefix
                + f"Approval required before I can run {len(pending_batch)} write actions.\n"
                + "Planned command(s):\n"
                + f"{cmd_block}\n"
                + "Reply `yes` to approve all or `no` to cancel."
            )

        if self._approval.should_promote_to_batch(decision):
            promoted = self._approval.promote_pending_to_batch(self.tools)
            if promoted and self._approval.pending_action is not None:
                pending = self._approval.pending_action
                cmd_preview = format_command_preview(pending.tool.name, pending.args)
                cmd_block = commands_code_block(cmd_preview)
                return (
                    alert_prefix
                    + "I switched this to a batch restart as requested.\n"
                    + f"Approval required before I can run `{pending.tool.name}` with args {pending.args}.\n"
                    + "Planned command(s):\n"
                    + cmd_block
                    + "\nReply `yes` to approve or `no` to cancel."
                )

        if decision in {"yes", "y", "approve", "proceed", "do it", "run it", "ok"} or decision.startswith("approve "):
            tool = pending.tool
            args = pending.args
            cmd_preview = format_command_preview(tool.name, args)
            cmd_block = commands_code_block(cmd_preview)
            self._approval.pending_action = None
            try:
                if tw:
                    tw.emit({"trace_id": trace_id, "event": "approval.accept", "tool": tool.name, "args": args})
                with TraceSpan(tw, trace_id, "tool.invoke", {"tool": getattr(tool, "name", "<unknown>"), "args": args}) if tw else _NullContext():
                    result = tool.invoke(args)
                return (
                    alert_prefix
                    + "Write command(s) requested:\n"
                    + f"{cmd_block}\n\n"
                    + f"Approved. Executed `{tool.name}`.\n\n"
                    + f"Result:\n{result}"
                )
            except Exception as exc:
                if tw:
                    tw.emit({"trace_id": trace_id, "event": "approval.exec_error", "tool": tool.name, "error": str(exc)})
                return f"Approved, but execution failed: {exc}"

        if decision in {"no", "n", "cancel", "stop"}:
            pending_name = pending.tool.name
            self._approval.pending_action = None
            if tw:
                tw.emit({"trace_id": trace_id, "event": "approval.deny", "tool": pending_name})
            return alert_prefix + f"Cancelled. I will not run `{pending_name}`."

        cmd_preview = format_command_preview(pending.tool.name, pending.args)
        cmd_block = commands_code_block(cmd_preview)
        return (
            alert_prefix
            + f"Approval required before I can run `{pending.tool.name}` with args {pending.args}.\n"
            + "Planned command(s):\n"
            + f"{cmd_block}\n"
            + "Reply `yes` to approve or `no` to cancel."
        )

    def run_autonomous_scan(self, namespace: str | None = None, *, send_notifications: bool = True) -> dict:
        """Run one autonomous incident scan and optionally send alerts."""
        if self._autonomy is None:
            result = {
                "ok": False,
                "error": "Autonomous monitoring is disabled",
                "incident": None,
                "notifications": {},
            }
            self.last_autonomous_scan = result
            return result

        result = self._autonomy.run_scan(namespace=namespace, send_notifications=send_notifications)
        self.last_autonomous_scan = result
        return result

    @staticmethod
    def format_autonomous_scan(scan: dict) -> str:
        return format_autonomous_scan(scan)

    def clear_history(self) -> None:
        """Clear any pending action state."""
        self._approval.clear()
