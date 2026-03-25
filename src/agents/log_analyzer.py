"""Kubernetes diagnostics agent with controlled remediation actions."""

from __future__ import annotations

from typing import Any, Callable

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from ..autonomy import SimpleAutonomyEngine
from ..config import Config
from ..models import get_model
from ..tools import get_all_tools
from ..utils.llm_retry import invoke_with_retries
from ..utils.query_intent import QueryIntent, classify_query_intent
from ..utils.response import extract_response_text
from ..utils.tracing import JsonlTraceWriter, TraceSpan, new_trace_id, trace_config_from_env
from .planner import build_turn_plan, render_turn_plan_directive
from .approval import (
    ApprovalCoordinator,
    commands_code_block,
    format_command_preview,
    is_repairable_command_tool,
    normalize_command_for_tool,
    parse_command_repair_response,
    tool_result_indicates_failure,
)
from .autonomy_helpers import format_autonomous_scan, notification_was_sent
from .state import (
    IncidentState,
    OperatorIntentState,
    apply_turn_outcome_to_state,
    register_operator_follow_up,
    update_operator_intent_state,
)
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

        self._approval = ApprovalCoordinator()
        self._last_announced_incident_fingerprint: str | None = None
        self._incident_state = IncidentState()
        self._operator_intent_state = OperatorIntentState()
        self._turn_index = 0

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

    def process_query(
        self,
        user_input: str,
        chat_history: list | None = None,
        status_callback: Callable[[str], None] | None = None,
    ) -> str:
        """Process one user query and return the assistant response."""
        if chat_history is None:
            chat_history = []

        self._turn_index += 1
        turn_index = self._turn_index

        trace_id = new_trace_id()
        self.last_trace_id = trace_id
        tw = self._trace_writer
        turn_ended = False

        def end_turn(final_text: str) -> None:
            nonlocal turn_ended
            if not tw or turn_ended:
                return
            tw.emit({"trace_id": trace_id, "event": "turn.end", "final_len": len(final_text or "")})
            turn_ended = True

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
            self._notify_status(status_callback, "Reviewing the request and current operating context...")
            update_operator_intent_state(
                operator_intent_state=self._operator_intent_state,
                user_input=user_input,
                turn_index=turn_index,
                incident_state=self._incident_state,
                approval_pending=self._approval.has_pending(),
            )

            if Config.AUTONOMY_ENABLED and Config.AUTONOMY_SCAN_ON_USER_TURN:
                self._notify_status(status_callback, "Checking recent cluster health signals in the background...")
            alert_prefix = self._build_alert_prefix(self._operator_intent_state)

            approval_response = self._handle_pending_approval(
                user_input,
                alert_prefix,
                trace_id,
                status_callback=status_callback,
            )
            if approval_response is not None:
                end_turn(approval_response)
                return approval_response

            intent = classify_query_intent(user_input)
            self._notify_status(status_callback, "Planning the next step for this turn...")
            turn_plan = build_turn_plan(
                user_input=user_input,
                intent=intent,
                chat_history=chat_history,
                incident_state=self._incident_state,
                operator_intent_state=self._operator_intent_state,
                approval_pending=self._approval.has_pending(),
            )
            if turn_plan.reset_existing_context:
                self._incident_state.clear()
            if tw:
                tw.emit(
                    {
                        "trace_id": trace_id,
                        "event": "turn.plan",
                        "mode": turn_plan.mode,
                        "stage": turn_plan.stage,
                        "focus": turn_plan.focus,
                        "continue_existing": turn_plan.continue_existing,
                        "allow_broad_discovery": turn_plan.allow_broad_discovery,
                        "prefer_cached_reads": turn_plan.prefer_cached_reads,
                        "prefer_fresh_reads": turn_plan.prefer_fresh_reads,
                        "objectives": [
                            {
                                "key": objective.key,
                                "focus": objective.focus,
                                "required_categories": list(objective.required_categories),
                            }
                            for objective in getattr(turn_plan, "objectives", ())[:6]
                        ],
                        "operator_mode": self._operator_intent_state.mode,
                        "execution_policy": self._operator_intent_state.execution_policy,
                    }
                )
            prompt_input = self._prepare_prompt_input(
                user_input=user_input,
                chat_history=chat_history,
                intent_mode=intent.mode,
                turn_plan=turn_plan,
                incident_state=self._incident_state,
                operator_intent_state=self._operator_intent_state,
            )

            selected_tools = self._select_tools_for_turn(
                intent=intent,
                user_input=user_input,
                turn_plan=turn_plan,
                operator_intent_state=self._operator_intent_state,
            )
            self._notify_status(status_callback, "Selecting the most relevant tools for this step...")
            selected_tools_by_name = {tool.name: tool for tool in selected_tools}
            llm_with_tools = self.model.get_llm_with_tools(selected_tools)
            if tw:
                tw.emit(
                    {
                        "trace_id": trace_id,
                        "event": "turn.tools_selected",
                        "count": len(selected_tools),
                        "mode": intent.mode,
                        "tools": [tool.name for tool in selected_tools],
                    }
                )

            messages = self.prompt.format_messages(chat_history=chat_history, input=prompt_input)

            self._notify_status(status_callback, "Asking the model to choose the next actions...")
            response = invoke_with_retries(
                llm_with_tools,
                messages,
                trace_writer=tw,
                trace_id=trace_id,
                event="llm.invoke",
            )

            if hasattr(response, "tool_calls") and response.tool_calls:
                self._notify_status(status_callback, "Running diagnostics and gathering evidence...")
                outcome = handle_tool_calls(
                    response=response,
                    user_input=prompt_input,
                    chat_history=chat_history,
                    prompt=self.prompt,
                    llm=self.llm,
                    llm_with_tools=llm_with_tools,
                    tools=selected_tools,
                    tools_by_name=selected_tools_by_name,
                    approval=self._approval,
                    turn_plan=turn_plan,
                    incident_state=self._incident_state,
                    operator_intent_state=self._operator_intent_state,
                    trace_writer=tw,
                    trace_id=trace_id,
                    status_callback=status_callback,
                )
                final = outcome.final_text
                if not (final or "").strip():
                    trace_hint = f" Trace ID: {trace_id}" if (tw and trace_id) else ""
                    final = (
                        "I got an empty response from the model after tool execution. "
                        f"Provider={Config.LLM_PROVIDER}, Model={Config.get_active_model_name()}."
                        + trace_hint
                    )
                self._incident_state = apply_turn_outcome_to_state(
                    incident_state=self._incident_state,
                    user_input=user_input,
                    intent_mode=intent.mode,
                    turn_plan=turn_plan,
                    outcome=outcome,
                    final_text=final,
                    turn_index=turn_index,
                )
                register_operator_follow_up(
                    operator_intent_state=self._operator_intent_state,
                    final_text=final,
                    turn_plan=turn_plan,
                    approval_pending=self._approval.has_pending(),
                )
                final = alert_prefix + final
                end_turn(final)
                return final

            self._notify_status(status_callback, "Drafting the response...")
            final = extract_response_text(response)
            if not (final or "").strip():
                trace_hint = f" Trace ID: {trace_id}" if (tw and trace_id) else ""
                final = (
                    "I got an empty response from the model (no text content). "
                    f"Provider={Config.LLM_PROVIDER}, Model={Config.get_active_model_name()}."
                    + trace_hint
                )
            self._incident_state = apply_turn_outcome_to_state(
                incident_state=self._incident_state,
                user_input=user_input,
                intent_mode=intent.mode,
                turn_plan=turn_plan,
                outcome=None,
                final_text=final,
                turn_index=turn_index,
            )
            register_operator_follow_up(
                operator_intent_state=self._operator_intent_state,
                final_text=final,
                turn_plan=turn_plan,
                approval_pending=self._approval.has_pending(),
            )
            final = alert_prefix + final
            end_turn(final)
            return final
        except Exception as exc:
            if tw:
                tw.emit({"trace_id": trace_id, "event": "turn.error", "error": str(exc)})
            error_text = f"Error processing query: {exc}"
            end_turn(error_text)
            return error_text
        finally:
            if not turn_ended:
                end_turn("")

    @staticmethod
    def _prepare_prompt_input(
        *,
        user_input: str,
        chat_history: list,
        intent_mode: str,
        turn_plan: Any,
        incident_state: IncidentState,
        operator_intent_state: OperatorIntentState,
    ) -> str:
        """Augment prompt input with execution behavior hints when useful."""
        directives: list[str] = []
        if Config.DEEP_INITIAL_INVESTIGATION and not chat_history and intent_mode == "incident_rca":
            directives.append(
                "Perform a comprehensive read-only investigation before answering. "
                "Use multiple relevant diagnostic tools in this turn, avoid shallow interim summaries, "
                "and provide one consolidated diagnosis with clear remediation steps."
            )
        elif any(aspect in set(getattr(turn_plan, "requested_aspects", ()) or ()) for aspect in {"inventory", "capacity", "cost", "optimization", "explanation"}):
            directives.append(
                "This is a direct analytical or inventory question. Continue read-only investigation in this turn until you can answer the user's request directly."
            )
            directives.append(
                "Do not ask for permission to perform additional read-only checks. Only ask for approval when a mutating action is ready."
            )
            directives.append(
                "Prefer one or a few aggregated read commands that cover the full requested scope. Avoid one-read-per-resource fanout when a cluster-wide or account-wide query can produce the table or totals directly."
            )
        if len(tuple(getattr(turn_plan, "objectives", ()) or ())) > 1:
            directives.append(
                "This request has multiple linked objectives. Complete all of them in this same turn before you stop."
            )

        plan_block = render_turn_plan_directive(
            turn_plan=turn_plan,
            incident_state=incident_state,
            operator_intent_state=operator_intent_state,
        )
        extra_blocks: list[str] = []
        if directives:
            extra_blocks.append("[Agent directive]\n" + "\n".join(f"- {item}" for item in directives))
        if plan_block:
            extra_blocks.append(plan_block)

        if not extra_blocks:
            return user_input
        return f"{user_input}\n\n" + "\n\n".join(extra_blocks)

    def _select_tools_for_turn(
        self,
        *,
        intent: QueryIntent,
        user_input: str,
        turn_plan: Any,
        operator_intent_state: OperatorIntentState,
    ) -> list[Any]:
        """Choose a tool subset to reduce prompt cost and off-target tool drift."""
        if not Config.AGENT_ENABLE_INTENT_TOOL_FILTER:
            return self.tools

        selected_names = set(getattr(turn_plan, "preferred_tools", ()) or ())
        selected_names.update({"kubectl_readonly", "helm_readonly", "aws_cli_readonly"})

        if getattr(turn_plan, "stage", "") == "scope":
            selected_names.update({"k8s_list_namespaces", "k8s_get_events"})
        if getattr(turn_plan, "allow_broad_discovery", False):
            selected_names.update({"k8s_list_pods", "k8s_get_events"})
        if getattr(turn_plan, "focus", "") == "service":
            selected_names.update({"k8s_list_services", "k8s_list_ingresses"})
        elif getattr(turn_plan, "focus", "") == "node":
            selected_names.update({"k8s_list_nodes", "k8s_top_nodes", "k8s_describe_node"})
        elif getattr(turn_plan, "focus", "") == "workload":
            selected_names.update(
                {
                    "k8s_list_deployments",
                    "k8s_list_statefulsets",
                    "k8s_list_daemonsets",
                    "k8s_list_pods",
                }
            )
        elif getattr(turn_plan, "focus", "") == "storage":
            selected_names.update({"k8s_get_pvcs", "k8s_list_pvs"})
        elif getattr(turn_plan, "focus", "") == "aws":
            selected_names.update({"aws_cli_readonly", "k8s_list_nodes"})

        selected_names.update(
            {
                "restart_kubernetes_pod",
                "restart_kubernetes_pods_batch",
                "scale_kubernetes_deployment",
                "scale_kubernetes_statefulset",
                "scale_kubernetes_workloads_batch",
                "rollout_restart_kubernetes_deployment",
                "rollout_restart_kubernetes_statefulset",
                "rollout_restart_kubernetes_daemonset",
                "rollout_restart_kubernetes_workloads_batch",
            }
        )

        if intent.mode == "command" or getattr(turn_plan, "stage", "") == "execute":
            selected_names.update(
                {
                    "kubectl_execute",
                    "helm_execute",
                    "aws_cli_execute",
                }
            )

        selected = [tool for tool in self.tools if tool.name in selected_names]
        return selected or self.tools

    def _build_alert_prefix(self, operator_intent_state: OperatorIntentState) -> str:
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

    @staticmethod
    def _notify_status(status_callback: Callable[[str], None] | None, text: str) -> None:
        """Best-effort UI status updates without coupling the agent to Streamlit."""
        if not callable(status_callback):
            return
        try:
            status_callback(text)
        except Exception:
            return

    def _propose_repaired_command(
        self,
        *,
        tool_name: str,
        tool_args: dict[str, Any],
        failure_text: str,
        trace_id: str,
        status_callback: Callable[[str], None] | None = None,
    ) -> tuple[dict[str, Any] | None, str]:
        """Generate a corrected command body for a failed approved execute tool."""
        if not is_repairable_command_tool(tool_name):
            return None, ""

        original_command = str(tool_args.get("command") or "").strip()
        if not original_command:
            return None, ""

        self._notify_status(
            status_callback,
            "Approved command failed. Analyzing the error and drafting a corrected command...",
        )

        command_prefix = {
            "aws_cli_execute": "aws",
            "kubectl_execute": "kubectl",
            "helm_execute": "helm",
        }.get(tool_name, "command")
        reason = str(tool_args.get("reason") or "").strip()

        repair_messages = [
            SystemMessage(
                content=(
                    "You repair failed infrastructure CLI commands after an approved write action.\n"
                    "Only fix mistakes directly evidenced by the failure output.\n"
                    "Do not broaden scope, change the target resource, or add extra operations.\n"
                    f"For `{tool_name}`, return only the command body after `{command_prefix} `, not the binary itself.\n"
                    "If you cannot confidently repair it, say so.\n"
                    "Return exactly four lines:\n"
                    "STATUS: repair|cannot_repair\n"
                    "SUMMARY: <one short sentence>\n"
                    "COMMAND: <single-line corrected command body or blank>\n"
                    "REASON: <short explanation>\n"
                )
            ),
            HumanMessage(
                content=(
                    f"Tool: {tool_name}\n"
                    f"Command prefix: {command_prefix}\n"
                    f"Original command body: {original_command}\n"
                    f"Operator reason: {reason or '(none provided)'}\n"
                    f"Failure output:\n{failure_text}"
                )
            ),
        ]

        repair_response = invoke_with_retries(
            self.llm,
            repair_messages,
            trace_writer=self._trace_writer,
            trace_id=trace_id,
            event="llm.invoke.command_repair",
        )
        parsed = parse_command_repair_response(extract_response_text(repair_response))
        if not parsed or parsed.get("status") != "repair":
            return None, ""

        repaired_command = normalize_command_for_tool(tool_name, parsed.get("command", ""))
        if not repaired_command or repaired_command == original_command:
            return None, ""

        summary = parsed.get("summary") or parsed.get("reason") or "I found a likely correction."
        return {**tool_args, "command": repaired_command}, summary

    def _handle_pending_approval(
        self,
        user_input: str,
        alert_prefix: str,
        trace_id: str,
        *,
        status_callback: Callable[[str], None] | None = None,
    ) -> str | None:
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
                self._notify_status(status_callback, f"Executing {len(pending_batch)} approved write action(s)...")
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
                        if tool_result_indicates_failure(result):
                            failures += 1
                            if tw:
                                tw.emit(
                                    {
                                        "trace_id": trace_id,
                                        "event": "approval.exec_result_failure",
                                        "tool": tool.name,
                                        "batch_index": idx,
                                        "batch_size": len(actions),
                                    }
                                )
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

        if decision in {"yes", "y", "approve", "proceed", "do it", "run it", "ok"} or decision.startswith("approve "):
            tool = pending.tool
            args = pending.args
            cmd_preview = format_command_preview(tool.name, args)
            cmd_block = commands_code_block(cmd_preview)
            self._approval.pending_action = None
            try:
                self._notify_status(status_callback, f"Executing approved command via `{tool.name}`...")
                if tw:
                    tw.emit({"trace_id": trace_id, "event": "approval.accept", "tool": tool.name, "args": args})
                with TraceSpan(tw, trace_id, "tool.invoke", {"tool": getattr(tool, "name", "<unknown>"), "args": args}) if tw else _NullContext():
                    result = tool.invoke(args)
                if tool_result_indicates_failure(result):
                    if tw:
                        tw.emit({"trace_id": trace_id, "event": "approval.exec_result_failure", "tool": tool.name, "args": args})
                    repaired_args, repair_summary = self._propose_repaired_command(
                        tool_name=tool.name,
                        tool_args=args,
                        failure_text=str(result),
                        trace_id=trace_id,
                        status_callback=status_callback,
                    )
                    if repaired_args is not None:
                        self._approval.set_pending_action(tool, repaired_args)
                        repaired_preview = commands_code_block(format_command_preview(tool.name, repaired_args))
                        self._notify_status(status_callback, "Corrected command prepared. Waiting for fresh approval.")
                        if tw:
                            tw.emit(
                                {
                                    "trace_id": trace_id,
                                    "event": "approval.repair_proposed",
                                    "tool": tool.name,
                                    "original_args": args,
                                    "repaired_args": repaired_args,
                                }
                            )
                        return (
                            alert_prefix
                            + "The approved command failed, so I analyzed the error and prepared a corrected command.\n\n"
                            + f"Original command(s):\n{cmd_block}\n\n"
                            + f"Failure details:\n```text\n{str(result).strip()}\n```\n\n"
                            + (f"Likely fix: {repair_summary}\n\n" if repair_summary else "")
                            + "Corrected command(s):\n"
                            + f"{repaired_preview}\n\n"
                            + "This still requires fresh approval. Reply `yes` to approve or `no` to cancel."
                        )
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
                repaired_args, repair_summary = self._propose_repaired_command(
                    tool_name=tool.name,
                    tool_args=args,
                    failure_text=str(exc),
                    trace_id=trace_id,
                    status_callback=status_callback,
                )
                if repaired_args is not None:
                    self._approval.set_pending_action(tool, repaired_args)
                    repaired_preview = commands_code_block(format_command_preview(tool.name, repaired_args))
                    self._notify_status(status_callback, "Corrected command prepared. Waiting for fresh approval.")
                    if tw:
                        tw.emit(
                            {
                                "trace_id": trace_id,
                                "event": "approval.repair_proposed",
                                "tool": tool.name,
                                "original_args": args,
                                "repaired_args": repaired_args,
                                "source": "exception",
                            }
                        )
                    return (
                        alert_prefix
                        + "The approved command failed, so I analyzed the error and prepared a corrected command.\n\n"
                        + f"Original command(s):\n{cmd_block}\n\n"
                        + f"Failure details:\n```text\n{exc}\n```\n\n"
                        + (f"Likely fix: {repair_summary}\n\n" if repair_summary else "")
                        + "Corrected command(s):\n"
                        + f"{repaired_preview}\n\n"
                        + "This still requires fresh approval. Reply `yes` to approve or `no` to cancel."
                    )
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
        self._incident_state.clear()
        self._operator_intent_state.clear()
        self._last_announced_incident_fingerprint = None
        self.last_trace_id = None

    @property
    def operator_intent_state(self) -> OperatorIntentState:
        return self._operator_intent_state
