"""Kubernetes diagnostics agent with controlled remediation actions."""

from __future__ import annotations

from typing import Any, Callable

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
    attempt_known_command_repair,
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
            update_operator_intent_state(
                operator_intent_state=self._operator_intent_state,
                user_input=user_input,
                turn_index=turn_index,
                incident_state=self._incident_state,
                approval_pending=self._approval.has_pending(),
            )

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
                        "operator_mode": self._operator_intent_state.mode,
                        "execution_policy": self._operator_intent_state.execution_policy,
                    }
                )
            if Config.AGENT_ENABLE_DIRECT_READ_ROUTER and intent.mode == "chat":
                reply = "Ready. Ask for diagnostics or a specific cluster action."
                final_chat = alert_prefix + reply
                end_turn(final_chat)
                return final_chat

            if Config.AGENT_ENABLE_DIRECT_READ_ROUTER:
                direct = self._handle_direct_read_request(
                    intent=intent,
                    user_input=user_input,
                    incident_state=self._incident_state,
                )
                if direct is not None:
                    final_direct = alert_prefix + direct
                    self._incident_state = apply_turn_outcome_to_state(
                        incident_state=self._incident_state,
                        user_input=user_input,
                        intent_mode=intent.mode,
                        turn_plan=turn_plan,
                        outcome=None,
                        final_text=direct,
                        turn_index=turn_index,
                    )
                    register_operator_follow_up(
                        operator_intent_state=self._operator_intent_state,
                        final_text=direct,
                        turn_plan=turn_plan,
                        approval_pending=self._approval.has_pending(),
                    )
                    end_turn(final_direct)
                    return final_direct

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

            response = invoke_with_retries(
                llm_with_tools,
                messages,
                trace_writer=tw,
                trace_id=trace_id,
                event="llm.invoke",
            )

            if hasattr(response, "tool_calls") and response.tool_calls:
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

        # If user already said they don't know/missing details, do not keep asking the same question.
        combined_user_text = " ".join(
            [user_input]
            + [
                str(getattr(msg, "content", ""))
                for msg in chat_history
                if str(getattr(msg, "type", "")).lower() == "human"
            ]
        ).lower()
        user_unknown_markers = (
            "i don't know",
            "i dont know",
            "no idea",
            "not sure",
            "i'm not sure",
            "im not sure",
            "stop asking",
            "i have no values",
            "i have no values.yaml",
            "i dont have the secret",
            "i don't have the secret",
            "no backup",
        )
        if any(marker in combined_user_text for marker in user_unknown_markers):
            directives.append(
                "The user already said they do not know the missing details. "
                "Do NOT repeat the same clarifying question. Continue autonomous investigation using read-only tools and derive/recover evidence directly where possible."
            )
            directives.append(
                "Prioritize alternative evidence paths before asking anything else: "
                "k8s_list_secrets, k8s_get_resource_yaml, k8s_describe_pod, k8s_get_pod_logs, kubectl_readonly, helm_readonly, aws_cli_readonly."
            )

        if "stop asking" in combined_user_text:
            directives.append(
                "Do not ask clarifying questions in this turn unless there is an absolute hard blocker after exhausting all read-only evidence paths."
            )

        if operator_intent_state.is_following_proposed_plan():
            pending_kind = str(getattr(operator_intent_state, "pending_step_kind", "") or "")
            directives.append(
                "The user asked you to continue with the previously proposed next step. "
                "Do not restate the issue summary unless new evidence changes it."
            )
            if operator_intent_state.pending_step_summary:
                directives.append(f"Continue this next step: {operator_intent_state.pending_step_summary}")
            if pending_kind == "implementation":
                directives.append(
                    "This follow-up is an implementation step. Use the minimum necessary read-only checks to construct exact changes, "
                    "then propose or queue the concrete write action for approval instead of repeating diagnosis."
                )
            elif pending_kind == "prepare":
                directives.append(
                    "This follow-up is a preparation step. Turn the proposed fix into concrete commands, patches, or an approval-ready plan."
                )
            elif pending_kind == "investigate":
                directives.append(
                    "This follow-up is an approved investigation step. Run the requested diagnostic reads now and summarize only the new findings."
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

    @staticmethod
    def _user_requested_execution(user_input: str) -> bool:
        lowered = str(user_input or "").strip().lower()
        if not lowered:
            return False
        if lowered.startswith(("aws ", "kubectl ", "helm ")):
            return True
        action_prefixes = ("restart ", "scale ", "rollout ", "delete ", "patch ", "apply ")
        return any(lowered.startswith(prefix) for prefix in action_prefixes)

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
        elif getattr(turn_plan, "focus", "") == "storage":
            selected_names.update({"k8s_get_pvcs", "k8s_list_pvs"})
        elif getattr(turn_plan, "focus", "") == "aws":
            selected_names.update({"aws_cli_readonly", "k8s_list_nodes"})

        if (
            intent.mode == "command"
            or getattr(turn_plan, "stage", "") == "execute"
            or (
                operator_intent_state.is_following_proposed_plan()
                and getattr(turn_plan, "stage", "") in {"command", "execute"}
            )
            or LogAnalyzerAgent._user_requested_execution(user_input)
        ):
            selected_names.update(
                {
                    "kubectl_execute",
                    "helm_execute",
                    "aws_cli_execute",
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

        selected = [tool for tool in self.tools if tool.name in selected_names]
        return selected or self.tools

    def _handle_direct_read_request(
        self,
        *,
        intent: QueryIntent,
        user_input: str,
        incident_state: IncidentState | None = None,
    ) -> str | None:
        """Run simple read requests deterministically without LLM/tool loops."""
        if intent.mode != "direct_read":
            return None

        inherited_namespace = ""
        if incident_state is not None and incident_state.active:
            inherited_namespace = str(incident_state.namespace or "").strip()

        namespace = intent.namespace or ("all" if intent.all_namespaces else inherited_namespace or Config.K8S_DEFAULT_NAMESPACE)
        if intent.resource in {"pods", "deployments", "statefulsets", "daemonsets", "services", "ingresses", "hpa"}:
            namespace = namespace or "all"
        if intent.resource == "pods" and ("cluster" in user_input.lower() or "all pods" in user_input.lower()):
            namespace = "all"

        mapping: dict[str, tuple[str, dict[str, Any]]] = {
            "pods": ("k8s_list_pods", {"namespace": namespace or "all"}),
            "nodes": ("k8s_list_nodes", {}),
            "namespaces": ("k8s_list_namespaces", {}),
            "deployments": ("k8s_list_deployments", {"namespace": namespace or "all"}),
            "statefulsets": ("k8s_list_statefulsets", {"namespace": namespace or "all"}),
            "daemonsets": ("k8s_list_daemonsets", {"namespace": namespace or "all"}),
            "services": ("k8s_list_services", {"namespace": namespace or "all"}),
            "ingresses": ("k8s_list_ingresses", {"namespace": namespace or "all"}),
            "hpa": ("k8s_list_hpa", {"namespace": namespace or "all"}),
            "events": ("k8s_get_events", {"namespace": namespace or "all", "since_minutes": 60, "limit": 200}),
            "pvcs": ("k8s_get_pvcs", {"namespace": namespace or Config.K8S_DEFAULT_NAMESPACE}),
            "pvs": ("k8s_list_pvs", {}),
        }

        selected = mapping.get(intent.resource)
        if not selected:
            return None

        tool_name, args = selected
        tool = self.tools_by_name.get(tool_name)
        if not tool:
            return f"❌ Tool '{tool_name}' is not available."
        raw = str(tool.invoke(args)).strip()
        title = f"Requested Data: {intent.resource or 'result'}"
        return f"### {title}\n```text\n{raw}\n```"

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
            if operator_intent_state.is_following_proposed_plan():
                prefix = (
                    "Background context: autonomous monitoring still sees an active incident.\n"
                    f"Severity: {incident.get('severity', 'unknown')} | "
                    f"Confidence: {incident.get('confidence_score', 0)}/100 | "
                    f"Impact: {incident.get('impact_score', 0)}/100\n"
                    f"Summary: {incident.get('issue_summary', '')}\n"
                    "Use this as context only. Prioritize continuing the user's requested next step over repeating the diagnosis.\n\n"
                )
            else:
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

        repaired = attempt_known_command_repair(tool_name, original_command, failure_text)
        if repaired and repaired != original_command:
            return (
                {**tool_args, "command": repaired},
                "I corrected the command structure to match the API schema reported in the error.",
            )

        command_prefix = {
            "aws_cli_execute": "aws",
            "kubectl_execute": "kubectl",
            "helm_execute": "helm",
        }.get(tool_name, "command")
        reason = str(tool_args.get("reason") or "").strip()

        repair_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
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
                    ),
                ),
                (
                    "user",
                    (
                        f"Tool: {tool_name}\n"
                        f"Command prefix: {command_prefix}\n"
                        f"Original command body: {original_command}\n"
                        f"Operator reason: {reason or '(none provided)'}\n"
                        f"Failure output:\n{failure_text}"
                    ),
                ),
            ]
        )

        repair_messages = repair_prompt.format_messages()
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
