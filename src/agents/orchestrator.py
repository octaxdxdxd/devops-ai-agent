"""Main AIOps orchestrator — routes queries, manages state, executes actions."""

from __future__ import annotations

import logging
import shlex
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ..config import Config
from ..infra.k8s_client import K8sClient
from ..infra.aws_client import AWSClient
from ..infra.topology import TopologyBuilder, TopologyCache
from ..models import get_model
from ..tools.registry import ToolRegistry
from ..tools.output import compress_output
from ..tracing import Tracer, TraceStore
from .base import StatusCallback, extract_token_usage
from .intent import classify_intent
from .lookup import handle_lookup
from .diagnose import handle_diagnose
from .action import PendingAction, _validate_single_kubectl_command, format_action_step_preview, handle_action
from .explain import handle_explain
from .read_policy import classify_read_scope, select_read_tools

log = logging.getLogger(__name__)

HEALTH_SCAN_PROMPT = """\
Analyze this Kubernetes cluster health data and identify issues.

Pods (all namespaces):
{pods}

Events (recent warnings/errors):
{events}

Nodes:
{nodes}

Provide a concise health report:
1. **Critical Issues**: crash loops, OOM kills, node failures, unschedulable pods
2. **Warnings**: high restart counts, pending pods, resource pressure
3. **Overall Status**: healthy / degraded / critical
Be specific — name the affected resources."""


@dataclass
class OperatorIntentState:
    mode: str = "incident_response"
    execution_policy: str = "approval_required"
    pinned_constraints: list[str] = field(default_factory=list)
    last_user_instruction: str = ""
    pending_step_summary: str = ""
    pending_step_kind: str = ""


class AIOpsAgent:
    """Top-level agent that routes queries through the hybrid architecture."""

    def __init__(
        self,
        model_provider: str | None = None,
        model_name: str | None = None,
    ) -> None:
        # LLM setup
        self._model_wrapper = get_model(model_provider, model_name)
        self.model_name = model_name or Config.get_active_model_name()

        # Infrastructure clients
        self.k8s = K8sClient()
        self.aws = AWSClient()

        # Tools
        self.tools = ToolRegistry(self.k8s, self.aws)

        # Topology
        builder = TopologyBuilder(self.k8s, self.aws)
        self.topology_cache = TopologyCache(builder, ttl_seconds=300)

        # Tracing
        self.tracer = Tracer()
        self.trace_store = TraceStore()
        self.last_trace_id: str | None = None

        # State
        self.operator_intent_state = OperatorIntentState()
        self.last_autonomous_scan: dict | None = None
        self.pending_actions: list[PendingAction] = []
        self._chat_history: list = []

    # ── Main entry point ─────────────────────────────────────────────────

    def process_query(
        self,
        user_input: str,
        chat_history: list | None = None,
        status_callback: StatusCallback | None = None,
    ) -> str:
        """Process a user query: classify → route → handle → return response text."""
        cb = status_callback or (lambda _: None)
        self.pending_actions = []
        self.aws.set_status_callback(cb)

        # Start trace
        trace_id = self.tracer.start(user_input)
        self.last_trace_id = trace_id
        cb("Classifying request...")

        history = chat_history or []

        try:
            # Step 1: Intent classification
            llm = self._model_wrapper.get_llm()
            intent_result = classify_intent(user_input, llm, self.model_name, self.tracer, chat_history=history)
            intent = intent_result.intent
            self.tracer.current_trace.intent = intent
            cb(f"Intent: {intent}")

            if intent_result.needs_clarification:
                clarification = (
                    intent_result.clarification_prompt.strip()
                    or "Please restate the resource, problem, or next step you want me to work on."
                )
                self.tracer.step(
                    "selector",
                    "orchestrator",
                    input_summary=user_input[:200],
                    output_summary="blocked_for_clarification",
                )
                trace = self.tracer.finish("answered")
                if trace:
                    self.trace_store.save(trace)
                return clarification

            self.operator_intent_state.last_user_instruction = user_input[:200]
            self.operator_intent_state.pending_step_kind = intent

            # Step 2: Route to handler
            if intent == "lookup":
                response = self._handle_lookup(user_input, history, cb)
            elif intent == "diagnose":
                response = self._handle_diagnose(user_input, history, cb)
            elif intent == "action":
                response = self._handle_action(user_input, history, cb)
            elif intent == "explain":
                response = self._handle_explain(user_input, history, cb)
            else:
                response = self._handle_diagnose(user_input, history, cb)

            # Determine outcome
            outcome = "answered"
            if self.pending_actions:
                outcome = "action_proposed"
            trace = self.tracer.finish(outcome)
            if trace:
                self.trace_store.save(trace)

            self.operator_intent_state.pending_step_summary = ""
            self.operator_intent_state.pending_step_kind = ""

            return response

        except Exception as exc:
            log.exception("Agent error processing query")
            trace = self.tracer.finish("error")
            if trace:
                self.tracer.step("error", "orchestrator", error=str(exc))
                self.trace_store.save(trace)
            return f"Agent encountered an error: {exc}"
        finally:
            self.aws.clear_status_callback()

    # ── Handler dispatch ─────────────────────────────────────────────────

    def _select_read_tools(self, user_input: str, history: list):
        llm = self._model_wrapper.get_llm()
        scope = classify_read_scope(user_input, history, llm, self.model_name, self.tracer)
        return select_read_tools(
            user_input,
            history,
            self.tools.k8s_read_tools,
            self.tools.aws_read_tools,
            scope=scope,
        )

    def _handle_lookup(self, user_input: str, history: list, cb: StatusCallback) -> str:
        cb("Looking up data...")
        selection = self._select_read_tools(user_input, history)
        capability_families = ["Kubernetes" if family == "k8s" else "AWS" for family in selection.capability_families]
        llm_with_tools = (
            self._model_wrapper.get_llm_with_tools(selection.tools)
            if selection.tools
            else self._model_wrapper.get_llm()
        )
        tool_map = {t.name: t for t in selection.tools}
        return handle_lookup(
            user_input, history, llm_with_tools, tool_map,
            self.model_name, self.tracer, cb,
            capability_prompt=selection.capability_prompt,
            require_live_inspection=selection.require_live_inspection,
            available_capability_families=capability_families,
            insufficient_tool_names=selection.insufficient_tool_names,
            specialization=selection.specialization,
        )

    def _handle_diagnose(self, user_input: str, history: list, cb: StatusCallback) -> str:
        cb("Starting investigation...")
        selection = self._select_read_tools(user_input, history)
        capability_families = ["Kubernetes" if family == "k8s" else "AWS" for family in selection.capability_families]
        llm_with_tools = (
            self._model_wrapper.get_llm_with_tools(selection.tools)
            if selection.tools
            else self._model_wrapper.get_llm()
        )
        tool_map = {t.name: t for t in selection.tools}
        return handle_diagnose(
            user_input, history, llm_with_tools, tool_map,
            self.model_name, self.tracer, self.topology_cache, cb,
            capability_prompt=selection.capability_prompt,
            require_live_inspection=selection.require_live_inspection,
            available_capability_families=capability_families,
            insufficient_tool_names=selection.insufficient_tool_names,
        )

    def _handle_action(self, user_input: str, history: list, cb: StatusCallback) -> str:
        cb("Planning action...")
        llm_with_tools = self._model_wrapper.get_llm_with_tools(self.tools.read_tools)
        read_tool_map = {t.name: t for t in self.tools.read_tools}
        write_tool_map = {t.name: t for t in self.tools.write_tools}
        response_text, actions = handle_action(
            user_input, history, llm_with_tools, read_tool_map, write_tool_map,
            self.model_name, self.tracer, cb,
        )
        self.pending_actions = actions
        if actions:
            self.operator_intent_state.pending_step_summary = actions[0].description[:100]
            self.operator_intent_state.pending_step_kind = "action_pending_approval"
        return response_text

    def _handle_explain(self, user_input: str, history: list, cb: StatusCallback) -> str:
        cb("Analyzing...")
        selection = self._select_read_tools(user_input, history)
        capability_families = ["Kubernetes" if family == "k8s" else "AWS" for family in selection.capability_families]
        llm_with_tools = (
            self._model_wrapper.get_llm_with_tools(selection.tools)
            if selection.tools
            else self._model_wrapper.get_llm()
        )
        tool_map = {t.name: t for t in selection.tools}
        return handle_explain(
            user_input, history, llm_with_tools, tool_map,
            self.model_name, self.tracer, self.topology_cache, cb,
            capability_prompt=selection.capability_prompt,
            require_live_inspection=selection.require_live_inspection,
            available_capability_families=capability_families,
            insufficient_tool_names=selection.insufficient_tool_names,
        )

    # ── Action execution (post-approval) ─────────────────────────────────

    def execute_action(self, action_id: str, status_callback: StatusCallback | None = None) -> str:
        """Execute a previously approved action and run verification."""
        cb = status_callback or (lambda _: None)
        action = self._find_pending_action(action_id)
        if not action:
            return "Action not found or already executed."

        trace_id = self.tracer.start(f"Execute action: {action.description}")
        self.last_trace_id = trace_id

        results: list[str] = []
        had_error = False
        action.status = "executed"

        for cmd in action.commands:
            step_label, step_preview, step_language = format_action_step_preview(cmd)
            cb(step_preview)

            # New format: {"command": "kubectl ...", "display": "..."}
            command_str = cmd.get("command", "")
            if command_str:
                result = self._run_approved_command(command_str)
                trace_tool_name = "kubectl"
            else:
                # Legacy format: {"tool": "...", "args": {...}, "display": "..."}
                tool_name = cmd.get("tool", "")
                if tool_name in ("shell", "k8s_run_kubectl"):
                    legacy_cmd = cmd.get("args", {}).get("command", cmd.get("display", ""))
                    if not legacy_cmd.startswith("kubectl"):
                        legacy_cmd = f"kubectl {legacy_cmd}"
                    result = self._run_approved_command(legacy_cmd)
                    trace_tool_name = "kubectl"
                elif tool_name:
                    result = self.tools.execute(tool_name, cmd.get("args", {}))
                    trace_tool_name = tool_name
                else:
                    result = f"ERROR: No command or tool specified in: {cmd}"
                    trace_tool_name = "unknown"

            self.tracer.step(
                "tool_call", "action",
                tool_name=trace_tool_name,
                tool_args=cmd,
                tool_result_preview=result[:300],
            )
            if result.startswith("ERROR:"):
                had_error = True
            result_parts: list[str] = []
            if step_label:
                result_parts.append(f"**Step**: {step_label}")
            result_parts.append(f"```{step_language}\n{step_preview}\n```")
            result_parts.append(f"```\n{result}\n```")
            results.append("\n".join(result_parts))

        # Verification
        if action.verification:
            verification_label, verification_preview, verification_language = format_action_step_preview(action.verification)
            cb(verification_preview)
            v_cmd = action.verification.get("command", "")
            if v_cmd:
                v_result = self._run_approved_command(v_cmd)
                trace_tool_name = "kubectl"
            else:
                # Legacy format: {"tool": "...", "args": {...}}
                v_tool = action.verification.get("tool", "")
                v_args = action.verification.get("args", {})
                v_result = self.tools.execute(v_tool, v_args) if v_tool else "No verification command"
                trace_tool_name = v_tool or "unknown"

            self.tracer.step(
                "verify", "action",
                tool_name=trace_tool_name,
                tool_args=action.verification,
                tool_result_preview=v_result[:300],
            )
            if v_result.startswith("ERROR:"):
                had_error = True
            verification_parts = ["\n## Verification"]
            if verification_label:
                verification_parts.append(f"**Step**: {verification_label}")
            verification_parts.append(f"```{verification_language}\n{verification_preview}\n```")
            verification_parts.append(f"```\n{v_result}\n```")
            results.append("\n".join(verification_parts))
            if not had_error:
                action.status = "verified"

        # Post-remediation analysis
        cb("Analyzing results...")
        combined = "\n\n".join(results)
        analysis = self._post_remediation_analysis(action, combined)
        if analysis:
            results.append(f"\n## Post-Execution Analysis\n{analysis}")

        if had_error:
            action.status = "failed"

        trace = self.tracer.finish("action_failed" if had_error else "action_executed")
        if trace:
            self.trace_store.save(trace)

        self.operator_intent_state.pending_step_summary = ""
        self.operator_intent_state.pending_step_kind = ""

        return "\n\n".join(results)

    def _run_approved_command(self, command: str) -> str:
        """Execute a user-approved kubectl command."""
        command = command.strip()
        command_error = _validate_single_kubectl_command(command, "approved command")
        if command_error:
            return f"ERROR: {command_error}"
        parts = shlex.split(command)
        # parts[0] is 'kubectl', rest are args
        return self.k8s.run(parts[1:])

    def _post_remediation_analysis(self, action: PendingAction, execution_output: str) -> str:
        """Quick LLM call to analyze whether the remediation worked."""
        try:
            llm = self._model_wrapper.get_llm()
            messages = [
                SystemMessage(content="You are verifying an infrastructure change. Be concise."),
                HumanMessage(content=(
                    f"Action: {action.description}\n"
                    f"Expected outcome: {action.expected_outcome}\n"
                    f"Execution output:\n{compress_output(execution_output, max_chars=2000)}\n\n"
                    "Did the change succeed? Is the expected outcome achieved? "
                    "Any concerns? Answer in 2-3 sentences."
                )),
            ]
            resp = llm.invoke(messages)
            tokens_in, tokens_out = extract_token_usage(resp)
            self.tracer.step(
                "llm_call", "action",
                input_summary="post-remediation analysis",
                output_summary=(resp.content or "")[:200],
                tokens_in=tokens_in,
                tokens_out=tokens_out,
            )
            return (resp.content or "").strip()
        except Exception as exc:
            return f"Post-execution analysis failed: {exc}"

    def reject_action(self, action_id: str) -> None:
        action = self._find_pending_action(action_id)
        if action:
            action.status = "rejected"
            self.operator_intent_state.pending_step_summary = ""
            self.operator_intent_state.pending_step_kind = ""

    def _find_pending_action(self, action_id: str) -> PendingAction | None:
        for a in self.pending_actions:
            if a.id == action_id:
                return a
        return None

    # ── Autonomous health scan ───────────────────────────────────────────

    def run_autonomous_scan(self, send_notifications: bool = True) -> dict:
        return self.capture_autonomous_scan(namespace=None, send_notifications=send_notifications)

    def capture_autonomous_scan(
        self,
        namespace: str | None = None,
        send_notifications: bool = True,
    ) -> dict:
        """Run a comprehensive health scan using kubectl + LLM analysis."""
        trace_id = self.tracer.start("autonomous_health_scan")

        try:
            ns = namespace or Config.AUTONOMY_NAMESPACE or None

            # Gather health data
            pods = self.k8s.get_resources("pods", all_namespaces=not ns, namespace=ns)
            events = self.k8s.get_events(all_namespaces=not ns, namespace=ns, field_selector="type=Warning")
            nodes = self.k8s.get_resources("nodes")

            self.tracer.step("tool_call", "health_scan", tool_name="k8s_get_resources", output_summary="pods gathered")
            self.tracer.step("tool_call", "health_scan", tool_name="k8s_get_events", output_summary="events gathered")
            self.tracer.step("tool_call", "health_scan", tool_name="k8s_get_resources", output_summary="nodes gathered")

            # LLM analysis
            prompt = HEALTH_SCAN_PROMPT.format(
                pods=compress_output(pods, max_lines=80, max_chars=4000),
                events=compress_output(events, max_lines=40, max_chars=2000),
                nodes=compress_output(nodes, max_lines=20, max_chars=1000),
            )
            llm = self._model_wrapper.get_llm()
            response = llm.invoke([
                SystemMessage(content="You are a Kubernetes health scanner. Be concise and accurate."),
                HumanMessage(content=prompt),
            ])
            tokens_in, tokens_out = extract_token_usage(response)
            self.tracer.step(
                "llm_call", "health_scan",
                llm_model=self.model_name,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                output_summary=(response.content or "")[:300],
            )

            report = (response.content or "").strip()
            scan = {
                "ok": True,
                "completed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "report": report,
                "incident": None,
                "notifications": {},
            }
            self._cache_autonomous_scan(scan)
            trace = self.tracer.finish("answered")
            if trace:
                self.trace_store.save(trace)
            return scan

        except Exception as exc:
            log.exception("Health scan failed")
            scan = {
                "ok": False,
                "completed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "error": str(exc),
                "report": f"Health scan failed: {exc}",
                "incident": None,
                "notifications": {},
            }
            self._cache_autonomous_scan(scan)
            trace = self.tracer.finish("error")
            if trace:
                self.trace_store.save(trace)
            return scan

    def format_autonomous_scan(self, scan: dict | None) -> str:
        if not scan:
            return "No scan data available."
        if scan.get("error"):
            return f"**Scan Error**: {scan['error']}"
        report = scan.get("report", "No report generated.")
        completed = scan.get("completed_at", "")
        header = f"*Scan completed at {completed}*\n\n" if completed else ""
        return f"{header}{report}"

    def _cache_autonomous_scan(self, scan: dict) -> None:
        self.last_autonomous_scan = scan

    # ── Lifecycle ────────────────────────────────────────────────────────

    def clear_history(self) -> None:
        self._chat_history = []
        self.pending_actions = []
        self.last_trace_id = None
        self.operator_intent_state = OperatorIntentState()
        self.topology_cache.invalidate()
