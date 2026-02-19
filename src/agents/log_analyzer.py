"""
Kubernetes Diagnostics Agent with controlled remediation actions.
"""
from contextlib import nullcontext
import json
import re
import time

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from ..models import get_model
from ..tools import get_all_tools, is_write_tool
from ..utils.response import extract_response_text
from ..utils.tracing import JsonlTraceWriter, TraceSpan, new_trace_id, trace_config_from_env
from ..config import Config
from ..autonomy import SimpleAutonomyEngine


class LogAnalyzerAgent:
    """
    AI Ops agent for Kubernetes diagnostics and remediation guidance.
    
    Capabilities:
    - Diagnose incidents from cluster state, events, and pod logs
    - Detect critical issues (OOMKilled, CrashLoopBackOff, etc.)
    - Recommend and execute approved remediation actions
    - Maintain conversation history
    """
    
    def __init__(self, model_provider: str | None = None, model_name: str | None = None):
        """Initialize the agent"""
        # Initialize model
        self.model = get_model(provider=model_provider, model_name=model_name)
        self.llm = self.model.get_llm()
        
        # Get all tools (log readers + K8s actions)
        self.tools = get_all_tools()
        self._tools_by_name = {tool.name: tool for tool in self.tools}
        
        # Bind tools to model
        self.llm_with_tools = self.model.get_llm_with_tools(self.tools)

        # Pending write action (requires explicit user approval)
        self._pending_action: dict | None = None
        self._recent_restart_candidates: list[str] = []
        self._recent_restart_namespace: str | None = None
        self._recent_restart_reason: str = ""
        self._last_announced_incident_fingerprint: str | None = None

        # Autonomous production alert monitor
        self._autonomy = SimpleAutonomyEngine() if Config.AUTONOMY_ENABLED else None
        self.last_autonomous_scan: dict | None = None
        
        # Tracing
        self.last_trace_id: str | None = None
        self._trace_writer = None
        try:
            cfg = trace_config_from_env(default_dir=Config.TRACE_DIR)
            if cfg.enabled:
                self._trace_writer = JsonlTraceWriter(cfg)
        except Exception:
            self._trace_writer = None
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", Config.get_system_prompt()),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ])

    _APPROVE_WORDS = {"yes", "y", "approve", "proceed", "do it", "run it", "ok"}
    _CANCEL_WORDS = {"no", "n", "cancel", "stop"}

    @staticmethod
    def _empty_response_message(stage: str, *, trace_id: str | None = None) -> str:
        trace_hint = f" Trace ID: {trace_id}" if trace_id else ""
        return (
            f"I got an empty response from the model ({stage}). "
            f"Provider={Config.LLM_PROVIDER}, Model={Config.get_active_model_name()}."
            + trace_hint
        )
    
    def process_query(self, user_input: str, chat_history: list = None) -> str:
        """
        Process a user query and return the response.
        
        Args:
            user_input: User's question or command
            chat_history: List of previous messages (HumanMessage, AIMessage)
        
        Returns:
            String containing the agent's response
        """
        if chat_history is None:
            chat_history = []
        
        trace_id = new_trace_id()
        self.last_trace_id = trace_id
        tw = self._trace_writer
        if tw:
            tw.emit({
                "trace_id": trace_id,
                "event": "turn.start",
                "provider": Config.LLM_PROVIDER,
                "model": Config.get_active_model_name(),
                "chat_history_len": len(chat_history),
                "user_input": user_input,
            })
        
        try:
            alert_prefix = ""
            if self._autonomy is not None and Config.AUTONOMY_SCAN_ON_USER_TURN and self._pending_action is None:
                scan = self.run_autonomous_scan(send_notifications=True)
                if scan.get("ok") and scan.get("incident", {}).get("should_alert"):
                    inc = scan.get("incident", {})
                    notif = scan.get("notifications", {})
                    fingerprint = str(inc.get("fingerprint") or "")
                    suppressed = bool((notif or {}).get("suppressed"))
                    if (not suppressed) and fingerprint and fingerprint != self._last_announced_incident_fingerprint:
                        alert_prefix = (
                            "🚨 Autonomous alert monitor detected an incident before handling your request.\n"
                            f"Severity: {inc.get('severity', 'unknown')} | "
                            f"Confidence: {inc.get('confidence_score', 0)}/100 | "
                            f"Impact: {inc.get('impact_score', 0)}/100\n"
                            f"Summary: {inc.get('issue_summary', '')}\n"
                            f"Notification result: {notif}\n\n"
                        )
                        self._last_announced_incident_fingerprint = fingerprint
                else:
                    self._last_announced_incident_fingerprint = None

            # If there is a pending write action, require an explicit confirmation.
            if self._pending_action is not None:
                decision = user_input.strip().lower()
                if self._should_promote_to_batch(decision):
                    promoted = self._promote_pending_to_batch()
                    if promoted:
                        pending_name = self._pending_action["tool"].name
                        pending_args = self._pending_action["args"]
                        cmd_preview = self._format_command_preview(pending_name, pending_args)
                        return (
                            alert_prefix
                            + "I switched this to a batch restart as requested.\n"
                            + f"Approval required before I can run `{pending_name}` with args {pending_args}.\n"
                            + "Planned command(s):\n"
                            + cmd_preview
                            + "\nReply `yes` to approve or `no` to cancel."
                        )

                if decision in self._APPROVE_WORDS or decision.startswith("approve "):
                    tool = self._pending_action["tool"]
                    args = self._pending_action["args"]
                    cmd_preview = self._format_command_preview(tool.name, args)
                    self._pending_action = None
                    try:
                        if tw:
                            tw.emit({"trace_id": trace_id, "event": "approval.accept", "tool": tool.name, "args": args})
                        with TraceSpan(tw, trace_id, "tool.invoke", {"tool": getattr(tool, "name", "<unknown>") , "args": args}) if tw else nullcontext():
                            result = tool.invoke(args)
                        return (
                            alert_prefix +
                            "Write command(s) to execute:\n"
                            f"{cmd_preview}\n\n"
                            f"Approved. Executed `{tool.name}`.\n\n"
                            f"Executed command(s):\n{cmd_preview}\n\n"
                            f"Result:\n{result}"
                        )
                    except Exception as e:
                        if tw:
                            tw.emit({"trace_id": trace_id, "event": "approval.exec_error", "tool": tool.name, "error": str(e)})
                        return f"Approved, but execution failed: {e}"

                if decision in self._CANCEL_WORDS:
                    pending_name = self._pending_action["tool"].name
                    self._pending_action = None
                    if tw:
                        tw.emit({"trace_id": trace_id, "event": "approval.deny", "tool": pending_name})
                    return alert_prefix + f"Cancelled. I will not run `{pending_name}`."

                pending_name = self._pending_action["tool"].name
                pending_args = self._pending_action["args"]
                cmd_preview = self._format_command_preview(pending_name, pending_args)
                return alert_prefix + self._format_approval_request(pending_name, pending_args, cmd_preview)

            # Format messages for the prompt
            messages = self.prompt.format_messages(
                chat_history=chat_history,
                input=user_input
            )
            
            # Get response from LLM with tools
            t0 = time.perf_counter()
            response = None
            try:
                response = self.llm_with_tools.invoke(messages)
            finally:
                if tw:
                    tw.emit({
                        "trace_id": trace_id,
                        "event": "llm.invoke",
                        "duration_ms": (time.perf_counter() - t0) * 1000.0,
                        "has_tool_calls": bool(getattr(response, 'tool_calls', None)),
                        "usage": getattr(response, 'usage_metadata', None) or getattr(response, 'response_metadata', None),
                    })
            
            # Check if model wants to use tools
            if hasattr(response, 'tool_calls') and response.tool_calls:
                final = self._handle_tool_calls(response, user_input, chat_history, trace_id=trace_id)
                if tw:
                    tw.emit({"trace_id": trace_id, "event": "turn.end", "final_len": len(final)})
                if not (final or "").strip():
                    return self._empty_response_message(
                        "after tool execution",
                        trace_id=trace_id if tw else None,
                    )
                return alert_prefix + final
            else:
                # Direct response without tools
                final = extract_response_text(response)
                if tw:
                    tw.emit({"trace_id": trace_id, "event": "turn.end", "final_len": len(final)})
                if not (final or "").strip():
                    return self._empty_response_message(
                        "no text content",
                        trace_id=trace_id if tw else None,
                    )
                return alert_prefix + final
        
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            print(f"\n{error_msg}")
            import traceback
            traceback.print_exc()
            if tw:
                tw.emit({"trace_id": trace_id, "event": "turn.error", "error": str(e)})
            return error_msg

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
        if not scan.get("ok"):
            return f"Autonomous scan failed: {scan.get('error', 'unknown error')}"
        incident = scan.get("incident", {}) or {}
        notifications = scan.get("notifications", {}) or {}
        lines = [
            "**Autonomous Alert Scan**",
            f"- Severity: {incident.get('severity', 'unknown')}",
            f"- Confidence: {incident.get('confidence_score', 0)}/100",
            f"- Impact: {incident.get('impact_score', 0)}/100",
            f"- Should alert: {incident.get('should_alert', False)}",
            f"- Summary: {incident.get('issue_summary', '(none)')}",
        ]
        anomalies = incident.get("anomalies", []) or []
        if anomalies:
            lines.append("- Anomalies:")
            for item in anomalies[:5]:
                lines.append(f"  • {item}")
        evidence = incident.get("evidence", []) or []
        if evidence:
            lines.append("- Evidence:")
            for item in evidence[:5]:
                lines.append(f"  • {item}")
        if notifications:
            lines.append(f"- Notifications: {notifications}")
        return "\n".join(lines)

    @staticmethod
    def _extract_pod_candidates_from_text(text: str) -> list[str]:
        raw = text or ""
        tokens = re.findall(r"\b[a-z0-9]+(?:-[a-z0-9]+){2,}\b", raw)
        out: list[str] = []
        seen: set[str] = set()
        for token in tokens:
            if token in seen:
                continue
            seen.add(token)
            out.append(token)
        return out[:20]

    @staticmethod
    def _is_batch_intent(decision: str) -> bool:
        text = (decision or "").lower()
        return ("all" in text and ("restart" in text or "do" in text)) or "at once" in text or "batch" in text

    def _should_promote_to_batch(self, decision: str) -> bool:
        if not self._pending_action:
            return False
        pending_tool = self._pending_action["tool"].name
        return pending_tool == "restart_kubernetes_pod" and self._is_batch_intent(decision)

    def _promote_pending_to_batch(self) -> bool:
        if not self._pending_action:
            return False
        pending_tool = self._pending_action["tool"].name
        if pending_tool != "restart_kubernetes_pod":
            return False

        candidates = self._recent_restart_candidates or []
        if len(candidates) <= 1:
            return False

        namespace = self._recent_restart_namespace or self._pending_action["args"].get("namespace") or Config.K8S_DEFAULT_NAMESPACE
        reason = self._recent_restart_reason or self._pending_action["args"].get("reason") or "Batch restart requested by user"

        batch_tool = self._tools_by_name.get("restart_kubernetes_pods_batch")
        if batch_tool is None:
            return False

        self._pending_action = {
            "tool": batch_tool,
            "args": {
                "pod_names": candidates,
                "namespace": namespace,
                "reason": reason,
            },
        }
        return True

    @staticmethod
    def _format_approval_request(tool_name: str, tool_args: dict, cmd_preview: str) -> str:
        return (
            f"Approval required before I can run `{tool_name}` with args {tool_args}.\n"
            "Planned command(s):\n"
            f"{cmd_preview}\n"
            "Reply `yes` to approve or `no` to cancel."
        )

    @staticmethod
    def _format_command_preview(tool_name: str, tool_args: dict) -> str:
        namespace = str(tool_args.get("namespace") or Config.K8S_DEFAULT_NAMESPACE)
        base = ["kubectl", "-n", namespace, "delete", "pod"]
        if tool_name == "restart_kubernetes_pod":
            pod_name = str(tool_args.get("pod_name") or "<pod>")
            return "- " + " ".join(base + [pod_name, "--wait=false", "--ignore-not-found=true"])
        if tool_name == "restart_kubernetes_pods_batch":
            pod_names = [str(p) for p in (tool_args.get("pod_names") or []) if str(p).strip()]
            if not pod_names:
                return "- (no pods provided)"
            return "\n".join(
                "- " + " ".join(base + [pod, "--wait=false", "--ignore-not-found=true"])
                for pod in pod_names
            )
        if tool_name == "scale_kubernetes_deployment":
            name = str(tool_args.get("deployment_name") or "<deployment>")
            replicas = str(tool_args.get("replicas") if tool_args.get("replicas") is not None else "<replicas>")
            return "- " + " ".join(["kubectl", "-n", namespace, "scale", "deployment", name, "--replicas", replicas])
        if tool_name == "scale_kubernetes_statefulset":
            name = str(tool_args.get("statefulset_name") or "<statefulset>")
            replicas = str(tool_args.get("replicas") if tool_args.get("replicas") is not None else "<replicas>")
            return "- " + " ".join(["kubectl", "-n", namespace, "scale", "statefulset", name, "--replicas", replicas])
        if tool_name == "scale_kubernetes_workloads_batch":
            changes = tool_args.get("changes") or []
            previews: list[str] = []
            for item in changes:
                if not isinstance(item, dict):
                    continue
                kind = str(item.get("kind") or "<kind>")
                name = str(item.get("name") or "<name>")
                replicas = str(item.get("replicas") if item.get("replicas") is not None else "<replicas>")
                previews.append("- " + " ".join(["kubectl", "-n", namespace, "scale", kind, name, "--replicas", replicas]))
            return "\n".join(previews) if previews else "- (no workload changes provided)"
        return "- command preview unavailable for this write tool"
    
    def _handle_tool_calls(self, response, user_input: str, chat_history: list, *, trace_id: str | None = None) -> str:
        """
        Handle tool calls from the model with iterative execution.
        
        Args:
            response: LLM response containing tool calls
            user_input: Original user input
            chat_history: Conversation history
        
        Returns:
            Final response after executing tools
        """
        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

        tw = self._trace_writer
        
        # Keep track of all messages for the agent loop
        messages = self.prompt.format_messages(
            chat_history=chat_history,
            input=user_input
        )
        
        # Agent loop: continue until no more tool calls
        max_iterations = getattr(Config, "MAX_ITERATIONS", 5)
        max_tool_calls = getattr(Config, "MAX_TOOL_CALLS_PER_TURN", 12)
        max_duplicate_tool_calls = getattr(Config, "MAX_DUPLICATE_TOOL_CALLS", 2)
        iteration = 0
        total_tool_calls = 0
        call_signature_counts: dict[str, int] = {}
        current_response = response
        
        while iteration < max_iterations:
            iteration += 1
            
            # Check if there are tool calls
            if not (hasattr(current_response, 'tool_calls') and current_response.tool_calls):
                # No more tool calls, return the final response
                return extract_response_text(current_response)
            
            # Execute each tool call
            tool_messages = []
            for tool_call in current_response.tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call['args']

                # Hard budget on total tool calls in a single turn.
                total_tool_calls += 1
                if total_tool_calls > max_tool_calls:
                    if tw and trace_id:
                        tw.emit({
                            "trace_id": trace_id,
                            "event": "tool_loop.call_budget_hit",
                            "max_tool_calls": max_tool_calls,
                            "attempted_tool": tool_name,
                        })
                    messages.append(AIMessage(content=current_response.content, tool_calls=current_response.tool_calls))
                    messages.append(
                        HumanMessage(
                            content=(
                                "Stop calling tools now. You have enough evidence. "
                                "Provide your best final incident summary using existing tool results."
                            )
                        )
                    )
                    forced = self.llm.invoke(messages)
                    forced_text = extract_response_text(forced)
                    if (forced_text or "").strip():
                        return forced_text
                    return "I stopped tool execution due to safety budget limits. Please narrow the request (service/pod/namespace/time window)."

                # Duplicate suppression: repeated identical tool+args usually indicates loopiness.
                try:
                    signature = f"{tool_name}:{json.dumps(tool_args, sort_keys=True, ensure_ascii=False)}"
                except Exception:
                    signature = f"{tool_name}:{str(tool_args)}"
                call_signature_counts[signature] = call_signature_counts.get(signature, 0) + 1
                if call_signature_counts[signature] > max_duplicate_tool_calls:
                    if tw and trace_id:
                        tw.emit({
                            "trace_id": trace_id,
                            "event": "tool_loop.duplicate_suppressed",
                            "tool": tool_name,
                            "args": tool_args,
                            "count": call_signature_counts[signature],
                            "max_duplicate_tool_calls": max_duplicate_tool_calls,
                        })
                    tool_messages.append(
                        ToolMessage(
                            content=(
                                "Duplicate tool call suppressed to avoid loops. "
                                "Use previous tool results and provide a final answer."
                            ),
                            tool_call_id=tool_call['id']
                        )
                    )
                    continue
                
                if tw and trace_id:
                    tw.emit({"trace_id": trace_id, "event": "tool.request", "tool": tool_name, "args": tool_args})
                
                tool_func = self._tools_by_name.get(tool_name)
                
                if tool_func:
                    # Enforce explicit approval for write tools.
                    if is_write_tool(tool_name):
                        if tool_name == "restart_kubernetes_pod":
                            ns = str(tool_args.get("namespace") or Config.K8S_DEFAULT_NAMESPACE)
                            reason = str(tool_args.get("reason") or "")
                            content_text = extract_response_text(current_response)
                            candidates = self._extract_pod_candidates_from_text(content_text)
                            requested = str(tool_args.get("pod_name") or "").strip()
                            if requested and requested not in candidates:
                                candidates.insert(0, requested)
                            self._recent_restart_candidates = candidates[:20]
                            self._recent_restart_namespace = ns
                            self._recent_restart_reason = reason

                        cmd_preview = self._format_command_preview(tool_name, tool_args)
                        self._pending_action = {
                            "tool": tool_func,
                            "args": tool_args,
                        }
                        if tw and trace_id:
                            tw.emit({"trace_id": trace_id, "event": "tool.requires_approval", "tool": tool_name, "args": tool_args})
                        return (
                            f"I recommend running `{tool_name}` with args {tool_args}, but it requires approval.\n"
                            "Planned command(s):\n"
                            f"{cmd_preview}\n"
                            "If you want all suggested pods restarted in one operation, reply: `do all at once`.\n"
                            "Would you like me to proceed? (yes/no)"
                        )
                    try:
                        t_tool0 = time.perf_counter()
                        result = tool_func.invoke(tool_args)
                        if tw and trace_id:
                            tw.emit({
                                "trace_id": trace_id,
                                "event": "tool.result",
                                "tool": tool_name,
                                "duration_ms": (time.perf_counter() - t_tool0) * 1000.0,
                                "result_type": type(result).__name__,
                                "result_len": len(str(result)),
                            })
                        tool_messages.append(
                            ToolMessage(
                                content=str(result),
                                tool_call_id=tool_call['id']
                            )
                        )
                    except Exception as e:
                        error_msg = f"Error: {str(e)}"
                        if tw and trace_id:
                            tw.emit({"trace_id": trace_id, "event": "tool.error", "tool": tool_name, "error": str(e)})
                        tool_messages.append(
                            ToolMessage(
                                content=error_msg,
                                tool_call_id=tool_call['id']
                            )
                        )
            
            # Add AI response and tool results to messages
            messages.append(AIMessage(content=current_response.content, tool_calls=current_response.tool_calls))
            messages.extend(tool_messages)
            
            # Get next response from LLM (might call more tools or finish)
            t1 = time.perf_counter()
            current_response = self.llm_with_tools.invoke(messages)
            if tw and trace_id:
                tw.emit({
                    "trace_id": trace_id,
                    "event": "llm.invoke",
                    "duration_ms": (time.perf_counter() - t1) * 1000.0,
                    "has_tool_calls": bool(getattr(current_response, 'tool_calls', None)),
                    "usage": getattr(current_response, 'usage_metadata', None) or getattr(current_response, 'response_metadata', None),
                })


        # If we hit max iterations and the model is still requesting tools, force a final answer
        # using the base LLM (tools disabled) so the UI doesn't look "stuck".
        if hasattr(current_response, 'tool_calls') and current_response.tool_calls:
            if tw and trace_id:
                tw.emit({
                    "trace_id": trace_id,
                    "event": "tool_loop.max_iterations_hit",
                    "max_iterations": max_iterations,
                    "remaining_tool_calls": len(current_response.tool_calls),
                })

            messages.append(
                HumanMessage(
                    content=(
                        "Stop calling tools now. Provide your best incident summary based only on the tool results already retrieved. "
                        "If the evidence is insufficient, ask ONE specific clarifying question instead of calling more tools."
                    )
                )
            )
            t_force = time.perf_counter()
            forced = self.llm.invoke(messages)
            if tw and trace_id:
                tw.emit({
                    "trace_id": trace_id,
                    "event": "llm.invoke.force_final",
                    "duration_ms": (time.perf_counter() - t_force) * 1000.0,
                    "usage": getattr(forced, 'usage_metadata', None) or getattr(forced, 'response_metadata', None),
                })
            forced_text = extract_response_text(forced)
            if (forced_text or "").strip():
                return forced_text

        # If we hit max iterations, return what we have
        final = extract_response_text(current_response)
        if not (final or "").strip():
            return self._empty_response_message(
                "end of the tool loop",
                trace_id=trace_id if tw else None,
            )
        return final


    def clear_history(self):
        """Clear any pending action. Streamlit history is managed externally."""
        self._pending_action = None
        self._recent_restart_candidates = []
        self._recent_restart_namespace = None
        self._recent_restart_reason = ""