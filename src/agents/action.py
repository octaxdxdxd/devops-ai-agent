"""Action handler — proposes infrastructure changes that require user approval."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool as langchain_tool

from ..config import Config
from ..tools.registry import ToolRegistry
from ..tracing.tracer import Tracer
from .base import StatusCallback, extract_token_usage

log = logging.getLogger(__name__)

_ACTION_SYSTEM_PROMPT_TEMPLATE = """\
You are an infrastructure operations assistant that proposes safe changes.
Today's date is {today}.

WORKFLOW:
1. Use read tools to understand the current state of the affected resources.
2. Analyze what change is needed and assess risk.
3. Call the 'propose_action' tool with your complete plan. NEVER execute changes directly.

RESOURCE DISCOVERY:
- When looking for a specific resource by name, try MULTIPLE search strategies:
  1. Search by namespace matching the resource name (e.g. "nexus" → namespace "nexus")
  2. Search across all namespaces with all_namespaces=True
  3. Try the resource's parent (deployment, statefulset) not just pods
- NEVER give up after a single failed search. Try at least 2-3 different approaches.
- Use k8s_get_resources with namespace parameter, then try all_namespaces=True if no results.

COMMAND FORMAT for propose_action:
- commands_json must be valid JSON (use double quotes).
- Each entry: {{"command": "kubectl ...", "display": "short description"}}
- Every command MUST be a complete kubectl command starting with "kubectl".
- Do NOT reference tool names. Write the actual kubectl commands.
- Examples:
  - {{"command": "kubectl patch pv foo -p '{{\\"spec\\":{{\\"persistentVolumeReclaimPolicy\\":\\"Retain\\"}}}}'", "display": "Set PV foo reclaim policy to Retain"}}
  - {{"command": "kubectl scale deployment bar --replicas=3 -n prod", "display": "Scale bar to 3 replicas"}}
  - {{"command": "kubectl rollout restart deployment baz -n staging", "display": "Rolling restart baz"}}
- verification_command: a single kubectl command to check that the change worked.

RULES:
- Always investigate current state before proposing changes.
- Be confident — don't say "I cannot" when you have tools to find the information.
- Assess risk honestly: LOW (non-destructive, easily reversible), MEDIUM (service impact possible), HIGH (data loss risk or wide blast radius), CRITICAL (irreversible or affects production data).
- Include exact kubectl commands the user can review.
- Include a verification command to confirm success after execution.
- If the request is ambiguous or dangerous, explain concerns and ask for clarification instead of proposing.
- Propose the MINIMAL change needed. Do not over-engineer.
- For pod restarts: prefer 'kubectl rollout restart' on the parent deployment/statefulset over deleting individual pods."""


@dataclass
class PendingAction:
    id: str
    description: str
    commands: list[dict]        # [{"tool": "k8s_scale", "args": {...}, "display": "kubectl scale ..."}]
    risk: str                   # LOW, MEDIUM, HIGH, CRITICAL
    expected_outcome: str
    verification: dict | None = None   # {"tool": "...", "args": {...}}
    status: str = "pending"     # pending | approved | rejected | executed | verified | failed


def _create_propose_action_tool(captured: list[dict]):
    """Create the propose_action tool that captures plans without executing."""

    @langchain_tool
    def propose_action(
        description: str,
        commands_json: str,
        risk: str,
        expected_outcome: str,
        verification_command: str = "",
    ) -> str:
        """Propose an infrastructure change for user approval. Do NOT execute changes directly.

        Args:
            description: Clear description of what this action does and why
            commands_json: JSON array of kubectl commands. Each entry: {"command":"kubectl ...","display":"short description"}. Every command must start with 'kubectl'. Use double quotes for JSON.
            risk: Risk level: LOW, MEDIUM, HIGH, or CRITICAL
            expected_outcome: What should happen after successful execution
            verification_command: A kubectl command to verify success, e.g. 'kubectl get pv -o wide' (optional)
        """
        try:
            commands = json.loads(commands_json)
        except (json.JSONDecodeError, TypeError):
            # Fallback: try to fix single-quoted JSON from LLM
            try:
                commands = json.loads(commands_json.replace("'", '"'))
            except Exception:
                commands = [{"command": commands_json, "display": commands_json}]

        verification = None
        if verification_command:
            verification = {"command": verification_command.strip()}

        captured.append({
            "description": description,
            "commands": commands,
            "risk": risk.upper(),
            "expected_outcome": expected_outcome,
            "verification": verification,
        })

        return (
            "Action proposal captured. The user will see your plan with Approve/Reject buttons. "
            "Now provide a clear summary of the proposed change for the user."
        )

    return propose_action


def handle_action(
    user_input: str,
    chat_history: list,
    llm_with_tools,
    read_tool_map: dict,
    model_name: str,
    tracer: Tracer,
    status_callback: StatusCallback | None = None,
) -> tuple[str, list[PendingAction]]:
    """Handle an action request. Returns (response_text, pending_actions)."""
    cb = status_callback or (lambda _: None)
    cb("Planning infrastructure change...")

    # Capture proposals from the LLM
    captured_proposals: list[dict] = []
    propose_tool = _create_propose_action_tool(captured_proposals)

    # Give the LLM read tools + the propose_action tool
    all_tools = list(read_tool_map.values()) + [propose_tool]
    tool_map = {t.name: t for t in all_tools}

    # Rebind the LLM with all tools including propose_action
    llm_rebound = llm_with_tools  # The caller should bind with read_tools + propose_action
    # Actually re-bind here:
    base_llm = _get_base_llm(llm_with_tools)
    if base_llm:
        llm_rebound = base_llm.bind_tools(all_tools)

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    system_prompt = _ACTION_SYSTEM_PROMPT_TEMPLATE.format(today=today)

    messages = [
        SystemMessage(content=system_prompt),
        *chat_history[-4:],
        HumanMessage(content=user_input),
    ]

    # Run tool loop with the action budget
    max_steps = Config.ACTION_MAX_STEPS
    for step in range(max_steps):
        t0 = time.monotonic()
        try:
            response: AIMessage = llm_rebound.invoke(messages)
        except Exception as exc:
            tracer.step("error", "action", error=str(exc))
            return f"Failed to plan action: {exc}", []
        elapsed = int((time.monotonic() - t0) * 1000)

        tokens_in, tokens_out = extract_token_usage(response)
        tracer.step(
            "llm_call", "action",
            input_summary=f"step {step + 1}/{max_steps}",
            output_summary=(response.content or "")[:200],
            llm_model=model_name,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            duration_ms=elapsed,
        )

        messages.append(response)

        if not response.tool_calls:
            break  # LLM produced final answer

        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            tool_id = tc.get("id", tool_name)

            tool = tool_map.get(tool_name)
            if not tool:
                result = f"Unknown tool: {tool_name}"
            else:
                cb(f"Running {tool_name}...")
                t1 = time.monotonic()
                try:
                    result = str(tool.invoke(tool_args))
                except Exception as exc:
                    result = f"Tool error: {exc}"
                tracer.step(
                    "tool_call", "action",
                    tool_name=tool_name,
                    tool_args=tool_args,
                    tool_result_preview=result[:300],
                    duration_ms=int((time.monotonic() - t1) * 1000),
                )

            messages.append(ToolMessage(content=result, tool_call_id=tool_id))

    # Build pending actions from captured proposals
    pending_actions: list[PendingAction] = []
    for proposal in captured_proposals:
        action = PendingAction(
            id=uuid4().hex[:10],
            description=proposal["description"],
            commands=proposal["commands"],
            risk=proposal["risk"],
            expected_outcome=proposal["expected_outcome"],
            verification=proposal.get("verification"),
        )
        pending_actions.append(action)
        tracer.step(
            "approval",
            "action",
            output_summary=f"Proposed: {action.description} (risk={action.risk})",
        )

    response_text = (response.content or "").strip() if response else ""

    # If the LLM didn't use propose_action, try to extract from the text
    if not pending_actions and response_text:
        extracted = _extract_actions_from_text(response_text)
        pending_actions.extend(extracted)

    return response_text, pending_actions


def _get_base_llm(llm_with_tools):
    """Try to get the underlying LLM from a tool-bound wrapper."""
    # LangChain's RunnableBinding stores the bound object
    if hasattr(llm_with_tools, "bound"):
        return llm_with_tools.bound
    if hasattr(llm_with_tools, "first"):
        return llm_with_tools.first
    return None


def _extract_actions_from_text(text: str) -> list[PendingAction]:
    """Fallback: extract code blocks as proposed commands from response text."""
    actions = []
    code_blocks = re.findall(r"```(?:bash|sh|shell)?\n(.*?)```", text, re.DOTALL)
    if not code_blocks:
        return actions

    commands = []
    for block in code_blocks:
        for line in block.strip().splitlines():
            line = line.strip()
            if line and line.startswith("kubectl"):
                commands.append({"command": line, "display": line})

    if commands:
        actions.append(PendingAction(
            id=uuid4().hex[:10],
            description="Extracted from response",
            commands=commands,
            risk="MEDIUM",
            expected_outcome="See response text for details",
        ))
    return actions
