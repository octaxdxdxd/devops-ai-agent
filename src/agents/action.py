"""Action handler — proposes infrastructure changes that require user approval."""

from __future__ import annotations

import ast
import json
import logging
import re
import shlex
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool as langchain_tool

from ..config import Config
from ..policy import guard_k8s_read_tool, guard_k8s_write_tool, guard_tool_invocation
from ..tools.command_preview import render_action_step_preview, render_tool_call_preview
from ..tracing.tracer import Tracer
from .base import StatusCallback, extract_token_usage

log = logging.getLogger(__name__)

_ACTION_SYSTEM_PROMPT_TEMPLATE = """\
You propose safe infrastructure changes that require operator approval.
Today's date is {today}.

Workflow:
- Inspect current state with read tools before proposing anything.
- If a lookup fails, try other namespaces, labels, parents, or AWS regions.
- Call `propose_action` once you have the minimal safe plan.
- Never execute changes directly.
- If a read tool says a kubectl command is blocked because it is mutating, do not suggest manual kubectl commands. Convert it into a typed write-tool proposal with `propose_action`.

Proposal format:
- `commands_json` must be valid JSON.
- Each step must be either:
  - `{{"command":"kubectl ...","display":"kubectl ..."}}`
  - `{{"tool":"actual_write_tool_name","args":{{...}},"display":"operator-facing summary"}}`
- Read tools are for investigation only. Do not put read tools like `aws_describe_service` or `k8s_get_resources` into `commands_json`.
- Prefer typed write tools over raw `kubectl` commands whenever a matching tool exists.
- Use real names you discovered. Never use placeholders like `<name>`, `REPLACE_ME`, or "replace this".
- If Kubernetes cannot do the change, use the right AWS/K8s write tool instead of inventing a kubectl resource.
- Add a `verification_command` when a kubectl check makes sense.

Allowed write tools for proposal steps:
{write_tool_help}

Rules:
- Propose the minimum effective change.
- Prefer `kubectl rollout restart` on a parent workload over deleting a pod directly.
- For `k8s_exec_in_pod`, prefer a single direct command when possible. Only use `sh -lc` when multiple shell operations are genuinely required.
- If the target pod has multiple containers, inspect the pod spec first and pass the exact container name.
- Do not repeat the same read tool call with identical arguments. If a path stays empty or errors, switch discovery paths or state the uncertainty.
- Risk must be LOW, MEDIUM, HIGH, or CRITICAL.
- After proposing, summarize briefly without placeholders or fill-in-the-blank notes."""

_PLACEHOLDER_RE = re.compile(
    r"<[^>\n]+>|"
    r"\b(?:placeholder|replace[-_ ]?me|fill[-_ ]?me|example[-_ ]?(?:name|id))\b",
    re.IGNORECASE,
)
_SHELL_OPERATOR_TOKENS = ("&&", "||", "|", ">", "<", ";", "&", "$(", "`", "\n", "\r")


@dataclass
class PendingAction:
    id: str
    description: str
    commands: list[dict]        # [{"command": "kubectl ..."}] or [{"tool": "aws_update_auto_scaling", ...}]
    risk: str                   # LOW, MEDIUM, HIGH, CRITICAL
    expected_outcome: str
    verification: dict | None = None   # {"command": "kubectl ..."} or {"tool": "...", "args": {...}}
    status: str = "pending"     # pending | approved | rejected | executed | verified | failed


def _contains_placeholder(value: str) -> bool:
    return bool(_PLACEHOLDER_RE.search(str(value or "")))


def _validate_single_kubectl_command(command: str, field_name: str) -> str | None:
    text = str(command or "").strip()
    if not text:
        return f"{field_name} must not be empty."
    if _contains_placeholder(text):
        return f"{field_name} contains a placeholder."
    if not text.startswith("kubectl "):
        return f"{field_name} must start with 'kubectl '."
    if any(token in text for token in _SHELL_OPERATOR_TOKENS):
        return f"{field_name} must be a single kubectl command without shell operators."
    try:
        parts = shlex.split(text)
    except ValueError as exc:
        return f"{field_name} has invalid shell syntax: {exc}"
    if not parts or parts[0] != "kubectl":
        return f"{field_name} must start with 'kubectl '."
    return None


def _normalize_action_step(raw_step: object) -> dict:
    if isinstance(raw_step, str):
        text = raw_step.strip()
        return {"command": text, "display": text} if text else {}

    if not isinstance(raw_step, dict):
        text = str(raw_step).strip()
        return {"display": text} if text else {}

    step: dict[str, object] = {}
    parse_error = str(raw_step.get("_parse_error", "") or "").strip()
    command = str(raw_step.get("command", "") or "").strip()
    display = str(raw_step.get("display", "") or "").strip()
    tool_name = str(raw_step.get("tool", "") or "").strip()
    args = raw_step.get("args", {})
    if tool_name and not display and isinstance(args, dict):
        nested_display = str(args.get("display", "") or "").strip()
        if nested_display:
            display = nested_display
            args = {key: value for key, value in args.items() if key != "display"}

    if parse_error:
        step["_parse_error"] = parse_error
    if command:
        step["command"] = command
    if display:
        step["display"] = display
    if tool_name:
        step["tool"] = tool_name
        step["args"] = args if isinstance(args, dict) else {}
    return step


def _normalize_action_steps(raw_steps: object) -> list[dict]:
    if isinstance(raw_steps, list):
        return [step for item in raw_steps if (step := _normalize_action_step(item))]

    step = _normalize_action_step(raw_steps)
    return [step] if step else []


def _format_tool_step(tool_name: str, args: dict) -> str:
    try:
        rendered_args = json.dumps(args or {}, sort_keys=True)
    except TypeError:
        rendered_args = str(args)
    return f"{tool_name} {rendered_args}".strip()


def format_action_step_preview(step: dict) -> tuple[str, str, str]:
    """Return a human label, code/text preview, and syntax hint for one action step."""
    return render_action_step_preview(step)


def _looks_like_serialized_steps(value: object) -> bool:
    text = str(value or "").strip()
    return text.startswith("[") or text.startswith("{")


def _parse_jsonish_step_objects(raw_text: str) -> list[object]:
    text = str(raw_text or "").strip()
    if not text.startswith("["):
        return []

    in_string = False
    escape = False
    brace_depth = 0
    object_start: int | None = None
    chunks: list[str] = []

    for index, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue
        if char == "{":
            if brace_depth == 0:
                object_start = index
            brace_depth += 1
            continue
        if char == "}":
            if brace_depth > 0:
                brace_depth -= 1
                if brace_depth == 0 and object_start is not None:
                    chunks.append(text[object_start : index + 1])
                    object_start = None

    if object_start is not None:
        partial = text[object_start:].rstrip()
        if partial.endswith("]"):
            partial = partial[:-1].rstrip()
        if partial.endswith(","):
            partial = partial[:-1].rstrip()
        if brace_depth > 0:
            partial += "}" * brace_depth
        chunks.append(partial)

    parsed_steps: list[object] = []
    for chunk in chunks:
        candidate = chunk.strip()
        if not candidate:
            continue
        try:
            parsed_steps.append(json.loads(candidate))
            continue
        except (json.JSONDecodeError, TypeError):
            pass
        try:
            parsed_steps.append(ast.literal_eval(candidate))
        except (SyntaxError, ValueError):
            continue
    return parsed_steps


def _parse_proposed_steps(commands_json: object) -> list[dict]:
    try:
        raw_steps = json.loads(commands_json)
    except (json.JSONDecodeError, TypeError):
        try:
            raw_steps = ast.literal_eval(commands_json)
        except (SyntaxError, ValueError):
            salvaged_steps = _parse_jsonish_step_objects(str(commands_json or ""))
            if salvaged_steps:
                raw_steps = salvaged_steps
            elif _looks_like_serialized_steps(commands_json):
                raw_steps = [{"_parse_error": "commands_json must be valid JSON or a recoverable JSON-like array of action steps."}]
            else:
                raw_steps = commands_json
    return _normalize_action_steps(raw_steps)


def _first_sentence(text: str) -> str:
    cleaned = " ".join(str(text or "").split())
    if not cleaned:
        return ""
    match = re.search(r"([.!?])\s", cleaned)
    return cleaned[: match.start() + 1] if match else cleaned


def _build_write_tool_help(write_tool_map: dict[str, object]) -> str:
    if not write_tool_map:
        return "- No write tools are available."

    lines: list[str] = []
    for tool_name in sorted(write_tool_map):
        tool = write_tool_map[tool_name]
        arg_names = ", ".join(getattr(tool, "args", {}).keys()) or "no args"
        description = _first_sentence(getattr(tool, "description", ""))
        if description:
            lines.append(f"- `{tool_name}`: {description} Args: {arg_names}.")
        else:
            lines.append(f"- `{tool_name}`: Args: {arg_names}.")
    return "\n".join(lines)


def _validate_proposed_steps(steps: list[dict], allowed_write_tool_names: set[str]) -> str | None:
    if not steps:
        return "commands_json must contain at least one action step."

    for index, step in enumerate(steps, start=1):
        parse_error = str(step.get("_parse_error", "") or "").strip()
        command = str(step.get("command", "") or "").strip()
        tool_name = str(step.get("tool", "") or "").strip()
        display = str(step.get("display", "") or "").strip()
        combined_text = " ".join(part for part in (command, tool_name, display) if part)

        if parse_error:
            return parse_error
        if _contains_placeholder(combined_text):
            return f"step {index} contains a placeholder. Discover the real identifier first."
        if command and tool_name:
            return f"step {index} must use either 'command' or 'tool', not both."
        if command:
            command_error = _validate_single_kubectl_command(command, f"step {index} command")
            if command_error:
                return command_error
            policy_error = guard_k8s_write_tool("approved_kubectl_command", command=command)
            if policy_error:
                return f"step {index}: {policy_error}"
            continue
        if tool_name:
            if tool_name not in allowed_write_tool_names:
                allowed = ", ".join(sorted(allowed_write_tool_names)) or "no write tools available"
                return f"step {index} tool '{tool_name}' is not an allowed write tool. Allowed tools: {allowed}."
            policy_error = guard_tool_invocation(tool_name, step.get("args", {}), write=True)
            if policy_error:
                return f"step {index}: {policy_error}"
            continue
        return f"step {index} must include either 'command' or 'tool'."

    return None


def _validate_verification_command(verification_command: str) -> str | None:
    command = verification_command.strip()
    if not command:
        return None
    syntax_error = _validate_single_kubectl_command(command, "verification_command")
    if syntax_error:
        return syntax_error
    return guard_k8s_read_tool("verification_command", command=command)


def _create_propose_action_tool(captured: list[dict], allowed_write_tool_names: set[str]):
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
            commands_json: JSON array of approved steps. Each entry is either {"command":"kubectl ...","display":"..."} or {"tool":"actual_write_tool_name","args":{...},"display":"..."}.
            risk: Risk level: LOW, MEDIUM, HIGH, or CRITICAL
            expected_outcome: What should happen after successful execution
            verification_command: A kubectl command to verify success, e.g. 'kubectl get pv -o wide' (optional)
        """
        commands = _parse_proposed_steps(commands_json)
        validation_error = _validate_proposed_steps(commands, allowed_write_tool_names)
        if validation_error:
            return f"Proposal rejected: {validation_error}"

        verification_error = _validate_verification_command(verification_command)
        if verification_error:
            return f"Proposal rejected: {verification_error}"

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
            "Now provide a brief summary without placeholders or fill-in-the-blank instructions."
        )

    return propose_action


def handle_action(
    user_input: str,
    chat_history: list,
    llm_with_tools,
    read_tool_map: dict,
    write_tool_map: dict,
    model_name: str,
    tracer: Tracer,
    status_callback: StatusCallback | None = None,
) -> tuple[str, list[PendingAction]]:
    """Handle an action request. Returns (response_text, pending_actions)."""
    cb = status_callback or (lambda _: None)

    # Capture proposals from the LLM
    captured_proposals: list[dict] = []
    propose_tool = _create_propose_action_tool(captured_proposals, set(write_tool_map))
    tool_result_cache: dict[str, str] = {}

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
    system_prompt = _ACTION_SYSTEM_PROMPT_TEMPLATE.format(
        today=today,
        write_tool_help=_build_write_tool_help(write_tool_map),
    )

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
                cache_key = ""
                cache_hit = False
                if tool_name != "propose_action":
                    cache_key = json.dumps({"tool": tool_name, "args": tool_args}, sort_keys=True, default=str)
                    cache_hit = cache_key in tool_result_cache
                if cache_hit:
                    result = tool_result_cache[cache_key]
                else:
                    if tool_name == "propose_action":
                        cb("Preparing approval request...")
                    else:
                        _, preview, _ = render_tool_call_preview(tool_name, tool_args)
                        cb(preview)
                    t1 = time.monotonic()
                    try:
                        result = str(tool.invoke(tool_args))
                    except Exception as exc:
                        result = f"Tool error: {exc}"
                    if cache_key:
                        tool_result_cache[cache_key] = result
                tracer.step(
                    "tool_call", "action",
                    tool_name=tool_name,
                    tool_args=tool_args,
                    tool_result_preview=result[:300],
                    duration_ms=0 if cache_hit else int((time.monotonic() - t1) * 1000),
                    output_summary="cache hit" if cache_hit else "",
                )

            messages.append(ToolMessage(content=result, tool_call_id=tool_id))
            planner_feedback = _planner_feedback_for_tool_result(tool_name, tool_args, result)
            if planner_feedback:
                messages.append(SystemMessage(content=planner_feedback))

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
    return response_text, pending_actions


def _get_base_llm(llm_with_tools):
    """Try to get the underlying LLM from a tool-bound wrapper."""
    # LangChain's RunnableBinding stores the bound object
    if hasattr(llm_with_tools, "bound"):
        return llm_with_tools.bound
    if hasattr(llm_with_tools, "first"):
        return llm_with_tools.first
    return None


def _planner_feedback_for_tool_result(tool_name: str, tool_args: dict, result: str) -> str:
    text = str(result or "").strip()
    if tool_name != "k8s_run_kubectl":
        return ""
    if "blocked in read-only mode" not in text or "typed write tool" not in text:
        return ""

    blocked_command = str(tool_args.get("command", "") or "").strip()
    lines = [
        "Planner note: the previous raw kubectl command was blocked because it is a mutating operation.",
        "Do not retry the same read-only tool and do not provide manual kubectl examples.",
        "Use `propose_action` with the typed write tool and args named in the tool result.",
    ]
    if blocked_command:
        lines.append(f"Blocked command: {blocked_command}")
    return "\n".join(lines)
