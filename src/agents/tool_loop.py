"""Tool-call execution loop for the AI Ops agent."""

from __future__ import annotations

import json
import re
import shlex
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ..config import Config
from ..tools import is_write_tool
from ..utils.command_intent import CommandIntent, classify_command_intent, target_tool_for_intent
from ..utils.llm_retry import invoke_with_retries
from ..utils.response import extract_response_text
from .approval import ApprovalCoordinator, PendingAction, commands_code_block, format_command_preview


_AWS_ARN_RE = re.compile(r"\barn:aws[a-z-]*:[^\s'\"]+\b", re.IGNORECASE)
_AWS_ID_TOKEN_RE = re.compile(
    r"\b(?:ami|i|lt|sg|subnet|vpc|vol|snap|eni|igw|nat|rtb|eipalloc|eipassoc|vpce|pcx|tgw|fs|db)-[0-9a-zA-Z]{6,}\b",
    re.IGNORECASE,
)


def _resolve_tool_call(
    *,
    tool_name: str,
    tool_args: Any,
    tool_lookup: dict[str, Any],
) -> tuple[str, Any, CommandIntent | None, bool]:
    """Route command tools by command intent; returns effective tool + args."""
    if not isinstance(tool_args, dict):
        return tool_name, tool_args, None, False

    command_text = str(tool_args.get("command") or "").strip()
    if not command_text:
        return tool_name, tool_args, None, False

    intent = classify_command_intent(tool_name, command_text)
    if intent.family == "unknown":
        return tool_name, tool_args, None, False

    effective_name = tool_name
    target_name = target_tool_for_intent(intent)
    if target_name and target_name in tool_lookup:
        effective_name = target_name

    normalized_args = dict(tool_args)
    if intent.normalized_command:
        normalized_args["command"] = intent.normalized_command

    routed = effective_name != tool_name
    return effective_name, normalized_args, intent, routed


def _requires_explicit_approval(*, tool_name: str, intent: CommandIntent | None) -> bool:
    """Approval is determined by command mutability when available; else tool policy."""
    if intent is not None:
        return intent.is_mutating
    return is_write_tool(tool_name)


_VERBOSE_CONTEXT_TOOLS = {
    "k8s_describe_pod",
    "k8s_describe_deployment",
    "k8s_describe_node",
    "k8s_get_pod_logs",
    "k8s_get_events",
    "k8s_get_resource_yaml",
    "k8s_get_pod_scheduling_report",
    "kubectl_readonly",
    "aws_cli_readonly",
}
_SIGNAL_LINE_RE = re.compile(
    r"(warning|error|fail|backoff|crashloop|oom|evict|pending|notready|unschedul|reason|message|event|status|condition)",
    re.IGNORECASE,
)


def _tool_context_budget(tool_name: str, default_max: int) -> int:
    """Return a per-tool context budget for prompt injection."""
    if tool_name in {"k8s_get_pod_logs", "k8s_get_resource_yaml", "k8s_describe_node", "kubectl_readonly"}:
        return max(800, min(default_max, 1800))
    if tool_name in {
        "k8s_describe_pod",
        "k8s_describe_deployment",
        "k8s_get_events",
        "k8s_get_pod_scheduling_report",
        "aws_cli_readonly",
    }:
        return max(900, min(default_max, 2200))
    return default_max


def _compact_high_signal_lines(text: str, *, max_lines: int = 120) -> str:
    """Extract likely high-signal lines to preserve RCA quality with lower token usage."""
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text

    selected: list[str] = []
    for line in lines:
        raw = line.strip()
        if not raw:
            continue
        if raw.endswith(":") or _SIGNAL_LINE_RE.search(raw):
            selected.append(line)
            if len(selected) >= max_lines:
                break

    if not selected:
        # Fallback to tail if no obvious signal lines were found.
        selected = lines[-max_lines:]

    return "\n".join(selected)


def _tool_result_to_message_content(result: Any, *, tool_name: str) -> tuple[str, int]:
    """Convert tool result to model context with truncation for token control."""
    text = str(result)
    max_chars_default = max(0, int(getattr(Config, "AGENT_TOOL_RESULT_MAX_CHARS", 5000)))
    max_chars = _tool_context_budget(tool_name, max_chars_default)

    if tool_name in _VERBOSE_CONTEXT_TOOLS and len(text) > max_chars:
        compacted = _compact_high_signal_lines(text)
        if compacted and len(compacted) < len(text):
            text = (
                "[Context compressed before model injection to reduce tokens while preserving high-signal evidence]\n"
                + compacted
            )

    if max_chars and len(text) > max_chars:
        marker = "\n... [middle truncated before sending to model; see full tool output in logs/trace] ...\n"
        marker_len = len(marker)

        if max_chars <= marker_len + 64:
            trimmed = text[:max_chars]
        else:
            head = int(max_chars * 0.7)
            tail = max_chars - head - marker_len
            if tail < 128:
                tail = 128
                head = max_chars - tail - marker_len
            if head < 64:
                head = 64
                tail = max_chars - head - marker_len

            trimmed = text[:head] + marker + text[-tail:]

        return trimmed, len(text)
    return text, len(text)


def _message_content_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(item) for item in content)
    return str(content or "")


def _build_evidence_corpus(messages: list[Any]) -> str:
    """Collect prior-turn/tool evidence text (excluding current model draft)."""
    parts: list[str] = []
    total = 0
    max_chars = 200_000
    for msg in messages:
        text = _message_content_text(msg)
        if not text:
            continue
        remaining = max_chars - total
        if remaining <= 0:
            break
        if len(text) > remaining:
            text = text[:remaining]
        parts.append(text)
        total += len(text)
    return "\n".join(parts)


def _extract_aws_reference_tokens(command: str) -> list[str]:
    """Extract likely AWS resource identifiers (IDs/ARNs) from command text."""
    raw = (command or "").strip()
    if not raw:
        return []

    out: list[str] = []
    seen: set[str] = set()

    for match in _AWS_ARN_RE.findall(raw):
        key = match.lower()
        if key not in seen:
            seen.add(key)
            out.append(match)

    for match in _AWS_ID_TOKEN_RE.findall(raw):
        key = match.lower()
        if key not in seen:
            seen.add(key)
            out.append(match)

    try:
        tokens = shlex.split(raw)
    except ValueError:
        tokens = raw.split()

    if tokens and tokens[0].lower() == "aws":
        tokens = tokens[1:]

    def add_value(value: str) -> None:
        for chunk in re.split(r"[,\s]+", value.strip()):
            item = chunk.strip().strip("'\"")
            if not item:
                continue
            if _AWS_ARN_RE.fullmatch(item) or _AWS_ID_TOKEN_RE.fullmatch(item):
                key = item.lower()
                if key not in seen:
                    seen.add(key)
                    out.append(item)

    i = 0
    while i < len(tokens):
        token = str(tokens[i]).strip()
        low = token.lower()
        if not low.startswith("--"):
            i += 1
            continue

        value = ""
        if "=" in token:
            flag, value = token.split("=", 1)
        else:
            flag = token
            if i + 1 < len(tokens):
                next_token = str(tokens[i + 1]).strip()
                if not next_token.startswith("--"):
                    value = next_token
                    i += 1

        flag_low = flag.lower()
        if "id" in flag_low or "arn" in flag_low:
            add_value(value)
        i += 1

    return out


def _validate_aws_write_grounding(command: str, evidence_corpus: str) -> tuple[list[str], str] | None:
    """Block write commands that introduce unverified AWS IDs/ARNs."""
    refs = _extract_aws_reference_tokens(command)
    if not refs:
        return None

    corpus_low = (evidence_corpus or "").lower()
    unresolved = [ref for ref in refs if ref.lower() not in corpus_low]
    if not unresolved:
        return None

    message = (
        "Write command blocked: unverified AWS resource identifiers found in the proposed write command.\n"
        f"Unverified identifiers: {', '.join(unresolved[:8])}\n"
        "Run read-only AWS checks first to verify these resources exist and are correct, then propose the write again."
    )
    return unresolved, message


def handle_tool_calls(
    *,
    response: Any,
    user_input: str,
    chat_history: list,
    prompt: Any,
    llm: Any,
    llm_with_tools: Any,
    tools: list,
    tools_by_name: dict[str, Any] | None = None,
    approval: ApprovalCoordinator,
    trace_writer: Any = None,
    trace_id: str | None = None,
) -> str:
    """Execute iterative tool calls until the model produces a final response."""
    tw = trace_writer

    messages = prompt.format_messages(chat_history=chat_history, input=user_input)
    tool_lookup = tools_by_name or {tool.name: tool for tool in tools}

    max_iterations = getattr(Config, "MAX_ITERATIONS", 5)
    max_tool_calls = getattr(Config, "MAX_TOOL_CALLS_PER_TURN", 12)
    max_duplicate_tool_calls = getattr(Config, "MAX_DUPLICATE_TOOL_CALLS", 2)

    iteration = 0
    total_tool_calls = 0
    call_signature_counts: dict[str, int] = {}
    current_response = response

    while iteration < max_iterations:
        iteration += 1

        if not (hasattr(current_response, "tool_calls") and current_response.tool_calls):
            return extract_response_text(current_response)

        resolved_calls: list[dict[str, Any]] = []
        for tool_call in current_response.tool_calls:
            original_name = str(tool_call["name"])
            original_args = tool_call["args"]
            effective_name, effective_args, intent, routed = _resolve_tool_call(
                tool_name=original_name,
                tool_args=original_args,
                tool_lookup=tool_lookup,
            )
            resolved_calls.append(
                {
                    "tool_call_id": tool_call["id"],
                    "original_name": original_name,
                    "original_args": original_args,
                    "name": effective_name,
                    "args": effective_args,
                    "intent": intent,
                    "routed": routed,
                    "requires_approval": _requires_explicit_approval(tool_name=effective_name, intent=intent),
                }
            )

        tool_messages: list[ToolMessage] = []
        for call_idx, resolved in enumerate(resolved_calls):
            tool_call_id = resolved["tool_call_id"]
            original_name = resolved["original_name"]
            original_args = resolved["original_args"]
            tool_name = resolved["name"]
            tool_args = resolved["args"]
            tool_intent = resolved["intent"]
            requires_approval = bool(resolved["requires_approval"])

            total_tool_calls += 1
            if total_tool_calls > max_tool_calls:
                if tw and trace_id:
                    tw.emit(
                        {
                            "trace_id": trace_id,
                            "event": "tool_loop.call_budget_hit",
                            "max_tool_calls": max_tool_calls,
                            "attempted_tool": tool_name,
                            "requested_tool": original_name,
                        }
                    )

                messages.append(AIMessage(content=current_response.content, tool_calls=current_response.tool_calls))
                messages.append(
                    HumanMessage(
                        content=(
                            "Stop calling tools now. You have enough evidence. "
                            "Provide your best final incident summary using existing tool results."
                        )
                    )
                )

                forced = invoke_with_retries(
                    llm,
                    messages,
                    trace_writer=tw,
                    trace_id=trace_id,
                    event="llm.invoke.force_budget",
                )
                forced_text = extract_response_text(forced)
                if (forced_text or "").strip():
                    return forced_text

                return "I stopped tool execution due to safety budget limits. Please narrow the request (service/pod/namespace/time window)."

            try:
                signature = f"{tool_name}:{json.dumps(tool_args, sort_keys=True, ensure_ascii=False)}"
            except Exception:
                signature = f"{tool_name}:{str(tool_args)}"

            call_signature_counts[signature] = call_signature_counts.get(signature, 0) + 1
            if call_signature_counts[signature] > max_duplicate_tool_calls:
                if tw and trace_id:
                    tw.emit(
                        {
                            "trace_id": trace_id,
                            "event": "tool_loop.duplicate_suppressed",
                            "tool": tool_name,
                            "requested_tool": original_name,
                            "args": tool_args,
                            "count": call_signature_counts[signature],
                            "max_duplicate_tool_calls": max_duplicate_tool_calls,
                        }
                    )
                tool_messages.append(
                    ToolMessage(
                        content=(
                            "Duplicate tool call suppressed to avoid loops. "
                            "Use previous tool results and provide a final answer."
                        ),
                        tool_call_id=tool_call_id,
                    )
                )
                continue

            if tw and trace_id:
                if resolved["routed"]:
                    intent_payload: dict[str, Any] = {}
                    if tool_intent is not None:
                        intent_payload = {
                            "family": tool_intent.family,
                            "verb": tool_intent.verb,
                            "is_mutating": tool_intent.is_mutating,
                        }
                    tw.emit(
                        {
                            "trace_id": trace_id,
                            "event": "tool.route",
                            "requested_tool": original_name,
                            "requested_args": original_args,
                            "tool": tool_name,
                            "args": tool_args,
                            "intent": intent_payload,
                        }
                    )
                tw.emit(
                    {
                        "trace_id": trace_id,
                        "event": "tool.request",
                        "tool": tool_name,
                        "requested_tool": original_name,
                        "args": tool_args,
                    }
                )

            tool_func = tool_lookup.get(tool_name)
            if tool_func is None:
                tool_messages.append(
                    ToolMessage(
                        content=f"Error: unknown tool '{tool_name}'.",
                        tool_call_id=tool_call_id,
                    )
                )
                continue

            if requires_approval:
                evidence_corpus = _build_evidence_corpus(messages)
                pending_actions: list[PendingAction] = []
                blocked_messages: list[ToolMessage] = []

                for pending in resolved_calls[call_idx:]:
                    pending_name = str(pending["name"])
                    pending_args = pending["args"]
                    pending_intent = pending["intent"]
                    pending_tool_call_id = str(pending["tool_call_id"])

                    if not bool(pending["requires_approval"]):
                        continue

                    pending_tool = tool_lookup.get(pending_name)
                    if pending_tool is None:
                        blocked_messages.append(
                            ToolMessage(
                                content=f"Error: unknown write tool '{pending_name}'.",
                                tool_call_id=pending_tool_call_id,
                            )
                        )
                        continue

                    if not isinstance(pending_args, dict):
                        blocked_messages.append(
                            ToolMessage(
                                content=f"Error: invalid args for write tool '{pending_name}'.",
                                tool_call_id=pending_tool_call_id,
                            )
                        )
                        continue

                    if pending_intent is not None and pending_intent.family == "aws" and pending_intent.is_mutating:
                        command_text = str((pending_args or {}).get("command") or "")
                        grounding_issue = _validate_aws_write_grounding(command_text, evidence_corpus)
                        if grounding_issue is not None:
                            unresolved, guard_msg = grounding_issue
                            if tw and trace_id:
                                tw.emit(
                                    {
                                        "trace_id": trace_id,
                                        "event": "tool.write_blocked_unverified_ids",
                                        "tool": pending_name,
                                        "unverified_identifiers": unresolved,
                                    }
                                )
                            blocked_messages.append(
                                ToolMessage(
                                    content=guard_msg,
                                    tool_call_id=pending_tool_call_id,
                                )
                            )
                            continue

                    pending_actions.append(PendingAction(tool=pending_tool, args=pending_args))

                if not pending_actions:
                    tool_messages.extend(blocked_messages)
                    break

                if len(pending_actions) > 1:
                    approval.set_pending_actions(pending_actions)
                    preview_lines: list[str] = []
                    for action in pending_actions:
                        preview = format_command_preview(action.tool.name, action.args)
                        if preview:
                            preview_lines.extend(preview.splitlines())
                    cmd_preview = "\n".join(preview_lines) if preview_lines else "- command preview unavailable"
                    cmd_block = commands_code_block(cmd_preview)
                    if tw and trace_id:
                        tw.emit(
                            {
                                "trace_id": trace_id,
                                "event": "tool.requires_approval",
                                "tool": "<batch>",
                                "batch_size": len(pending_actions),
                                "tools": [action.tool.name for action in pending_actions],
                            }
                        )
                    return (
                        f"I recommend running {len(pending_actions)} write actions as one approved plan.\n"
                        "Planned command(s):\n"
                        f"{cmd_block}\n"
                        "Would you like me to proceed with all of them? (yes/no)"
                    )

                single = pending_actions[0]
                single_name = single.tool.name
                single_args = single.args

                if single_name == "restart_kubernetes_pod":
                    content_text = extract_response_text(current_response)
                    approval.record_restart_context(single_args, content_text)

                cmd_preview = format_command_preview(single_name, single_args)
                approval.set_pending_action(single.tool, single_args)

                if tw and trace_id:
                    tw.emit(
                        {
                            "trace_id": trace_id,
                            "event": "tool.requires_approval",
                            "tool": single_name,
                            "args": single_args,
                        }
                    )

                batch_prompt = ""
                if approval.should_offer_batch_prompt(single_name, single_args):
                    batch_prompt = "If you want all suggested pods restarted in one operation, reply: `do all at once`.\n"

                cmd_block = commands_code_block(cmd_preview)
                return (
                    f"I recommend running `{single_name}` with args {single_args}, but it requires approval.\n"
                    "Planned command(s):\n"
                    f"{cmd_block}\n"
                    f"{batch_prompt}"
                    "Would you like me to proceed? (yes/no)"
                )

            try:
                result = tool_func.invoke(tool_args)
                result_content, raw_result_len = _tool_result_to_message_content(result, tool_name=tool_name)
                if tw and trace_id:
                    tw.emit(
                        {
                            "trace_id": trace_id,
                            "event": "tool.result",
                            "tool": tool_name,
                            "requested_tool": original_name,
                            "result_type": type(result).__name__,
                            "result_len": raw_result_len,
                            "result_injected_len": len(result_content),
                        }
                    )
                tool_messages.append(ToolMessage(content=result_content, tool_call_id=tool_call_id))
            except Exception as exc:
                if tw and trace_id:
                    tw.emit(
                        {
                            "trace_id": trace_id,
                            "event": "tool.error",
                            "tool": tool_name,
                            "requested_tool": original_name,
                            "error": str(exc),
                        }
                    )
                tool_messages.append(ToolMessage(content=f"Error: {exc}", tool_call_id=tool_call_id))

        messages.append(AIMessage(content=current_response.content, tool_calls=current_response.tool_calls))
        messages.extend(tool_messages)

        current_response = invoke_with_retries(
            llm_with_tools,
            messages,
            trace_writer=tw,
            trace_id=trace_id,
            event="llm.invoke",
        )

    if hasattr(current_response, "tool_calls") and current_response.tool_calls:
        if tw and trace_id:
            tw.emit(
                {
                    "trace_id": trace_id,
                    "event": "tool_loop.max_iterations_hit",
                    "max_iterations": max_iterations,
                    "remaining_tool_calls": len(current_response.tool_calls),
                }
            )

        messages.append(
            HumanMessage(
                content=(
                    "Stop calling tools now. Provide your best incident summary based only on the tool results already retrieved. "
                    "If the evidence is insufficient, ask ONE specific clarifying question instead of calling more tools."
                )
            )
        )

        forced = invoke_with_retries(
            llm,
            messages,
            trace_writer=tw,
            trace_id=trace_id,
            event="llm.invoke.force_final",
        )

        forced_text = extract_response_text(forced)
        if (forced_text or "").strip():
            return forced_text

    final = extract_response_text(current_response)
    if not (final or "").strip():
        trace_hint = f" Trace ID: {trace_id}" if (tw and trace_id) else ""
        return (
            "I got an empty response from the model at the end of the tool loop. "
            f"Provider={Config.LLM_PROVIDER}, Model={Config.get_active_model_name()}."
            + trace_hint
        )

    return final
