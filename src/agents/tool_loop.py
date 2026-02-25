"""Tool-call execution loop for the AI Ops agent."""

from __future__ import annotations

import json
import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ..config import Config
from ..tools import is_write_tool
from ..utils.response import extract_response_text
from .approval import ApprovalCoordinator, commands_code_block, format_command_preview


def _tool_result_to_message_content(result: Any) -> tuple[str, int]:
    """Convert tool result to model context with truncation for token control."""
    text = str(result)
    max_chars = max(0, int(getattr(Config, "AGENT_TOOL_RESULT_MAX_CHARS", 5000)))
    if max_chars and len(text) > max_chars:
        return (
            text[:max_chars]
            + "\n... [truncated before sending to model; see tool output in logs/trace if needed]",
            len(text),
        )
    return text, len(text)


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

        tool_messages: list[ToolMessage] = []
        for tool_call in current_response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            total_tool_calls += 1
            if total_tool_calls > max_tool_calls:
                if tw and trace_id:
                    tw.emit(
                        {
                            "trace_id": trace_id,
                            "event": "tool_loop.call_budget_hit",
                            "max_tool_calls": max_tool_calls,
                            "attempted_tool": tool_name,
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

                forced = llm.invoke(messages)
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
                        tool_call_id=tool_call["id"],
                    )
                )
                continue

            if tw and trace_id:
                tw.emit({"trace_id": trace_id, "event": "tool.request", "tool": tool_name, "args": tool_args})

            tool_func = tool_lookup.get(tool_name)
            if tool_func is None:
                tool_messages.append(
                    ToolMessage(
                        content=f"Error: unknown tool '{tool_name}'.",
                        tool_call_id=tool_call["id"],
                    )
                )
                continue

            if is_write_tool(tool_name):
                if tool_name == "restart_kubernetes_pod":
                    content_text = extract_response_text(current_response)
                    approval.record_restart_context(tool_args, content_text)

                cmd_preview = format_command_preview(tool_name, tool_args)
                approval.set_pending_action(tool_func, tool_args)

                if tw and trace_id:
                    tw.emit(
                        {
                            "trace_id": trace_id,
                            "event": "tool.requires_approval",
                            "tool": tool_name,
                            "args": tool_args,
                        }
                    )

                batch_prompt = ""
                if approval.should_offer_batch_prompt(tool_name, tool_args):
                    batch_prompt = "If you want all suggested pods restarted in one operation, reply: `do all at once`.\n"

                cmd_block = commands_code_block(cmd_preview)
                return (
                    f"I recommend running `{tool_name}` with args {tool_args}, but it requires approval.\n"
                    "Planned command(s):\n"
                    f"{cmd_block}\n"
                    f"{batch_prompt}"
                    "Would you like me to proceed? (yes/no)"
                )

            try:
                t_tool0 = time.perf_counter()
                result = tool_func.invoke(tool_args)
                result_content, raw_result_len = _tool_result_to_message_content(result)
                if tw and trace_id:
                    tw.emit(
                        {
                            "trace_id": trace_id,
                            "event": "tool.result",
                            "tool": tool_name,
                            "duration_ms": (time.perf_counter() - t_tool0) * 1000.0,
                            "result_type": type(result).__name__,
                            "result_len": raw_result_len,
                            "result_injected_len": len(result_content),
                        }
                    )
                tool_messages.append(ToolMessage(content=result_content, tool_call_id=tool_call["id"]))
            except Exception as exc:
                if tw and trace_id:
                    tw.emit({"trace_id": trace_id, "event": "tool.error", "tool": tool_name, "error": str(exc)})
                tool_messages.append(ToolMessage(content=f"Error: {exc}", tool_call_id=tool_call["id"]))

        messages.append(AIMessage(content=current_response.content, tool_calls=current_response.tool_calls))
        messages.extend(tool_messages)

        t1 = time.perf_counter()
        current_response = llm_with_tools.invoke(messages)
        if tw and trace_id:
            tw.emit(
                {
                    "trace_id": trace_id,
                    "event": "llm.invoke",
                    "duration_ms": (time.perf_counter() - t1) * 1000.0,
                    "has_tool_calls": bool(getattr(current_response, "tool_calls", None)),
                    "usage": getattr(current_response, "usage_metadata", None)
                    or getattr(current_response, "response_metadata", None),
                }
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

        t_force = time.perf_counter()
        forced = llm.invoke(messages)
        if tw and trace_id:
            tw.emit(
                {
                    "trace_id": trace_id,
                    "event": "llm.invoke.force_final",
                    "duration_ms": (time.perf_counter() - t_force) * 1000.0,
                    "usage": getattr(forced, "usage_metadata", None) or getattr(forced, "response_metadata", None),
                }
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
