"""Agent handler base class and shared tool-calling loop."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from ..tools.command_preview import render_tool_call_preview
from ..tracing.tracer import Tracer

log = logging.getLogger(__name__)

StatusCallback = Callable[[str], None]


def _tool_cache_key(tool_name: str, tool_args: Any) -> str:
    payload = {"tool": str(tool_name or "").strip(), "args": tool_args or {}}
    return json.dumps(payload, sort_keys=True, default=str)


def extract_token_usage(response: AIMessage) -> tuple[int, int]:
    """Extract (tokens_in, tokens_out) from a LangChain AIMessage."""
    # usage_metadata (newer LangChain)
    um = getattr(response, "usage_metadata", None)
    if um and isinstance(um, dict):
        return um.get("input_tokens", 0), um.get("output_tokens", 0)
    # response_metadata (openai-style)
    rm = getattr(response, "response_metadata", None) or {}
    usage = rm.get("token_usage") or rm.get("usage") or {}
    return usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)


def _is_data_bearing_tool_result(result: str) -> bool:
    text = str(result or "").strip()
    if not text or text == "(empty output)":
        return False
    if text.startswith("ERROR:") or text.startswith("Tool error:") or text.startswith("Unknown tool:"):
        return False
    lowered = text.lower()
    if lowered.startswith("no resources found"):
        return False
    return True


def _tool_policy_retry_message(original_query: str, capability_families: list[str]) -> str:
    families = ", ".join(capability_families) if capability_families else "the bound read tools"
    return (
        "This request requires live inspection before you answer.\n\n"
        f"Available capability families in this turn: {families}.\n"
        f"Original query: {original_query}\n\n"
        "Call a relevant bound read tool first, then answer from the tool output. "
        "Do not claim a tool family is unavailable if it is bound in this turn. "
        "If the obvious path is empty, try an alternate namespace, label, parent resource, or region before answering. "
        "Only answer without a tool call if no bound tool can answer the question or every relevant tool failed, and say that explicitly."
    )


def _allows_explicit_no_tool_answer(text: str) -> bool:
    lowered = str(text or "").lower()
    required_phrases = (
        "no bound tool can answer",
        "none of the bound tools can answer",
        "none of the available tools can answer",
        "every relevant tool failed",
        "all relevant tools failed",
        "the bound tools cannot answer",
    )
    return any(phrase in lowered for phrase in required_phrases)


def run_tool_loop(
    *,
    messages: list,
    llm_with_tools,
    tool_map: dict[str, Any],
    max_steps: int,
    handler_name: str,
    model_name: str,
    tracer: Tracer,
    status_callback: StatusCallback | None = None,
    checkpoint_step: int | None = None,
    original_query: str = "",
    system_prompt: str = "",
    require_relevant_tool_call_before_answer: bool = False,
    available_capability_families: list[str] | None = None,
    insufficient_tool_names: set[str] | None = None,
) -> str:
    """Shared tool-calling loop used by diagnose, explain, and lookup handlers.

    Returns the final text response from the LLM.
    """
    cb = status_callback or (lambda _: None)
    tool_result_cache: dict[str, str] = {}
    capability_families = available_capability_families or []
    insufficient_names = set(insufficient_tool_names or set())
    has_sufficient_tool_call = False
    forced_tool_retry_sent = False

    for step in range(max_steps):
        t0 = time.monotonic()
        try:
            response: AIMessage = llm_with_tools.invoke(messages)
        except Exception as exc:
            tracer.step("error", handler_name, error=str(exc))
            return f"LLM call failed: {exc}"
        elapsed = int((time.monotonic() - t0) * 1000)

        tokens_in, tokens_out = extract_token_usage(response)
        tracer.step(
            "llm_call",
            handler_name,
            input_summary=f"step {step + 1}/{max_steps}",
            output_summary=(response.content or "")[:200],
            llm_model=model_name,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            duration_ms=elapsed,
        )

        messages.append(response)

        # If no tool calls, LLM has produced a final answer
        tool_calls = list(getattr(response, "tool_calls", []) or [])

        if not tool_calls:
            final_text = (response.content or "").strip()
            if require_relevant_tool_call_before_answer and tool_map and not has_sufficient_tool_call:
                if not forced_tool_retry_sent:
                    cb("Requesting live inspection...")
                    retry_message = _tool_policy_retry_message(original_query, capability_families)
                    tracer.step("checkpoint", handler_name, output_summary=retry_message[:300])
                    messages.append(HumanMessage(content=retry_message))
                    forced_tool_retry_sent = True
                    continue
                if _allows_explicit_no_tool_answer(final_text):
                    return final_text
                return (
                    "I wasn’t able to complete a live inspection with the bound read tools. "
                    "Please retry the request."
                )
            return final_text

        # Execute each tool call
        for tc in tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            tool_id = tc.get("id", tool_name)

            tool = tool_map.get(tool_name)
            if not tool:
                result = f"Unknown tool: {tool_name}"
                cache_hit = False
                tool_ms = 0
            else:
                cache_key = _tool_cache_key(tool_name, tool_args)
                cache_hit = cache_key in tool_result_cache
                if cache_hit:
                    result = tool_result_cache[cache_key]
                    tool_ms = 0
                else:
                    _, preview, _ = render_tool_call_preview(tool_name, tool_args)
                    cb(preview)
                    t1 = time.monotonic()
                    try:
                        result = str(tool.invoke(tool_args))
                    except Exception as exc:
                        result = f"Tool error: {exc}"
                    tool_ms = int((time.monotonic() - t1) * 1000)
                    tool_result_cache[cache_key] = result
                tracer.step(
                    "tool_call",
                    handler_name,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    tool_result_preview=result[:300],
                    output_summary="cache hit" if cache_hit else "",
                    duration_ms=tool_ms,
                )
                if tool_name not in insufficient_names and _is_data_bearing_tool_result(result):
                    has_sufficient_tool_call = True

            messages.append(ToolMessage(content=result, tool_call_id=tool_id))

        # Mid-investigation checkpoint: compress context to save tokens
        if (
            checkpoint_step is not None
            and step == checkpoint_step
            and len(messages) > 8
        ):
            cb("Compressing investigation context...")
            summary = _compress_context(messages, llm_with_tools, tracer, handler_name, model_name)
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(
                    content=(
                        f"Continuing investigation.\n\n"
                        f"Original query: {original_query}\n\n"
                        f"Findings so far:\n{summary}\n\n"
                        f"Continue investigating. Focus on remaining unknowns."
                    )
                ),
            ]
            tracer.step("checkpoint", handler_name, output_summary=summary[:300])

    # Max steps exhausted — ask for summary
    cb("Summarizing findings...")
    messages.append(
        HumanMessage(
            content=(
                "You have reached the maximum investigation steps. "
                "Summarize your findings so far. Clearly state what you confirmed, "
                "what remains unknown, and recommended next steps."
            )
        )
    )
    try:
        final: AIMessage = llm_with_tools.invoke(messages)
        tokens_in, tokens_out = extract_token_usage(final)
        tracer.step(
            "llm_call",
            handler_name,
            input_summary="final summary",
            llm_model=model_name,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )
        return (final.content or "").strip()
    except Exception as exc:
        return f"Failed to generate summary: {exc}"


def _compress_context(
    messages: list,
    llm,
    tracer: Tracer,
    handler_name: str,
    model_name: str,
) -> str:
    """Ask the LLM to summarize its investigation findings so far."""
    compress_messages = list(messages) + [
        HumanMessage(
            content=(
                "Summarize your investigation findings so far in concise bullet points. "
                "What have you confirmed? What hypotheses remain? What data is missing?"
            )
        )
    ]
    try:
        resp: AIMessage = llm.invoke(compress_messages)
        tokens_in, tokens_out = extract_token_usage(resp)
        tracer.step(
            "llm_call",
            handler_name,
            input_summary="checkpoint compression",
            llm_model=model_name,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )
        return (resp.content or "").strip()
    except Exception:
        # Fallback: manual extraction of tool results
        parts = []
        for msg in messages:
            if isinstance(msg, ToolMessage):
                parts.append(msg.content[:200])
        return "\n".join(parts[-5:])
