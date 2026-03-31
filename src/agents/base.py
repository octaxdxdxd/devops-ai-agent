"""Agent handler base class and shared tool-calling loop."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from ..tracing.tracer import Tracer

log = logging.getLogger(__name__)

StatusCallback = Callable[[str], None]


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
) -> str:
    """Shared tool-calling loop used by diagnose, explain, and lookup handlers.

    Returns the final text response from the LLM.
    """
    cb = status_callback or (lambda _: None)

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
        if not response.tool_calls:
            return (response.content or "").strip()

        # Execute each tool call
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
                tool_ms = int((time.monotonic() - t1) * 1000)
                tracer.step(
                    "tool_call",
                    handler_name,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    tool_result_preview=result[:300],
                    duration_ms=tool_ms,
                )

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
