"""Lookup handler — fast, direct data retrieval with minimal LLM usage."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage, SystemMessage

from ..config import Config
from ..tracing.tracer import Tracer
from .base import StatusCallback, run_tool_loop

log = logging.getLogger(__name__)

_LOOKUP_SYSTEM_PROMPT_TEMPLATE = """\
You are an infrastructure lookup assistant. The user wants specific data from Kubernetes or AWS.
Today's date is {today}.

Rules:
- Perform read-only lookups immediately; do not ask for permission first.
- Use tools before answering. If a lookup fails or is empty, try other namespaces, labels, related resources, or AWS regions.
- Do not repeat the same tool call with identical arguments. If a path is empty or errors twice, switch strategy or answer with uncertainty.
- Show all returned items that matter; do not silently drop rows or records.
- For AWS, be region-aware. For load balancers, check both `elbv2` and `elb` when relevant.
- For cost questions, use {today} for date math; "recent" means the last 30 days.
- This handler is read-only. If the user wants a change, direct them to the action flow.

Presentation:
- Answer the exact question directly.
- Use markdown tables for tabular results.
- If something is empty, say where you checked."""


def handle_lookup(
    user_input: str,
    chat_history: list,
    llm_with_tools,
    tool_map: dict,
    model_name: str,
    tracer: Tracer,
    status_callback: StatusCallback | None = None,
) -> str:
    """Handle a simple data retrieval query. Typically 1-2 tool calls."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    system_prompt = _LOOKUP_SYSTEM_PROMPT_TEMPLATE.format(today=today)

    messages = [
        SystemMessage(content=system_prompt),
        *chat_history[-8:],   # Enough history for follow-up context
        HumanMessage(content=user_input),
    ]

    return run_tool_loop(
        messages=messages,
        llm_with_tools=llm_with_tools,
        tool_map=tool_map,
        max_steps=Config.LOOKUP_MAX_STEPS,
        handler_name="lookup",
        model_name=model_name,
        tracer=tracer,
        status_callback=status_callback,
        system_prompt=system_prompt,
        original_query=user_input,
    )
