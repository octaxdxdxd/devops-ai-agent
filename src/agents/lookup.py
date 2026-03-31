"""Lookup handler — fast, direct data retrieval with minimal LLM usage."""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from ..config import Config
from ..tools.registry import ToolRegistry
from ..tracing.tracer import Tracer
from .base import StatusCallback, run_tool_loop

log = logging.getLogger(__name__)

LOOKUP_SYSTEM_PROMPT = """\
You are an infrastructure lookup assistant. The user wants specific data from Kubernetes or AWS.

CRITICAL BEHAVIORAL RULES:
- NEVER ask the user for confirmation before executing read-only lookups. Just do it.
- NEVER say "Would you like me to..." or "Shall I..." — execute the query directly.
- If the user says "yes", "do it", or similar, look at the conversation history to determine what they agreed to, then execute it immediately.
- When the user asks a question, answer it by calling tools. Do not respond with text-only messages unless you have already retrieved and presented the data.

DATA ACCURACY RULES:
- Present ALL data returned by tools. Never summarize away items, counts, or records.
- If a tool returns a list of 6 items, you MUST show all 6 — not "here is one of them".
- For AWS: be REGION-AWARE. If results are empty in one region, try other common regions (us-east-1, us-west-2, eu-west-1, eu-central-1) before concluding.
- For load balancers: check BOTH 'elbv2' (ALB/NLB) AND 'elb' (Classic) services.
- If results are empty, say "No results in region X" — never "There are no X".
- Use aws_describe_service or k8s_run_kubectl for queries not covered by other specific tools.

PRESENTATION:
- After getting tool results, present them clearly and concisely.
- If the data is tabular, use markdown tables.
- If a tool returns an error, report it clearly."""


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
    messages = [
        SystemMessage(content=LOOKUP_SYSTEM_PROMPT),
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
        system_prompt=LOOKUP_SYSTEM_PROMPT,
        original_query=user_input,
    )
