"""Lookup handler — fast, direct data retrieval with minimal LLM usage."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage, SystemMessage

from ..config import Config
from ..tracing.tracer import Tracer
from .base import StatusCallback, run_tool_loop
from .read_policy import ReadScopeResult, select_read_tools

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
- For Lambda/EventBridge schedule questions (frequency, last run, next run), use `aws_inspect_lambda_schedules`.
- For schedule questions, do not infer actor/history from caller identity and do not use CloudTrail unless the user explicitly asks for audit/history/actor data.
- For schedule questions, treat `schedule_expression` as configuration context only. Never derive `next_run_time` directly from it.
- If `next_run_confidence` is not `high`, say the next run is unknown.
- Prioritize `schedules` in the answer and mention `related_schedules` only as secondary context.
- For CloudTrail/event-history/audit questions, use `aws_audit_cloudtrail` instead of `aws_describe_service`.
- CloudTrail `Username` and `EventName` lookups are exact-match server-side, and CloudTrail accepts only one lookup attribute per request.
- For delete-style CloudTrail searches, use `event_name_prefix="Delete"` rather than assuming CloudTrail supports `Delete*` or partial `EventName` matching.
- If a CloudTrail search is empty, report the region and principal variants that were checked.
- For cost questions, use {today} for date math; "recent" means the last 30 days.
- This handler is read-only. If the user wants a change, direct them to the action flow.

{capability_prompt}

Presentation:
- Answer the exact question directly.
- Use markdown tables for tabular results.
- If something is empty, say where you checked."""


def select_lookup_tools(user_input: str, chat_history: list, read_tools: list) -> list:
    k8s_tools = [tool for tool in read_tools if getattr(tool, "name", "").startswith("k8s_")]
    aws_tools = [tool for tool in read_tools if getattr(tool, "name", "").startswith("aws_")]
    return select_read_tools(
        user_input,
        chat_history,
        k8s_tools,
        aws_tools,
        scope=ReadScopeResult(backend="mixed", specialization="none", confidence="low"),
    ).tools


def handle_lookup(
    user_input: str,
    chat_history: list,
    llm_with_tools,
    tool_map: dict,
    model_name: str,
    tracer: Tracer,
    status_callback: StatusCallback | None = None,
    capability_prompt: str = "",
    require_live_inspection: bool = False,
    available_capability_families: list[str] | None = None,
    insufficient_tool_names: set[str] | None = None,
    specialization: str = "none",
) -> str:
    """Handle a simple data retrieval query. Typically 1-2 tool calls."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    system_prompt = _LOOKUP_SYSTEM_PROMPT_TEMPLATE.format(
        today=today,
        capability_prompt=capability_prompt.strip() or "Capabilities in this turn:\n- No live read tools are bound.",
    )
    if specialization == "schedule":
        system_prompt += (
            "\n\nSchedule-specific rules:\n"
            "- Call `aws_inspect_lambda_schedules` first.\n"
            "- Use explicit region hints from the user when present.\n"
            "- Extract concrete function/rule name hints from the user request and avoid passing tag names like Owner, Discipline, or Purpose.\n"
            "- Do not call `aws_audit_cloudtrail` for schedule/frequency/last-run questions unless the user explicitly asks about audit history or who made a change.\n"
            "- Prefer one schedule-tool call and answer directly from its output.\n"
            "- Report configured schedule and observed recent runs separately."
        )

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
        checkpoint_step=Config.LOOKUP_CHECKPOINT_STEP,
        system_prompt=system_prompt,
        original_query=user_input,
        require_relevant_tool_call_before_answer=require_live_inspection,
        available_capability_families=available_capability_families,
        insufficient_tool_names=insufficient_tool_names,
    )
