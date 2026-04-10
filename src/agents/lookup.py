"""Lookup handler — fast, direct data retrieval with minimal LLM usage."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ..config import Config
from ..tracing.tracer import Tracer
from .base import StatusCallback, run_tool_loop

log = logging.getLogger(__name__)

_CLOUDTRAIL_KEYWORDS = (
    "cloudtrail",
    "event history",
    "lookup events",
    "lookupevents",
    "audit trail",
    "audit log",
)
_CLOUDTRAIL_AUDIT_PATTERNS = (
    "deleted by",
    "created by",
    "modified by",
    "updated by",
    "who deleted",
    "who created",
    "who changed",
    "who modified",
)
_PRINCIPAL_HINT_RE = re.compile(r"[\w.+-]+@[\w.-]+\.\w+")
_SCHEDULE_LOOKUP_PATTERNS = (
    "how frequent",
    "how often",
    "last run",
    "last ran",
    "when will it run next",
    "when does it run next",
    "run next",
    "runs every",
    "schedule",
    "cron",
    "eventbridge",
    "lambda runs",
)

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

Presentation:
- Answer the exact question directly.
- Use markdown tables for tabular results.
- If something is empty, say where you checked."""


def _message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content or "")


def is_cloudtrail_lookup_query(user_input: str, chat_history: list | None = None) -> bool:
    combined_parts = [str(user_input or "")]
    for message in (chat_history or [])[-6:]:
        combined_parts.append(_message_text(message))
    text = " ".join(combined_parts).lower()

    if any(keyword in text for keyword in _CLOUDTRAIL_KEYWORDS):
        return True
    if any(pattern in text for pattern in _CLOUDTRAIL_AUDIT_PATTERNS) and (
        "aws" in text or "resource" in text or bool(_PRINCIPAL_HINT_RE.search(text))
    ):
        return True
    if "event name" in text and any(token in text for token in ("delete", "create", "update")):
        return True
    return False


def is_schedule_lookup_query(user_input: str, chat_history: list | None = None) -> bool:
    combined_parts = [str(user_input or "")]
    for message in (chat_history or [])[-6:]:
        combined_parts.append(_message_text(message))
    text = " ".join(combined_parts).lower()

    if any(pattern in text for pattern in _SCHEDULE_LOOKUP_PATTERNS):
        return True
    if "lambda" in text and any(token in text for token in ("frequency", "next run", "last run", "scheduled")):
        return True
    return False


def select_lookup_tools(user_input: str, chat_history: list, read_tools: list) -> list:
    is_schedule_query = is_schedule_lookup_query(user_input, chat_history)
    is_cloudtrail_query = is_cloudtrail_lookup_query(user_input, chat_history)

    if is_schedule_query and not is_cloudtrail_query:
        wanted = {"aws_inspect_lambda_schedules", "aws_get_caller_identity"}
        selected = [tool for tool in read_tools if getattr(tool, "name", "") in wanted]
        return selected or read_tools

    if is_cloudtrail_query and not is_schedule_query:
        wanted = {"aws_audit_cloudtrail", "aws_get_caller_identity"}
        selected = [tool for tool in read_tools if getattr(tool, "name", "") in wanted]
        return selected or read_tools

    if is_cloudtrail_query and is_schedule_query:
        wanted = {"aws_inspect_lambda_schedules", "aws_audit_cloudtrail", "aws_get_caller_identity"}
        selected = [tool for tool in read_tools if getattr(tool, "name", "") in wanted]
        return selected or read_tools

    return read_tools


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
    if is_schedule_lookup_query(user_input, chat_history) and not is_cloudtrail_lookup_query(user_input, chat_history):
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
    )
