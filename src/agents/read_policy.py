"""Shared read-path scope classification and tool selection."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ..tracing.tracer import Tracer
from .base import extract_token_usage

log = logging.getLogger(__name__)

_READ_SCOPE_SYSTEM_PROMPT = """\
Classify the read-only investigation scope for an infrastructure request.

Decide:
- backend: one of "k8s", "aws", "mixed"
- specialization: one of "none", "schedule", "cloudtrail", "identity"
- confidence: one of "high", "low"

Rules:
- Use only the current user request plus recent user-authored context. Ignore assistant messages.
- If the current user request is explicit, prioritize it over older context.
- Use specialization "schedule" only for explicit Lambda/EventBridge schedule/frequency/last-run/next-run requests.
- Use specialization "cloudtrail" only for explicit audit/event-history/deleted-by/who-changed requests.
- Use specialization "identity" only for explicit account/caller-identity/ARN questions.
- Never infer topic continuity from vague acknowledgements. Ambiguous acknowledgements should already have been stopped upstream; if they still reach you, return backend "mixed", specialization "none", confidence "low".
- Use confidence "low" when the safe choice is to keep the broad read tool set.

Output ONLY valid JSON:
{"backend":"k8s","specialization":"none","confidence":"high"}"""


@dataclass(slots=True)
class ReadScopeResult:
    backend: str = "mixed"
    specialization: str = "none"
    confidence: str = "low"
    context_snippet: str = ""


@dataclass(slots=True)
class ReadToolSelection:
    tools: list
    backend: str
    specialization: str = "none"
    confidence: str = "low"
    capability_families: list[str] = field(default_factory=list)
    insufficient_tool_names: set[str] = field(default_factory=set)
    require_live_inspection: bool = False
    capability_prompt: str = ""
    context_snippet: str = ""


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


def _message_role(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("role", "")).lower()
    role = getattr(message, "type", "") or getattr(message, "role", "")
    return str(role).lower()


def _recent_user_context(chat_history: list | None, limit: int = 4) -> list[str]:
    context: list[str] = []
    for message in chat_history or []:
        if _message_role(message) not in {"human", "user"}:
            continue
        text = _message_text(message).strip()
        if text:
            context.append(text)
    return context[-limit:]


def _context_snippet(user_input: str, recent_user_context: list[str]) -> str:
    parts = [text for text in recent_user_context if text and text != str(user_input or "").strip()]
    parts.append(str(user_input or "").strip())
    snippet = " | ".join(part for part in parts if part)
    return snippet[:240]


def _parse_scope(raw: str, *, context_snippet: str = "") -> ReadScopeResult:
    text = str(raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        lines = text.splitlines()
        text = "\n".join(lines[1:] if lines and lines[0].isalpha() else lines).strip()
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return ReadScopeResult(context_snippet=context_snippet)

    backend = str(data.get("backend", "mixed") or "mixed").strip().lower()
    specialization = str(data.get("specialization", "none") or "none").strip().lower()
    confidence = str(data.get("confidence", "low") or "low").strip().lower()

    if backend not in {"k8s", "aws", "mixed"}:
        backend = "mixed"
    if specialization not in {"none", "schedule", "cloudtrail", "identity"}:
        specialization = "none"
    if confidence not in {"high", "low"}:
        confidence = "low"

    return ReadScopeResult(
        backend=backend,
        specialization=specialization,
        confidence=confidence,
        context_snippet=context_snippet,
    )


def classify_read_scope(
    user_input: str,
    chat_history: list | None,
    llm,
    model_name: str,
    tracer: Tracer,
) -> ReadScopeResult:
    recent_user_context = _recent_user_context(chat_history)
    snippet = _context_snippet(user_input, recent_user_context)

    messages = [SystemMessage(content=_READ_SCOPE_SYSTEM_PROMPT)]
    if recent_user_context:
        messages.append(
            HumanMessage(
                content=(
                    "Recent user-authored context only:\n"
                    + "\n".join(f"- {line}" for line in recent_user_context)
                    + f"\n\nCurrent user request:\n{user_input}"
                )
            )
        )
    else:
        messages.append(HumanMessage(content=f"Current user request:\n{user_input}"))

    t0 = time.monotonic()
    try:
        response = llm.invoke(messages)
    except Exception as exc:
        log.warning("Read scope classification failed: %s", exc)
        tracer.step("error", "orchestrator", error=str(exc))
        scope = ReadScopeResult(context_snippet=snippet)
        tracer.step(
            "selector",
            "orchestrator",
            input_summary=user_input[:200],
            output_summary=(
                f"backend={scope.backend}, specialization={scope.specialization}, "
                f"confidence={scope.confidence}, mode=fallback, context={snippet}"
            ),
        )
        return scope

    elapsed = int((time.monotonic() - t0) * 1000)
    tokens_in, tokens_out = extract_token_usage(response)
    raw = (response.content or "").strip()
    scope = _parse_scope(raw, context_snippet=snippet)

    tracer.step(
        "selector",
        "orchestrator",
        input_summary=user_input[:200],
        output_summary=(
            f"backend={scope.backend}, specialization={scope.specialization}, "
            f"confidence={scope.confidence}, mode=classified, context={scope.context_snippet}"
        ),
        llm_model=model_name,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        duration_ms=elapsed,
    )
    return scope


def _named_tools(tools: list) -> dict[str, object]:
    return {getattr(tool, "name", ""): tool for tool in tools}


def build_capability_prompt(selection: ReadToolSelection) -> str:
    if not selection.tools:
        return (
            "Capabilities in this turn:\n"
            "- No live read tools are bound.\n"
            "- If data is missing, say so clearly."
        )

    lines = ["Capabilities in this turn:"]
    if "k8s" in selection.capability_families:
        lines.append("- Kubernetes read tools available.")
    if "aws" in selection.capability_families:
        lines.append("- AWS read tools available.")
    tool_names = ", ".join(f"`{getattr(tool, 'name', '')}`" for tool in selection.tools if getattr(tool, "name", ""))
    if tool_names:
        lines.append(f"- Bound tools: {tool_names}.")
    lines.extend(
        [
            "- Do not claim a tool family is unavailable if tools from that family are bound.",
            "- For concrete resource inspection questions, call a relevant tool and answer from its output.",
            "- If a tool path is empty, try an obvious alternate path before answering.",
        ]
    )
    return "\n".join(lines)


def select_read_tools(
    user_input: str,
    chat_history: list,
    k8s_tools: list,
    aws_tools: list,
    *,
    scope: ReadScopeResult | None = None,
) -> ReadToolSelection:
    resolved_scope = scope or ReadScopeResult()
    aws_by_name = _named_tools(aws_tools)

    selected_k8s = list(k8s_tools) if resolved_scope.backend in {"k8s", "mixed"} else []

    selected_aws: list = []
    if resolved_scope.backend in {"aws", "mixed"}:
        if resolved_scope.specialization == "schedule":
            if "aws_inspect_lambda_schedules" in aws_by_name:
                selected_aws = [aws_by_name["aws_inspect_lambda_schedules"]]
        elif resolved_scope.specialization == "cloudtrail":
            if "aws_audit_cloudtrail" in aws_by_name:
                selected_aws = [aws_by_name["aws_audit_cloudtrail"]]
        elif resolved_scope.specialization == "identity":
            if "aws_get_caller_identity" in aws_by_name:
                selected_aws = [aws_by_name["aws_get_caller_identity"]]
        else:
            selected_aws = [tool for tool in aws_tools if getattr(tool, "name", "") != "aws_get_caller_identity"]

    if resolved_scope.confidence == "low":
        selected_k8s = list(k8s_tools)
        selected_aws = [tool for tool in aws_tools if getattr(tool, "name", "") != "aws_get_caller_identity"]

    selected_tools = [*selected_k8s, *selected_aws]
    capability_families: list[str] = []
    if selected_k8s:
        capability_families.append("k8s")
    if selected_aws:
        capability_families.append("aws")

    insufficient_tool_names: set[str] = set()
    if resolved_scope.specialization != "identity" and "aws_get_caller_identity" in aws_by_name:
        insufficient_tool_names.add("aws_get_caller_identity")

    selection = ReadToolSelection(
        tools=selected_tools,
        backend=resolved_scope.backend,
        specialization=resolved_scope.specialization,
        confidence=resolved_scope.confidence,
        capability_families=capability_families,
        insufficient_tool_names=insufficient_tool_names,
        require_live_inspection=bool(selected_tools),
        context_snippet=resolved_scope.context_snippet,
    )
    selection.capability_prompt = build_capability_prompt(selection)
    return selection
