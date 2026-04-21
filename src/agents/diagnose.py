"""Diagnose handler — deep investigation and root cause analysis."""

from __future__ import annotations

from datetime import datetime, timezone

from langchain_core.messages import HumanMessage, SystemMessage

from ..config import Config
from ..tracing.tracer import Tracer
from .base import StatusCallback, run_tool_loop

_DIAGNOSE_SYSTEM_PROMPT_TEMPLATE = """\
You are a senior SRE performing deep infrastructure investigation.
Today's date is {today}.

Rules:
- Investigate directly with tools; do not ask the user for permission for read-only checks.
- Answer the user’s exact question first. Only switch into RCA mode when they report a problem or ask why something is happening.
- Try multiple discovery paths before giving up: namespace, name, labels, parent resources, and AWS region.
- Do not repeat the same tool call with identical arguments. If the same path stays empty or errors, note that evidence and move to a different path.
- Do not stop at the first symptom-level explanation. Keep investigating until you either identify the most likely root cause or can clearly explain what evidence is still missing.
- Every conclusion must be tied to specific evidence from tool output.
- Separate confirmed facts from hypotheses and call out what is still unknown.
- For AWS, be region-aware. For load balancers, check both `elbv2` and `elb` when relevant.
- For CloudTrail, prefer selective filters such as exact event names, resource names, or usernames. Avoid broad EventSource/resource-type scans over long windows unless the user explicitly asks for audit-history spelunking.
- This handler is read-only, but when the evidence is strong, you should still state the minimum safe fix clearly and precisely so it can be turned into an action proposal.

{capability_prompt}

OUTPUT FORMAT for RCA (only when investigating problems):
## Root Cause Analysis
**Symptom**: [what was reported]
**Root Cause**: [identified cause with evidence]
**Evidence Chain**: [step-by-step how you reached this conclusion]
**Impact**: [what's affected and severity]
**Recommended Next Steps**: [actionable remediation steps]
**Exact Fix**: [minimum safe change, or state why no safe fix should be proposed yet]"""


def _diagnose_completion_feedback(final_text: str, meta: dict[str, object]) -> str | None:
    text = str(final_text or "").strip()
    lower = text.lower()
    sufficient_calls = int(meta.get("sufficient_tool_call_count", 0) or 0)

    required_markers = [
        "root cause analysis",
        "root cause",
        "evidence",
        "recommended next steps",
        "exact fix",
    ]
    missing_markers = [marker for marker in required_markers if marker not in lower]
    if missing_markers:
        return (
            "Your answer is incomplete for an RCA request. Re-open the investigation and produce a fuller RCA. "
            "You must include: Root Cause Analysis, Root Cause, Evidence Chain, Recommended Next Steps, and Exact Fix. "
            "Investigate further before answering."
        )

    if sufficient_calls < 2:
        return (
            "You have not gathered enough distinct evidence yet. Continue the investigation with additional tool calls "
            "that test the leading root-cause hypothesis and rule out close alternatives."
        )

    uncertain_phrases = (
        "not sure",
        "unclear",
        "still unknown",
        "needs more investigation",
        "cannot determine",
        "could be",
        "might be",
    )
    if any(phrase in lower for phrase in uncertain_phrases):
        return (
            "Your RCA is still too uncertain. Continue investigating the unresolved unknowns and either identify the "
            "most likely root cause with evidence or explicitly state the exact blocking evidence you failed to obtain."
        )

    if "**exact fix**" in lower and ("no safe fix" in lower or "cannot safely propose" in lower):
        return None

    fix_section = lower.split("exact fix", 1)[-1] if "exact fix" in lower else ""
    weak_fix_phrases = (
        "investigate further",
        "check logs",
        "look into",
        "continue investigating",
    )
    if any(phrase in fix_section for phrase in weak_fix_phrases):
        return (
            "The Exact Fix section is too vague. Continue investigating until you can state the minimum safe remediation "
            "clearly, or explicitly say why no safe remediation should be proposed."
        )

    return None


def handle_diagnose(
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
) -> str:
    """Handle a diagnostic / RCA query with structured investigation protocol."""
    cb = status_callback or (lambda _: None)

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    system_prompt = _DIAGNOSE_SYSTEM_PROMPT_TEMPLATE.format(
        today=today,
        capability_prompt=capability_prompt.strip() or "Capabilities in this turn:\n- No live read tools are bound.",
    )

    messages = [
        SystemMessage(content=system_prompt),
        *chat_history[-6:],
        HumanMessage(content=f"User query: {user_input}"),
    ]

    return run_tool_loop(
        messages=messages,
        llm_with_tools=llm_with_tools,
        tool_map=tool_map,
        max_steps=Config.DIAGNOSE_MAX_STEPS,
        handler_name="diagnose",
        model_name=model_name,
        tracer=tracer,
        status_callback=status_callback,
        checkpoint_step=Config.DIAGNOSE_CHECKPOINT_STEP,
        original_query=user_input,
        system_prompt=system_prompt,
        require_relevant_tool_call_before_answer=require_live_inspection,
        available_capability_families=available_capability_families,
        insufficient_tool_names=insufficient_tool_names,
        min_sufficient_tool_calls_before_answer=2,
        final_answer_feedback=_diagnose_completion_feedback,
    )
