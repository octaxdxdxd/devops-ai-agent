"""Diagnose handler — deep investigation and root cause analysis."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage, SystemMessage

from ..config import Config
from ..infra.topology import TopologyCache
from ..tracing.tracer import Tracer
from .base import StatusCallback, run_tool_loop

log = logging.getLogger(__name__)

_DIAGNOSE_SYSTEM_PROMPT_TEMPLATE = """\
You are a senior SRE performing deep infrastructure investigation.
Today's date is {today}.

Rules:
- Investigate directly with tools; do not ask the user for permission for read-only checks.
- Answer the user’s exact question first. Only switch into RCA mode when they report a problem or ask why something is happening.
- Try multiple discovery paths before giving up: namespace, name, labels, parent resources, and AWS region.
- Do not repeat the same tool call with identical arguments. If the same path stays empty or errors, note that evidence and move to a different path.
- Every conclusion must be tied to specific evidence from tool output.
- Separate confirmed facts from hypotheses and call out what is still unknown.
- For AWS, be region-aware. For load balancers, check both `elbv2` and `elb` when relevant.
- For CloudTrail, prefer selective filters such as exact event names, resource names, or usernames. Avoid broad EventSource/resource-type scans over long windows unless the user explicitly asks for audit-history spelunking.
- This handler is read-only. If a fix is needed, describe it and point the user to the action flow.

{capability_prompt}

OUTPUT FORMAT for RCA (only when investigating problems):
## Root Cause Analysis
**Symptom**: [what was reported]
**Root Cause**: [identified cause with evidence]
**Evidence Chain**: [step-by-step how you reached this conclusion]
**Impact**: [what's affected and severity]
**Recommended Next Steps**: [actionable remediation steps]"""


def handle_diagnose(
    user_input: str,
    chat_history: list,
    llm_with_tools,
    tool_map: dict,
    model_name: str,
    tracer: Tracer,
    topology_cache: TopologyCache | None = None,
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

    # Build context preamble with topology if available
    context_parts = []
    if topology_cache:
        cb("Building infrastructure topology...")
        try:
            topo = topology_cache.get()
            summary = topo.to_summary(max_nodes=40, max_edges=25)
            context_parts.append(f"Current infrastructure topology:\n{summary}")
        except Exception as exc:
            log.warning("Topology build failed: %s", exc)

    preamble = ""
    if context_parts:
        preamble = "\n\n".join(context_parts) + "\n\n"

    messages = [
        SystemMessage(content=system_prompt),
        *chat_history[-6:],
        HumanMessage(content=f"{preamble}User query: {user_input}"),
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
    )
