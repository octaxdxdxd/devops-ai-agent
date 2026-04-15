"""Explain handler — context-stuffed analysis and insight generation."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage, SystemMessage

from ..config import Config
from ..infra.topology import TopologyCache
from ..tracing.tracer import Tracer
from .base import StatusCallback, run_tool_loop

log = logging.getLogger(__name__)

_EXPLAIN_SYSTEM_PROMPT_TEMPLATE = """\
You are an infrastructure analyst providing clear, accurate explanations and insights.
Today's date is {today}.

Use tools first, then explain what the data means.

Rules:
- Do the lookup work yourself; do not ask the user for permission for read-only checks.
- Base the answer on real tool output, not guesses.
- If a query is empty or errors, try the obvious fix: other namespaces, labels, parents, or AWS regions.
- Do not repeat the same tool call with identical arguments. If repeated paths stay empty or error, explain the uncertainty instead of restating the same evidence.
- For AWS, be region-aware. For load balancers, check both `elbv2` and `elb` when relevant.
- For cost questions, use {today} for date math; "recent" means the last 30 days.
- Quantify where possible and say what data is missing if the picture is incomplete.
- This handler is read-only. If a change is needed, explain it and point the user to the action flow.

{capability_prompt}

Presentation:
- Answer the user’s actual question first.
- Be concise, structured, and explicit about important evidence."""


def handle_explain(
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
    """Handle an analysis/explanation query. Fewer steps than diagnose."""
    cb = status_callback or (lambda _: None)

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    system_prompt = _EXPLAIN_SYSTEM_PROMPT_TEMPLATE.format(
        today=today,
        capability_prompt=capability_prompt.strip() or "Capabilities in this turn:\n- No live read tools are bound.",
    )

    context_parts = []
    if topology_cache:
        cb("Loading infrastructure topology...")
        try:
            topo = topology_cache.get()
            summary = topo.to_summary(max_nodes=30, max_edges=20)
            context_parts.append(f"Current infrastructure topology:\n{summary}")
        except Exception as exc:
            log.warning("Topology build failed: %s", exc)

    preamble = ""
    if context_parts:
        preamble = "\n\n".join(context_parts) + "\n\n"

    messages = [
        SystemMessage(content=system_prompt),
        *chat_history[-4:],
        HumanMessage(content=f"{preamble}User question: {user_input}"),
    ]

    return run_tool_loop(
        messages=messages,
        llm_with_tools=llm_with_tools,
        tool_map=tool_map,
        max_steps=Config.EXPLAIN_MAX_STEPS,
        handler_name="explain",
        model_name=model_name,
        tracer=tracer,
        status_callback=status_callback,
        system_prompt=system_prompt,
        original_query=user_input,
        require_relevant_tool_call_before_answer=require_live_inspection,
        available_capability_families=available_capability_families,
        insufficient_tool_names=insufficient_tool_names,
    )
