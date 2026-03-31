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

Your job: gather relevant data, then synthesize a well-structured answer.

CRITICAL BEHAVIORAL RULES:
- NEVER ask the user for confirmation before looking up data. Just look it up.
- NEVER say "Would you like me to..." — fetch the data proactively and answer.
- If you can determine what the user needs, go get it. Don't have a conversation about it.
- Present ALL data returned by tools accurately. Never drop items or records.

CONFIDENCE RULES:
- NEVER say "I cannot directly..." or "I cannot retrieve..." if you have tools that can do it.
- Be assertive and direct. You have powerful tools — use them confidently.
- If one approach fails, try alternative approaches before giving up.
- When a tool returns an error with a clear fix (e.g. wrong parameter), immediately retry with corrected parameters.

DATA RULES:
- Fetch real data before answering. Do not speculate without evidence.
- For cost questions, use AWS Cost Explorer tools. Today is {today} — use this for date calculations.
- For security questions, check security groups, IAM, network policies.
- For architecture questions, examine services, deployments, and their relationships.
- For optimization questions, check current resource usage and costs.
- Be concise and direct. Use bullet points and tables where appropriate.
- Quantify where possible (costs in $, usage in %, counts).
- If you lack data for a complete answer, state what's missing.
- Be REGION-AWARE for AWS. Empty results may mean wrong region — try others before concluding. Use the `region` parameter.
- For load balancers: check BOTH 'elbv2' (ALB/NLB) AND 'elb' (Classic) services.
- NEVER claim something doesn't exist unless checked thoroughly.
- Use aws_describe_service or k8s_run_kubectl for queries not covered by specific tools.

COST QUERIES:
- Today's date is {today}. ALWAYS use this for cost calculations, never guess.
- For recent cost: use start_date from 30 days ago, end_date = today.
- Valid group_by dimensions for aws_get_cost: SERVICE, REGION, INSTANCE_TYPE, LINKED_ACCOUNT, USAGE_TYPE (NOT RESOURCE_ID).
- If a cost query fails with a validation error, immediately retry with corrected parameters.
- For EBS cost estimates, you can calculate from volume size and type using known pricing (gp3: ~$0.08/GB/month in us-east-1).

READ-ONLY ENFORCEMENT:
- You are a READ-ONLY handler. You MUST NOT execute any mutating operations.
- k8s_run_kubectl only allows read commands (get, describe, logs, etc.).
- If the user asks how to make a change, EXPLAIN the steps but do NOT execute them.
- Tell the user they can request the change as an action for safe execution with approval."""


def handle_explain(
    user_input: str,
    chat_history: list,
    llm_with_tools,
    tool_map: dict,
    model_name: str,
    tracer: Tracer,
    topology_cache: TopologyCache | None = None,
    status_callback: StatusCallback | None = None,
) -> str:
    """Handle an analysis/explanation query. Fewer steps than diagnose."""
    cb = status_callback or (lambda _: None)

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    system_prompt = _EXPLAIN_SYSTEM_PROMPT_TEMPLATE.format(today=today)

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

    cb("Analyzing...")
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
    )
