"""Explain handler — context-stuffed analysis and insight generation."""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from ..config import Config
from ..infra.topology import TopologyCache
from ..tracing.tracer import Tracer
from .base import StatusCallback, run_tool_loop

log = logging.getLogger(__name__)

EXPLAIN_SYSTEM_PROMPT = """\
You are an infrastructure analyst providing clear, accurate explanations and insights.

Your job: gather relevant data, then synthesize a well-structured answer.

CRITICAL BEHAVIORAL RULES:
- NEVER ask the user for confirmation before looking up data. Just look it up.
- NEVER say "Would you like me to..." — fetch the data proactively and answer.
- If you can determine what the user needs, go get it. Don't have a conversation about it.
- Present ALL data returned by tools accurately. Never drop items or records.

DATA RULES:
- Fetch real data before answering. Do not speculate without evidence.
- For cost questions, use AWS Cost Explorer tools.
- For security questions, check security groups, IAM, network policies.
- For architecture questions, examine services, deployments, and their relationships.
- For optimization questions, check current resource usage and costs.
- Be concise and direct. Use bullet points and tables where appropriate.
- Quantify where possible (costs in $, usage in %, counts).
- If you lack data for a complete answer, state what's missing.
- Be REGION-AWARE for AWS. Empty results may mean wrong region — try others before concluding. Use the `region` parameter.
- For load balancers: check BOTH 'elbv2' (ALB/NLB) AND 'elb' (Classic) services.
- NEVER claim something doesn't exist unless checked thoroughly.
- Use aws_describe_service or k8s_run_kubectl for queries not covered by specific tools."""


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
        SystemMessage(content=EXPLAIN_SYSTEM_PROMPT),
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
        system_prompt=EXPLAIN_SYSTEM_PROMPT,
        original_query=user_input,
    )
