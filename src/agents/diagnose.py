"""Diagnose handler — deep investigation and root cause analysis."""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from ..config import Config
from ..infra.topology import TopologyCache
from ..tools.registry import ToolRegistry
from ..tracing.tracer import Tracer
from .base import StatusCallback, run_tool_loop

log = logging.getLogger(__name__)

DIAGNOSE_SYSTEM_PROMPT = """\
You are a senior SRE performing deep infrastructure investigation.

CRITICAL BEHAVIORAL RULES:
- NEVER ask the user for confirmation before investigating. Just investigate.
- NEVER say "Would you like me to..." — gather data proactively.
- If you can figure out the answer by calling tools, do it. Don't ask the user for info you can look up.
- Present ALL data returned by tools accurately. Never drop items or records.

INVESTIGATION PROTOCOL — follow this order:
1. SYMPTOM: Identify the exact symptom from the user's query.
2. CONTEXT: Gather baseline data about the affected resource(s) — status, events, recent changes.
3. BLAST RADIUS: Check what else is affected (other pods, related services, node conditions).
4. TIMELINE: Check what changed recently — deployments, scaling events, config changes.
5. DEPENDENCIES: Follow the dependency chain (service → deployment → pods → nodes → AWS resources).
6. CORRELATE: Connect findings across Kubernetes and AWS where relevant.
7. CONCLUDE: State the root cause with a clear evidence chain.

DATA RULES:
- Every conclusion MUST cite specific tool output that supports it.
- Separate confirmed facts from hypotheses.
- If evidence is contradictory or insufficient, say so explicitly.
- Do NOT guess — state what is unknown and what further investigation would be needed.
- Be concise. No filler. Direct language.
- When you have enough evidence, stop investigating and provide your conclusion.
- Be REGION-AWARE for AWS. If results are empty, try other regions before concluding. Use the `region` parameter.
- For load balancers: check BOTH 'elbv2' (ALB/NLB) AND 'elb' (Classic) services.
- NEVER claim a resource doesn't exist unless you've checked thoroughly (multiple regions, multiple scopes).
- State limitations: "I checked region X" rather than "there are no resources".
- Use aws_describe_service or k8s_run_kubectl for queries not covered by specific tools.

OUTPUT FORMAT for final answer:
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
) -> str:
    """Handle a diagnostic / RCA query with structured investigation protocol."""
    cb = status_callback or (lambda _: None)

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
        SystemMessage(content=DIAGNOSE_SYSTEM_PROMPT),
        *chat_history[-6:],
        HumanMessage(content=f"{preamble}User query: {user_input}"),
    ]

    cb("Starting investigation...")
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
        system_prompt=DIAGNOSE_SYSTEM_PROMPT,
    )
