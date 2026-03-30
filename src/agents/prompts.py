"""Prompt builders for the rebuilt AI Ops agent."""

from __future__ import annotations

from textwrap import dedent
from typing import Any

from .parsing import safe_json_dumps


INVESTIGATION_PROFILES = {
    "general_investigation": "General cross-system investigation when the request is broad or ambiguous.",
    "restore_workloads": "Restore workloads by checking Kubernetes scheduling, nodes, and AWS backing compute.",
    "cluster_health": "Broad cluster health review across namespaces, nodes, pods, and warning events.",
    "service_outage": "Investigate service reachability, backing workloads, and platform dependencies.",
    "inventory": "Collect cluster/account inventory and direct factual tables or mappings.",
    "capacity": "Assess compute/resource capacity and bottlenecks across Kubernetes and AWS.",
    "helm_release": "Investigate Helm releases, their workloads, and reconciliation failures.",
    "direct_command": "Handle an explicit raw kubectl/aws/helm command request.",
}


ACTION_CATALOG: dict[str, dict[str, str]] = {
    "k8s": {
        "cluster_overview": "Cluster-wide namespace, node, pod, workload, service, storage, and event overview.",
        "namespace_overview": "Namespace-focused workload, pod, service, ingress, and event overview. Params: namespace.",
        "node_overview": "Detailed node status, pressure, readiness, and scheduling blockers.",
        "pod_overview": "Pod health, restart, and scheduling overview. Params optional: namespace.",
        "workload_overview": "Deployment/statefulset/daemonset overview. Params optional: namespace.",
        "service_overview": "Service and ingress overview. Params optional: namespace.",
        "storage_overview": "PVC/PV overview. Params optional: namespace.",
        "event_overview": "Cluster or namespace warning-event overview. Params optional: namespace.",
        "resource_details": "Detailed JSON for one resource. Params: kind, name, optional namespace.",
        "raw_read": "A custom read-only kubectl command. Params: command.",
        "raw_write": "A custom mutating kubectl command that always requires approval. Params: command.",
    },
    "aws": {
        "identity": "Current AWS caller/account identity.",
        "regions": "Reachable AWS regions for investigation.",
        "eks_overview": "EKS clusters and nodegroups across configured regions.",
        "ec2_overview": "EC2 instance inventory across configured regions. Params optional: states, regions.",
        "asg_overview": "Auto Scaling Group inventory across configured regions.",
        "compute_backing_overview": "Combined EC2, ASG, and EKS/nodegroup backing compute picture.",
        "raw_read": "A custom read-only AWS CLI command. Params: command, optional all_regions.",
        "raw_write": "A custom mutating AWS CLI command that always requires approval. Params: command.",
    },
    "helm": {
        "release_overview": "Helm releases across namespaces.",
        "release_details": "Detailed Helm release state. Params: release_name, namespace.",
        "raw_read": "A custom read-only Helm command. Params: command.",
        "raw_write": "A custom mutating Helm command that always requires approval. Params: command.",
    },
}


def build_turn_analysis_prompt(*, system_prompt: str, user_input: str, active_case: dict[str, Any] | None, chat_history: list[Any]) -> str:
    history_lines: list[str] = []
    for message in chat_history[-8:]:
        role = getattr(message, "type", None) or getattr(message, "role", None) or message.__class__.__name__
        content = getattr(message, "content", "")
        history_lines.append(f"{role}: {content}")

    case_block = safe_json_dumps(active_case) if active_case else "null"
    history_block = "\n".join(history_lines) if history_lines else "(no recent chat history)"
    profiles_block = safe_json_dumps(INVESTIGATION_PROFILES)
    return dedent(
        f"""
        {system_prompt}

        You are deciding how to handle the next user turn.
        Output JSON only.

        Allowed profiles:
        {profiles_block}

        Return this JSON shape:
        {{
          "decision": "start_new_case" | "continue_case",
          "goal": "short precise statement of what success means",
          "desired_outcome": "what the user ultimately wants to be true",
          "profile": "one allowed profile name",
          "domains": ["k8s", "aws", "helm"],
          "notes": ["important guidance for the investigator"]
        }}

        Rules:
        - Prefer continuing the active case unless the new user request is clearly unrelated.
        - Infer the user's intended outcome, not just the literal wording.
        - If the user cares about workloads running, cluster nodes, EC2 backing, or Helm releases, choose a profile that drives a long investigation instead of a clarification loop.
        - Do not ask the user for namespace or region here.

        Active case:
        {case_block}

        Recent chat history:
        {history_block}

        New user message:
        {user_input}
        """
    ).strip()


def build_planning_prompt(
    *,
    system_prompt: str,
    case_snapshot: dict[str, Any],
    latest_user_message: str,
) -> str:
    action_catalog_block = safe_json_dumps(ACTION_CATALOG)
    return dedent(
        f"""
        {system_prompt}

        You are planning the next investigation step for an infrastructure case.
        Output JSON only.

        Available actions:
        {action_catalog_block}

        Return this JSON shape:
        {{
          "assistant_status": "short present-progress sentence for the UI",
          "phase": "observe|correlate|verify|remediate|synthesize",
          "working_summary": "what you currently think is happening",
          "hypotheses": ["short hypothesis", "..."],
          "gaps": ["missing signal", "..."],
          "actions": [
            {{
              "family": "k8s|aws|helm",
              "mode": "read|write",
              "action": "action name from catalog",
              "params": {{}},
              "reason": "why this action is the best next step",
              "expected_outcome": "what signal this should produce"
            }}
          ],
          "stop": true,
          "stop_reason": "resolved|blocked|needs_approval|enough_evidence",
          "answer": "only when stop=true and no further read action is needed",
          "approval_summary": "only when stop_reason=needs_approval"
        }}

        Rules:
        - Prefer 1-3 high-value read actions per step.
        - Keep investigating until you can answer with correlated evidence, not just one symptom.
        - Do not ask for namespace, region, or resource names until discovery actions were genuinely exhausted.
        - If AWS compute looks empty in one region, use region-discovery or cross-region inventory before concluding nothing exists.
        - If the user wants things restored, optimize for the end state, not a narrow literal interpretation.
        - Propose writes only after enough evidence exists to justify them.
        - If a write is needed, stop with stop_reason=needs_approval and provide concrete write actions.

        Case snapshot:
        {safe_json_dumps(case_snapshot)}

        Latest user message:
        {latest_user_message}
        """
    ).strip()


def build_integration_prompt(
    *,
    system_prompt: str,
    case_snapshot: dict[str, Any],
    executed_actions: list[dict[str, Any]],
    observations: list[dict[str, Any]],
) -> str:
    return dedent(
        f"""
        {system_prompt}

        Integrate the latest infrastructure observations into the case.
        Output JSON only.

        Return this JSON shape:
        {{
          "summary": "updated case summary grounded in the new observations",
          "phase": "observe|correlate|verify|remediate|synthesize",
          "entities": [
            {{
              "kind": "resource kind",
              "name": "resource name",
              "namespace": "",
              "scope": "",
              "provider_id": "",
              "attrs": {{}}
            }}
          ],
          "findings": [
            {{
              "claim": "evidence-backed finding",
              "confidence": 0,
              "verified": true,
              "entity_refs": ["kind:namespace:name", "..."]
            }}
          ],
          "hypotheses": ["current strongest theories"],
          "gaps": ["remaining missing signals"]
        }}

        Case snapshot before integration:
        {safe_json_dumps(case_snapshot)}

        Executed actions:
        {safe_json_dumps(executed_actions)}

        New observations:
        {safe_json_dumps(observations)}
        """
    ).strip()


def build_synthesis_prompt(*, system_prompt: str, case_snapshot: dict[str, Any], include_next_steps: bool = True) -> str:
    next_step_instruction = (
        "Include the single best next step if more work would still help."
        if include_next_steps
        else "Do not add speculative next steps."
    )
    return dedent(
        f"""
        {system_prompt}

        Write the user-facing case answer from the stored evidence only.
        If the user wanted a change and approval is still required, say so clearly and do not pretend the change already happened.

        Structure:
        **Bottom Line:** short direct answer
        **What I Found:**
        - concise evidence-backed findings
        **Why I Believe This:**
        - key correlations across Kubernetes, AWS, and Helm when relevant
        **Recommended Next Step:** short practical next action

        {next_step_instruction}
        Keep it crisp and avoid repeating the full case chronology.

        Case snapshot:
        {safe_json_dumps(case_snapshot)}
        """
    ).strip()
