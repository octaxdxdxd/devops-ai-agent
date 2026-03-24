"""Turn planning helpers for incident investigations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..utils.query_intent import QueryIntent
from .state import IncidentState, OperatorIntentState


_FOLLOW_UP_MARKERS = (
    "is it healthy",
    "is it better",
    "is it fixed",
    "is it resolved",
    "is it stable",
    "check again",
    "recheck",
    "verify",
    "what next",
    "what changed",
    "and now",
    "status now",
    "healthy now",
    "did that fix",
)
_SERVICE_MARKERS = (
    "service",
    "ingress",
    "load balancer",
    "target group",
    "elb",
    "alb",
    "nlb",
    "unreachable",
    "dns",
    "gateway",
    "route",
    "502",
    "503",
)
_NODE_MARKERS = (
    "node",
    "nodegroup",
    "asg",
    "autoscaling",
    "pending",
    "schedule",
    "unschedul",
    "notready",
    "evict",
)
_STORAGE_MARKERS = (
    "pvc",
    "pv",
    "volume",
    "mount",
    "storage",
    "attachvolume",
    "persistentvolume",
)
_POD_MARKERS = (
    "pod",
    "container",
    "crashloop",
    "oom",
    "logs",
    "log",
    "imagepull",
    "backoff",
    "restart",
)
_AWS_MARKERS = ("aws", "ec2", "eks", "elb", "elbv2", "autoscaling", "target group")


@dataclass(frozen=True)
class TurnPlan:
    """Structured execution plan for one user turn."""

    mode: str
    stage: str
    focus: str
    continue_existing: bool
    reset_existing_context: bool
    prefer_cached_reads: bool
    prefer_fresh_reads: bool
    allow_broad_discovery: bool
    required_categories: tuple[str, ...]
    preferred_tools: tuple[str, ...]
    notes: tuple[str, ...]


def _looks_like_follow_up(user_input: str) -> bool:
    lowered = str(user_input or "").strip().lower()
    if not lowered:
        return False
    if len(lowered.split()) <= 6 and any(token in lowered for token in {"now", "again", "still", "healthy", "fixed"}):
        return True
    return any(marker in lowered for marker in _FOLLOW_UP_MARKERS)


def _detect_focus(user_input: str, incident_state: IncidentState) -> str:
    lowered = str(user_input or "").lower()

    if any(marker in lowered for marker in _STORAGE_MARKERS):
        return "storage"
    if any(marker in lowered for marker in _NODE_MARKERS):
        return "node"
    if any(marker in lowered for marker in _SERVICE_MARKERS):
        return "service"
    if any(marker in lowered for marker in _POD_MARKERS):
        return "pod"
    if any(marker in lowered for marker in _AWS_MARKERS):
        return "aws"
    if _looks_like_follow_up(lowered) and incident_state.active:
        if incident_state.last_focus and incident_state.last_focus != "general":
            return incident_state.last_focus
        if incident_state.services or incident_state.ingresses:
            return "service"
        if incident_state.nodes:
            return "node"
        if incident_state.pods or incident_state.workloads:
            return "pod"
    return "general"


def _required_categories_for_focus(*, focus: str, stage: str, has_known_scope: bool) -> tuple[str, ...]:
    if stage == "verify":
        if focus == "service":
            return ("service_network", "pod_health")
        if focus == "node":
            return ("node_health", "events")
        if focus == "storage":
            return ("storage", "events")
        return ("pod_health", "events")

    if focus == "service":
        return ("service_network", "pod_health")
    if focus == "node":
        return ("node_health", "events")
    if focus == "storage":
        return ("storage", "events")
    if focus == "pod":
        return ("pod_health", "events")
    if focus == "aws":
        return ("aws",)
    if has_known_scope:
        return ("pod_health", "events")
    return ("discovery_cluster", "pod_health", "events")


def _preferred_tools_for_focus(focus: str) -> tuple[str, ...]:
    mapping = {
        "service": (
            "k8s_list_services",
            "k8s_list_ingresses",
            "k8s_get_resource_yaml",
            "k8s_list_pods",
            "k8s_find_pods",
            "k8s_describe_pod",
            "k8s_get_pod_logs",
            "aws_cli_readonly",
        ),
        "node": (
            "k8s_list_nodes",
            "k8s_top_nodes",
            "k8s_describe_node",
            "k8s_get_events",
            "k8s_get_pod_scheduling_report",
            "k8s_list_pods",
            "aws_cli_readonly",
        ),
        "storage": (
            "k8s_get_pvcs",
            "k8s_list_pvs",
            "k8s_describe_pvc",
            "k8s_describe_pv",
            "k8s_get_events",
            "k8s_get_pod_scheduling_report",
        ),
        "pod": (
            "k8s_find_pods",
            "k8s_list_pods",
            "k8s_describe_pod",
            "k8s_get_pod_logs",
            "k8s_top_pods",
            "k8s_get_crashloop_pods",
            "k8s_get_events",
        ),
        "aws": ("aws_cli_readonly", "k8s_list_nodes", "k8s_list_pods"),
    }
    return mapping.get(
        focus,
        (
            "k8s_find_pods",
            "k8s_list_pods",
            "k8s_get_events",
            "k8s_describe_pod",
            "aws_cli_readonly",
        ),
    )


def _mentions_existing_scope(user_input: str, incident_state: IncidentState) -> bool:
    if not incident_state.active:
        return False
    lowered = str(user_input or "").lower()
    return any(token in lowered for token in incident_state.known_scope_tokens())


def build_turn_plan(
    *,
    user_input: str,
    intent: QueryIntent,
    chat_history: list[Any] | None,
    incident_state: IncidentState,
    operator_intent_state: OperatorIntentState,
    approval_pending: bool = False,
) -> TurnPlan:
    """Derive one structured plan for the current turn."""
    del chat_history  # The current heuristics rely on incident_state + current turn text.

    mode = intent.mode
    focus = _detect_focus(user_input, incident_state)
    follow_up_action = operator_intent_state.is_following_proposed_plan()
    if follow_up_action and operator_intent_state.pending_step_focus:
        focus = operator_intent_state.pending_step_focus
    follow_up = _looks_like_follow_up(user_input)
    mentions_scope = _mentions_existing_scope(user_input, incident_state)
    continue_existing = bool(follow_up_action or (incident_state.active and (follow_up or mentions_scope)))
    reset_existing_context = bool(
        incident_state.active
        and not continue_existing
        and mode in {"incident_rca", "general", "command"}
        and str(user_input or "").strip()
    )

    has_known_scope = bool(
        incident_state.namespace
        or incident_state.pods
        or incident_state.services
        or incident_state.workloads
        or incident_state.nodes
    )
    known_scope_available = has_known_scope and not reset_existing_context
    explicit_scope = bool(intent.namespace or mentions_scope)

    if approval_pending:
        stage = "execute"
    elif follow_up_action:
        stage = operator_intent_state.pending_step_stage or "collect"
    elif mode == "direct_read":
        stage = "direct_read"
    elif mode == "command":
        stage = "command"
    elif continue_existing and follow_up:
        stage = "verify"
    elif continue_existing:
        stage = "collect"
    elif explicit_scope or known_scope_available:
        stage = "collect"
    else:
        stage = "scope"

    prefer_fresh_reads = stage == "verify"
    prefer_cached_reads = (follow_up_action or continue_existing) and not prefer_fresh_reads and mode in {"incident_rca", "general", "command"}
    allow_broad_discovery = stage == "scope" and not follow_up_action

    required_categories = _required_categories_for_focus(
        focus=focus,
        stage=stage,
        has_known_scope=(known_scope_available or explicit_scope),
    )
    preferred_tools = _preferred_tools_for_focus(focus)

    notes: list[str] = []
    if continue_existing:
        notes.append("Continue the existing incident instead of restarting cluster-wide discovery.")
    if follow_up_action:
        notes.append("Continue with the previously proposed next step instead of restating the issue.")
        if operator_intent_state.pending_step_kind == "implementation":
            notes.append("Move from proposal into implementation planning/execution using the minimum required reads.")
        elif operator_intent_state.pending_step_kind == "prepare":
            notes.append("Turn the approved next step into concrete commands or patches without re-diagnosing.")
    if known_scope_available and stage != "scope":
        notes.append("Reuse the known namespace/resource scope before calling broad inventory tools.")
    if prefer_cached_reads:
        notes.append("Prefer evidence already gathered in this incident before re-running the same read.")
    if prefer_fresh_reads:
        notes.append("This is a verification follow-up: run fresh targeted health checks for the affected resources.")
    if not allow_broad_discovery:
        notes.append("Avoid re-listing namespaces/nodes/all pods unless the scope changed or evidence is stale.")
    if required_categories:
        notes.append("Stop and summarize once the required evidence categories are covered.")
    return TurnPlan(
        mode=mode,
        stage=stage,
        focus=focus,
        continue_existing=continue_existing,
        reset_existing_context=reset_existing_context,
        prefer_cached_reads=prefer_cached_reads,
        prefer_fresh_reads=prefer_fresh_reads,
        allow_broad_discovery=allow_broad_discovery,
        required_categories=required_categories,
        preferred_tools=preferred_tools,
        notes=tuple(notes),
    )


def render_turn_plan_directive(
    *,
    turn_plan: TurnPlan,
    incident_state: IncidentState,
    operator_intent_state: OperatorIntentState,
) -> str:
    """Render the plan and prior state into a prompt block."""
    lines = [
        "[Execution plan]",
        f"- Stage: {turn_plan.stage}",
        f"- Focus: {turn_plan.focus}",
    ]
    if turn_plan.required_categories:
        lines.append(f"- Target evidence categories: {', '.join(turn_plan.required_categories)}")
    if turn_plan.preferred_tools:
        lines.append(f"- Preferred tools: {', '.join(turn_plan.preferred_tools[:8])}")
    for note in turn_plan.notes[:6]:
        lines.append(f"- {note}")

    state_block = incident_state.render_context_block()
    if state_block:
        lines.append(state_block)
    operator_block = operator_intent_state.render_context_block()
    if operator_block:
        lines.append(operator_block)
    return "\n".join(lines)
