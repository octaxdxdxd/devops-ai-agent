"""Turn planning helpers for incident investigations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..utils.query_intent import QueryIntent
from .state import IncidentState, OperatorIntentState


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
_WORKLOAD_MARKERS = (
    "workload",
    "workloads",
    "deployment",
    "deployments",
    "statefulset",
    "statefulsets",
    "daemonset",
    "daemonsets",
    "replicaset",
    "replicasets",
    "cronjob",
    "cronjobs",
    "job",
    "jobs",
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
_CAPACITY_MARKERS = (
    "cpu",
    "ram",
    "memory",
    "capacity",
    "allocatable",
    "instance",
    "instances",
    "vcpu",
)
_COST_MARKERS = ("cost", "costs", "billing", "bill", "price", "prices", "monthly", "spend", "savings")
_OPTIMIZATION_MARKERS = ("optimized", "optimised", "optimize", "optimise", "rightsize", "improve", "saving", "opportunity")
_EXPLANATION_MARKERS = ("how does", "step by step", "flow", "explain", "architecture", "traffic")
_INVENTORY_MARKERS = ("list", "show", "get", "what kind", "what are", "do i have")


@dataclass(frozen=True)
class PlanObjective:
    """One concrete sub-goal within a turn."""

    key: str
    description: str
    focus: str
    required_categories: tuple[str, ...]
    preferred_tools: tuple[str, ...]
    minimum_successful_calls: int = 1


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
    requested_aspects: tuple[str, ...] = ()
    objectives: tuple[PlanObjective, ...] = ()


def _looks_like_follow_up(user_input: str) -> bool:
    del user_input
    return False


def _detect_focus(user_input: str, incident_state: IncidentState) -> str:
    del incident_state
    lowered = str(user_input or "").lower()

    if any(marker in lowered for marker in _COST_MARKERS):
        return "aws"
    if any(marker in lowered for marker in _STORAGE_MARKERS):
        return "storage"
    if any(marker in lowered for marker in _POD_MARKERS) and not any(marker in lowered for marker in _NODE_MARKERS):
        return "pod"
    if any(marker in lowered for marker in _NODE_MARKERS):
        return "node"
    if any(marker in lowered for marker in _CAPACITY_MARKERS):
        return "node"
    if any(marker in lowered for marker in _WORKLOAD_MARKERS):
        return "workload"
    if any(marker in lowered for marker in _SERVICE_MARKERS):
        return "service"
    if any(marker in lowered for marker in _AWS_MARKERS):
        return "aws"
    return "general"


def _looks_like_inventory_request(user_input: str) -> bool:
    lowered = str(user_input or "").strip().lower()
    if not lowered:
        return False
    return any(marker in lowered for marker in _INVENTORY_MARKERS)


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def _derive_requested_aspects(user_input: str, intent: QueryIntent, focus: str) -> tuple[str, ...]:
    lowered = str(user_input or "").strip().lower()
    aspects: list[str] = []

    if intent.mode == "incident_rca":
        aspects.append("diagnosis")
    if _looks_like_inventory_request(lowered):
        aspects.append("inventory")
    if any(marker in lowered for marker in _CAPACITY_MARKERS):
        aspects.append("capacity")
    if any(marker in lowered for marker in _COST_MARKERS):
        aspects.append("cost")
    if any(marker in lowered for marker in _OPTIMIZATION_MARKERS):
        aspects.append("optimization")
    if any(marker in lowered for marker in _EXPLANATION_MARKERS):
        aspects.append("explanation")
    if aspects and focus in {"aws", "node", "service", "workload", "storage", "pod"} and "inventory" not in aspects:
        if any(aspect in {"capacity", "cost", "optimization"} for aspect in aspects):
            aspects.insert(0, "inventory")

    if not aspects:
        if focus in {"service", "node", "pod", "storage", "workload"}:
            aspects.append("analysis")
        else:
            aspects.append("diagnosis" if intent.mode == "incident_rca" else "analysis")

    ordered: list[str] = []
    seen: set[str] = set()
    for aspect in aspects:
        if aspect not in seen:
            seen.add(aspect)
            ordered.append(aspect)
    return tuple(ordered)


def _objective_focus_for_aspect(*, aspect: str, base_focus: str, user_input: str) -> str:
    lowered = str(user_input or "").lower()

    if aspect == "cost":
        return "aws"
    if aspect == "capacity":
        if _contains_any(lowered, _AWS_MARKERS + _COST_MARKERS) and base_focus == "aws":
            return "aws"
        if _contains_any(lowered, _POD_MARKERS) and not _contains_any(lowered, _NODE_MARKERS):
            return "pod"
        if _contains_any(lowered, _CAPACITY_MARKERS + _NODE_MARKERS) or base_focus == "node":
            return "node"
        return base_focus
    if aspect == "inventory":
        if _contains_any(lowered, _AWS_MARKERS + _COST_MARKERS):
            return "aws"
        if _contains_any(lowered, _POD_MARKERS) and not _contains_any(lowered, _NODE_MARKERS):
            return "pod"
        if _contains_any(lowered, _CAPACITY_MARKERS + _NODE_MARKERS):
            return "node"
        if _contains_any(lowered, _WORKLOAD_MARKERS):
            return "workload"
        if _contains_any(lowered, _SERVICE_MARKERS):
            return "service"
        if _contains_any(lowered, _STORAGE_MARKERS):
            return "storage"
        if _contains_any(lowered, _POD_MARKERS):
            return "pod"
        return base_focus
    if aspect == "optimization":
        if _contains_any(lowered, _CAPACITY_MARKERS + _NODE_MARKERS):
            return "node"
        if _contains_any(lowered, _SERVICE_MARKERS):
            return "service"
        if _contains_any(lowered, _STORAGE_MARKERS):
            return "storage"
        if _contains_any(lowered, _WORKLOAD_MARKERS):
            return "workload"
        if _contains_any(lowered, _AWS_MARKERS + _COST_MARKERS):
            return "aws"
        return base_focus
    if aspect == "explanation":
        if _contains_any(lowered, _SERVICE_MARKERS + _EXPLANATION_MARKERS):
            return "service"
        if _contains_any(lowered, _WORKLOAD_MARKERS):
            return "workload"
        if _contains_any(lowered, _NODE_MARKERS + _CAPACITY_MARKERS):
            return "node"
        return base_focus
    return base_focus


def _required_categories_for_objective(
    *,
    aspect: str,
    focus: str,
    stage: str,
    has_known_scope: bool,
) -> tuple[str, ...]:
    if aspect == "inventory":
        if focus == "workload":
            return ("workload_health",)
        if focus == "node":
            return ("node_health",)
        if focus == "storage":
            return ("storage",)
        if focus == "service":
            return ("service_network", "workload_health")
        if focus == "pod":
            return ("pod_health",)
        if focus == "aws":
            return ("aws",)
        return ("discovery_cluster",)
    if aspect == "capacity":
        if focus == "aws":
            return ("aws", "node_health")
        if focus == "pod":
            return ("pod_health",)
        if focus == "node":
            return ("node_health",)
        return _required_categories_for_focus(
            focus=focus,
            stage=stage,
            has_known_scope=has_known_scope,
            requested_aspects=("capacity",),
        )
    if aspect == "cost":
        return ("aws",)
    if aspect == "optimization":
        if focus == "node":
            return ("node_health", "pod_health")
        if focus == "service":
            return ("service_network", "workload_health")
        if focus == "storage":
            return ("storage", "events")
        if focus == "workload":
            return ("workload_health", "pod_health")
        if focus == "aws":
            return ("aws", "node_health", "pod_health")
        return _required_categories_for_focus(
            focus=focus,
            stage=stage,
            has_known_scope=has_known_scope,
            requested_aspects=("optimization",),
        )
    if aspect == "explanation":
        if focus == "service":
            return ("service_network", "workload_health")
        return _required_categories_for_focus(
            focus=focus,
            stage=stage,
            has_known_scope=has_known_scope,
            requested_aspects=("explanation",),
        )
    return _required_categories_for_focus(
        focus=focus,
        stage=stage,
        has_known_scope=has_known_scope,
        requested_aspects=(aspect,),
    )


def _preferred_tools_for_objective(*, aspect: str, focus: str) -> tuple[str, ...]:
    if aspect == "cost":
        return ("aws_cli_readonly",)
    if aspect == "capacity" and focus == "aws":
        return ("aws_cli_readonly", "k8s_list_nodes", "k8s_list_pods")
    return _preferred_tools_for_focus(focus)


def _minimum_calls_for_objective(*, aspect: str, focus: str, stage: str) -> int:
    if stage == "verify":
        return 1
    if aspect == "inventory":
        if focus in {"workload", "pod", "storage", "aws"}:
            return 1
        return 2
    if aspect in {"capacity", "optimization", "explanation"}:
        return 2
    if aspect == "cost":
        return 1
    if aspect == "diagnosis":
        return 2
    return 2


def _objective_description(aspect: str, focus: str) -> str:
    mapping = {
        "inventory": f"Inventory the requested {focus} resources.",
        "capacity": "Compute the relevant CPU / memory / capacity totals.",
        "cost": "Retrieve or estimate the cost breakdown.",
        "optimization": "Assess whether the current setup is well optimized and suggest improvements.",
        "explanation": "Explain the architecture or traffic flow step by step.",
        "diagnosis": "Diagnose the issue and state the next best action.",
        "analysis": f"Analyze the requested {focus} resources.",
    }
    return mapping.get(aspect, f"Complete the {aspect} objective.")


def _build_objectives(
    *,
    user_input: str,
    base_focus: str,
    stage: str,
    has_known_scope: bool,
    requested_aspects: tuple[str, ...],
) -> tuple[PlanObjective, ...]:
    aspects = requested_aspects or ("analysis",)
    objectives: list[PlanObjective] = []
    seen: set[tuple[str, str]] = set()
    for aspect in aspects:
        objective_focus = _objective_focus_for_aspect(aspect=aspect, base_focus=base_focus, user_input=user_input)
        key = (aspect, objective_focus)
        if key in seen:
            continue
        seen.add(key)
        objectives.append(
            PlanObjective(
                key=aspect,
                description=_objective_description(aspect, objective_focus),
                focus=objective_focus,
                required_categories=_required_categories_for_objective(
                    aspect=aspect,
                    focus=objective_focus,
                    stage=stage,
                    has_known_scope=has_known_scope,
                ),
                preferred_tools=_preferred_tools_for_objective(aspect=aspect, focus=objective_focus),
                minimum_successful_calls=_minimum_calls_for_objective(
                    aspect=aspect,
                    focus=objective_focus,
                    stage=stage,
                ),
            )
        )
    return tuple(objectives)


def _required_categories_for_focus(
    *,
    focus: str,
    stage: str,
    has_known_scope: bool,
    requested_aspects: tuple[str, ...],
) -> tuple[str, ...]:
    if stage == "verify":
        if focus == "service":
            return ("service_network", "pod_health")
        if focus == "node":
            return ("node_health", "events")
        if focus == "storage":
            return ("storage", "events")
        if focus == "workload":
            return ("workload_health",)
        return ("pod_health", "events")

    if focus == "service":
        if "explanation" in requested_aspects:
            return ("service_network", "pod_health", "workload_health")
        return ("service_network", "pod_health")
    if focus == "node":
        if "capacity" in requested_aspects or "optimization" in requested_aspects:
            return ("node_health", "pod_health")
        return ("node_health", "events")
    if focus == "storage":
        return ("storage", "events")
    if focus == "workload":
        if "inventory" in requested_aspects:
            return ("workload_health",)
        return ("workload_health", "pod_health")
    if focus == "pod":
        return ("pod_health", "events")
    if focus == "aws":
        if "cost" in requested_aspects and ("capacity" in requested_aspects or "optimization" in requested_aspects):
            return ("aws", "node_health", "pod_health")
        if "cost" in requested_aspects or "capacity" in requested_aspects:
            return ("aws", "node_health")
        return ("aws",)
    if "inventory" in requested_aspects:
        return ("discovery_cluster", "workload_health")
    if has_known_scope:
        return ("pod_health", "events")
    return ("discovery_cluster", "pod_health", "events")


def _preferred_tools_for_focus(focus: str) -> tuple[str, ...]:
    mapping = {
        "workload": (
            "k8s_list_deployments",
            "k8s_list_statefulsets",
            "k8s_list_daemonsets",
            "k8s_list_pods",
            "helm_readonly",
            "kubectl_readonly",
        ),
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
    requested_aspects = _derive_requested_aspects(user_input, intent, focus)
    inventory_request = "inventory" in requested_aspects
    mentions_scope = _mentions_existing_scope(user_input, incident_state)
    continue_existing = bool(incident_state.active and mentions_scope)
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
    elif mode == "command":
        stage = "command"
    elif continue_existing:
        stage = "collect"
    elif inventory_request:
        stage = "collect"
    elif explicit_scope or known_scope_available:
        stage = "collect"
    else:
        stage = "scope"

    prefer_fresh_reads = False
    prefer_cached_reads = continue_existing and mode in {"incident_rca", "general", "command"}
    allow_broad_discovery = stage == "scope" or inventory_request
    objectives = _build_objectives(
        user_input=user_input,
        base_focus=focus,
        stage=stage,
        has_known_scope=(known_scope_available or explicit_scope),
        requested_aspects=requested_aspects,
    )

    required_categories = tuple(
        dict.fromkeys(category for objective in objectives for category in objective.required_categories)
    )
    preferred_tools = tuple(dict.fromkeys(tool for objective in objectives for tool in objective.preferred_tools))

    notes: list[str] = []
    if continue_existing:
        notes.append("Continue the existing incident instead of restarting cluster-wide discovery.")
    if known_scope_available and stage != "scope":
        notes.append("Reuse the known namespace/resource scope before calling broad inventory tools.")
    if prefer_cached_reads:
        notes.append("Prefer evidence already gathered in this incident before re-running the same read.")
    if prefer_fresh_reads:
        notes.append("This is a verification follow-up: run fresh targeted health checks for the affected resources.")
    if not allow_broad_discovery:
        notes.append("Avoid re-listing namespaces/nodes/all pods unless the scope changed or evidence is stale.")
    if inventory_request:
        notes.append("This is a direct inventory/capacity question: return the concrete list or totals requested, not only a health summary.")
    if set(requested_aspects).intersection({"inventory", "capacity", "cost", "optimization", "explanation"}):
        notes.append("Prefer aggregated read commands that cover all relevant resources at once. Avoid one read per pod/service/node when a cluster-wide list, table, or JSON query can answer the question.")
    if objectives:
        notes.append("Complete every objective in this turn before summarizing.")
    if required_categories:
        notes.append("Stop and summarize once every objective has the evidence it needs.")
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
        requested_aspects=requested_aspects,
        objectives=objectives,
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
    if turn_plan.objectives:
        for idx, objective in enumerate(turn_plan.objectives[:5], start=1):
            lines.append(
                f"- Objective {idx}: {objective.description} (focus={objective.focus}; evidence={', '.join(objective.required_categories) or 'none'})"
            )
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
