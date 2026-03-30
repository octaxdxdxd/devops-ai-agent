"""Structured incident state and turn outcome helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..config import Config
from ..utils.payload_shape import looks_like_structured_payload


_AWS_ARN_RE = re.compile(r"\barn:aws[a-z-]*:[^\s'\"]+\b", re.IGNORECASE)
_AWS_ID_RE = re.compile(
    r"\b(?:ami|i|lt|sg|subnet|vpc|vol|snap|eni|igw|nat|rtb|eipalloc|eipassoc|vpce|pcx|tgw|fs|db)-[0-9a-zA-Z]{6,}\b",
    re.IGNORECASE,
)
_NAME_RE = re.compile(r"\b[a-z0-9][a-z0-9\.-]*\b", re.IGNORECASE)
_RESOURCE_PATTERNS: dict[str, tuple[str, ...]] = {
    "namespace": (r"\bnamespace\s+([a-z0-9][a-z0-9-]*)\b",),
    "pod": (r"\bpod(?:s)?\s+([a-z0-9][a-z0-9\.-]*)\b",),
    "service": (r"\bservice(?:s)?\s+([a-z0-9][a-z0-9\.-]*)\b", r"\bsvc\s+([a-z0-9][a-z0-9\.-]*)\b"),
    "deployment": (r"\bdeployment(?:s)?\s+([a-z0-9][a-z0-9\.-]*)\b",),
    "statefulset": (r"\bstatefulset(?:s)?\s+([a-z0-9][a-z0-9\.-]*)\b",),
    "daemonset": (r"\bdaemonset(?:s)?\s+([a-z0-9][a-z0-9\.-]*)\b",),
    "ingress": (r"\bingress(?:es)?\s+([a-z0-9][a-z0-9\.-]*)\b",),
    "node": (r"\bnode(?:s)?\s+([a-z0-9][a-z0-9\.-]*)\b",),
}
_FOLLOW_UP_PROMPT_RE = re.compile(
    r"would you like me to proceed(?:\s+with)?\s*(.+?)\?\s*\(yes/no\)",
    re.IGNORECASE | re.DOTALL,
)
_RECOMMENDED_ACTION_RE = re.compile(
    r"(?:recommended action:|i recommend the following actions:)\s*(.+?)(?:would you like me to proceed|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_FOLLOW_UP_EXECUTE_MARKERS = (
    "drain",
    "cordon",
    "uncordon",
    "terminate",
    "restart",
    "reboot",
    "delete",
    "remove",
    "scale",
    "increase",
    "decrease",
    "rollout",
    "patch",
    "apply",
    "replace",
    "update",
    "modify",
    "set desired capacity",
    "desired capacity",
    "evict",
)
_FOLLOW_UP_COLLECT_MARKERS = (
    "check",
    "checking",
    "inspect",
    "investigate",
    "review",
    "look at",
    "collect",
    "gather",
    "verify",
    "confirm",
    "read",
    "describe",
    "list",
    "show",
)
_FOLLOW_UP_NODE_MARKERS = ("node", "nodes", "ec2", "instance", "instances", "asg", "autoscaling", "cpu", "memory")
_FOLLOW_UP_SERVICE_MARKERS = ("service", "services", "ingress", "load balancer", "target group", "alb", "nlb")
_FOLLOW_UP_STORAGE_MARKERS = ("pvc", "pv", "volume", "storage")
_FOLLOW_UP_WORKLOAD_MARKERS = ("deployment", "statefulset", "daemonset", "workload")
_FOLLOW_UP_POD_MARKERS = ("pod", "pods", "container", "containers")
_AFFIRMATIVE_REPLIES = ("yes", "y", "ok", "okay", "sure", "approve", "approved", "proceed", "do it", "run it", "go ahead")
_NEGATIVE_REPLIES = ("no", "n", "cancel", "stop", "don't", "do not")


@dataclass
class ToolExecutionRecord:
    """Structured summary of one tool call inside a turn."""

    tool_name: str
    requested_tool: str
    args: Any
    semantic_key: str
    success: bool
    result_excerpt: str = ""
    summary: str = ""
    from_cache: bool = False
    capability: str = ""
    evidence_categories: tuple[str, ...] = ()


@dataclass
class ToolLoopOutcome:
    """Return payload from the tool loop."""

    final_text: str
    records: list[ToolExecutionRecord] = field(default_factory=list)
    max_iterations_hit: bool = False
    stopped_reason: str = ""


@dataclass
class CachedToolResult:
    """Cached read-only evidence reused across follow-up turns."""

    semantic_key: str
    tool_name: str
    content: str
    summary: str
    turn_index: int


@dataclass
class IncidentState:
    """Long-lived in-memory incident context for one chat session."""

    active: bool = False
    turn_index: int = 0
    opened_turn: int = 0
    last_updated_turn: int = 0
    last_intent_mode: str = ""
    last_stage: str = ""
    last_focus: str = "general"
    last_user_input: str = ""
    summary: str = ""
    severity: str = ""
    confidence_score: int = 0
    namespace: str = ""
    pods: list[str] = field(default_factory=list)
    services: list[str] = field(default_factory=list)
    workloads: list[str] = field(default_factory=list)
    ingresses: list[str] = field(default_factory=list)
    nodes: list[str] = field(default_factory=list)
    aws_resources: list[str] = field(default_factory=list)
    evidence_notes: list[str] = field(default_factory=list)
    recent_tool_records: list[ToolExecutionRecord] = field(default_factory=list)
    cached_tool_results: dict[str, CachedToolResult] = field(default_factory=dict)

    def clear(self) -> None:
        self.active = False
        self.turn_index = 0
        self.opened_turn = 0
        self.last_updated_turn = 0
        self.last_intent_mode = ""
        self.last_stage = ""
        self.last_focus = "general"
        self.last_user_input = ""
        self.summary = ""
        self.severity = ""
        self.confidence_score = 0
        self.namespace = ""
        self.pods = []
        self.services = []
        self.workloads = []
        self.ingresses = []
        self.nodes = []
        self.aws_resources = []
        self.evidence_notes = []
        self.recent_tool_records = []
        self.cached_tool_results = {}

    def known_scope_tokens(self) -> set[str]:
        tokens: set[str] = set()
        for value in [self.namespace, self.summary, *self.pods, *self.services, *self.workloads, *self.ingresses, *self.nodes]:
            for token in _NAME_RE.findall(str(value).lower()):
                if len(token) >= 3:
                    tokens.add(token)
        return tokens

    def render_context_block(self) -> str:
        if not self.active:
            return ""

        lines: list[str] = ["[Existing incident context]"]
        if self.summary:
            lines.append(f"- Last summary: {self.summary}")
        if self.namespace:
            lines.append(f"- Known namespace: {self.namespace}")
        if self.workloads:
            lines.append(f"- Known workloads: {', '.join(self.workloads[:4])}")
        if self.services:
            lines.append(f"- Known services: {', '.join(self.services[:4])}")
        if self.pods:
            lines.append(f"- Known pods: {', '.join(self.pods[:4])}")
        if self.nodes:
            lines.append(f"- Known nodes: {', '.join(self.nodes[:4])}")
        if self.aws_resources:
            lines.append(f"- Known AWS refs: {', '.join(self.aws_resources[:4])}")
        if self.evidence_notes:
            lines.append("- Prior evidence to reuse before rediscovery:")
            lines.extend(f"  - {item}" for item in self.evidence_notes[:6])
        return "\n".join(lines)


@dataclass
class OperatorIntentState:
    """Persistent operator intent separate from incident facts."""

    mode: str = "incident_response"
    execution_policy: str = "approval_required"
    pinned_constraints: list[str] = field(default_factory=list)
    last_user_instruction: str = ""
    source_turn: int = 0
    pending_step_summary: str = ""
    pending_step_focus: str = "general"
    pending_step_stage: str = ""
    pending_step_kind: str = ""
    awaiting_follow_up: bool = False
    approved_proposed_plan: bool = False

    def clear(self) -> None:
        self.mode = "incident_response"
        self.execution_policy = "approval_required"
        self.pinned_constraints = []
        self.last_user_instruction = ""
        self.source_turn = 0
        self.pending_step_summary = ""
        self.pending_step_focus = "general"
        self.pending_step_stage = ""
        self.pending_step_kind = ""
        self.awaiting_follow_up = False
        self.approved_proposed_plan = False

    def blocks_write_actions(self) -> bool:
        return self.execution_policy in {"no_write", "prepare_only"}

    def prefers_prevention_planning(self) -> bool:
        return False

    def is_following_proposed_plan(self) -> bool:
        return self.approved_proposed_plan and bool(self.pending_step_summary)

    def allows_direct_write_execution(self) -> bool:
        return (
            self.execution_policy == "approved_follow_up"
            and self.is_following_proposed_plan()
            and self.pending_step_stage == "execute"
        )

    def render_context_block(self) -> str:
        lines = [
            "[Operator intent]",
            f"- Mode: {self.mode}",
            f"- Execution policy: {self.execution_policy}",
        ]
        if self.pinned_constraints:
            lines.append(f"- Constraints: {', '.join(self.pinned_constraints)}")
        if self.last_user_instruction:
            lines.append(f"- Latest operator instruction: {self.last_user_instruction}")
        if self.pending_step_summary:
            lines.append(f"- Pending step: {self.pending_step_summary}")
        if self.pending_step_kind:
            lines.append(f"- Pending step kind: {self.pending_step_kind}")
        if self.is_following_proposed_plan():
            lines.append("- Approved follow-up plan is active for this turn.")
        return "\n".join(lines)
def _clear_pending_step(operator_intent_state: OperatorIntentState) -> None:
    operator_intent_state.pending_step_summary = ""
    operator_intent_state.pending_step_focus = "general"
    operator_intent_state.pending_step_stage = ""
    operator_intent_state.pending_step_kind = ""
    operator_intent_state.awaiting_follow_up = False
    operator_intent_state.approved_proposed_plan = False


def _normalize_reply(text: str) -> str:
    lowered = str(text or "").strip().lower()
    lowered = re.sub(r"[`'\"]", "", lowered)
    lowered = re.sub(r"[.!?]+$", "", lowered)
    return " ".join(lowered.split())


def _matches_reply(text: str, prefixes: tuple[str, ...]) -> bool:
    normalized = _normalize_reply(text)
    return any(normalized == prefix or normalized.startswith(prefix + " ") for prefix in prefixes)


def _normalize_follow_up_summary(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "").strip())
    return cleaned.strip(" .")


def _infer_follow_up_focus(summary: str, default_focus: str) -> str:
    lowered = str(summary or "").lower()
    if any(marker in lowered for marker in _FOLLOW_UP_NODE_MARKERS):
        return "node"
    if any(marker in lowered for marker in _FOLLOW_UP_SERVICE_MARKERS):
        return "service"
    if any(marker in lowered for marker in _FOLLOW_UP_STORAGE_MARKERS):
        return "storage"
    if any(marker in lowered for marker in _FOLLOW_UP_WORKLOAD_MARKERS):
        return "workload"
    if any(marker in lowered for marker in _FOLLOW_UP_POD_MARKERS):
        return "pod"
    return str(default_focus or "general") or "general"


def _classify_follow_up_step(summary: str) -> tuple[str, str]:
    lowered = str(summary or "").lower()
    if any(marker in lowered for marker in _FOLLOW_UP_EXECUTE_MARKERS):
        return "execute", "execute"
    if any(marker in lowered for marker in _FOLLOW_UP_COLLECT_MARKERS):
        return "collect", "collect"
    return "collect", "collect"


def _extract_follow_up_summary(final_text: str) -> str:
    prompt_match = _FOLLOW_UP_PROMPT_RE.search(str(final_text or ""))
    if prompt_match:
        return _normalize_follow_up_summary(prompt_match.group(1))

    block_match = _RECOMMENDED_ACTION_RE.search(str(final_text or ""))
    if not block_match:
        return ""

    lines = [
        re.sub(r"^[\-\*\d\.\)\s]+", "", line).strip()
        for line in block_match.group(1).splitlines()
        if str(line).strip()
    ]
    lines = [line for line in lines if line]
    if not lines:
        return ""
    return _normalize_follow_up_summary("; ".join(lines[:3]))


def _extract_follow_up_step(final_text: str, turn_plan: Any) -> dict[str, str] | None:
    text = str(final_text or "")
    if "yes/no" not in text.lower() or "proceed" not in text.lower():
        return None

    summary = _extract_follow_up_summary(text)
    if not summary:
        return None

    kind, stage = _classify_follow_up_step(summary)
    default_focus = str(getattr(turn_plan, "focus", "general") or "general")
    return {
        "summary": summary,
        "focus": _infer_follow_up_focus(summary, default_focus),
        "kind": kind,
        "stage": stage,
    }


def register_operator_follow_up(
    *,
    operator_intent_state: OperatorIntentState,
    final_text: str,
    turn_plan: Any,
    approval_pending: bool = False,
) -> OperatorIntentState:
    """Track proposed next-step follow-ups from assistant responses."""
    if approval_pending:
        _clear_pending_step(operator_intent_state)
        operator_intent_state.mode = "incident_response"
        operator_intent_state.execution_policy = "approval_required"
        operator_intent_state.pinned_constraints = []
        return operator_intent_state

    follow_up = _extract_follow_up_step(final_text, turn_plan)
    if follow_up is not None:
        operator_intent_state.mode = "follow_up_action"
        operator_intent_state.execution_policy = "approval_required"
        operator_intent_state.pending_step_summary = follow_up["summary"]
        operator_intent_state.pending_step_focus = follow_up["focus"]
        operator_intent_state.pending_step_stage = follow_up["stage"]
        operator_intent_state.pending_step_kind = follow_up["kind"]
        operator_intent_state.awaiting_follow_up = True
        operator_intent_state.approved_proposed_plan = False
        operator_intent_state.pinned_constraints = []
        return operator_intent_state

    _clear_pending_step(operator_intent_state)
    if operator_intent_state.mode == "follow_up_action":
        operator_intent_state.mode = "incident_response"
        operator_intent_state.execution_policy = "approval_required"
        operator_intent_state.pinned_constraints = []
    return operator_intent_state


def update_operator_intent_state(
    *,
    operator_intent_state: OperatorIntentState,
    user_input: str,
    turn_index: int,
    incident_state: IncidentState | None = None,
    approval_pending: bool = False,
) -> OperatorIntentState:
    """Keep only explicit operator metadata; do not infer workflow from wording."""
    text = str(user_input or "").strip()
    if not text:
        return operator_intent_state

    operator_intent_state.last_user_instruction = text
    operator_intent_state.source_turn = turn_index

    if approval_pending:
        return operator_intent_state

    if operator_intent_state.awaiting_follow_up:
        if _matches_reply(text, _NEGATIVE_REPLIES):
            operator_intent_state.mode = "incident_response"
            operator_intent_state.execution_policy = "approval_required"
            operator_intent_state.pinned_constraints = []
            _clear_pending_step(operator_intent_state)
            return operator_intent_state

        if _matches_reply(text, _AFFIRMATIVE_REPLIES):
            operator_intent_state.mode = "follow_up_action"
            operator_intent_state.awaiting_follow_up = False
            operator_intent_state.approved_proposed_plan = True
            summary = str(operator_intent_state.pending_step_summary or "").strip()
            if operator_intent_state.pending_step_stage == "execute":
                operator_intent_state.execution_policy = "approved_follow_up"
                operator_intent_state.pinned_constraints = (
                    [f"Execute only this approved next step: {summary}"] if summary else []
                )
            else:
                operator_intent_state.execution_policy = "approval_required"
                operator_intent_state.pinned_constraints = (
                    [f"Continue this approved next step: {summary}"] if summary else []
                )
            return operator_intent_state

    operator_intent_state.mode = "incident_response"
    operator_intent_state.execution_policy = "approval_required"
    operator_intent_state.pinned_constraints = []
    _clear_pending_step(operator_intent_state)
    return operator_intent_state


def _add_unique(items: list[str], values: list[str], *, limit: int = 12) -> None:
    seen = {str(item).strip().lower() for item in items if str(item).strip()}
    for value in values:
        clean = str(value).strip()
        key = clean.lower()
        if not clean or key in seen:
            continue
        seen.add(key)
        items.append(clean)
        if len(items) >= limit:
            break


def _extract_first_sentence(text: str) -> str:
    raw = " ".join(str(text or "").strip().split())
    if not raw:
        return ""
    match = re.split(r"(?<=[\.\!\?])\s+", raw, maxsplit=1)
    return match[0][:240].strip()


def _trim_excerpt(text: str, *, max_chars: int = 600) -> str:
    clean = str(text or "").strip()
    if looks_like_structured_payload(clean):
        max_chars = max(max_chars, int(getattr(Config, "AGENT_STRUCTURED_TOOL_RESULT_MAX_CHARS", 12000)))
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 3].rstrip() + "..."


def _extract_namespace_from_command(command: str) -> str:
    text = str(command or "")
    for pattern in (
        r"(?:^|\s)-n\s+([a-z0-9][a-z0-9-]*)\b",
        r"(?:^|\s)--namespace\s+([a-z0-9][a-z0-9-]*)\b",
        r"(?:^|\s)--namespace=([a-z0-9][a-z0-9-]*)\b",
    ):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return ""


def _extract_scope_from_text(text: str) -> dict[str, list[str]]:
    lowered = str(text or "").lower()
    out: dict[str, list[str]] = {
        "namespaces": [],
        "pods": [],
        "services": [],
        "workloads": [],
        "ingresses": [],
        "nodes": [],
    }

    namespace = ""
    for pattern in _RESOURCE_PATTERNS["namespace"]:
        match = re.search(pattern, lowered)
        if match:
            namespace = match.group(1).strip()
            break
    if namespace and namespace not in {"all", "any", "auto"}:
        out["namespaces"].append(namespace)

    for pattern in _RESOURCE_PATTERNS["pod"]:
        out["pods"].extend(match.group(1).strip() for match in re.finditer(pattern, lowered))
    for pattern in _RESOURCE_PATTERNS["service"]:
        out["services"].extend(match.group(1).strip() for match in re.finditer(pattern, lowered))
    for pattern in _RESOURCE_PATTERNS["deployment"]:
        out["workloads"].extend(f"deployment/{match.group(1).strip()}" for match in re.finditer(pattern, lowered))
    for pattern in _RESOURCE_PATTERNS["statefulset"]:
        out["workloads"].extend(f"statefulset/{match.group(1).strip()}" for match in re.finditer(pattern, lowered))
    for pattern in _RESOURCE_PATTERNS["daemonset"]:
        out["workloads"].extend(f"daemonset/{match.group(1).strip()}" for match in re.finditer(pattern, lowered))
    for pattern in _RESOURCE_PATTERNS["ingress"]:
        out["ingresses"].extend(match.group(1).strip() for match in re.finditer(pattern, lowered))
    for pattern in _RESOURCE_PATTERNS["node"]:
        out["nodes"].extend(match.group(1).strip() for match in re.finditer(pattern, lowered))

    return out


def _extract_aws_refs(text: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for match in _AWS_ARN_RE.findall(str(text or "")):
        key = match.lower()
        if key not in seen:
            seen.add(key)
            out.append(match)
    for match in _AWS_ID_RE.findall(str(text or "")):
        key = match.lower()
        if key not in seen:
            seen.add(key)
            out.append(match)
    return out


def _merge_scope_from_args(state: IncidentState, args: Any) -> None:
    if not isinstance(args, dict):
        return

    namespace = str(args.get("namespace") or "").strip()
    if namespace and namespace.lower() not in {"all", "any", "auto"}:
        state.namespace = namespace

    command = str(args.get("command") or "").strip()
    if command and not state.namespace:
        cmd_ns = _extract_namespace_from_command(command)
        if cmd_ns and cmd_ns.lower() not in {"all", "any", "auto"}:
            state.namespace = cmd_ns

    for key, value in args.items():
        if key in {"reason", "limit"}:
            continue
        if key == "pod_name":
            _add_unique(state.pods, [str(value)])
        elif key == "pod_names" and isinstance(value, list):
            _add_unique(state.pods, [str(item) for item in value])
        elif key == "service_name":
            _add_unique(state.services, [str(value)])
        elif key in {"deployment_name", "statefulset_name", "daemonset_name"}:
            prefix = key.replace("_name", "")
            _add_unique(state.workloads, [f"{prefix}/{value}"])
        elif key == "node_name":
            _add_unique(state.nodes, [str(value)])
        elif key == "name" and args.get("kind"):
            _add_unique(state.workloads, [f"{args.get('kind')}/{value}"])
        elif key == "workloads" and isinstance(value, list):
            workloads = []
            for item in value:
                if not isinstance(item, dict):
                    continue
                kind = str(item.get("kind") or "").strip()
                name = str(item.get("name") or "").strip()
                if kind and name:
                    workloads.append(f"{kind}/{name}")
            _add_unique(state.workloads, workloads)
        elif key == "changes" and isinstance(value, list):
            workloads = []
            for item in value:
                if not isinstance(item, dict):
                    continue
                kind = str(item.get("kind") or "").strip()
                name = str(item.get("name") or "").strip()
                if kind and name:
                    workloads.append(f"{kind}/{name}")
            _add_unique(state.workloads, workloads)

    if command:
        scope = _extract_scope_from_text(command)
        if scope["namespaces"] and not state.namespace:
            state.namespace = scope["namespaces"][0]
        _add_unique(state.pods, scope["pods"])
        _add_unique(state.services, scope["services"])
        _add_unique(state.workloads, scope["workloads"])
        _add_unique(state.ingresses, scope["ingresses"])
        _add_unique(state.nodes, scope["nodes"])
        _add_unique(state.aws_resources, _extract_aws_refs(command))


def _merge_scope_from_text(state: IncidentState, text: str) -> None:
    scope = _extract_scope_from_text(text)
    if scope["namespaces"] and not state.namespace:
        state.namespace = scope["namespaces"][0]
    _add_unique(state.pods, scope["pods"])
    _add_unique(state.services, scope["services"])
    _add_unique(state.workloads, scope["workloads"])
    _add_unique(state.ingresses, scope["ingresses"])
    _add_unique(state.nodes, scope["nodes"])
    _add_unique(state.aws_resources, _extract_aws_refs(text))


def _update_summary_fields(state: IncidentState, final_text: str) -> None:
    summary_match = re.search(r"\*\*Issue Summary:\*\*\s*(.+)", final_text or "", re.IGNORECASE)
    severity_match = re.search(r"\*\*Severity:\*\*\s*([A-Za-z0-9]+)", final_text or "", re.IGNORECASE)
    confidence_match = re.search(r"\*\*Confidence Score:\*\*\s*([0-9]{1,3})", final_text or "", re.IGNORECASE)

    if summary_match:
        state.summary = summary_match.group(1).strip()
    elif not state.summary:
        state.summary = _extract_first_sentence(final_text)

    if severity_match:
        state.severity = severity_match.group(1).strip()
    if confidence_match:
        try:
            state.confidence_score = int(confidence_match.group(1))
        except ValueError:
            state.confidence_score = 0


def apply_turn_outcome_to_state(
    *,
    incident_state: IncidentState,
    user_input: str,
    intent_mode: str,
    turn_plan: Any,
    outcome: ToolLoopOutcome | None,
    final_text: str,
    turn_index: int,
) -> IncidentState:
    """Merge a completed turn into the structured incident state."""
    state = incident_state

    if getattr(turn_plan, "reset_existing_context", False):
        state.clear()

    keep_active = bool(
        getattr(turn_plan, "continue_existing", False)
        or intent_mode in {"incident_rca", "general", "command"}
    )
    if not keep_active:
        if intent_mode == "chat":
            state.clear()
        return state

    if not state.active:
        state.active = True
        state.opened_turn = turn_index

    state.turn_index = turn_index
    state.last_updated_turn = turn_index
    state.last_intent_mode = intent_mode
    state.last_stage = str(getattr(turn_plan, "stage", "") or "")
    state.last_focus = str(getattr(turn_plan, "focus", "general") or "general")
    state.last_user_input = str(user_input or "")

    _merge_scope_from_text(state, user_input)

    records = list((outcome.records if outcome else []) or [])
    successful_records = [record for record in records if record.success]
    for record in successful_records:
        _merge_scope_from_args(state, record.args)
        _merge_scope_from_text(state, record.summary)
        _merge_scope_from_text(state, record.result_excerpt)

    _update_summary_fields(state, final_text)

    evidence_limit = max(4, int(getattr(Config, "INCIDENT_STATE_MAX_EVIDENCE", 8)))
    note_candidates = [record.summary for record in successful_records if record.summary]
    if final_text:
        note_candidates.append(_extract_first_sentence(final_text))
    merged_notes: list[str] = []
    _add_unique(merged_notes, [item for item in state.evidence_notes if item])
    _add_unique(merged_notes, [item for item in note_candidates if item], limit=evidence_limit)
    state.evidence_notes = merged_notes[:evidence_limit]

    state.recent_tool_records = successful_records[-12:]

    cache_limit = max(8, int(getattr(Config, "INCIDENT_STATE_MAX_CACHE_ENTRIES", 24)))
    for record in successful_records:
        if not record.result_excerpt:
            continue
        state.cached_tool_results[record.semantic_key] = CachedToolResult(
            semantic_key=record.semantic_key,
            tool_name=record.tool_name,
            content=_trim_excerpt(record.result_excerpt, max_chars=1200),
            summary=record.summary,
            turn_index=turn_index,
        )
    if len(state.cached_tool_results) > cache_limit:
        ordered = sorted(state.cached_tool_results.values(), key=lambda item: item.turn_index)
        trimmed = ordered[-cache_limit:]
        state.cached_tool_results = {item.semantic_key: item for item in trimmed}

    return state
