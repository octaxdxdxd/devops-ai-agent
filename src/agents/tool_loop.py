"""Tool-call execution loop for the AI Ops agent."""

from __future__ import annotations

import json
import re
import shlex
from typing import Any, Callable

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ..config import Config
from ..tools import is_write_tool
from ..utils.command_intent import CommandIntent, classify_command_intent, target_tool_for_intent
from ..utils.llm_retry import invoke_with_retries
from ..utils.payload_shape import looks_like_structured_payload
from ..utils.response import extract_response_text
from .approval import ApprovalCoordinator, PendingAction, commands_code_block, format_command_preview
from .planner import TurnPlan
from .state import IncidentState, OperatorIntentState, ToolExecutionRecord, ToolLoopOutcome


_AWS_ARN_RE = re.compile(r"\barn:aws[a-z-]*:[^\s'\"]+\b", re.IGNORECASE)
_AWS_ID_TOKEN_RE = re.compile(
    r"\b(?:ami|i|lt|sg|subnet|vpc|vol|snap|eni|igw|nat|rtb|eipalloc|eipassoc|vpce|pcx|tgw|fs|db)-[0-9a-zA-Z]{6,}\b",
    re.IGNORECASE,
)
_SEMANTIC_IGNORE_ARGS_BY_TOOL: dict[str, set[str]] = {
    "k8s_list_pods": {"limit"},
    "k8s_find_pods": {"limit"},
    "k8s_get_events": {"limit"},
}
_BROAD_DISCOVERY_TOOLS = {
    "k8s_list_namespaces",
    "k8s_list_nodes",
    "k8s_top_nodes",
    "k8s_list_pods",
    "k8s_find_pods",
    "k8s_list_deployments",
    "k8s_list_statefulsets",
    "k8s_list_daemonsets",
    "k8s_list_services",
    "k8s_list_ingresses",
    "k8s_get_events",
    "k8s_get_crashloop_pods",
}


def _operator_write_guard_message(operator_intent_state: OperatorIntentState, *, tool_name: str) -> str | None:
    """Block write actions when the operator has explicitly asked for planning-only behavior."""
    if operator_intent_state.execution_policy == "no_write":
        return (
            f"Write action `{tool_name}` blocked by operator intent. "
            "The user asked for planning/explanation only with no execution."
        )
    if operator_intent_state.execution_policy == "prepare_only":
        return (
            f"Write action `{tool_name}` blocked by operator intent. "
            "The user asked to prepare commands/change plans only, not execute them."
        )
    return None


def _preapproved_follow_up_write_enabled(operator_intent_state: OperatorIntentState) -> bool:
    return bool(operator_intent_state.allows_direct_write_execution())


def _notify_status(status_callback: Callable[[str], None] | None, text: str) -> None:
    """Best-effort status updates for the UI layer."""
    if not callable(status_callback):
        return
    try:
        status_callback(text)
    except Exception:
        return


def _resolve_tool_call(
    *,
    tool_name: str,
    tool_args: Any,
    tool_lookup: dict[str, Any],
) -> tuple[str, Any, CommandIntent | None, bool]:
    """Route command tools by command intent; returns effective tool + args."""
    if not isinstance(tool_args, dict):
        return tool_name, tool_args, None, False

    command_text = str(tool_args.get("command") or "").strip()
    if not command_text:
        return tool_name, tool_args, None, False

    intent = classify_command_intent(tool_name, command_text)
    if intent.family == "unknown":
        return tool_name, tool_args, None, False

    effective_name = tool_name
    target_name = target_tool_for_intent(intent)
    if target_name and target_name in tool_lookup:
        effective_name = target_name

    normalized_args = dict(tool_args)
    if intent.normalized_command:
        normalized_args["command"] = intent.normalized_command

    routed = effective_name != tool_name
    return effective_name, normalized_args, intent, routed


def _requires_explicit_approval(*, tool_name: str, intent: CommandIntent | None) -> bool:
    """Approval is determined by command mutability when available; else tool policy."""
    if intent is not None:
        return intent.requires_approval
    return is_write_tool(tool_name)


def _command_guard_message(*, requested_tool: str, intent: CommandIntent) -> str | None:
    if intent.is_blocked:
        return f"Command blocked for `{requested_tool}`: {intent.reason}"
    if intent.is_sensitive_read:
        return (
            f"Sensitive command blocked for `{requested_tool}`: {intent.reason} "
            "Use purpose-built diagnostic tools instead of generic command execution."
        )
    return None


_VERBOSE_CONTEXT_TOOLS = {
    "k8s_describe_pod",
    "k8s_describe_deployment",
    "k8s_describe_node",
    "k8s_get_pod_logs",
    "k8s_get_events",
    "k8s_get_resource_yaml",
    "k8s_get_pod_scheduling_report",
    "kubectl_readonly",
    "aws_cli_readonly",
}
_SIGNAL_LINE_RE = re.compile(
    r"(warning|error|fail|backoff|crashloop|oom|evict|pending|notready|unschedul|reason|message|event|status|condition)",
    re.IGNORECASE,
)


def _tool_context_budget(tool_name: str, default_max: int, *, structured_payload: bool = False) -> int:
    """Return a per-tool context budget for prompt injection."""
    if structured_payload:
        return max(
            default_max,
            int(getattr(Config, "AGENT_STRUCTURED_TOOL_RESULT_MAX_CHARS", 12000)),
        )
    if tool_name in {"k8s_get_pod_logs", "k8s_get_resource_yaml", "k8s_describe_node", "kubectl_readonly"}:
        return max(800, min(default_max, 1800))
    if tool_name in {
        "k8s_describe_pod",
        "k8s_describe_deployment",
        "k8s_get_events",
        "k8s_get_pod_scheduling_report",
        "aws_cli_readonly",
    }:
        return max(900, min(default_max, 2200))
    return default_max


def _compact_high_signal_lines(text: str, *, max_lines: int = 120) -> str:
    """Extract likely high-signal lines to preserve RCA quality with lower token usage."""
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text

    selected: list[str] = []
    for line in lines:
        raw = line.strip()
        if not raw:
            continue
        if raw.endswith(":") or _SIGNAL_LINE_RE.search(raw):
            selected.append(line)
            if len(selected) >= max_lines:
                break

    if not selected:
        # Fallback to tail if no obvious signal lines were found.
        selected = lines[-max_lines:]

    return "\n".join(selected)


def _tool_result_to_message_content(result: Any, *, tool_name: str) -> tuple[str, int]:
    """Convert tool result to model context with truncation for token control."""
    text = str(result)
    max_chars_default = max(0, int(getattr(Config, "AGENT_TOOL_RESULT_MAX_CHARS", 5000)))
    structured_payload = looks_like_structured_payload(text)
    max_chars = _tool_context_budget(tool_name, max_chars_default, structured_payload=structured_payload)

    if tool_name in _VERBOSE_CONTEXT_TOOLS and not structured_payload and len(text) > max_chars:
        compacted = _compact_high_signal_lines(text)
        if compacted and len(compacted) < len(text):
            text = (
                "[Context compressed before model injection to reduce tokens while preserving high-signal evidence]\n"
                + compacted
            )

    if max_chars and len(text) > max_chars:
        marker = "\n... [middle truncated before sending to model; see full tool output in logs/trace] ...\n"
        marker_len = len(marker)

        if max_chars <= marker_len + 64:
            trimmed = text[:max_chars]
        else:
            head = int(max_chars * 0.7)
            tail = max_chars - head - marker_len
            if tail < 128:
                tail = 128
                head = max_chars - tail - marker_len
            if head < 64:
                head = 64
                tail = max_chars - head - marker_len

            trimmed = text[:head] + marker + text[-tail:]

        return trimmed, len(text)
    return text, len(text)


def _message_content_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(item) for item in content)
    return str(content or "")


def _build_evidence_corpus(messages: list[Any]) -> str:
    """Collect prior-turn/tool evidence text (excluding current model draft)."""
    parts: list[str] = []
    total = 0
    max_chars = 200_000
    for msg in messages:
        text = _message_content_text(msg)
        if not text:
            continue
        remaining = max_chars - total
        if remaining <= 0:
            break
        if len(text) > remaining:
            text = text[:remaining]
        parts.append(text)
        total += len(text)
    return "\n".join(parts)


def _extract_aws_reference_tokens(command: str) -> list[str]:
    """Extract likely AWS resource identifiers (IDs/ARNs) from command text."""
    raw = (command or "").strip()
    if not raw:
        return []

    out: list[str] = []
    seen: set[str] = set()

    for match in _AWS_ARN_RE.findall(raw):
        key = match.lower()
        if key not in seen:
            seen.add(key)
            out.append(match)

    for match in _AWS_ID_TOKEN_RE.findall(raw):
        key = match.lower()
        if key not in seen:
            seen.add(key)
            out.append(match)

    try:
        tokens = shlex.split(raw)
    except ValueError:
        tokens = raw.split()

    if tokens and tokens[0].lower() == "aws":
        tokens = tokens[1:]

    def add_value(value: str) -> None:
        for chunk in re.split(r"[,\s]+", value.strip()):
            item = chunk.strip().strip("'\"")
            if not item:
                continue
            if _AWS_ARN_RE.fullmatch(item) or _AWS_ID_TOKEN_RE.fullmatch(item):
                key = item.lower()
                if key not in seen:
                    seen.add(key)
                    out.append(item)

    i = 0
    while i < len(tokens):
        token = str(tokens[i]).strip()
        low = token.lower()
        if not low.startswith("--"):
            i += 1
            continue

        value = ""
        if "=" in token:
            flag, value = token.split("=", 1)
        else:
            flag = token
            if i + 1 < len(tokens):
                next_token = str(tokens[i + 1]).strip()
                if not next_token.startswith("--"):
                    value = next_token
                    i += 1

        flag_low = flag.lower()
        if "id" in flag_low or "arn" in flag_low:
            add_value(value)
        i += 1

    return out


def _validate_aws_write_grounding(command: str, evidence_corpus: str) -> tuple[list[str], str] | None:
    """Block write commands that introduce unverified AWS IDs/ARNs."""
    refs = _extract_aws_reference_tokens(command)
    if not refs:
        return None

    corpus_low = (evidence_corpus or "").lower()
    unresolved = [ref for ref in refs if ref.lower() not in corpus_low]
    if not unresolved:
        return None

    message = (
        "Write command blocked: unverified AWS resource identifiers found in the proposed write command.\n"
        f"Unverified identifiers: {', '.join(unresolved[:8])}\n"
        "Run read-only AWS checks first to verify these resources exist and are correct, then propose the write again."
    )
    return unresolved, message


def _semantic_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _semantic_jsonable(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, list):
        return [_semantic_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_semantic_jsonable(item) for item in value]
    return value


def _semantic_tool_signature(tool_name: str, tool_args: Any, intent: CommandIntent | None = None) -> str:
    """Normalize tool calls so near-duplicate investigations collapse together."""
    if isinstance(tool_args, dict) and "command" in tool_args:
        command_text = ""
        if intent is not None and intent.normalized_command:
            command_text = intent.normalized_command
        else:
            command_text = str(tool_args.get("command") or "").strip()
        return f"{tool_name}:command:{command_text}"

    if not isinstance(tool_args, dict):
        return f"{tool_name}:{str(tool_args)}"

    ignored = set(_SEMANTIC_IGNORE_ARGS_BY_TOOL.get(tool_name, set()))
    normalized: dict[str, Any] = {}
    for key, value in sorted(tool_args.items(), key=lambda item: str(item[0])):
        if key in {"reason"} or key in ignored:
            continue
        normalized[str(key)] = _semantic_jsonable(value)
    return f"{tool_name}:{json.dumps(normalized, sort_keys=True, ensure_ascii=False)}"


def _single_target_read_fanout_signature(
    tool_name: str,
    tool_args: Any,
    intent: CommandIntent | None = None,
) -> str | None:
    """Return a coarse signature for readonly one-resource fanout patterns."""
    if tool_name != "kubectl_readonly" or not isinstance(tool_args, dict):
        return None
    if intent is not None and intent.capability != "safe_read":
        return None

    command_text = ""
    if intent is not None and intent.normalized_command:
        command_text = intent.normalized_command
    else:
        command_text = str(tool_args.get("command") or "").strip()
    if not command_text:
        return None

    try:
        tokens = shlex.split(command_text)
    except Exception:
        tokens = command_text.split()
    if len(tokens) < 3:
        return None

    verb = tokens[0].lower()
    if verb not in {"get", "describe", "top"}:
        return None

    if any(flag in tokens for flag in ("-A", "--all-namespaces")):
        return None

    if verb == "top":
        resource = tokens[1].lower()
        name = tokens[2]
    else:
        resource = tokens[1].lower()
        name = tokens[2]

    if not resource or not name or name.startswith("-"):
        return None
    if "," in resource or "," in name:
        return None

    normalized_resource = resource.split("/", 1)[0].rstrip("s")
    if normalized_resource not in {"pod", "node", "service", "deployment", "statefulset", "daemonset", "ingress", "pvc", "pv"}:
        return None
    return f"{tool_name}:single-target-read:{normalized_resource}"


def _should_prefer_aggregated_reads(turn_plan: TurnPlan | None) -> bool:
    if turn_plan is None:
        return False
    aspects = set(getattr(turn_plan, "requested_aspects", ()) or ())
    return bool(aspects.intersection({"inventory", "capacity", "cost", "optimization", "explanation"}))


def _trim_record_excerpt(text: Any, *, max_chars: int = 1200) -> str:
    clean = str(text or "").strip()
    if looks_like_structured_payload(clean):
        max_chars = max(max_chars, int(getattr(Config, "AGENT_STRUCTURED_TOOL_RESULT_MAX_CHARS", 12000)))
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 3].rstrip() + "..."


def _summarize_tool_result(tool_name: str, result_text: Any) -> str:
    lines = [line.strip() for line in str(result_text or "").splitlines() if line.strip()]
    for line in lines:
        if line in {"```", "```bash", "```text"}:
            continue
        if line.startswith("Planned command:"):
            continue
        if line.startswith("Output:"):
            continue
        return _trim_record_excerpt(line, max_chars=240)
    if not lines:
        return f"{tool_name} returned no output."
    return _trim_record_excerpt(lines[0], max_chars=240)


def _evidence_categories_for_call(
    *,
    tool_name: str,
    tool_args: Any,
    intent: CommandIntent | None = None,
) -> tuple[str, ...]:
    categories: set[str] = set()
    args = tool_args if isinstance(tool_args, dict) else {}
    namespace = str(args.get("namespace") or "").strip().lower() if isinstance(args, dict) else ""

    if tool_name in {"k8s_list_namespaces", "k8s_find_pods"}:
        categories.update({"discovery_cluster", "pod_health"})
    if tool_name == "k8s_list_pods":
        categories.add("pod_health")
        if namespace in {"", "all"}:
            categories.add("discovery_cluster")
    if tool_name in {"k8s_describe_pod", "k8s_get_pod_logs", "k8s_top_pods", "k8s_get_crashloop_pods"}:
        categories.add("pod_health")
    if tool_name in {
        "k8s_list_deployments",
        "k8s_describe_deployment",
        "k8s_list_statefulsets",
        "k8s_list_daemonsets",
        "helm_readonly",
    }:
        categories.add("workload_health")
    if tool_name in {"k8s_list_services", "k8s_list_ingresses"}:
        categories.add("service_network")
    if tool_name == "k8s_get_resource_yaml":
        kind = str(args.get("kind") or "").strip().lower()
        if kind in {"service", "ingress"}:
            categories.add("service_network")
        else:
            categories.add("workload_health")
    if tool_name in {"k8s_get_events", "k8s_get_pod_scheduling_report"}:
        categories.add("events")
    if tool_name in {"k8s_list_nodes", "k8s_top_nodes", "k8s_describe_node"}:
        categories.add("node_health")
    if tool_name in {"k8s_get_pvcs", "k8s_list_pvs", "k8s_describe_pvc", "k8s_describe_pv"}:
        categories.add("storage")
    if tool_name == "aws_cli_readonly":
        categories.add("aws")
        command = str(args.get("command") or "").lower()
        if any(token in command for token in {"elbv2", "load-balancer", "target-group", "listener"}):
            categories.add("service_network")
        if any(token in command for token in {"autoscaling", "ec2", "eks", "nodegroup"}):
            categories.add("node_health")
    if tool_name == "kubectl_readonly":
        command = str(args.get("command") or "").lower()
        if any(token in command for token in {"service", "svc", "ingress"}):
            categories.add("service_network")
        if any(token in command for token in {"pod", "logs", "container"}):
            categories.add("pod_health")
        if any(token in command for token in {"node", "taint", "drain"}):
            categories.add("node_health")
        if any(token in command for token in {"pvc", "pv", "volume"}):
            categories.add("storage")
        if not categories and intent is not None and intent.verb in {"get", "describe"}:
            categories.add("discovery_cluster")
    if not categories:
        categories.add("general")
    return tuple(sorted(categories))


def _cached_result_for_call(
    *,
    incident_state: IncidentState | None,
    turn_plan: TurnPlan | None,
    semantic_key: str,
    tool_name: str,
    requires_approval: bool,
) -> str | None:
    if incident_state is None or turn_plan is None:
        return None
    if requires_approval or turn_plan.prefer_fresh_reads or not turn_plan.prefer_cached_reads:
        return None

    cached = incident_state.cached_tool_results.get(semantic_key)
    if cached is None:
        return None

    if turn_plan.allow_broad_discovery and tool_name not in _BROAD_DISCOVERY_TOOLS:
        return None

    content = cached.content or cached.summary or "(cached evidence unavailable)"
    return (
        f"Reusing evidence already gathered earlier in this incident for `{tool_name}` to avoid redundant rediscovery.\n"
        f"Cached summary: {cached.summary or '(no summary)'}\n"
        f"Cached result:\n{content}"
    )


def _plan_has_enough_evidence(turn_plan: TurnPlan | None, records: list[ToolExecutionRecord]) -> bool:
    if turn_plan is None:
        return False
    if turn_plan.mode not in {"incident_rca", "general"}:
        return False

    successful = [record for record in records if record.success]
    if not successful:
        return False

    categories: set[str] = set()
    fresh_categories: set[str] = set()
    fresh_count = 0
    for record in successful:
        categories.update(record.evidence_categories)
        if not record.from_cache:
            fresh_count += 1
            fresh_categories.update(record.evidence_categories)

    objectives = tuple(getattr(turn_plan, "objectives", ()) or ())
    if objectives:
        return all(
            _objective_has_enough_evidence(
                turn_plan=turn_plan,
                objective=objective,
                successful_records=successful,
            )
            for objective in objectives
        )

    required = set(turn_plan.required_categories)
    requested_aspects = set(getattr(turn_plan, "requested_aspects", ()) or ())
    minimum_calls = _minimum_successful_calls_for_plan(turn_plan)
    if turn_plan.prefer_fresh_reads and not fresh_count:
        return False
    if turn_plan.prefer_fresh_reads and required and required.issubset(fresh_categories):
        return fresh_count >= minimum_calls
    if required and required.issubset(categories):
        return len(successful) >= minimum_calls

    if "inventory" in requested_aspects and "workload_health" in categories and len(successful) >= minimum_calls:
        return True

    if turn_plan.continue_existing and len(categories) >= 2 and len(successful) >= minimum_calls:
        return True
    return False


def _objective_has_enough_evidence(
    *,
    turn_plan: TurnPlan,
    objective: Any,
    successful_records: list[ToolExecutionRecord],
) -> bool:
    required = set(getattr(objective, "required_categories", ()) or ())
    relevant_records: list[ToolExecutionRecord] = []
    relevant_categories: set[str] = set()
    relevant_fresh_categories: set[str] = set()
    fresh_count = 0

    for record in successful_records:
        record_categories = set(record.evidence_categories)
        if required and not record_categories.intersection(required):
            continue
        relevant_records.append(record)
        relevant_categories.update(record_categories)
        if not record.from_cache:
            fresh_count += 1
            relevant_fresh_categories.update(record_categories)

    if turn_plan.prefer_fresh_reads and not fresh_count:
        return False
    if required and not required.issubset(relevant_categories):
        return False
    if turn_plan.prefer_fresh_reads and required and not required.issubset(relevant_fresh_categories):
        return False

    minimum_calls = max(1, int(getattr(objective, "minimum_successful_calls", 1)))
    return len(relevant_records) >= minimum_calls


def _minimum_successful_calls_for_plan(turn_plan: TurnPlan) -> int:
    objectives = tuple(getattr(turn_plan, "objectives", ()) or ())
    if objectives:
        return max(int(getattr(objective, "minimum_successful_calls", 1)) for objective in objectives)

    aspects = set(getattr(turn_plan, "requested_aspects", ()) or ())
    if turn_plan.stage == "verify":
        return 1

    minimum = 2
    if turn_plan.focus == "workload" and aspects == {"inventory"}:
        minimum = 1
    if aspects.intersection({"capacity", "cost", "optimization", "explanation"}):
        minimum = max(minimum, 3)
    if len(aspects) >= 3:
        minimum = max(minimum, 4)
    return min(minimum, 4)


def _missing_requested_aspects(
    turn_plan: TurnPlan | None,
    records: list[ToolExecutionRecord],
) -> tuple[str, ...]:
    if turn_plan is None:
        return ()
    successful_records = [record for record in records if record.success]
    missing: list[str] = []
    objectives = tuple(getattr(turn_plan, "objectives", ()) or ())
    if objectives:
        for objective in objectives:
            if not _objective_has_enough_evidence(
                turn_plan=turn_plan,
                objective=objective,
                successful_records=successful_records,
            ):
                key = str(getattr(objective, "key", "") or "").strip() or "analysis"
                if key not in missing:
                    missing.append(key)
        return tuple(missing)

    aspects = tuple(getattr(turn_plan, "requested_aspects", ()) or ())
    if not _plan_has_enough_evidence(turn_plan, successful_records):
        for aspect in aspects:
            if aspect == "analysis":
                continue
            missing.append(aspect)
    return tuple(missing)


def _force_readonly_completion_prompt(turn_plan: TurnPlan | None, final_text: str, *, missing_aspects: tuple[str, ...]) -> str:
    reasons: list[str] = []
    if missing_aspects:
        reasons.append(f"You have not fully answered these parts of the request yet: {', '.join(missing_aspects)}.")
    stage = str(getattr(turn_plan, "stage", "") or "")
    focus = str(getattr(turn_plan, "focus", "") or "")
    return (
        "Continue this turn now.\n"
        f"Current stage={stage}, focus={focus}.\n"
        + " ".join(reasons)
        + " Use additional read-only tools if needed, then answer the user's full request directly. "
        "Only stop and ask for approval if a mutating action is actually required."
    )


def _plan_synthesis_prompt(turn_plan: TurnPlan, incident_state: IncidentState | None) -> str:
    scope_lines: list[str] = []
    if incident_state is not None:
        if incident_state.namespace:
            scope_lines.append(f"namespace={incident_state.namespace}")
        if incident_state.workloads:
            scope_lines.append(f"workloads={', '.join(incident_state.workloads[:3])}")
        if incident_state.services:
            scope_lines.append(f"services={', '.join(incident_state.services[:3])}")
        if incident_state.pods:
            scope_lines.append(f"pods={', '.join(incident_state.pods[:3])}")

    scope_text = ", ".join(scope_lines) if scope_lines else "no locked scope"
    aspects = set(getattr(turn_plan, "requested_aspects", ()) or ())
    objectives = tuple(getattr(turn_plan, "objectives", ()) or ())
    objectives_text = "; ".join(objective.description for objective in objectives[:5]) if objectives else ""
    if aspects.intersection({"inventory", "capacity", "cost", "optimization", "explanation"}) and "diagnosis" not in aspects:
        return (
            "Stop calling tools now and answer the user's request directly.\n"
            f"Current stage: {turn_plan.stage}. Focus: {turn_plan.focus}. Scope: {scope_text}.\n"
            + (
                f"Complete these objectives in the final answer: {objectives_text}.\n"
                if objectives_text
                else ""
            )
            + 
            "Use only the evidence already collected in this turn plus the carried incident context. "
            "Do not restart discovery. Answer every part of the question with concrete lists, totals, costs, explanations, or recommendations as requested. "
            "Do not fall back to an incident summary unless the user actually asked for diagnosis."
        )
    return (
        "Stop calling tools now and synthesize the current incident.\n"
        f"Current stage: {turn_plan.stage}. Focus: {turn_plan.focus}. Scope: {scope_text}.\n"
        + (
            f"Complete these objectives in the final answer: {objectives_text}.\n"
            if objectives_text
            else ""
        )
        +
        "Use only the evidence already collected in this turn plus the carried incident context. "
        "Do not restart discovery. Provide one consolidated diagnosis, confidence, and the next best action."
    )


def handle_tool_calls(
    *,
    response: Any,
    user_input: str,
    chat_history: list,
    prompt: Any,
    llm: Any,
    llm_with_tools: Any,
    tools: list,
    tools_by_name: dict[str, Any] | None = None,
    approval: ApprovalCoordinator,
    turn_plan: TurnPlan | None = None,
    incident_state: IncidentState | None = None,
    operator_intent_state: OperatorIntentState | None = None,
    trace_writer: Any = None,
    trace_id: str | None = None,
    status_callback: Callable[[str], None] | None = None,
) -> ToolLoopOutcome:
    """Execute iterative tool calls until the model produces a final response."""
    tw = trace_writer

    messages = prompt.format_messages(chat_history=chat_history, input=user_input)
    tool_lookup = tools_by_name or {tool.name: tool for tool in tools}

    max_iterations = getattr(Config, "MAX_ITERATIONS", 5)
    max_tool_calls = getattr(Config, "MAX_TOOL_CALLS_PER_TURN", 12)
    max_duplicate_tool_calls = getattr(Config, "MAX_DUPLICATE_TOOL_CALLS", 2)
    max_semantic_duplicate_tool_calls = getattr(
        Config,
        "MAX_SEMANTIC_DUPLICATE_TOOL_CALLS",
        max_duplicate_tool_calls,
    )
    max_read_fanout_calls = max(3, int(getattr(Config, "MAX_SINGLE_TARGET_READ_FANOUT_CALLS", 4)))

    iteration = 0
    total_tool_calls = 0
    call_signature_counts: dict[str, int] = {}
    semantic_signature_counts: dict[str, int] = {}
    read_fanout_signature_counts: dict[str, int] = {}
    execution_records: list[ToolExecutionRecord] = []
    current_response = response
    operator_intent_state = operator_intent_state or OperatorIntentState()
    completeness_retries = 0
    max_completeness_retries = max(0, int(getattr(Config, "MAX_COMPLETENESS_RETRIES", 2)))

    while iteration < max_iterations:
        iteration += 1

        if not (hasattr(current_response, "tool_calls") and current_response.tool_calls):
            final_text = extract_response_text(current_response)
            missing_aspects = _missing_requested_aspects(turn_plan, execution_records)
            if (
                completeness_retries < max_completeness_retries
                and missing_aspects
                and iteration < max_iterations
            ):
                completeness_retries += 1
                if tw and trace_id:
                    tw.emit(
                        {
                            "trace_id": trace_id,
                            "event": "tool_loop.force_readonly_continue",
                            "reason": "missing_aspects",
                            "missing_aspects": list(missing_aspects),
                            "retry": completeness_retries,
                        }
                    )
                _notify_status(status_callback, "Continuing read-only investigation to finish the answer...")
                messages.append(AIMessage(content=current_response.content))
                messages.append(
                    HumanMessage(
                        content=_force_readonly_completion_prompt(
                            turn_plan,
                            final_text,
                            missing_aspects=missing_aspects,
                        )
                    )
                )
                current_response = invoke_with_retries(
                    llm_with_tools,
                    messages,
                    trace_writer=tw,
                    trace_id=trace_id,
                    event="llm.invoke.force_completeness",
                )
                continue

            return ToolLoopOutcome(
                final_text=final_text,
                records=execution_records,
            )

        resolved_calls: list[dict[str, Any]] = []
        for tool_call in current_response.tool_calls:
            original_name = str(tool_call["name"])
            original_args = tool_call["args"]
            effective_name, effective_args, intent, routed = _resolve_tool_call(
                tool_name=original_name,
                tool_args=original_args,
                tool_lookup=tool_lookup,
            )
            resolved_calls.append(
                {
                    "tool_call_id": tool_call["id"],
                    "original_name": original_name,
                    "original_args": original_args,
                    "name": effective_name,
                    "args": effective_args,
                    "intent": intent,
                    "routed": routed,
                    "requires_approval": _requires_explicit_approval(tool_name=effective_name, intent=intent),
                }
            )

        tool_messages: list[ToolMessage] = []
        for call_idx, resolved in enumerate(resolved_calls):
            tool_call_id = resolved["tool_call_id"]
            original_name = resolved["original_name"]
            original_args = resolved["original_args"]
            tool_name = resolved["name"]
            tool_args = resolved["args"]
            tool_intent = resolved["intent"]
            requires_approval = bool(resolved["requires_approval"])
            preapproved_follow_up_write = requires_approval and _preapproved_follow_up_write_enabled(operator_intent_state)
            if preapproved_follow_up_write:
                requires_approval = False
            guard_message = (
                _command_guard_message(requested_tool=original_name, intent=tool_intent)
                if tool_intent is not None
                else None
            )

            total_tool_calls += 1
            if total_tool_calls > max_tool_calls:
                _notify_status(status_callback, "Tool budget reached. Drafting the best answer from current evidence...")
                if tw and trace_id:
                    tw.emit(
                        {
                            "trace_id": trace_id,
                            "event": "tool_loop.call_budget_hit",
                            "max_tool_calls": max_tool_calls,
                            "attempted_tool": tool_name,
                            "requested_tool": original_name,
                        }
                    )

                messages.append(AIMessage(content=current_response.content, tool_calls=current_response.tool_calls))
                messages.append(
                    HumanMessage(
                        content=(
                            "Stop calling tools now. You have enough evidence. "
                            "Provide your best final incident summary using existing tool results."
                        )
                    )
                )

                forced = invoke_with_retries(
                    llm,
                    messages,
                    trace_writer=tw,
                    trace_id=trace_id,
                    event="llm.invoke.force_budget",
                )
                forced_text = extract_response_text(forced)
                if (forced_text or "").strip():
                    return ToolLoopOutcome(
                        final_text=forced_text,
                        records=execution_records,
                        stopped_reason="tool_budget_hit",
                    )

                return ToolLoopOutcome(
                    final_text=(
                        "I stopped tool execution due to safety budget limits. "
                        "Please narrow the request (service/pod/namespace/time window)."
                    ),
                    records=execution_records,
                    stopped_reason="tool_budget_hit",
                )

            try:
                signature = f"{tool_name}:{json.dumps(tool_args, sort_keys=True, ensure_ascii=False)}"
            except Exception:
                signature = f"{tool_name}:{str(tool_args)}"

            semantic_signature = _semantic_tool_signature(tool_name, tool_args, tool_intent)
            read_fanout_signature = _single_target_read_fanout_signature(tool_name, tool_args, tool_intent)

            call_signature_counts[signature] = call_signature_counts.get(signature, 0) + 1
            if call_signature_counts[signature] > max_duplicate_tool_calls:
                if tw and trace_id:
                    tw.emit(
                        {
                            "trace_id": trace_id,
                            "event": "tool_loop.duplicate_suppressed",
                            "tool": tool_name,
                            "requested_tool": original_name,
                            "args": tool_args,
                            "count": call_signature_counts[signature],
                            "max_duplicate_tool_calls": max_duplicate_tool_calls,
                        }
                    )
                tool_messages.append(
                    ToolMessage(
                        content=(
                            "Duplicate tool call suppressed to avoid loops. "
                            "Pick a different tool/arguments and continue investigation. "
                            "Good pivots: inspect different related resources, logs, events, manifests, provider metadata, or other relevant read-only CLI tools."
                        ),
                        tool_call_id=tool_call_id,
                    )
                )
                execution_records.append(
                    ToolExecutionRecord(
                        tool_name=tool_name,
                        requested_tool=original_name,
                        args=tool_args,
                        semantic_key=semantic_signature,
                        success=False,
                        summary="Duplicate tool call suppressed inside the same turn.",
                        capability=tool_intent.capability if tool_intent is not None else "",
                        evidence_categories=_evidence_categories_for_call(
                            tool_name=tool_name,
                            tool_args=tool_args,
                            intent=tool_intent,
                        ),
                    )
                )
                continue

            semantic_signature_counts[semantic_signature] = semantic_signature_counts.get(semantic_signature, 0) + 1
            if semantic_signature_counts[semantic_signature] > max_semantic_duplicate_tool_calls:
                if tw and trace_id:
                    tw.emit(
                        {
                            "trace_id": trace_id,
                            "event": "tool_loop.semantic_duplicate_suppressed",
                            "tool": tool_name,
                            "requested_tool": original_name,
                            "args": tool_args,
                            "semantic_key": semantic_signature,
                            "count": semantic_signature_counts[semantic_signature],
                            "max_semantic_duplicate_tool_calls": max_semantic_duplicate_tool_calls,
                        }
                    )
                tool_messages.append(
                    ToolMessage(
                        content=(
                            "Near-duplicate investigation step suppressed. "
                            "Reuse the evidence already gathered and pivot to a different target, scope, or tool."
                        ),
                        tool_call_id=tool_call_id,
                    )
                )
                execution_records.append(
                    ToolExecutionRecord(
                        tool_name=tool_name,
                        requested_tool=original_name,
                        args=tool_args,
                        semantic_key=semantic_signature,
                        success=False,
                        summary="Semantic duplicate tool call suppressed to reduce loopiness.",
                        capability=tool_intent.capability if tool_intent is not None else "",
                        evidence_categories=_evidence_categories_for_call(
                            tool_name=tool_name,
                            tool_args=tool_args,
                            intent=tool_intent,
                        ),
                    )
                )
                continue

            if tool_intent is not None and tool_intent.capability == "write":
                target_name = target_tool_for_intent(tool_intent)
                if target_name and target_name not in tool_lookup:
                    tool_messages.append(
                        ToolMessage(
                            content=(
                                f"Command requires `{target_name}`, but that tool is not available in this turn. "
                                "Retry with the execute-capable tool set."
                            ),
                            tool_call_id=tool_call_id,
                        )
                    )
                    continue

            if guard_message:
                if tw and trace_id:
                    tw.emit(
                        {
                            "trace_id": trace_id,
                            "event": "tool.command_blocked",
                            "requested_tool": original_name,
                            "tool": tool_name,
                            "args": tool_args,
                            "capability": tool_intent.capability if tool_intent is not None else "",
                            "reason": tool_intent.reason if tool_intent is not None else "",
                        }
                    )
                tool_messages.append(ToolMessage(content=guard_message, tool_call_id=tool_call_id))
                execution_records.append(
                    ToolExecutionRecord(
                        tool_name=tool_name,
                        requested_tool=original_name,
                        args=tool_args,
                        semantic_key=semantic_signature,
                        success=False,
                        summary=guard_message,
                        capability=tool_intent.capability if tool_intent is not None else "",
                        evidence_categories=_evidence_categories_for_call(
                            tool_name=tool_name,
                            tool_args=tool_args,
                            intent=tool_intent,
                        ),
                    )
                )
                continue

            if tw and trace_id:
                if resolved["routed"]:
                    intent_payload: dict[str, Any] = {}
                    if tool_intent is not None:
                        intent_payload = {
                            "family": tool_intent.family,
                            "verb": tool_intent.verb,
                            "is_mutating": tool_intent.is_mutating,
                        }
                    tw.emit(
                        {
                            "trace_id": trace_id,
                            "event": "tool.route",
                            "requested_tool": original_name,
                            "requested_args": original_args,
                            "tool": tool_name,
                            "args": tool_args,
                            "intent": intent_payload,
                        }
                    )
                tw.emit(
                    {
                        "trace_id": trace_id,
                        "event": "tool.request",
                        "tool": tool_name,
                        "requested_tool": original_name,
                        "args": tool_args,
                    }
                )

            tool_func = tool_lookup.get(tool_name)
            if tool_func is None:
                tool_messages.append(
                    ToolMessage(
                        content=(
                            f"Error: unknown tool '{tool_name}'. "
                            "Use other relevant read-only diagnostics or generic CLI tools that can gather the same evidence."
                        ),
                        tool_call_id=tool_call_id,
                    )
                )
                execution_records.append(
                    ToolExecutionRecord(
                        tool_name=tool_name,
                        requested_tool=original_name,
                        args=tool_args,
                        semantic_key=semantic_signature,
                        success=False,
                        summary=f"Unknown tool requested: {tool_name}.",
                        capability=tool_intent.capability if tool_intent is not None else "",
                        evidence_categories=_evidence_categories_for_call(
                            tool_name=tool_name,
                            tool_args=tool_args,
                            intent=tool_intent,
                        ),
                    )
                )
                continue

            if _should_prefer_aggregated_reads(turn_plan) and read_fanout_signature:
                read_fanout_signature_counts[read_fanout_signature] = read_fanout_signature_counts.get(read_fanout_signature, 0) + 1
                if read_fanout_signature_counts[read_fanout_signature] > max_read_fanout_calls:
                    if tw and trace_id:
                        tw.emit(
                            {
                                "trace_id": trace_id,
                                "event": "tool_loop.read_fanout_suppressed",
                                "tool": tool_name,
                                "requested_tool": original_name,
                                "args": tool_args,
                                "fanout_key": read_fanout_signature,
                                "count": read_fanout_signature_counts[read_fanout_signature],
                                "max_single_target_read_fanout_calls": max_read_fanout_calls,
                            }
                        )
                    tool_messages.append(
                        ToolMessage(
                            content=(
                                "Single-resource read fanout suppressed. "
                                "For inventory, capacity, optimization, cost, or explanation requests, switch to one or a few aggregated read commands "
                                "that cover all relevant resources at once instead of inspecting each resource individually."
                            ),
                            tool_call_id=tool_call_id,
                        )
                    )
                    execution_records.append(
                        ToolExecutionRecord(
                            tool_name=tool_name,
                            requested_tool=original_name,
                            args=tool_args,
                            semantic_key=semantic_signature,
                            success=False,
                            summary="Single-target readonly fanout suppressed to encourage aggregated reads.",
                            capability=tool_intent.capability if tool_intent is not None else "",
                            evidence_categories=_evidence_categories_for_call(
                                tool_name=tool_name,
                                tool_args=tool_args,
                                intent=tool_intent,
                            ),
                        )
                    )
                    continue

            cached_result = _cached_result_for_call(
                incident_state=incident_state,
                turn_plan=turn_plan,
                semantic_key=semantic_signature,
                tool_name=tool_name,
                requires_approval=requires_approval,
            )
            if cached_result:
                _notify_status(status_callback, f"Reusing cached evidence from `{tool_name}`...")
                if tw and trace_id:
                    tw.emit(
                        {
                            "trace_id": trace_id,
                            "event": "tool.cache_hit",
                            "tool": tool_name,
                            "requested_tool": original_name,
                            "semantic_key": semantic_signature,
                        }
                    )
                tool_messages.append(ToolMessage(content=cached_result, tool_call_id=tool_call_id))
                execution_records.append(
                    ToolExecutionRecord(
                        tool_name=tool_name,
                        requested_tool=original_name,
                        args=tool_args,
                        semantic_key=semantic_signature,
                        success=True,
                        from_cache=True,
                        result_excerpt=_trim_record_excerpt(cached_result),
                        summary=_summarize_tool_result(tool_name, cached_result),
                        capability=tool_intent.capability if tool_intent is not None else "",
                        evidence_categories=_evidence_categories_for_call(
                            tool_name=tool_name,
                            tool_args=tool_args,
                            intent=tool_intent,
                        ),
                    )
                )
                continue

            if preapproved_follow_up_write and tw and trace_id:
                tw.emit(
                    {
                        "trace_id": trace_id,
                        "event": "tool.write_preapproved_follow_up",
                        "tool": tool_name,
                        "requested_tool": original_name,
                        "args": tool_args,
                    }
                )

            if preapproved_follow_up_write and tool_intent is not None and tool_intent.family == "aws" and tool_intent.is_mutating:
                evidence_corpus = _build_evidence_corpus(messages)
                grounding_issue = _validate_aws_write_grounding(str((tool_args or {}).get("command") or ""), evidence_corpus)
                if grounding_issue is not None:
                    unresolved, guard_msg = grounding_issue
                    if tw and trace_id:
                        tw.emit(
                            {
                                "trace_id": trace_id,
                                "event": "tool.write_blocked_unverified_ids",
                                "tool": tool_name,
                                "unverified_identifiers": unresolved,
                                "source": "preapproved_follow_up",
                            }
                        )
                    tool_messages.append(ToolMessage(content=guard_msg, tool_call_id=tool_call_id))
                    execution_records.append(
                        ToolExecutionRecord(
                            tool_name=tool_name,
                            requested_tool=original_name,
                            args=tool_args,
                            semantic_key=semantic_signature,
                            success=False,
                            summary=guard_msg,
                            capability=tool_intent.capability if tool_intent is not None else "",
                            evidence_categories=_evidence_categories_for_call(
                                tool_name=tool_name,
                                tool_args=tool_args,
                                intent=tool_intent,
                            ),
                        )
                    )
                    continue

            if requires_approval:
                operator_guard = _operator_write_guard_message(operator_intent_state, tool_name=tool_name)
                if operator_guard is not None:
                    if tw and trace_id:
                        tw.emit(
                            {
                                "trace_id": trace_id,
                                "event": "tool.write_blocked_operator_intent",
                                "tool": tool_name,
                                "requested_tool": original_name,
                                "policy": operator_intent_state.execution_policy,
                            }
                        )
                    tool_messages.append(ToolMessage(content=operator_guard, tool_call_id=tool_call_id))
                    execution_records.append(
                        ToolExecutionRecord(
                            tool_name=tool_name,
                            requested_tool=original_name,
                            args=tool_args,
                            semantic_key=semantic_signature,
                            success=False,
                            summary=operator_guard,
                            capability=tool_intent.capability if tool_intent is not None else "",
                            evidence_categories=_evidence_categories_for_call(
                                tool_name=tool_name,
                                tool_args=tool_args,
                                intent=tool_intent,
                            ),
                        )
                    )
                    continue

                evidence_corpus = _build_evidence_corpus(messages)
                pending_actions: list[PendingAction] = []
                blocked_messages: list[ToolMessage] = []

                for pending in resolved_calls[call_idx:]:
                    pending_name = str(pending["name"])
                    pending_args = pending["args"]
                    pending_intent = pending["intent"]
                    pending_tool_call_id = str(pending["tool_call_id"])

                    if not bool(pending["requires_approval"]):
                        continue

                    pending_tool = tool_lookup.get(pending_name)
                    if pending_tool is None:
                        blocked_messages.append(
                            ToolMessage(
                                content=f"Error: unknown write tool '{pending_name}'.",
                                tool_call_id=pending_tool_call_id,
                            )
                        )
                        continue

                    if not isinstance(pending_args, dict):
                        blocked_messages.append(
                            ToolMessage(
                                content=f"Error: invalid args for write tool '{pending_name}'.",
                                tool_call_id=pending_tool_call_id,
                            )
                        )
                        continue

                    if pending_intent is not None and pending_intent.family == "aws" and pending_intent.is_mutating:
                        command_text = str((pending_args or {}).get("command") or "")
                        grounding_issue = _validate_aws_write_grounding(command_text, evidence_corpus)
                        if grounding_issue is not None:
                            unresolved, guard_msg = grounding_issue
                            if tw and trace_id:
                                tw.emit(
                                    {
                                        "trace_id": trace_id,
                                        "event": "tool.write_blocked_unverified_ids",
                                        "tool": pending_name,
                                        "unverified_identifiers": unresolved,
                                    }
                                )
                            blocked_messages.append(
                                ToolMessage(
                                    content=guard_msg,
                                    tool_call_id=pending_tool_call_id,
                                )
                            )
                            continue

                    pending_actions.append(PendingAction(tool=pending_tool, args=pending_args))

                if not pending_actions:
                    tool_messages.extend(blocked_messages)
                    break

                if len(pending_actions) > 1:
                    _notify_status(
                        status_callback,
                        f"Prepared {len(pending_actions)} write actions. Waiting for approval...",
                    )
                    approval.set_pending_actions(pending_actions)
                    preview_lines: list[str] = []
                    for action in pending_actions:
                        preview = format_command_preview(action.tool.name, action.args)
                        if preview:
                            preview_lines.extend(preview.splitlines())
                    cmd_preview = "\n".join(preview_lines) if preview_lines else "- command preview unavailable"
                    cmd_block = commands_code_block(cmd_preview)
                    if tw and trace_id:
                        tw.emit(
                            {
                                "trace_id": trace_id,
                                "event": "tool.requires_approval",
                                "tool": "<batch>",
                                "batch_size": len(pending_actions),
                                "tools": [action.tool.name for action in pending_actions],
                            }
                        )
                    return ToolLoopOutcome(
                        final_text=(
                            f"I recommend running {len(pending_actions)} write actions as one approved plan.\n"
                            "Planned command(s):\n"
                            f"{cmd_block}\n"
                            "Would you like me to proceed with all of them? (yes/no)"
                        ),
                        records=execution_records,
                        stopped_reason="approval_requested",
                    )

                single = pending_actions[0]
                single_name = single.tool.name
                single_args = single.args

                cmd_preview = format_command_preview(single_name, single_args)
                approval.set_pending_action(single.tool, single_args)
                _notify_status(status_callback, f"Prepared `{single_name}`. Waiting for approval...")

                if tw and trace_id:
                    tw.emit(
                        {
                            "trace_id": trace_id,
                            "event": "tool.requires_approval",
                            "tool": single_name,
                            "args": single_args,
                        }
                    )

                cmd_block = commands_code_block(cmd_preview)
                return ToolLoopOutcome(
                    final_text=(
                        f"I recommend running `{single_name}` with args {single_args}, but it requires approval.\n"
                        "Planned command(s):\n"
                        f"{cmd_block}\n"
                        "Would you like me to proceed? (yes/no)"
                    ),
                    records=execution_records,
                    stopped_reason="approval_requested",
                )

            try:
                _notify_status(status_callback, f"Running `{tool_name}`...")
                result = tool_func.invoke(tool_args)
                result_content, raw_result_len = _tool_result_to_message_content(result, tool_name=tool_name)
                if tw and trace_id:
                    tw.emit(
                        {
                            "trace_id": trace_id,
                            "event": "tool.result",
                            "tool": tool_name,
                            "requested_tool": original_name,
                            "result_type": type(result).__name__,
                            "result_len": raw_result_len,
                            "result_injected_len": len(result_content),
                        }
                    )
                tool_messages.append(ToolMessage(content=result_content, tool_call_id=tool_call_id))
                execution_records.append(
                    ToolExecutionRecord(
                        tool_name=tool_name,
                        requested_tool=original_name,
                        args=tool_args,
                        semantic_key=semantic_signature,
                        success=True,
                        result_excerpt=_trim_record_excerpt(result_content),
                        summary=_summarize_tool_result(tool_name, result_content),
                        capability=tool_intent.capability if tool_intent is not None else "",
                        evidence_categories=_evidence_categories_for_call(
                            tool_name=tool_name,
                            tool_args=tool_args,
                            intent=tool_intent,
                        ),
                    )
                )
            except Exception as exc:
                if tw and trace_id:
                    tw.emit(
                        {
                            "trace_id": trace_id,
                            "event": "tool.error",
                            "tool": tool_name,
                            "requested_tool": original_name,
                            "error": str(exc),
                        }
                    )
                tool_messages.append(ToolMessage(content=f"Error: {exc}", tool_call_id=tool_call_id))
                execution_records.append(
                    ToolExecutionRecord(
                        tool_name=tool_name,
                        requested_tool=original_name,
                        args=tool_args,
                        semantic_key=semantic_signature,
                        success=False,
                        result_excerpt=_trim_record_excerpt(str(exc)),
                        summary=f"{tool_name} failed: {exc}",
                        capability=tool_intent.capability if tool_intent is not None else "",
                        evidence_categories=_evidence_categories_for_call(
                            tool_name=tool_name,
                            tool_args=tool_args,
                            intent=tool_intent,
                        ),
                    )
                )

        messages.append(AIMessage(content=current_response.content, tool_calls=current_response.tool_calls))
        messages.extend(tool_messages)

        if turn_plan is not None and _plan_has_enough_evidence(turn_plan, execution_records):
            if tw and trace_id:
                tw.emit(
                    {
                        "trace_id": trace_id,
                        "event": "tool_loop.plan_satisfied",
                        "stage": turn_plan.stage if turn_plan is not None else "",
                        "focus": turn_plan.focus if turn_plan is not None else "",
                        "records": len(execution_records),
                    }
                )
            _notify_status(status_callback, "Enough evidence collected. Drafting the response...")
            messages.append(HumanMessage(content=_plan_synthesis_prompt(turn_plan, incident_state)))
            forced = invoke_with_retries(
                llm,
                messages,
                trace_writer=tw,
                trace_id=trace_id,
                event="llm.invoke.plan_satisfied",
            )
            forced_text = extract_response_text(forced)
            if (forced_text or "").strip():
                missing_aspects = _missing_requested_aspects(turn_plan, execution_records)
                if (
                    completeness_retries < max_completeness_retries
                    and missing_aspects
                    and iteration < max_iterations
                ):
                    completeness_retries += 1
                    if tw and trace_id:
                        tw.emit(
                            {
                                "trace_id": trace_id,
                                "event": "tool_loop.force_readonly_continue",
                                "reason": "missing_aspects",
                                "missing_aspects": list(missing_aspects),
                                "retry": completeness_retries,
                            }
                        )
                    _notify_status(status_callback, "Collecting a bit more evidence to finish the answer...")
                    messages.append(AIMessage(content=forced.content))
                    messages.append(
                        HumanMessage(
                            content=_force_readonly_completion_prompt(
                                turn_plan,
                                forced_text,
                                missing_aspects=missing_aspects,
                            )
                        )
                    )
                    current_response = invoke_with_retries(
                        llm_with_tools,
                        messages,
                        trace_writer=tw,
                        trace_id=trace_id,
                        event="llm.invoke.force_completeness",
                    )
                    continue
                return ToolLoopOutcome(
                    final_text=forced_text,
                    records=execution_records,
                    stopped_reason="plan_satisfied",
                )

        current_response = invoke_with_retries(
            llm_with_tools,
            messages,
            trace_writer=tw,
            trace_id=trace_id,
            event="llm.invoke",
        )

    if hasattr(current_response, "tool_calls") and current_response.tool_calls:
        _notify_status(status_callback, "Investigation took too many steps. Drafting a best-effort answer...")
        if tw and trace_id:
            tw.emit(
                {
                    "trace_id": trace_id,
                    "event": "tool_loop.max_iterations_hit",
                    "max_iterations": max_iterations,
                    "remaining_tool_calls": len(current_response.tool_calls),
                }
            )

        messages.append(
            HumanMessage(
                content=(
                    "Stop calling tools now. Provide your best incident summary based only on the tool results already retrieved. "
                    "If evidence is insufficient, clearly state what remains unverified and give the best next actions from existing evidence."
                )
            )
        )

        forced = invoke_with_retries(
            llm,
            messages,
            trace_writer=tw,
            trace_id=trace_id,
            event="llm.invoke.force_final",
        )

        forced_text = extract_response_text(forced)
        if (forced_text or "").strip():
            return ToolLoopOutcome(
                final_text=forced_text,
                records=execution_records,
                max_iterations_hit=True,
                stopped_reason="max_iterations_hit",
            )

    final = extract_response_text(current_response)
    if not (final or "").strip():
        trace_hint = f" Trace ID: {trace_id}" if (tw and trace_id) else ""
        return ToolLoopOutcome(
            final_text=(
                "I got an empty response from the model at the end of the tool loop. "
                f"Provider={Config.LLM_PROVIDER}, Model={Config.get_active_model_name()}."
                + trace_hint
            ),
            records=execution_records,
            max_iterations_hit=bool(hasattr(current_response, "tool_calls") and current_response.tool_calls),
        )

    return ToolLoopOutcome(final_text=final, records=execution_records)
