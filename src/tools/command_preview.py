"""Render tool invocations and action steps as CLI-equivalent previews."""

from __future__ import annotations

import json
import shlex

from ..config import Config


def _quote(value: object) -> str:
    return shlex.quote(str(value))


def _json_arg(value: object) -> str:
    return _quote(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str))


def _hyphenate(name: str) -> str:
    return str(name or "").strip().replace("_", "-")


def _parse_json_arg(raw_value: object) -> object | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, (dict, list)):
        return raw_value
    text = str(raw_value).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return text


def _append_arg(parts: list[str], flag: str, value: object) -> None:
    if value is None or value == "":
        return
    if value is True:
        parts.append(flag)
        return
    parts.extend([flag, _quote(value)])


def _append_json_arg(parts: list[str], flag: str, value: object | None) -> None:
    if value in (None, "", {}, []):
        return
    parts.extend([flag, _json_arg(value)])


def _default_namespace(namespace: str = "", all_namespaces: bool = False) -> str:
    if all_namespaces:
        return ""
    return str(namespace or Config.K8S_DEFAULT_NAMESPACE or "default").strip() or "default"


def _render_aws_cli(service: str, operation: str, params: object | None = None, region: str = "") -> str:
    parts = ["aws", str(service or "").strip(), _hyphenate(operation)]
    _append_arg(parts, "--region", str(region or "").strip())
    _append_json_arg(parts, "--cli-input-json", params)
    return " ".join(parts)


def _render_aws_update_auto_scaling(args: dict) -> str:
    parts = ["aws", "autoscaling", "update-auto-scaling-group"]
    _append_arg(parts, "--auto-scaling-group-name", args.get("asg_name", ""))
    if args.get("min_size", -1) is not None and int(args.get("min_size", -1)) >= 0:
        _append_arg(parts, "--min-size", args.get("min_size"))
    if args.get("max_size", -1) is not None and int(args.get("max_size", -1)) >= 0:
        _append_arg(parts, "--max-size", args.get("max_size"))
    if args.get("desired", -1) is not None and int(args.get("desired", -1)) >= 0:
        _append_arg(parts, "--desired-capacity", args.get("desired"))
    return " ".join(parts)


def _render_aws_describe_instances(args: dict) -> str:
    params: dict[str, object] = {}
    filters = _parse_json_arg(args.get("filters_json"))
    instance_ids = _parse_json_arg(args.get("instance_ids_json"))
    if filters:
        params["Filters"] = filters
    if instance_ids:
        params["InstanceIds"] = instance_ids
    return _render_aws_cli("ec2", "describe_instances", params or None, str(args.get("region", "") or ""))


def _render_aws_describe_service(args: dict) -> str:
    return _render_aws_cli(
        str(args.get("service", "") or ""),
        str(args.get("operation", "") or ""),
        _parse_json_arg(args.get("params_json")),
        str(args.get("region", "") or ""),
    )


def _render_aws_audit_cloudtrail(args: dict) -> str:
    parts = ["aws", "cloudtrail", "lookup-events"]
    region = str(args.get("region", "") or "").strip()
    start_time = str(args.get("start_time", "") or "").strip()
    end_time = str(args.get("end_time", "") or "").strip()
    principal = str(args.get("principal", "") or "").strip()
    event_name_exact = str(args.get("event_name_exact", "") or "").strip()
    event_source = str(args.get("event_source", "") or "").strip()
    resource_name = str(args.get("resource_name", "") or "").strip()
    resource_type = str(args.get("resource_type", "") or "").strip()

    _append_arg(parts, "--region", region)
    _append_arg(parts, "--start-time", start_time)
    _append_arg(parts, "--end-time", end_time)

    lookup_attribute = ""
    if principal:
        lookup_attribute = f"AttributeKey=Username,AttributeValue={principal}"
    elif event_name_exact:
        lookup_attribute = f"AttributeKey=EventName,AttributeValue={event_name_exact}"
    elif event_source:
        lookup_attribute = f"AttributeKey=EventSource,AttributeValue={event_source}"
    elif resource_name:
        lookup_attribute = f"AttributeKey=ResourceName,AttributeValue={resource_name}"
    elif resource_type:
        lookup_attribute = f"AttributeKey=ResourceType,AttributeValue={resource_type}"
    _append_arg(parts, "--lookup-attributes", lookup_attribute)
    try:
        max_results = int(args.get("max_events", 200) or 200)
    except (TypeError, ValueError):
        max_results = 200
    _append_arg(parts, "--max-results", min(max(max_results, 1), 50))

    filters: list[str] = []
    event_name_prefix = str(args.get("event_name_prefix", "") or "").strip()
    contains_text = str(args.get("contains_text", "") or "").strip()
    if event_name_prefix:
        filters.append(f"event_name_prefix={event_name_prefix}")
    if contains_text:
        filters.append(f"contains_text={contains_text}")
    if filters:
        return f"{' '.join(parts)}  # client-side filters: {', '.join(filters)}"
    return " ".join(parts)


def _render_aws_inspect_lambda_schedules(args: dict) -> str:
    hints = _parse_json_arg(args.get("name_hints_json"))
    regions = _parse_json_arg(args.get("regions_json"))
    hint_list = [str(item or "").strip() for item in (hints or []) if str(item or "").strip()]
    region_list = [str(item or "").strip() for item in (regions or []) if str(item or "").strip()]

    first_region = region_list[0] if region_list else "<default-region>"
    first_hint = hint_list[0] if hint_list else "<name-hint>"
    parts = ["aws", "events", "list-rules", "--region", _quote(first_region), "--name-prefix", _quote(first_hint)]

    notes: list[str] = ["then events list-targets-by-rule", "logs start-query/get-query-results", "cloudwatch get-metric-statistics"]
    if len(region_list) > 1:
        notes.append(f"regions={','.join(region_list)}")
    if len(hint_list) > 1:
        notes.append(f"hints={','.join(hint_list)}")
    return f"{' '.join(parts)}  # {', '.join(notes)}"


def _render_aws_list_resources(args: dict) -> str:
    params: dict[str, object] = {}
    resource_filters = _parse_json_arg(args.get("resource_type_filters_json"))
    if resource_filters:
        params["ResourceTypeFilters"] = resource_filters
    return _render_aws_cli("resourcegroupstaggingapi", "get_resources", params or None, "")


def _render_aws_get_alarms(args: dict) -> str:
    params: dict[str, object] = {}
    state = str(args.get("state", "") or "").strip()
    if state:
        params["StateValue"] = state
    return _render_aws_cli("cloudwatch", "describe_alarms", params or None, "")


def _render_aws_get_cloudwatch_metrics(args: dict) -> str:
    params = {
        "Namespace": args.get("namespace"),
        "MetricName": args.get("metric_name"),
        "Dimensions": _parse_json_arg(args.get("dimensions_json")) or [],
        "Period": args.get("period", 300),
        "Statistics": [args.get("stat", "Average")],
    }
    return _render_aws_cli("cloudwatch", "get_metric_statistics", params, "")


def _render_aws_get_cost(args: dict) -> str:
    params = {
        "TimePeriod": {
            "Start": str(args.get("start_date", "") or "<auto-start>"),
            "End": str(args.get("end_date", "") or "<auto-end>"),
        },
        "Granularity": args.get("granularity", "MONTHLY"),
        "Metrics": ["UnblendedCost", "UsageQuantity"],
    }
    group_by = str(args.get("group_by", "") or "").strip()
    if group_by:
        params["GroupBy"] = [{"Type": "DIMENSION", "Key": group_by}]
    return _render_aws_cli("ce", "get_cost_and_usage", params, "")


def _render_aws_describe_security_groups(args: dict) -> str:
    params: dict[str, object] = {}
    vpc_id = str(args.get("vpc_id", "") or "").strip()
    group_ids = _parse_json_arg(args.get("group_ids_json"))
    if vpc_id:
        params["Filters"] = [{"Name": "vpc-id", "Values": [vpc_id]}]
    if group_ids:
        params["GroupIds"] = group_ids
    return _render_aws_cli("ec2", "describe_security_groups", params or None, "")


def _render_kubectl_get_resources(args: dict) -> str:
    parts = ["kubectl", "get", str(args.get("kind", "") or "").strip()]
    name = str(args.get("name", "") or "").strip()
    if name:
        parts.append(_quote(name))
    label_selector = str(args.get("label_selector", "") or "").strip()
    if label_selector:
        parts.extend(["-l", _quote(label_selector)])
    all_namespaces = bool(args.get("all_namespaces"))
    if all_namespaces:
        parts.append("-A")
    else:
        _append_arg(parts, "-n", _default_namespace(str(args.get("namespace", "") or ""), all_namespaces=False))
    parts.extend(["-o", "wide"])
    return " ".join(parts)


def _render_kubectl_describe(args: dict) -> str:
    parts = [
        "kubectl",
        "describe",
        str(args.get("kind", "") or "").strip(),
        _quote(str(args.get("name", "") or "").strip()),
    ]
    _append_arg(parts, "-n", _default_namespace(str(args.get("namespace", "") or "")))
    return " ".join(parts)


def _render_kubectl_logs(args: dict) -> str:
    parts = ["kubectl", "logs", _quote(str(args.get("pod", "") or "").strip())]
    _append_arg(parts, "-n", _default_namespace(str(args.get("namespace", "") or "")))
    _append_arg(parts, "-c", str(args.get("container", "") or "").strip())
    _append_arg(parts, "--tail", args.get("tail_lines", 100))
    _append_arg(parts, "--since", str(args.get("since", "") or "").strip())
    return " ".join(parts)


def _render_kubectl_events(args: dict) -> str:
    parts = ["kubectl", "get", "events", "--sort-by=.lastTimestamp"]
    field_selector = str(args.get("field_selector", "") or "").strip()
    if field_selector:
        parts.extend(["--field-selector", _quote(field_selector)])
    all_namespaces = bool(args.get("all_namespaces"))
    if all_namespaces:
        parts.append("-A")
    else:
        _append_arg(parts, "-n", _default_namespace(str(args.get("namespace", "") or "")))
    return " ".join(parts)


def _render_kubectl_top(args: dict) -> str:
    parts = ["kubectl", "top", str(args.get("resource_type", "pods") or "pods").strip()]
    name = str(args.get("name", "") or "").strip()
    if name:
        parts.append(_quote(name))
    resource_type = str(args.get("resource_type", "pods") or "pods").strip()
    if resource_type != "nodes":
        _append_arg(parts, "-n", _default_namespace(str(args.get("namespace", "") or "")))
    return " ".join(parts)


def _render_kubectl_rollout_history(args: dict) -> str:
    parts = [
        "kubectl",
        "rollout",
        "history",
        f"{str(args.get('kind', '') or '').strip()}/{str(args.get('name', '') or '').strip()}",
    ]
    _append_arg(parts, "-n", _default_namespace(str(args.get("namespace", "") or "")))
    return " ".join(parts)


def _render_kubectl_yaml(args: dict) -> str:
    parts = [
        "kubectl",
        "get",
        str(args.get("kind", "") or "").strip(),
        _quote(str(args.get("name", "") or "").strip()),
        "-o",
        "yaml",
    ]
    _append_arg(parts, "-n", _default_namespace(str(args.get("namespace", "") or "")))
    return " ".join(parts)


def _render_kubectl_run(args: dict) -> str:
    command = str(args.get("command", "") or "").strip()
    if not command:
        return "kubectl"
    return command if command.startswith("kubectl ") else f"kubectl {command}"


def _render_k8s_scale(args: dict) -> str:
    parts = [
        "kubectl",
        "scale",
        f"{str(args.get('kind', '') or '').strip()}/{str(args.get('name', '') or '').strip()}",
        f"--replicas={args.get('replicas', 0)}",
    ]
    _append_arg(parts, "-n", _default_namespace(str(args.get("namespace", "") or "")))
    return " ".join(parts)


def _render_k8s_rollout(args: dict, operation: str) -> str:
    parts = [
        "kubectl",
        "rollout",
        operation,
        f"{str(args.get('kind', '') or '').strip()}/{str(args.get('name', '') or '').strip()}",
    ]
    _append_arg(parts, "-n", _default_namespace(str(args.get("namespace", "") or "")))
    return " ".join(parts)


def _render_k8s_delete(args: dict) -> str:
    parts = [
        "kubectl",
        "delete",
        str(args.get("kind", "") or "").strip(),
        _quote(str(args.get("name", "") or "").strip()),
    ]
    _append_arg(parts, "-n", _default_namespace(str(args.get("namespace", "") or "")))
    return " ".join(parts)


def _render_k8s_cordon(args: dict) -> str:
    operation = "uncordon" if bool(args.get("uncordon")) else "cordon"
    return f"kubectl {operation} {_quote(str(args.get('node', '') or '').strip())}"


def _render_k8s_drain(args: dict) -> str:
    grace_period = int(args.get("grace_period", 300) or 300)
    return (
        f"kubectl drain {_quote(str(args.get('node', '') or '').strip())} "
        f"--ignore-daemonsets --delete-emptydir-data --grace-period={grace_period} --force"
    )


def _render_k8s_exec(args: dict) -> str:
    parts = ["kubectl", "exec", _quote(str(args.get("pod", "") or "").strip())]
    _append_arg(parts, "-n", _default_namespace(str(args.get("namespace", "") or "")))
    _append_arg(parts, "-c", str(args.get("container", "") or "").strip())
    parts.append("--")
    command = str(args.get("command", "") or "").strip()
    if command:
        parts.append(command)
    return " ".join(parts)


def _fallback_preview(tool_name: str, args: dict) -> str:
    try:
        rendered_args = json.dumps(args or {}, sort_keys=True, default=str)
    except TypeError:
        rendered_args = str(args)
    return f"{tool_name} {rendered_args}".strip()


def _preview_for_tool(tool_name: str, args: dict) -> str:
    mapping = {
        "aws_describe_instances": lambda: _render_aws_describe_instances(args),
        "aws_describe_service": lambda: _render_aws_describe_service(args),
        "aws_inspect_lambda_schedules": lambda: _render_aws_inspect_lambda_schedules(args),
        "aws_audit_cloudtrail": lambda: _render_aws_audit_cloudtrail(args),
        "aws_get_cost": lambda: _render_aws_get_cost(args),
        "aws_get_cloudwatch_metrics": lambda: _render_aws_get_cloudwatch_metrics(args),
        "aws_get_alarms": lambda: _render_aws_get_alarms(args),
        "aws_describe_security_groups": lambda: _render_aws_describe_security_groups(args),
        "aws_get_iam_summary": lambda: "aws iam get-account-summary",
        "aws_list_resources": lambda: _render_aws_list_resources(args),
        "aws_get_caller_identity": lambda: "aws sts get-caller-identity",
        "aws_run_api_command": lambda: _render_aws_cli(
            str(args.get("service", "") or ""),
            str(args.get("operation", "") or ""),
            _parse_json_arg(args.get("params_json")),
            str(args.get("region", "") or ""),
        ),
        "aws_update_auto_scaling": lambda: _render_aws_update_auto_scaling(args),
        "k8s_get_resources": lambda: _render_kubectl_get_resources(args),
        "k8s_describe_resource": lambda: _render_kubectl_describe(args),
        "k8s_get_pod_logs": lambda: _render_kubectl_logs(args),
        "k8s_get_events": lambda: _render_kubectl_events(args),
        "k8s_get_resource_usage": lambda: _render_kubectl_top(args),
        "k8s_get_rollout_history": lambda: _render_kubectl_rollout_history(args),
        "k8s_get_contexts": lambda: "kubectl config get-contexts",
        "k8s_get_namespaces": lambda: "kubectl get namespaces",
        "k8s_get_resource_yaml": lambda: _render_kubectl_yaml(args),
        "k8s_run_kubectl": lambda: _render_kubectl_run(args),
        "k8s_scale": lambda: _render_k8s_scale(args),
        "k8s_rollout_restart": lambda: _render_k8s_rollout(args, "restart"),
        "k8s_rollout_undo": lambda: _render_k8s_rollout(args, "undo"),
        "k8s_delete_resource": lambda: _render_k8s_delete(args),
        "k8s_apply_manifest": lambda: "kubectl apply -f -",
        "k8s_cordon_node": lambda: _render_k8s_cordon(args),
        "k8s_drain_node": lambda: _render_k8s_drain(args),
        "k8s_exec_in_pod": lambda: _render_k8s_exec(args),
    }
    renderer = mapping.get(str(tool_name or "").strip())
    if renderer is None:
        return _fallback_preview(tool_name, args)
    return renderer()


def render_tool_call_preview(
    tool_name: str,
    tool_args: dict | None = None,
    *,
    display: str = "",
) -> tuple[str, str, str]:
    """Return a human label, preview command/text, and syntax hint for a tool invocation."""
    args = tool_args if isinstance(tool_args, dict) else {}
    preview = _preview_for_tool(str(tool_name or "").strip(), args)
    label = str(display or "").strip()
    if label == preview:
        label = ""
    language = "bash" if preview.startswith(("aws ", "kubectl ")) else "text"
    return label, preview, language


def render_action_step_preview(step: dict) -> tuple[str, str, str]:
    """Return a human label, code/text preview, and syntax hint for one action step."""
    display = str(step.get("display", "") or "").strip()
    command = str(step.get("command", "") or "").strip()
    if command:
        label = display if display and display != command else ""
        return label, command, "bash"

    tool_name = str(step.get("tool", "") or "").strip()
    if tool_name:
        return render_tool_call_preview(tool_name, step.get("args", {}), display=display)

    preview = display or str(step)
    language = "bash" if preview.startswith(("aws ", "kubectl ")) else "text"
    return "", preview, language
