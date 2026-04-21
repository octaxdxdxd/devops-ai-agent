"""Central policy checks for infrastructure reads, writes, and approvals."""

from __future__ import annotations

import shlex

from .config import Config


def _csv_values(raw: str) -> set[str]:
    return {
        value.strip()
        for value in str(raw or "").split(",")
        if value.strip()
    }


def _effective_namespace(namespace: str = "") -> str:
    return str(namespace or Config.K8S_DEFAULT_NAMESPACE or "default").strip() or "default"


def _normalize_region(region: str = "") -> str:
    return str(region or Config.AWS_CLI_DEFAULT_REGION or "").strip()


def _blocked_by_scope(
    *,
    value: str,
    allowed_values: set[str],
    blocked_values: set[str],
    scope_name: str,
) -> str | None:
    if not value:
        return None
    if allowed_values and value not in allowed_values:
        return f"Policy blocked access to {scope_name} '{value}'. Allowed {scope_name}s: {', '.join(sorted(allowed_values))}."
    if blocked_values and value in blocked_values:
        return f"Policy blocked access to {scope_name} '{value}'."
    return None


def _k8s_namespace_policy(namespace: str = "", all_namespaces: bool = False) -> str | None:
    allowed = _csv_values(Config.K8S_ALLOWED_NAMESPACES)
    blocked = _csv_values(Config.K8S_BLOCKED_NAMESPACES)
    if all_namespaces and allowed:
        return (
            "Policy blocked cluster-wide namespace access because "
            f"K8S_ALLOWED_NAMESPACES is restricted to: {', '.join(sorted(allowed))}."
        )
    if all_namespaces and blocked:
        return (
            "Policy blocked cluster-wide namespace access because "
            f"K8S_BLOCKED_NAMESPACES contains protected namespaces: {', '.join(sorted(blocked))}."
        )
    return _blocked_by_scope(
        value=_effective_namespace(namespace),
        allowed_values=allowed,
        blocked_values=blocked,
        scope_name="namespace",
    )


def _aws_region_policy(region: str = "") -> str | None:
    return _blocked_by_scope(
        value=_normalize_region(region),
        allowed_values=_csv_values(Config.AWS_ALLOWED_REGIONS),
        blocked_values=_csv_values(Config.AWS_BLOCKED_REGIONS),
        scope_name="region",
    )


def write_actions_allowed() -> str | None:
    posture = str(Config.COMMAND_SAFETY_POSTURE or "approval_required").strip().lower()
    if posture == "read_only":
        return "Policy blocked write access because COMMAND_SAFETY_POSTURE is set to 'read_only'."
    return None


def parse_kubectl_scope(command: str) -> tuple[str, bool]:
    namespace = ""
    all_namespaces = False
    try:
        parts = shlex.split(str(command or "").strip())
    except ValueError:
        return namespace, all_namespaces

    for index, part in enumerate(parts):
        if part in {"-A", "--all-namespaces"}:
            all_namespaces = True
        elif part in {"-n", "--namespace"} and index + 1 < len(parts):
            namespace = parts[index + 1]
        elif part.startswith("--namespace="):
            namespace = part.split("=", 1)[1]

    return namespace, all_namespaces


def guard_k8s_read_tool(
    tool_name: str,
    *,
    namespace: str = "",
    all_namespaces: bool = False,
    command: str = "",
) -> str | None:
    if tool_name == "k8s_run_kubectl" and not Config.K8S_CLI_ALLOW_ALL_READ:
        return "Policy blocked generic kubectl read commands because K8S_CLI_ALLOW_ALL_READ is disabled."
    if command:
        namespace, all_namespaces = parse_kubectl_scope(command)
    return _k8s_namespace_policy(namespace=namespace, all_namespaces=all_namespaces)


def guard_k8s_write_tool(
    tool_name: str,
    *,
    namespace: str = "",
    all_namespaces: bool = False,
    command: str = "",
) -> str | None:
    write_block = write_actions_allowed()
    if write_block:
        return write_block
    if command and not Config.K8S_CLI_ALLOW_ALL_WRITE:
        return "Policy blocked raw kubectl write commands because K8S_CLI_ALLOW_ALL_WRITE is disabled."
    if command:
        namespace, all_namespaces = parse_kubectl_scope(command)
    return _k8s_namespace_policy(namespace=namespace, all_namespaces=all_namespaces)


def guard_aws_read_tool(
    tool_name: str,
    *,
    region: str = "",
    service: str = "",
    operation: str = "",
) -> str | None:
    if tool_name == "aws_describe_service" and not Config.AWS_CLI_ALLOW_ALL_READ:
        return "Policy blocked generic AWS read API calls because AWS_CLI_ALLOW_ALL_READ is disabled."
    return _aws_region_policy(region)


def guard_aws_write_tool(
    tool_name: str,
    *,
    region: str = "",
    service: str = "",
    operation: str = "",
) -> str | None:
    write_block = write_actions_allowed()
    if write_block:
        return write_block
    if tool_name == "aws_run_api_command" and not Config.AWS_CLI_ALLOW_ALL_WRITE:
        return "Policy blocked generic AWS write API calls because AWS_CLI_ALLOW_ALL_WRITE is disabled."
    return _aws_region_policy(region)


def guard_tool_invocation(tool_name: str, args: dict | None, *, write: bool) -> str | None:
    payload = args if isinstance(args, dict) else {}
    name = str(tool_name or "").strip()
    if name.startswith("k8s_"):
        checker = guard_k8s_write_tool if write else guard_k8s_read_tool
        return checker(
            name,
            namespace=str(payload.get("namespace", "") or ""),
            all_namespaces=bool(payload.get("all_namespaces")),
            command=str(payload.get("command", "") or ""),
        )
    if name.startswith("aws_"):
        checker = guard_aws_write_tool if write else guard_aws_read_tool
        return checker(
            name,
            region=str(payload.get("region", "") or ""),
            service=str(payload.get("service", "") or ""),
            operation=str(payload.get("operation", "") or ""),
        )
    return None
