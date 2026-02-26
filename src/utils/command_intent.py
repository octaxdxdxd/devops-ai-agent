"""Command-family routing and mutability classification helpers."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Literal


CommandFamily = Literal["kubectl", "aws", "helm", "unknown"]


@dataclass(frozen=True)
class CommandIntent:
    family: CommandFamily
    verb: str
    is_mutating: bool
    normalized_command: str


_KUBECTL_OPTIONS_WITH_VALUE = {
    "-n",
    "--namespace",
    "-f",
    "--filename",
    "-l",
    "--selector",
    "--field-selector",
    "--context",
    "--kubeconfig",
    "-o",
    "--output",
    "--as",
    "--as-group",
    "--server",
    "--token",
    "--cluster",
    "--user",
    "--request-timeout",
}

_KUBECTL_MUTATING_VERBS = {
    "apply",
    "create",
    "delete",
    "edit",
    "patch",
    "replace",
    "scale",
    "rollout",
    "set",
    "cordon",
    "uncordon",
    "drain",
    "taint",
    "label",
    "annotate",
    "autoscale",
    "expose",
}
_KUBECTL_READONLY_VERBS = {
    "api-resources",
    "api-versions",
    "cluster-info",
    "describe",
    "diff",
    "events",
    "explain",
    "get",
    "logs",
    "top",
    "version",
    "wait",
}

_AWS_READONLY_OPERATION_EXCEPTIONS = {
    "start-query",
    "get-query-results",
    "filter-log-events",
    "generate-query",
}
_AWS_MUTATING_OPERATION_PREFIXES = (
    "add",
    "associate",
    "attach",
    "cancel",
    "copy",
    "create",
    "delete",
    "deregister",
    "detach",
    "disable",
    "disassociate",
    "enable",
    "execute",
    "export",
    "import",
    "modify",
    "patch",
    "promote",
    "publish",
    "put",
    "reboot",
    "register",
    "remove",
    "replace",
    "reset",
    "restore",
    "resume",
    "revoke",
    "rotate",
    "run",
    "send",
    "set",
    "start",
    "stop",
    "suspend",
    "tag",
    "terminate",
    "untag",
    "update",
    "upgrade",
)

_HELM_OPTIONS_WITH_VALUE = {
    "-n",
    "--namespace",
    "--kube-context",
    "--kubeconfig",
    "--repo",
    "--version",
    "--values",
    "-f",
    "--set",
    "--set-string",
    "--set-file",
    "--timeout",
    "--output",
}
_HELM_MUTATING_VERBS = {
    "install",
    "upgrade",
    "rollback",
    "uninstall",
    "delete",
    "repo",
    "plugin",
    "dependency",
    "package",
    "push",
    "registry",
}
_HELM_READONLY_VERBS = {
    "list",
    "get",
    "status",
    "history",
    "show",
    "search",
    "template",
    "lint",
    "version",
    "env",
    "completion",
}
_HELM_READONLY_SUBVERBS = {
    "repo": {"list"},
    "plugin": {"list"},
    "dependency": {"list"},
}
_HELM_MUTATING_SUBVERBS = {
    "repo": {"add", "remove", "rm", "update", "index"},
    "plugin": {"install", "uninstall", "update"},
    "registry": {"login", "logout"},
    "dependency": {"build", "update"},
}


def _normalize_family_from_tool(tool_name: str) -> CommandFamily:
    name = (tool_name or "").strip().lower()
    if name.startswith("kubectl_"):
        return "kubectl"
    if name.startswith("aws_cli_"):
        return "aws"
    if name.startswith("helm_"):
        return "helm"
    return "unknown"


def _split_tokens(command: str) -> tuple[list[str], str | None]:
    raw = (command or "").strip()
    if not raw:
        return [], None
    try:
        return shlex.split(raw), None
    except ValueError as exc:
        return [], f"invalid syntax: {exc}"


def _extract_first_positional(tokens: list[str], *, options_with_value: set[str]) -> str:
    i = 0
    while i < len(tokens):
        token = str(tokens[i]).strip()
        if not token:
            i += 1
            continue
        if token == "--":
            return str(tokens[i + 1]).strip().lower() if i + 1 < len(tokens) else ""
        if token.startswith("-"):
            if token in options_with_value:
                i += 2
                continue
            if token.startswith("--") and "=" in token:
                i += 1
                continue
            i += 1
            continue
        return token.lower()
    return ""


def _classify_kubectl(tokens: list[str]) -> CommandIntent:
    verb = _extract_first_positional(tokens, options_with_value=_KUBECTL_OPTIONS_WITH_VALUE)
    if not verb:
        is_mutating = True
    elif verb in _KUBECTL_MUTATING_VERBS:
        is_mutating = True
    elif verb in _KUBECTL_READONLY_VERBS:
        is_mutating = False
    else:
        # Conservative default: unknown kubectl verbs are treated as mutating.
        is_mutating = True
    return CommandIntent(
        family="kubectl",
        verb=verb,
        is_mutating=is_mutating,
        normalized_command=shlex.join(tokens),
    )


def _classify_aws(tokens: list[str]) -> CommandIntent:
    service = str(tokens[0]).strip().lower() if len(tokens) >= 1 else ""
    operation = str(tokens[1]).strip().lower() if len(tokens) >= 2 else ""
    if not operation:
        is_mutating = True
    elif operation in _AWS_READONLY_OPERATION_EXCEPTIONS:
        is_mutating = False
    else:
        is_mutating = operation.startswith(_AWS_MUTATING_OPERATION_PREFIXES)
    return CommandIntent(
        family="aws",
        verb=f"{service}:{operation}".strip(":"),
        is_mutating=is_mutating,
        normalized_command=shlex.join(tokens),
    )


def _classify_helm(tokens: list[str]) -> CommandIntent:
    verb = _extract_first_positional(tokens, options_with_value=_HELM_OPTIONS_WITH_VALUE)
    if not verb:
        is_mutating = True
    elif verb in _HELM_READONLY_VERBS:
        is_mutating = False
    elif verb in _HELM_MUTATING_VERBS:
        subverb = _extract_first_positional(tokens[1:], options_with_value=_HELM_OPTIONS_WITH_VALUE)
        readonly_subverbs = _HELM_READONLY_SUBVERBS.get(verb, set())
        mutating_subverbs = _HELM_MUTATING_SUBVERBS.get(verb, set())
        if subverb and subverb in readonly_subverbs:
            is_mutating = False
        elif subverb and subverb in mutating_subverbs:
            is_mutating = True
        elif verb in {"repo", "plugin", "dependency", "registry"}:
            # Default these grouped verbs to mutating unless we can prove read-only.
            is_mutating = True
        else:
            is_mutating = True
    else:
        # Conservative: unknown verbs are treated as mutating.
        is_mutating = True
    return CommandIntent(
        family="helm",
        verb=verb,
        is_mutating=is_mutating,
        normalized_command=shlex.join(tokens),
    )


def classify_command_intent(tool_name: str, command: str) -> CommandIntent:
    """Classify command family + mutability for routing and approval decisions."""
    default_family = _normalize_family_from_tool(tool_name)
    tokens, _err = _split_tokens(command)
    if not tokens:
        return CommandIntent(
            family=default_family,
            verb="",
            is_mutating=(default_family != "unknown"),
            normalized_command=(command or "").strip(),
        )

    first = str(tokens[0]).strip().lower()
    explicit_family: CommandFamily = default_family
    if first in {"kubectl", "aws", "helm"}:
        explicit_family = first  # type: ignore[assignment]
        tokens = tokens[1:]

    if explicit_family == "kubectl":
        return _classify_kubectl(tokens)
    if explicit_family == "aws":
        return _classify_aws(tokens)
    if explicit_family == "helm":
        return _classify_helm(tokens)

    return CommandIntent(
        family="unknown",
        verb="",
        is_mutating=True,
        normalized_command=(command or "").strip(),
    )


def target_tool_for_intent(intent: CommandIntent) -> str | None:
    if intent.family == "kubectl":
        return "kubectl_execute" if intent.is_mutating else "kubectl_readonly"
    if intent.family == "aws":
        return "aws_cli_execute" if intent.is_mutating else "aws_cli_readonly"
    if intent.family == "helm":
        return "helm_execute" if intent.is_mutating else "helm_readonly"
    return None
