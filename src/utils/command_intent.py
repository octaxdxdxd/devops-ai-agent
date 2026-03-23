"""Command-family routing and safety classification helpers."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Literal

from ..config import Config

CommandFamily = Literal["kubectl", "aws", "helm", "unknown"]
CommandCapability = Literal["safe_read", "sensitive_read", "write", "blocked"]


@dataclass(frozen=True)
class CommandIntent:
    family: CommandFamily
    verb: str
    is_mutating: bool
    normalized_command: str
    capability: CommandCapability
    reason: str = ""

    @property
    def requires_approval(self) -> bool:
        return self.capability == "write"

    @property
    def is_blocked(self) -> bool:
        return self.capability == "blocked"

    @property
    def is_sensitive_read(self) -> bool:
        return self.capability == "sensitive_read"


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
_KUBECTL_SENSITIVE_READ_VERBS = {
    "attach",
    "cp",
    "debug",
    "exec",
    "port-forward",
}
_KUBECTL_READONLY_VERBS = {
    "api-resources",
    "api-versions",
    "auth",
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
_AWS_SENSITIVE_READ_OPERATIONS = {
    "get-login-password",
    "get-secret-value",
    "batch-get-secret-value",
    "decrypt",
}
_AWS_MUTATING_OPERATION_PREFIXES = (
    "add",
    "associate",
    "attach",
    "cancel",
    "complete",
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
def _blocked_intent(
    *,
    family: CommandFamily,
    verb: str,
    normalized_command: str,
    reason: str,
) -> CommandIntent:
    return CommandIntent(
        family=family,
        verb=verb,
        is_mutating=False,
        normalized_command=normalized_command,
        capability="blocked",
        reason=reason,
    )


def _intent(
    *,
    family: CommandFamily,
    verb: str,
    normalized_command: str,
    capability: CommandCapability,
    reason: str = "",
) -> CommandIntent:
    return CommandIntent(
        family=family,
        verb=verb,
        is_mutating=(capability == "write"),
        normalized_command=normalized_command,
        capability=capability,
        reason=reason,
    )


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


def _powerful_posture() -> bool:
    return str(getattr(Config, "COMMAND_SAFETY_POSTURE", "powerful")).strip().lower() != "strict"


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


def _positional_tokens(tokens: list[str], *, options_with_value: set[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(tokens):
        token = str(tokens[i]).strip()
        if not token:
            i += 1
            continue
        if token == "--":
            out.extend(str(item).strip() for item in tokens[i + 1 :])
            break
        if token.startswith("-"):
            if token in options_with_value:
                i += 2
                continue
            i += 1
            continue
        out.append(token)
        i += 1
    return out


def _kubectl_targets_secret(tokens: list[str]) -> bool:
    positionals = _positional_tokens(tokens, options_with_value=_KUBECTL_OPTIONS_WITH_VALUE)
    if len(positionals) < 2:
        return False

    target = positionals[1].strip().lower()
    if target.startswith("secret/") or target.startswith("secrets/"):
        return True
    return target in {"secret", "secrets"}


def _kubectl_output_flag_value(tokens: list[str]) -> str:
    for i, raw in enumerate(tokens):
        token = str(raw).strip()
        if token in {"-o", "--output"} and i + 1 < len(tokens):
            return str(tokens[i + 1]).strip().lower()
        if token.startswith("-o="):
            return token.split("=", 1)[1].strip().lower()
        if token.startswith("--output="):
            return token.split("=", 1)[1].strip().lower()
    return ""


def _kubectl_secret_reads_sensitive(tokens: list[str], normalized_command: str) -> bool:
    if not _kubectl_targets_secret(tokens):
        return False

    output_value = _kubectl_output_flag_value(tokens)
    if output_value in {"json", "yaml"}:
        return True
    if output_value.startswith("jsonpath") or output_value.startswith("go-template"):
        return True

    lowered = normalized_command.lower()
    return ".data" in lowered or "base64" in lowered or "--template" in lowered


def _aws_sensitive_read(service: str, operation: str, tokens: list[str]) -> bool:
    if operation in _AWS_SENSITIVE_READ_OPERATIONS:
        return True
    if service == "ssm" and operation in {"get-parameter", "get-parameters"}:
        lowered = [str(token).strip().lower() for token in tokens]
        return "--with-decryption" in lowered
    return False


def _classify_kubectl(tokens: list[str]) -> CommandIntent:
    verb = _extract_first_positional(tokens, options_with_value=_KUBECTL_OPTIONS_WITH_VALUE)
    normalized = shlex.join(tokens)
    if not verb:
        return _blocked_intent(
            family="kubectl",
            verb=verb,
            normalized_command=normalized,
            reason="could not determine kubectl verb.",
        )
    if verb in _KUBECTL_SENSITIVE_READ_VERBS:
        if _powerful_posture():
            return _intent(
                family="kubectl",
                verb=verb,
                normalized_command=normalized,
                capability="write",
                reason=f"kubectl verb `{verb}` is treated as execute-capable in powerful posture.",
            )
        return _intent(
            family="kubectl",
            verb=verb,
            normalized_command=normalized,
            capability="sensitive_read",
            reason=(
                f"kubectl verb `{verb}` is a privileged/sensitive read operation and is blocked in generic command tools."
            ),
        )
    if verb in _KUBECTL_MUTATING_VERBS:
        return _intent(
            family="kubectl",
            verb=verb,
            normalized_command=normalized,
            capability="write",
        )
    if verb in _KUBECTL_READONLY_VERBS:
        if _kubectl_secret_reads_sensitive(tokens, normalized):
            if _powerful_posture():
                return _intent(
                    family="kubectl",
                    verb=verb,
                    normalized_command=normalized,
                    capability="safe_read",
                    reason="secret extraction is allowed in powerful posture.",
                )
            return _intent(
                family="kubectl",
                verb=verb,
                normalized_command=normalized,
                capability="sensitive_read",
                reason="secret value extraction is blocked in generic kubectl command tools.",
            )
        return _intent(
            family="kubectl",
            verb=verb,
            normalized_command=normalized,
            capability="safe_read",
        )
    if _powerful_posture():
        return _intent(
            family="kubectl",
            verb=verb,
            normalized_command=normalized,
            capability="write",
            reason=f"kubectl verb `{verb}` is treated as execute-capable in powerful posture.",
        )
    return _blocked_intent(
        family="kubectl",
        verb=verb,
        normalized_command=normalized,
        reason=f"kubectl verb `{verb}` is not supported in generic command tools.",
    )


def _classify_aws(tokens: list[str]) -> CommandIntent:
    service = str(tokens[0]).strip().lower() if len(tokens) >= 1 else ""
    operation = str(tokens[1]).strip().lower() if len(tokens) >= 2 else ""
    normalized = shlex.join(tokens)
    verb = f"{service}:{operation}".strip(":")
    if not service or not operation:
        return _blocked_intent(
            family="aws",
            verb=verb,
            normalized_command=normalized,
            reason="AWS command must include both service and operation.",
        )
    if _aws_sensitive_read(service, operation, tokens):
        if _powerful_posture():
            return _intent(
                family="aws",
                verb=verb,
                normalized_command=normalized,
                capability="safe_read",
                reason="sensitive AWS reads are allowed in powerful posture.",
            )
        return _intent(
            family="aws",
            verb=verb,
            normalized_command=normalized,
            capability="sensitive_read",
            reason="sensitive AWS secret/decryption reads are blocked in generic command tools.",
        )
    if operation in _AWS_READONLY_OPERATION_EXCEPTIONS:
        return _intent(
            family="aws",
            verb=verb,
            normalized_command=normalized,
            capability="safe_read",
        )
    if operation.startswith(_AWS_MUTATING_OPERATION_PREFIXES):
        return _intent(
            family="aws",
            verb=verb,
            normalized_command=normalized,
            capability="write",
        )
    return _intent(
        family="aws",
        verb=verb,
        normalized_command=normalized,
        capability="safe_read",
    )


def _classify_helm(tokens: list[str]) -> CommandIntent:
    verb = _extract_first_positional(tokens, options_with_value=_HELM_OPTIONS_WITH_VALUE)
    normalized = shlex.join(tokens)
    if _contains_shell_syntax(tokens, normalized):
        return _blocked_intent(
            family="helm",
            verb=verb,
            normalized_command=normalized,
            reason="shell-style operators are blocked in Helm command tools.",
        )
    if not verb:
        return _blocked_intent(
            family="helm",
            verb=verb,
            normalized_command=normalized,
            reason="could not determine Helm verb.",
        )
    if verb in _HELM_READONLY_VERBS:
        return _intent(
            family="helm",
            verb=verb,
            normalized_command=normalized,
            capability="safe_read",
        )
    if verb in _HELM_MUTATING_VERBS:
        subverb = _extract_first_positional(tokens[1:], options_with_value=_HELM_OPTIONS_WITH_VALUE)
        readonly_subverbs = _HELM_READONLY_SUBVERBS.get(verb, set())
        mutating_subverbs = _HELM_MUTATING_SUBVERBS.get(verb, set())
        if subverb and subverb in readonly_subverbs:
            return _intent(
                family="helm",
                verb=verb,
                normalized_command=normalized,
                capability="safe_read",
            )
        if subverb and subverb in mutating_subverbs:
            return _intent(
                family="helm",
                verb=verb,
                normalized_command=normalized,
                capability="write",
            )
        return _intent(
            family="helm",
            verb=verb,
            normalized_command=normalized,
            capability="write",
        )
    if _powerful_posture():
        return _intent(
            family="helm",
            verb=verb,
            normalized_command=normalized,
            capability="write",
            reason=f"Helm verb `{verb}` is treated as execute-capable in powerful posture.",
        )
    return _blocked_intent(
        family="helm",
        verb=verb,
        normalized_command=normalized,
        reason=f"Helm verb `{verb}` is not supported in generic command tools.",
    )


def classify_command_intent(tool_name: str, command: str) -> CommandIntent:
    """Classify command family + mutability for routing and approval decisions."""
    default_family = _normalize_family_from_tool(tool_name)
    tokens, _err = _split_tokens(command)
    if not tokens:
        return CommandIntent(
            family=default_family,
            verb="",
            is_mutating=False,
            normalized_command=(command or "").strip(),
            capability="blocked",
            reason="command is empty or invalid.",
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
        is_mutating=False,
        normalized_command=(command or "").strip(),
        capability="blocked",
        reason="could not determine command family.",
    )


def target_tool_for_intent(intent: CommandIntent) -> str | None:
    if intent.family == "kubectl":
        if intent.capability == "safe_read":
            return "kubectl_readonly"
        if intent.capability == "write":
            return "kubectl_execute"
        return None
    if intent.family == "aws":
        if intent.capability == "safe_read":
            return "aws_cli_readonly"
        if intent.capability == "write":
            return "aws_cli_execute"
        return None
    if intent.family == "helm":
        if intent.capability == "safe_read":
            return "helm_readonly"
        if intent.capability == "write":
            return "helm_execute"
        return None
    return None
