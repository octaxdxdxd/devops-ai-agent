"""Generic kubectl command tools.

These tools let the agent execute arbitrary kubectl commands directly.
Write commands are approval-gated by the agent policy layer.
"""

from __future__ import annotations

import shlex

from langchain_core.tools import tool

from ..config import Config
from .k8s_common import (
    ensure_kubectl_installed,
    kubectl_base_args,
    kubectl_not_found_msg,
    run_kubectl,
    truncate_text,
)


_OPTIONS_WITH_VALUE = {
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

_MUTATING_KUBECTL_VERBS = {
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


def _csv_to_set(value: str) -> set[str]:
    return {item.strip().lower() for item in (value or "").split(",") if item.strip()}


def _normalize_kubectl_command(command: str) -> tuple[list[str] | None, str | None]:
    raw = (command or "").strip()
    if not raw:
        return None, "❌ command is required. Example: `get pods -A`."

    try:
        tokens = shlex.split(raw)
    except ValueError as exc:
        return None, f"❌ Invalid kubectl command syntax: {exc}"

    if not tokens:
        return None, "❌ command is required."

    if tokens[0].lower() == "kubectl":
        tokens = tokens[1:]

    if not tokens:
        return None, "❌ command is required after `kubectl`."

    return tokens, None


def _extract_verb(tokens: list[str]) -> str:
    i = 0
    while i < len(tokens):
        token = str(tokens[i]).strip()
        if not token:
            i += 1
            continue

        if token == "--":
            if i + 1 < len(tokens):
                return str(tokens[i + 1]).strip().lower()
            return ""

        if token.startswith("-"):
            if token in _OPTIONS_WITH_VALUE:
                i += 2
                continue
            if "=" in token:
                i += 1
                continue
            i += 1
            continue

        return token.lower()

    return ""


def _build_kubectl_args(tokens: list[str]) -> list[str]:
    return [*kubectl_base_args(), *tokens]


def _command_result(command_args: list[str], *, reason: str = "") -> str:
    command_text = shlex.join(command_args)
    code, out, err = run_kubectl(command_args)
    if code != 0:
        return (
            f"Planned command:\n```bash\n{command_text}\n```\n"
            f"❌ kubectl command failed (exit={code}).\n"
            f"Details: {truncate_text(err or out or 'unknown error')}"
        )

    reason_line = f"\nReason: {reason}" if reason else ""
    return (
        f"Planned command:\n```bash\n{command_text}\n```\n"
        f"✅ kubectl command executed successfully.{reason_line}\n"
        f"Output:\n{truncate_text(out or '(no output)')}"
    )


@tool
def kubectl_readonly(command: str) -> str:
    """Run a kubectl read command.

    Examples:
    - get pods -A
    - describe pod my-pod -n prod
    - logs my-pod -n prod --tail 200

    When K8S_CLI_ALLOW_ALL_READ=1, non-mutating verb allowlist checks are bypassed.
    """
    if not ensure_kubectl_installed():
        return kubectl_not_found_msg()

    tokens, err = _normalize_kubectl_command(command)
    if err:
        return err

    assert tokens is not None
    verb = _extract_verb(tokens)
    if verb in _MUTATING_KUBECTL_VERBS:
        return (
            f"❌ kubectl verb `{verb}` is mutating and is blocked in kubectl_readonly. "
            "Use `kubectl_execute` (approval required)."
        )

    readonly_verbs = _csv_to_set(Config.K8S_CLI_READONLY_VERBS)
    if not Config.K8S_CLI_ALLOW_ALL_READ and verb not in readonly_verbs:
        return (
            f"❌ kubectl verb `{verb or 'unknown'}` is not allowed in kubectl_readonly. "
            "Use `kubectl_execute` for mutating commands (approval required), or enable K8S_CLI_ALLOW_ALL_READ=1."
        )

    args = _build_kubectl_args(tokens)
    return _command_result(args)


@tool
def kubectl_execute(command: str, reason: str = "") -> str:
    """Run any kubectl command (mutating-capable).

    IMPORTANT: This tool can change cluster state and should require explicit approval.

    Examples:
    - apply -f deployment.yaml
    - delete pod my-pod -n prod
    - patch deployment web -n prod --type merge -p '{"spec":{"replicas":3}}'

    When K8S_CLI_ALLOW_ALL_WRITE=1, verb allowlist checks are bypassed.
    """
    if not ensure_kubectl_installed():
        return kubectl_not_found_msg()

    tokens, err = _normalize_kubectl_command(command)
    if err:
        return err

    assert tokens is not None
    verb = _extract_verb(tokens)
    write_verbs = _csv_to_set(Config.K8S_CLI_WRITE_ALLOWLIST_VERBS)
    if not Config.K8S_CLI_ALLOW_ALL_WRITE and verb not in write_verbs:
        return (
            f"❌ kubectl verb `{verb or 'unknown'}` is not allowed in kubectl_execute. "
            "Enable K8S_CLI_ALLOW_ALL_WRITE=1 or add verb to K8S_CLI_WRITE_ALLOWLIST_VERBS."
        )

    args = _build_kubectl_args(tokens)
    command_text = shlex.join(args)

    if Config.K8S_DRY_RUN:
        return (
            "🧪 DRY RUN enabled; no mutation executed.\n"
            f"Planned command:\n```bash\n{command_text}\n```\n"
            f"Reason: {reason or 'n/a'}"
        )

    return _command_result(args, reason=reason)


def get_k8s_cli_tools() -> list:
    """Return generic kubectl tools."""
    return [kubectl_readonly, kubectl_execute]
