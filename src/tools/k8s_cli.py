"""Generic kubectl command tools.

These tools let the agent execute arbitrary kubectl commands directly.
Write commands are approval-gated by the agent policy layer.
"""

from __future__ import annotations

import shlex

from langchain_core.tools import tool

from ..config import Config
from ..utils.command_intent import classify_command_intent
from .k8s_common import (
    ensure_kubectl_installed,
    is_cluster_scoped_kind,
    kubectl_base_args,
    kubectl_not_found_msg,
    resolve_namespace_for_resource,
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
_ALL_NAMESPACE_FLAGS = {"-A", "--all-namespaces"}

_TARGET_ONLY_VERBS = {"exec", "logs", "attach", "port-forward"}
_SKIP_AUTONAMESPACE_VERBS = {
    "cp",
    "debug",
    "auth",
    "api-resources",
    "api-versions",
    "cluster-info",
    "config",
    "version",
    "completion",
    "plugin",
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

    first = tokens[0].lower()
    if first == "helm":
        return None, "❌ Invalid kubectl command: this looks like a Helm command. Use `helm_readonly` or `helm_execute`."
    if first == "aws":
        return None, "❌ Invalid kubectl command: this looks like an AWS CLI command. Use `aws_cli_readonly` or `aws_cli_execute`."

    return tokens, None


def _next_positional_index(tokens: list[str], start: int = 0) -> int | None:
    i = start
    while i < len(tokens):
        token = str(tokens[i]).strip()
        if not token:
            i += 1
            continue

        if token == "--":
            if i + 1 < len(tokens):
                return i + 1
            return None

        if token.startswith("-"):
            if token in _OPTIONS_WITH_VALUE:
                i += 2
                continue
            i += 1
            continue

        return i
    return None


def _extract_namespace(tokens: list[str]) -> tuple[str | None, int | None, bool]:
    for i, raw in enumerate(tokens):
        token = str(raw).strip()
        if token in {"-n", "--namespace"}:
            if i + 1 < len(tokens):
                return str(tokens[i + 1]).strip(), i, False
            return "", i, False
        if token.startswith("--namespace="):
            return token.split("=", 1)[1].strip(), i, True
    return None, None, False


def _uses_all_namespaces(tokens: list[str]) -> bool:
    lowered = {str(t).strip().lower() for t in tokens}
    return bool(lowered & {x.lower() for x in _ALL_NAMESPACE_FLAGS})


def _parse_target_kind_name(tokens: list[str]) -> tuple[str | None, str | None]:
    verb_idx = _next_positional_index(tokens, 0)
    if verb_idx is None:
        return None, None

    verb = str(tokens[verb_idx]).strip().lower()

    if verb in {"apply", "create"}:
        return None, None
    if verb in _SKIP_AUTONAMESPACE_VERBS:
        return None, None

    if verb in {"set", "rollout"}:
        sub_idx = _next_positional_index(tokens, verb_idx + 1)
        if sub_idx is None:
            return None, None
        target_idx = _next_positional_index(tokens, sub_idx + 1)
        if target_idx is None:
            return None, None
        target = str(tokens[target_idx]).strip()
        if "/" in target:
            kind, name = target.split("/", 1)
            return kind.strip(), name.strip()
        name_idx = _next_positional_index(tokens, target_idx + 1)
        if name_idx is None:
            return None, None
        return target, str(tokens[name_idx]).strip()

    if verb in _TARGET_ONLY_VERBS:
        target_idx = _next_positional_index(tokens, verb_idx + 1)
        if target_idx is None:
            return None, None
        target = str(tokens[target_idx]).strip()
        if not target:
            return None, None
        if "/" in target:
            kind, name = target.split("/", 1)
            return kind.strip(), name.strip()
        # For exec/logs/attach, bare target is usually a pod name.
        return "pod", target

    kind_idx = _next_positional_index(tokens, verb_idx + 1)
    if kind_idx is None:
        return None, None

    kind = str(tokens[kind_idx]).strip()
    if "/" in kind:
        kind_part, name_part = kind.split("/", 1)
        if name_part:
            return kind_part.strip(), name_part.strip()

    name_idx = _next_positional_index(tokens, kind_idx + 1)
    if name_idx is None:
        return None, None
    return kind, str(tokens[name_idx]).strip()


def _autoresolve_namespace(tokens: list[str]) -> tuple[list[str] | None, str | None]:
    if _uses_all_namespaces(tokens):
        return list(tokens), None

    namespace_hint, ns_idx, ns_eq = _extract_namespace(tokens)
    explicit_namespace = (namespace_hint or "").strip()
    if explicit_namespace and explicit_namespace.lower() not in {"auto", "any", "all"}:
        # Respect explicit namespace; do not reinterpret command grammar.
        return list(tokens), None

    kind, name = _parse_target_kind_name(tokens)
    if not kind or not name:
        return list(tokens), None
    if is_cluster_scoped_kind(kind):
        return list(tokens), None

    resolved_ns, resolve_err = resolve_namespace_for_resource(kind, name, namespace_hint or "")
    if resolve_err:
        return None, resolve_err
    if not resolved_ns:
        return None, f"❌ Could not resolve namespace for {kind} '{name}'."

    rewritten = list(tokens)
    if ns_idx is None:
        rewritten.extend(["-n", resolved_ns])
    elif ns_eq:
        rewritten[ns_idx] = f"--namespace={resolved_ns}"
    elif ns_idx + 1 < len(rewritten):
        rewritten[ns_idx + 1] = resolved_ns

    return rewritten, None


def _build_kubectl_args(tokens: list[str]) -> list[str]:
    return [*kubectl_base_args(), *tokens]


def _intent_message(intent, *, readonly: bool) -> str | None:
    if intent.is_blocked:
        return f"❌ {intent.reason}"
    if intent.is_sensitive_read:
        return f"❌ {intent.reason}"
    if readonly and intent.capability == "write":
        return (
            f"❌ kubectl verb `{intent.verb or 'unknown'}` is mutating and is blocked in kubectl_readonly. "
            "Use `kubectl_execute` (approval required)."
        )
    if not readonly and intent.capability == "safe_read":
        return (
            f"❌ kubectl verb `{intent.verb or 'unknown'}` is read-only. "
            "Use `kubectl_readonly` instead."
        )
    return None


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

    In the default powerful posture, secret extraction reads are allowed here and
    exec-class operations are expected to route to `kubectl_execute`.
    """
    if not ensure_kubectl_installed():
        return kubectl_not_found_msg()

    tokens, err = _normalize_kubectl_command(command)
    if err:
        return err

    assert tokens is not None
    intent = classify_command_intent("kubectl_readonly", shlex.join(tokens))
    intent_err = _intent_message(intent, readonly=True)
    if intent_err:
        return intent_err

    verb = intent.verb
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
    Read-only and sensitive-read commands are rejected here to keep generic execute
    limited to known mutating operations.

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
    intent = classify_command_intent("kubectl_execute", shlex.join(tokens))
    intent_err = _intent_message(intent, readonly=False)
    if intent_err:
        return intent_err

    verb = intent.verb
    write_verbs = _csv_to_set(Config.K8S_CLI_WRITE_ALLOWLIST_VERBS)
    if not Config.K8S_CLI_ALLOW_ALL_WRITE and verb not in write_verbs:
        return (
            f"❌ kubectl verb `{verb or 'unknown'}` is not allowed in kubectl_execute. "
            "Enable K8S_CLI_ALLOW_ALL_WRITE=1 or add verb to K8S_CLI_WRITE_ALLOWLIST_VERBS."
        )

    resolved_tokens, resolve_err = _autoresolve_namespace(tokens)
    if resolve_err:
        return resolve_err
    assert resolved_tokens is not None

    args = _build_kubectl_args(resolved_tokens)
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
