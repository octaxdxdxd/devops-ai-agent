"""Generic Helm command tools."""

from __future__ import annotations

import shlex
import shutil
import subprocess

from langchain_core.tools import tool

from ..config import Config
from ..utils.command_intent import classify_command_intent
from .k8s_common import truncate_text


def _helm_not_found_msg() -> str:
    return "❌ `helm` not found in PATH in this runtime. I cannot execute Helm commands here until it is available."


def _ensure_helm_installed() -> bool:
    return bool(shutil.which("helm"))


def _normalize_helm_command(command: str) -> tuple[list[str] | None, str | None]:
    raw = (command or "").strip()
    if not raw:
        return None, "❌ command is required. Example: `list -A`."
    try:
        tokens = shlex.split(raw)
    except ValueError as exc:
        return None, f"❌ Invalid helm command syntax: {exc}"

    if not tokens:
        return None, "❌ command is required."
    if tokens[0].lower() == "helm":
        tokens = tokens[1:]
    if not tokens:
        return None, "❌ command is required after `helm`."

    first = tokens[0].lower()
    if first == "kubectl":
        return None, "❌ Invalid helm command: this looks like a kubectl command. Use `kubectl_readonly` or `kubectl_execute`."
    if first == "aws":
        return None, "❌ Invalid helm command: this looks like an AWS CLI command. Use `aws_cli_readonly` or `aws_cli_execute`."

    return tokens, None


def _run_helm(args: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=max(10, Config.K8S_REQUEST_TIMEOUT_SEC + 20),
    )
    return proc.returncode, (proc.stdout or "").strip(), (proc.stderr or "").strip()


def _command_result(command_args: list[str], *, reason: str = "") -> str:
    command_text = shlex.join(command_args)
    code, out, err = _run_helm(command_args)
    if code != 0:
        return (
            f"Planned command:\n```bash\n{command_text}\n```\n"
            f"❌ helm command failed (exit={code}).\n"
            f"Details: {truncate_text(err or out or 'unknown error')}"
        )

    reason_line = f"\nReason: {reason}" if reason else ""
    return (
        f"Planned command:\n```bash\n{command_text}\n```\n"
        f"✅ helm command executed successfully.{reason_line}\n"
        f"Output:\n{truncate_text(out or '(no output)')}"
    )


def _intent_error(intent, *, readonly: bool) -> str | None:
    if intent.is_blocked:
        return f"❌ {intent.reason}"
    if intent.is_sensitive_read:
        return f"❌ {intent.reason}"
    if readonly and intent.capability == "write":
        return (
            f"❌ helm verb `{intent.verb or 'unknown'}` appears mutating and is blocked in helm_readonly. "
            "Use `helm_execute` (approval required)."
        )
    if not readonly and intent.capability == "safe_read":
        return (
            f"❌ helm verb `{intent.verb or 'unknown'}` is read-only. "
            "Use `helm_readonly` instead."
        )
    return None


@tool
def helm_readonly(command: str) -> str:
    """Run a Helm read command.

    Examples:
    - list -A
    - status gitlab -n gitlab
    - get values gitlab -n gitlab
    """
    if not _ensure_helm_installed():
        return _helm_not_found_msg()

    tokens, err = _normalize_helm_command(command)
    if err:
        return err
    assert tokens is not None

    intent = classify_command_intent("helm_readonly", shlex.join(tokens))
    intent_err = _intent_error(intent, readonly=True)
    if intent_err:
        return intent_err

    return _command_result(["helm", *tokens])


@tool
def helm_execute(command: str, reason: str = "") -> str:
    """Run mutating Helm commands only."""
    if not _ensure_helm_installed():
        return _helm_not_found_msg()

    tokens, err = _normalize_helm_command(command)
    if err:
        return err
    assert tokens is not None

    command_args = ["helm", *tokens]
    command_text = shlex.join(command_args)
    intent = classify_command_intent("helm_execute", shlex.join(tokens))
    intent_err = _intent_error(intent, readonly=False)
    if intent_err:
        return intent_err

    if Config.K8S_DRY_RUN and intent.is_mutating:
        return (
            "🧪 DRY RUN enabled; no mutation executed.\n"
            f"Planned command:\n```bash\n{command_text}\n```\n"
            f"Reason: {reason or 'n/a'}"
        )

    return _command_result(command_args, reason=reason)


def get_helm_tools() -> list:
    return [helm_readonly, helm_execute]
