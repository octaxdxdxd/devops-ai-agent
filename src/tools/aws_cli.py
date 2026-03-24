"""AWS CLI tools with safety guardrails.

Design:
- Read-only tool is default and constrained by allowlist.
- Write tool is separately exposed and intended to require explicit approval.
- All executions are audited to JSONL.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import shlex
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.tools import tool

from ..config import Config
from ..utils.command_intent import classify_command_intent
from .k8s_common import truncate_text


_SAFE_TOKEN = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_GLOBAL_SERVICES = {
    "route53",
    "cloudfront",
    "organizations",
    "iam",
    "account",
    "shield",
}
_META_KEYS = {"responsemetadata", "nexttoken", "marker", "istruncated", "continuationtoken"}
_READONLY_OPERATION_EXCEPTIONS = {
    "start-query",
    "get-query-results",
    "filter-log-events",
    "generate-query",
}
_MUTATING_OPERATION_PREFIXES = (
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


def _csv_to_list(value: str) -> list[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def _redact(text: str) -> str:
    out = text or ""
    out = re.sub(r"AKIA[0-9A-Z]{16}", "[REDACTED_AWS_ACCESS_KEY_ID]", out)
    out = re.sub(r"ASIA[0-9A-Z]{16}", "[REDACTED_AWS_ACCESS_KEY_ID]", out)
    out = re.sub(r"(?i)(aws_secret_access_key\s*[=:]\s*)([^\s]+)", r"\1[REDACTED]", out)
    out = re.sub(r"(?i)(aws_session_token\s*[=:]\s*)([^\s]+)", r"\1[REDACTED]", out)
    return out


def _ensure_aws_cli_installed() -> bool:
    return bool(shutil.which("aws"))


def _normalize_command(command: str) -> tuple[list[str] | None, str | None]:
    raw = (command or "").strip()
    if not raw:
        return None, "❌ command is required. Example: `sts get-caller-identity`."

    try:
        tokens = shlex.split(raw)
    except ValueError as exc:
        return None, f"❌ Invalid command syntax: {exc}"

    if not tokens:
        return None, "❌ command is required."

    if tokens[0].lower() == "aws":
        tokens = tokens[1:]

    if len(tokens) < 2:
        return None, "❌ Command must include service and operation (e.g., `ec2 describe-instances`)."

    service = (tokens[0] or "").strip().lower()
    operation = (tokens[1] or "").strip().lower()

    if service in {"kubectl", "k8s", "kubernetes"}:
        return (
            None,
            "❌ Invalid AWS CLI command: this looks like a Kubernetes command. "
            "Use Kubernetes tools (`kubectl_readonly` / `kubectl_execute`) instead.",
        )
    if service == "helm":
        return (
            None,
            "❌ Invalid AWS CLI command: this looks like a Helm command. "
            "Use Helm tools (`helm_readonly` / `helm_execute`) instead.",
        )

    if not _SAFE_TOKEN.fullmatch(service):
        return None, f"❌ Invalid AWS service token: {service!r}."
    if not _SAFE_TOKEN.fullmatch(operation):
        return None, f"❌ Invalid AWS operation token: {operation!r}."

    return tokens, None


def _is_action_allowed(service: str, operation: str, patterns: list[str]) -> bool:
    action = f"{service}:{operation}"
    action_lower = action.lower()
    for pattern in patterns:
        p = pattern.strip().lower()
        if not p:
            continue
        if ":" not in p:
            continue
        if fnmatch.fnmatch(action_lower, p):
            return True
    return False


def _looks_mutating_operation(operation: str) -> bool:
    op = (operation or "").strip().lower()
    if not op:
        return True
    if op in _READONLY_OPERATION_EXCEPTIONS:
        return False
    return op.startswith(_MUTATING_OPERATION_PREFIXES)


def _build_final_args(tokens: list[str]) -> list[str]:
    final = ["aws", "--no-cli-pager", *tokens]

    lowered = [str(item).lower() for item in tokens]
    if Config.AWS_CLI_PROFILE and "--profile" not in lowered:
        final.extend(["--profile", Config.AWS_CLI_PROFILE])
    if Config.AWS_CLI_DEFAULT_REGION and "--region" not in lowered:
        final.extend(["--region", Config.AWS_CLI_DEFAULT_REGION])

    return final


def _has_region_flag(tokens: list[str]) -> bool:
    lowered = [str(item).lower() for item in tokens]
    for i, item in enumerate(lowered):
        if item == "--region":
            return i + 1 < len(lowered)
        if item.startswith("--region="):
            return True
    return False


def _append_region(tokens: list[str], region: str) -> list[str]:
    if _has_region_flag(tokens):
        return list(tokens)
    return [*tokens, "--region", region]


def _resolve_preferred_region() -> str:
    for value in [
        Config.AWS_CLI_DEFAULT_REGION,
        os.getenv("AWS_REGION", "").strip(),
        os.getenv("AWS_DEFAULT_REGION", "").strip(),
    ]:
        if value:
            return value
    return ""


def _candidate_regions() -> list[str]:
    seen: set[str] = set()
    out: list[str] = []

    preferred = _resolve_preferred_region()
    if preferred:
        seen.add(preferred)
        out.append(preferred)

    for value in _csv_to_list(Config.AWS_CLI_FALLBACK_REGIONS):
        if value in seen:
            continue
        seen.add(value)
        out.append(value)

    limit = max(1, Config.AWS_CLI_AUTO_REGION_FANOUT_MAX)
    return out[:limit]


def _is_effectively_empty_json(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, list):
        if not value:
            return True
        return all(_is_effectively_empty_json(item) for item in value)
    if isinstance(value, dict):
        keys = [k for k in value.keys() if str(k).lower() not in _META_KEYS]
        if not keys:
            return True
        return all(_is_effectively_empty_json(value.get(k)) for k in keys)
    return False


def _is_effectively_empty_output(stdout: str) -> bool:
    text = (stdout or "").strip()
    if not text:
        return True
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return False
    return _is_effectively_empty_json(payload)


def _run_aws(final_args: list[str]) -> tuple[int, str, str, float]:
    t0 = time.perf_counter()
    proc = subprocess.run(
        final_args,
        capture_output=True,
        text=True,
        timeout=max(5, Config.AWS_CLI_TIMEOUT_SEC),
    )
    duration_ms = (time.perf_counter() - t0) * 1000.0
    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    return proc.returncode, stdout, stderr, duration_ms


def _bash_block(command_text: str) -> str:
    return f"```bash\n{command_text}\n```"


def _audit(
    *,
    mode: str,
    command: str,
    service: str,
    operation: str,
    allowed: bool,
    reason: str,
    exit_code: int | None,
    duration_ms: float | None,
    stdout_len: int,
    stderr_len: int,
) -> None:
    path = Path(Config.AWS_CLI_AUDIT_LOG)
    path.parent.mkdir(parents=True, exist_ok=True)

    event = {
        "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "schema": "aiops.aws_cli_audit.v1",
        "user": os.getenv("USER") or os.getenv("USERNAME") or "unknown",
        "mode": mode,
        "service": service,
        "operation": operation,
        "allowed": allowed,
        "reason": reason,
        "exit_code": exit_code,
        "duration_ms": duration_ms,
        "stdout_len": stdout_len,
        "stderr_len": stderr_len,
        "command": _redact(command),
    }

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def _intent_error(intent, *, readonly: bool) -> str | None:
    if intent.is_blocked:
        return f"❌ {intent.reason}"
    if intent.is_sensitive_read:
        return f"❌ {intent.reason}"
    if readonly and intent.capability == "write":
        return (
            f"❌ Action `{intent.verb or 'unknown'}` looks mutating and is blocked in `aws_cli_readonly`. "
            "Use `aws_cli_execute` (approval required)."
        )
    if not readonly and intent.capability == "safe_read":
        return (
            f"❌ Action `{intent.verb or 'unknown'}` is read-only. "
            "Use `aws_cli_readonly` instead."
        )
    return None


def _execute(tokens: list[str], *, mode: str, reason: str = "") -> str:
    service = tokens[0].lower()
    operation = tokens[1].lower()

    if Config.AWS_CLI_ENFORCE_BLOCKLIST:
        blocklist = _csv_to_list(Config.AWS_CLI_BLOCKLIST)
        if _is_action_allowed(service, operation, blocklist):
            _audit(
                mode=mode,
                command=" ".join(["aws", *tokens]),
                service=service,
                operation=operation,
                allowed=False,
                reason="blocked_by_blocklist",
                exit_code=None,
                duration_ms=None,
                stdout_len=0,
                stderr_len=0,
            )
            return f"❌ Action `{service}:{operation}` is blocked by policy."

    if mode == "readonly":
        if _looks_mutating_operation(operation):
            _audit(
                mode=mode,
                command=" ".join(["aws", *tokens]),
                service=service,
                operation=operation,
                allowed=False,
                reason="mutating_operation_in_readonly_mode",
                exit_code=None,
                duration_ms=None,
                stdout_len=0,
                stderr_len=0,
            )
            return (
                f"❌ Action `{service}:{operation}` looks mutating and is blocked in `aws_cli_readonly`. "
                "Use `aws_cli_execute` (approval required)."
            )

        if not Config.AWS_CLI_ALLOW_ALL_READ:
            readonly_allowlist = _csv_to_list(Config.AWS_CLI_READONLY_ALLOWLIST)
            if not _is_action_allowed(service, operation, readonly_allowlist):
                _audit(
                    mode=mode,
                    command=" ".join(["aws", *tokens]),
                    service=service,
                    operation=operation,
                    allowed=False,
                    reason="not_in_readonly_allowlist",
                    exit_code=None,
                    duration_ms=None,
                    stdout_len=0,
                    stderr_len=0,
                )
                return (
                    f"❌ Action `{service}:{operation}` is not in readonly allowlist. "
                    "Use `aws_cli_execute` (with approval) and allowlist configuration for mutating commands."
                )

    if mode == "write" and not Config.AWS_CLI_ALLOW_ALL_WRITE:
        write_allowlist = _csv_to_list(Config.AWS_CLI_WRITE_ALLOWLIST)
        if not _is_action_allowed(service, operation, write_allowlist):
            _audit(
                mode=mode,
                command=" ".join(["aws", *tokens]),
                service=service,
                operation=operation,
                allowed=False,
                reason="not_in_write_allowlist",
                exit_code=None,
                duration_ms=None,
                stdout_len=0,
                stderr_len=0,
            )
            return (
                f"❌ Action `{service}:{operation}` is not in write allowlist. "
                "Add an explicit allowlist pattern in AWS_CLI_WRITE_ALLOWLIST to permit it."
            )

    service_is_global = service in _GLOBAL_SERVICES
    region_provided = _has_region_flag(tokens)
    preferred_region = _resolve_preferred_region()

    if mode == "write" and Config.AWS_CLI_DRY_RUN:
        final_args = _build_final_args(tokens)
        command_text = " ".join(final_args)
        _audit(
            mode=mode,
            command=command_text,
            service=service,
            operation=operation,
            allowed=True,
            reason="dry_run",
            exit_code=0,
            duration_ms=0.0,
            stdout_len=0,
            stderr_len=0,
        )
        return (
            "🧪 AWS CLI dry-run mode enabled; no command executed.\n"
            f"Planned command:\n{_bash_block(command_text)}\n"
            f"Reason: {reason or 'n/a'}"
        )

    attempts: list[tuple[list[str], str]] = []
    if mode == "readonly" and not service_is_global and not region_provided:
        if preferred_region:
            attempts.append((_build_final_args(_append_region(tokens, preferred_region)), preferred_region))
        elif Config.AWS_CLI_REQUIRE_DEFAULT_REGION_FOR_REGIONAL:
            return (
                f"❌ `{service}:{operation}` is a regional AWS API and no region was provided.\n"
                "Set `AWS_CLI_DEFAULT_REGION` in .env or provide `--region <region>` in the command."
            )
        elif Config.AWS_CLI_AUTO_REGION_RETRY:
            candidate_regions = _candidate_regions()
            if candidate_regions:
                for r in candidate_regions:
                    attempts.append((_build_final_args(_append_region(tokens, r)), r))
            else:
                attempts.append((_build_final_args(tokens), "auto/default"))
        else:
            attempts.append((_build_final_args(tokens), "explicit/default"))
    else:
        attempts.append((_build_final_args(tokens), "explicit/default"))

    regions_checked: list[str] = []
    first_error: str | None = None
    saw_empty_success = False
    last_command_text = ""
    last_stdout = ""

    for final_args, attempt_region in attempts:
        command_text = " ".join(final_args)
        last_command_text = command_text
        regions_checked.append(attempt_region)

        try:
            code, stdout, stderr, duration_ms = _run_aws(final_args)
            last_stdout = stdout
            _audit(
                mode=mode,
                command=command_text,
                service=service,
                operation=operation,
                allowed=True,
                reason="executed",
                exit_code=code,
                duration_ms=duration_ms,
                stdout_len=len(stdout),
                stderr_len=len(stderr),
            )

            if code != 0:
                if not first_error:
                    first_error = (
                        f"Planned command:\n{_bash_block(command_text)}\n"
                        f"❌ AWS CLI command failed (exit={code}).\n"
                        f"Details: {truncate_text(_redact(stderr or stdout or 'unknown error'))}"
                    )
                continue

            if mode == "readonly" and len(attempts) > 1 and _is_effectively_empty_output(stdout):
                saw_empty_success = True
                continue

            output = stdout or "(no output)"
            region_suffix = ""
            if mode == "readonly" and len(attempts) > 1:
                region_suffix = f"\nRegions checked: {', '.join(regions_checked)}"
            return (
                f"Planned command:\n{_bash_block(command_text)}\n"
                f"✅ AWS CLI command executed successfully.{region_suffix}\n"
                f"Output:\n{truncate_text(_redact(output), max_chars=Config.K8S_OUTPUT_MAX_CHARS)}"
            )
        except subprocess.TimeoutExpired:
            _audit(
                mode=mode,
                command=command_text,
                service=service,
                operation=operation,
                allowed=True,
                reason="timeout",
                exit_code=124,
                duration_ms=float(Config.AWS_CLI_TIMEOUT_SEC) * 1000.0,
                stdout_len=0,
                stderr_len=0,
            )
            if not first_error:
                first_error = (
                    f"❌ AWS CLI command timed out after {Config.AWS_CLI_TIMEOUT_SEC}s.\n"
                    f"Command:\n{_bash_block(command_text)}"
                )
            continue

    if saw_empty_success and mode == "readonly" and len(attempts) == 1 and not service_is_global and not region_provided:
        if Config.AWS_CLI_PROMPT_FOR_REGION_ON_EMPTY:
            prompt_lines = [
                f"Planned command:\n{_bash_block(last_command_text)}",
                f"✅ No resources found in region `{regions_checked[0]}`.",
                "You can provide a different region (for example `--region us-west-2`) and I will rerun.",
            ]
            if Config.AWS_CLI_AUTO_REGION_RETRY:
                fanout_regions = _candidate_regions()
                if fanout_regions:
                    prompt_lines.append(f"If you want, I can also scan fallback regions: {', '.join(fanout_regions)}")
            return "\n".join(prompt_lines)

    if saw_empty_success and mode == "readonly" and len(attempts) > 1:
        details = truncate_text(_redact(last_stdout or "(no output)"), max_chars=Config.K8S_OUTPUT_MAX_CHARS)
        return (
            f"Planned command:\n{_bash_block(last_command_text)}\n"
            "✅ AWS CLI command executed, but no resources were found in checked regions.\n"
            f"Regions checked: {', '.join(regions_checked)}\n"
            f"Output:\n{details}"
        )

    return first_error or "❌ AWS CLI command did not complete successfully."


@tool
def aws_cli_readonly(command: str) -> str:
    """Run an AWS CLI readonly command.

    Input examples:
    - sts get-caller-identity
    - ec2 describe-instances --region us-east-1

    In the default powerful posture, secret/decryption reads are allowed here.
    """
    if not Config.AWS_CLI_ENABLED:
        return "❌ AWS CLI tools are disabled. Set AWS_CLI_ENABLED=1 to enable."
    if not _ensure_aws_cli_installed():
        return "❌ `aws` CLI not found in PATH in this runtime. I cannot execute AWS commands here until it is available."

    tokens, err = _normalize_command(command)
    if err:
        return err
    intent = classify_command_intent("aws_cli_readonly", shlex.join(tokens))
    intent_err = _intent_error(intent, readonly=True)
    if intent_err:
        return intent_err
    return _execute(tokens, mode="readonly")


@tool
def aws_cli_execute(command: str, reason: str = "") -> str:
    """Run an AWS CLI mutating-capable command.

    IMPORTANT: This is a mutating-capable tool and should require explicit user approval.
    Read-only and sensitive-read commands are rejected here to keep generic execute
    focused on explicit write operations. When AWS_CLI_ALLOW_ALL_WRITE=1, write
    allowlist checks are bypassed.
    """
    if not Config.AWS_CLI_ENABLED:
        return "❌ AWS CLI tools are disabled. Set AWS_CLI_ENABLED=1 to enable."
    if not _ensure_aws_cli_installed():
        return "❌ `aws` CLI not found in PATH in this runtime. I cannot execute AWS commands here until it is available."

    tokens, err = _normalize_command(command)
    if err:
        return err
    intent = classify_command_intent("aws_cli_execute", shlex.join(tokens))
    intent_err = _intent_error(intent, readonly=False)
    if intent_err:
        return intent_err

    reason = (reason or "").strip()
    return _execute(tokens, mode="write", reason=reason)


def get_aws_tools() -> list:
    return [aws_cli_readonly, aws_cli_execute]
