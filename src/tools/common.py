"""Shared command execution helpers for infrastructure connectors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import shlex
import subprocess
import time
from typing import Any


@dataclass
class CommandExecutionResult:
    command: str
    args: list[str]
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: float

    @property
    def ok(self) -> bool:
        return self.exit_code == 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToolObservation:
    family: str
    action: str
    summary: str
    structured: dict[str, Any]
    commands: list[str]
    raw_preview: str = ""
    ok: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def truncate_text(text: str, *, max_chars: int = 6000) -> str:
    clean = str(text or "").strip()
    if max_chars <= 0 or len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 3].rstrip() + "..."


def parse_json_output(text: str) -> Any | None:
    clean = str(text or "").strip()
    if not clean:
        return None
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        return None


def shell_split(command: str) -> list[str]:
    raw = str(command or "").strip()
    if not raw:
        raise ValueError("command is required")
    try:
        return shlex.split(raw)
    except ValueError as exc:
        raise ValueError(f"invalid command syntax: {exc}") from exc


def first_non_flag(tokens: list[str], *, skip_value_flags: set[str] | None = None) -> str:
    skip_value_flags = set(skip_value_flags or set())
    skip_next = False
    for token in tokens:
        if skip_next:
            skip_next = False
            continue
        if token in skip_value_flags:
            skip_next = True
            continue
        if token.startswith("--") and "=" not in token and token in skip_value_flags:
            skip_next = True
            continue
        if token.startswith("-"):
            continue
        return token
    return ""


def run_subprocess(args: list[str], *, timeout_sec: int) -> CommandExecutionResult:
    command = " ".join(args)
    t0 = time.perf_counter()
    completed = subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=max(5, int(timeout_sec)),
        check=False,
    )
    return CommandExecutionResult(
        command=command,
        args=list(args),
        exit_code=int(completed.returncode),
        stdout=str(completed.stdout or ""),
        stderr=str(completed.stderr or ""),
        duration_ms=(time.perf_counter() - t0) * 1000.0,
    )


def format_error_summary(result: CommandExecutionResult) -> str:
    details = truncate_text(result.stderr or result.stdout or "unknown error", max_chars=800)
    return f"`{result.command}` failed with exit code {result.exit_code}: {details}"
