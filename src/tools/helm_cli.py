"""Helm connector for semantic investigations and approval-gated writes."""

from __future__ import annotations

import shutil
from typing import Any

from ..config import Config
from .common import ToolObservation, first_non_flag, format_error_summary, parse_json_output, run_subprocess, shell_split, truncate_text


_READONLY_VERBS = {"get", "history", "list", "search", "show", "status", "template", "version"}
_FLAG_NAMES = {"-n", "--namespace", "--kube-context", "--kubeconfig", "--output"}


class HelmConnector:
    def __init__(self) -> None:
        self.family = "helm"

    @staticmethod
    def _ensure_binary() -> None:
        if not shutil.which("helm"):
            raise RuntimeError("helm is not installed or not on PATH")

    @staticmethod
    def _base_args() -> list[str]:
        args = ["helm"]
        if Config.K8S_KUBECONFIG:
            args.extend(["--kubeconfig", Config.K8S_KUBECONFIG])
        if Config.K8S_CONTEXT:
            args.extend(["--kube-context", Config.K8S_CONTEXT])
        return args

    def _run(self, tokens: list[str]) -> tuple[list[str], Any]:
        self._ensure_binary()
        args = [*self._base_args(), *tokens]
        result = run_subprocess(args, timeout_sec=Config.HELM_TIMEOUT_SEC)
        if not result.ok:
            raise RuntimeError(format_error_summary(result))
        payload = parse_json_output(result.stdout)
        return [result.command], payload if payload is not None else result.stdout

    def release_overview(self, *, all_namespaces: bool = True) -> ToolObservation:
        tokens = ["list", "-o", "json"]
        if all_namespaces:
            tokens.append("-A")
        commands, payload = self._run(tokens)
        releases = payload if isinstance(payload, list) else []
        structured_releases = []
        for item in releases:
            if not isinstance(item, dict):
                continue
            structured_releases.append(
                {
                    "name": item.get("name"),
                    "namespace": item.get("namespace"),
                    "revision": item.get("revision"),
                    "status": item.get("status"),
                    "chart": item.get("chart"),
                    "app_version": item.get("app_version"),
                }
            )
        summary = f"Found {len(structured_releases)} Helm release(s)."
        return ToolObservation(
            family=self.family,
            action="release_overview",
            summary=summary,
            structured={"releases": structured_releases[:80]},
            commands=commands,
            raw_preview=truncate_text(str(structured_releases[:16]), max_chars=1800),
        )

    def release_details(self, *, release_name: str, namespace: str) -> ToolObservation:
        status_commands, status_payload = self._run(["status", release_name, "-n", namespace, "-o", "json"])
        values_commands, values_payload = self._run(["get", "values", release_name, "-n", namespace, "-o", "yaml"])
        manifest_commands, manifest_payload = self._run(["get", "manifest", release_name, "-n", namespace])
        structured = {
            "release_name": release_name,
            "namespace": namespace,
            "status": status_payload,
            "values": values_payload,
            "manifest_preview": truncate_text(str(manifest_payload), max_chars=2400),
        }
        return ToolObservation(
            family=self.family,
            action="release_details",
            summary=f"Collected Helm release details for `{release_name}` in namespace `{namespace}`.",
            structured=structured,
            commands=[*status_commands, *values_commands, *manifest_commands],
            raw_preview=truncate_text(str(structured), max_chars=2600),
        )

    def raw_read(self, command: str) -> ToolObservation:
        tokens = shell_split(command)
        if tokens and tokens[0] == "helm":
            tokens = tokens[1:]
        verb = first_non_flag(tokens, skip_value_flags=_FLAG_NAMES)
        if verb not in _READONLY_VERBS:
            raise RuntimeError(f"`{command}` is not a recognized read-only Helm command")
        commands, payload = self._run(tokens)
        return ToolObservation(
            family=self.family,
            action="raw_read",
            summary=f"Executed read-only Helm command `{command}`.",
            structured={"command": command, "output": payload},
            commands=commands,
            raw_preview=truncate_text(str(payload), max_chars=2400),
        )

    def execute(self, command: str) -> ToolObservation:
        tokens = shell_split(command)
        if tokens and tokens[0] == "helm":
            tokens = tokens[1:]
        if Config.K8S_DRY_RUN:
            args = [*self._base_args(), *tokens]
            return ToolObservation(
                family=self.family,
                action="raw_write",
                summary="Helm dry-run mode is enabled; no mutating command was executed.",
                structured={"command": " ".join(args), "dry_run": True},
                commands=[" ".join(args)],
                raw_preview="dry run",
            )
        commands, payload = self._run(tokens)
        return ToolObservation(
            family=self.family,
            action="raw_write",
            summary=f"Executed mutating Helm command `{command}` successfully.",
            structured={"command": command, "output": payload},
            commands=commands,
            raw_preview=truncate_text(str(payload), max_chars=2400),
        )
