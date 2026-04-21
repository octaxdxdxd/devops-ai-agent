"""Kubernetes infrastructure client wrapping kubectl."""

from __future__ import annotations

import json
import logging
import shlex
import subprocess

from ..config import Config

log = logging.getLogger(__name__)


class K8sClient:
    """Thin subprocess wrapper around kubectl."""

    def __init__(self) -> None:
        self.context = Config.K8S_CONTEXT
        self.namespace = Config.K8S_DEFAULT_NAMESPACE or "default"

    # ── availability ─────────────────────────────────────────────────────

    def available(self) -> bool:
        result = self.run(["version", "--client"])
        return not result.startswith("ERROR:")

    # ── low-level execution ──────────────────────────────────────────────

    def _base_cmd(self) -> list[str]:
        cmd = ["kubectl"]
        if self.context:
            cmd.extend(["--context", self.context])
        return cmd

    def run(
        self,
        args: list[str],
        namespace: str | None = None,
        timeout: int = 30,
    ) -> str:
        cmd = self._base_cmd()
        if namespace:
            cmd.extend(["-n", namespace])
        cmd.extend(args)
        log.debug("kubectl: %s", " ".join(cmd))
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if result.returncode != 0:
                return f"ERROR: {result.stderr.strip()}"
            return result.stdout.strip()
        except subprocess.TimeoutExpired:
            return f"ERROR: Command timed out after {timeout}s"
        except FileNotFoundError:
            return "ERROR: kubectl not found in PATH"

    def run_json(
        self,
        args: list[str],
        namespace: str | None = None,
        timeout: int = 30,
    ) -> dict | list | str:
        output = self.run([*args, "-o", "json"], namespace=namespace, timeout=timeout)
        if output.startswith("ERROR:"):
            return output
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            return output

    # ── read operations ──────────────────────────────────────────────────

    def get_resources(
        self,
        kind: str,
        namespace: str | None = None,
        name: str | None = None,
        label_selector: str | None = None,
        all_namespaces: bool = False,
    ) -> str:
        args = ["get", kind]
        if name:
            args.append(name)
        if label_selector:
            args.extend(["-l", label_selector])
        if all_namespaces:
            args.append("--all-namespaces")
        args.extend(["-o", "wide"])
        ns = None if all_namespaces else (namespace or self.namespace)
        return self.run(args, namespace=ns)

    def describe(self, kind: str, name: str, namespace: str | None = None) -> str:
        return self.run(["describe", kind, name], namespace=namespace or self.namespace)

    def get_logs(
        self,
        pod: str,
        namespace: str | None = None,
        container: str | None = None,
        tail: int = 100,
        since: str | None = None,
    ) -> str:
        args = ["logs", pod]
        if container:
            args.extend(["-c", container])
        args.extend(["--tail", str(tail)])
        if since:
            args.extend(["--since", since])
        return self.run(args, namespace=namespace or self.namespace, timeout=15)

    def get_events(
        self,
        namespace: str | None = None,
        field_selector: str | None = None,
        all_namespaces: bool = False,
    ) -> str:
        args = ["get", "events", "--sort-by=.lastTimestamp"]
        if field_selector:
            args.extend(["--field-selector", field_selector])
        if all_namespaces:
            args.append("--all-namespaces")
        ns = None if all_namespaces else (namespace or self.namespace)
        return self.run(args, namespace=ns)

    def top(self, resource_type: str = "pods", namespace: str | None = None, name: str | None = None) -> str:
        args = ["top", resource_type]
        if name:
            args.append(name)
        return self.run(args, namespace=namespace or self.namespace)

    def rollout_history(self, kind: str, name: str, namespace: str | None = None) -> str:
        return self.run(
            ["rollout", "history", f"{kind}/{name}"],
            namespace=namespace or self.namespace,
        )

    def get_contexts(self) -> str:
        return self.run(["config", "get-contexts"])

    def get_namespaces(self) -> str:
        return self.run(["get", "namespaces"])

    def get_resource_yaml(self, kind: str, name: str, namespace: str | None = None) -> str:
        return self.run(
            ["get", kind, name, "-o", "yaml"],
            namespace=namespace or self.namespace,
        )

    def get_resource_json(self, kind: str, name: str, namespace: str | None = None) -> dict | list | str:
        ns = None if kind in {"node", "nodes", "namespace", "namespaces", "pv", "persistentvolume", "persistentvolumes"} else (namespace or self.namespace)
        return self.run_json(["get", kind, name], namespace=ns)

    def list_resources_json(
        self,
        kind: str,
        namespace: str | None = None,
        label_selector: str | None = None,
        all_namespaces: bool = False,
    ) -> dict | list | str:
        args = ["get", kind]
        if label_selector:
            args.extend(["-l", label_selector])
        if all_namespaces:
            args.append("--all-namespaces")
        ns = None if all_namespaces else (namespace or self.namespace)
        return self.run_json(args, namespace=ns)

    # ── write operations ─────────────────────────────────────────────────

    def scale(self, kind: str, name: str, replicas: int, namespace: str | None = None) -> str:
        return self.run(
            ["scale", f"{kind}/{name}", f"--replicas={replicas}"],
            namespace=namespace or self.namespace,
        )

    def rollout_restart(self, kind: str, name: str, namespace: str | None = None) -> str:
        return self.run(
            ["rollout", "restart", f"{kind}/{name}"],
            namespace=namespace or self.namespace,
        )

    def rollout_undo(self, kind: str, name: str, namespace: str | None = None) -> str:
        return self.run(
            ["rollout", "undo", f"{kind}/{name}"],
            namespace=namespace or self.namespace,
        )

    def delete_resource(self, kind: str, name: str, namespace: str | None = None) -> str:
        return self.run(["delete", kind, name], namespace=namespace or self.namespace)

    def apply_manifest(self, manifest: str) -> str:
        cmd = self._base_cmd() + ["apply", "-f", "-"]
        try:
            result = subprocess.run(
                cmd,
                input=manifest,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                return f"ERROR: {result.stderr.strip()}"
            return result.stdout.strip()
        except subprocess.TimeoutExpired:
            return "ERROR: Apply timed out after 30s"

    def cordon(self, node: str) -> str:
        return self.run(["cordon", node])

    def uncordon(self, node: str) -> str:
        return self.run(["uncordon", node])

    def drain(self, node: str, grace_period: int = 300) -> str:
        return self.run(
            [
                "drain", node,
                "--ignore-daemonsets",
                "--delete-emptydir-data",
                f"--grace-period={grace_period}",
                "--force",
            ],
            timeout=grace_period + 30,
        )

    def exec_command(
        self,
        pod: str,
        command: str,
        namespace: str | None = None,
        container: str | None = None,
    ) -> str:
        args = ["exec", pod]
        if container:
            args.extend(["-c", container])
        args.append("--")
        args.extend(shlex.split(command))
        return self.run(args, namespace=namespace or self.namespace, timeout=30)

    def rollout_status(self, kind: str, name: str, namespace: str | None = None, timeout_seconds: int = 300) -> str:
        return self.run(
            ["rollout", "status", f"{kind}/{name}", f"--timeout={timeout_seconds}s"],
            namespace=namespace or self.namespace,
            timeout=timeout_seconds + 30,
        )

    def restart_workload_safely(
        self,
        kind: str,
        name: str,
        namespace: str | None = None,
        *,
        timeout_seconds: int = 300,
    ) -> str:
        history = self.rollout_history(kind, name, namespace or self.namespace)
        if history.startswith("ERROR:"):
            return history
        restart_result = self.rollout_restart(kind, name, namespace or self.namespace)
        if restart_result.startswith("ERROR:"):
            return restart_result
        status_result = self.rollout_status(kind, name, namespace or self.namespace, timeout_seconds=timeout_seconds)
        return "\n".join([
            f"Precheck rollout history:\n{history}",
            f"Restart result:\n{restart_result}",
            f"Rollout status:\n{status_result}",
        ])

    def cleanup_terminated_pods(
        self,
        phases: list[str],
        namespace: str | None = None,
        *,
        all_namespaces: bool = False,
    ) -> str:
        scoped_namespace = None if all_namespaces else (namespace or self.namespace)
        summary: list[str] = []
        for phase in phases:
            selector = f"status.phase={phase}"
            before = self.run(
                ["get", "pods", "--field-selector", selector, "--no-headers"],
                namespace=scoped_namespace,
            )
            before_count = 0 if before.startswith("ERROR: No resources found") or before == "No resources found" else len([line for line in before.splitlines() if line.strip()])
            delete_args = ["delete", "pods", "--field-selector", selector, "--ignore-not-found"]
            if all_namespaces:
                delete_args.append("--all-namespaces")
                delete_result = self.run(delete_args, namespace=None)
            else:
                delete_result = self.run(delete_args, namespace=scoped_namespace)
            summary.append(f"{phase}: matched={before_count}\n{delete_result}")
        return "\n\n".join(summary)

    # ── convenience: execute a safe, pre-approved shell command ──────────

    def execute_shell_command(self, command: str) -> str:
        """Execute a pre-approved kubectl command string.

        Only kubectl commands are allowed.  Shell operators are rejected.
        """
        dangerous = ["|", ">", "<", ";", "&", "$(", "`"]
        if any(d in command for d in dangerous):
            return "ERROR: Command contains potentially dangerous shell operators"
        try:
            parts = shlex.split(command.strip())
        except ValueError as exc:
            return f"ERROR: Invalid command syntax: {exc}"
        if not parts or parts[0] != "kubectl":
            return "ERROR: Only kubectl commands are allowed"
        # Strip the leading 'kubectl' since _base_cmd already sets it
        return self.run(parts[1:])
