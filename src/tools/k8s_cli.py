"""Kubernetes connector for semantic investigations and approval-gated writes."""

from __future__ import annotations

from collections import Counter
import shutil
from typing import Any

from ..config import Config
from .common import ToolObservation, first_non_flag, format_error_summary, parse_json_output, run_subprocess, shell_split, truncate_text


_READONLY_VERBS = {
    "api-resources",
    "api-versions",
    "auth",
    "cluster-info",
    "config",
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
_GLOBAL_FLAG_NAMES = {"-n", "--namespace", "--context", "--kubeconfig", "-l", "--selector", "-o", "--output"}


def _container_restart_total(pod: dict[str, Any]) -> int:
    statuses = (pod.get("status") or {}).get("containerStatuses") or []
    return sum(int(item.get("restartCount") or 0) for item in statuses if isinstance(item, dict))


class KubernetesConnector:
    def __init__(self) -> None:
        self.family = "k8s"

    @staticmethod
    def _ensure_binary() -> None:
        if not shutil.which("kubectl"):
            raise RuntimeError("kubectl is not installed or not on PATH")

    @staticmethod
    def _base_args() -> list[str]:
        args = ["kubectl"]
        if Config.K8S_KUBECONFIG:
            args.extend(["--kubeconfig", Config.K8S_KUBECONFIG])
        if Config.K8S_CONTEXT:
            args.extend(["--context", Config.K8S_CONTEXT])
        args.extend(["--request-timeout", f"{Config.K8S_REQUEST_TIMEOUT_SEC}s"])
        return args

    def _run(self, tokens: list[str]) -> tuple[list[str], Any]:
        self._ensure_binary()
        args = [*self._base_args(), *tokens]
        result = run_subprocess(args, timeout_sec=Config.K8S_REQUEST_TIMEOUT_SEC + 10)
        if not result.ok:
            raise RuntimeError(format_error_summary(result))
        payload = parse_json_output(result.stdout)
        return [result.command], payload if payload is not None else result.stdout

    def _run_json(self, tokens: list[str]) -> tuple[list[str], dict[str, Any] | list[Any]]:
        commands, payload = self._run(tokens)
        if not isinstance(payload, (dict, list)):
            raise RuntimeError(f"`{' '.join(tokens)}` did not return JSON output")
        return commands, payload

    def _namespace_args(self, namespace: str | None, *, all_namespaces: bool = False) -> list[str]:
        if all_namespaces:
            return ["-A"]
        ns = str(namespace or "").strip()
        if ns:
            return ["-n", ns]
        if Config.K8S_DEFAULT_NAMESPACE:
            return ["-n", Config.K8S_DEFAULT_NAMESPACE]
        return []

    def list_namespaces(self) -> ToolObservation:
        commands, payload = self._run_json(["get", "namespaces", "-o", "json"])
        items = payload.get("items") if isinstance(payload, dict) else []
        names = [str((item.get("metadata") or {}).get("name") or "") for item in items if isinstance(item, dict)]
        return ToolObservation(
            family=self.family,
            action="list_namespaces",
            summary=f"Found {len(names)} namespaces.",
            structured={"namespace_count": len(names), "namespaces": names},
            commands=commands,
            raw_preview=truncate_text(str(names), max_chars=1200),
        )

    def node_overview(self) -> ToolObservation:
        commands, payload = self._run_json(["get", "nodes", "-o", "json"])
        items = payload.get("items") if isinstance(payload, dict) else []
        nodes: list[dict[str, Any]] = []
        ready_count = 0
        for item in items:
            if not isinstance(item, dict):
                continue
            metadata = item.get("metadata") or {}
            status = item.get("status") or {}
            ready = False
            pressure: list[str] = []
            for condition in status.get("conditions") or []:
                if not isinstance(condition, dict):
                    continue
                cond_type = str(condition.get("type") or "")
                cond_status = str(condition.get("status") or "")
                if cond_type == "Ready":
                    ready = cond_status == "True"
                if cond_type.endswith("Pressure") and cond_status == "True":
                    pressure.append(cond_type)
            if ready:
                ready_count += 1
            nodes.append(
                {
                    "name": metadata.get("name"),
                    "ready": ready,
                    "taints": [str(item.get("key")) for item in (item.get("spec") or {}).get("taints") or [] if isinstance(item, dict)],
                    "pressure": pressure,
                }
            )
        not_ready = [node for node in nodes if not node.get("ready")]
        summary = f"{ready_count}/{len(nodes)} nodes are Ready."
        if not_ready:
            summary += f" {len(not_ready)} node(s) are not Ready."
        return ToolObservation(
            family=self.family,
            action="node_overview",
            summary=summary,
            structured={
                "node_count": len(nodes),
                "ready_count": ready_count,
                "not_ready_count": len(not_ready),
                "nodes": nodes[:40],
            },
            commands=commands,
            raw_preview=truncate_text(str(nodes[:12]), max_chars=1800),
        )

    def pod_overview(self, *, namespace: str | None = None, all_namespaces: bool = False) -> ToolObservation:
        commands, payload = self._run_json(["get", "pods", *self._namespace_args(namespace, all_namespaces=all_namespaces), "-o", "json"])
        items = payload.get("items") if isinstance(payload, dict) else []
        phase_counts: Counter[str] = Counter()
        problem_pods: list[dict[str, Any]] = []
        restart_heavy: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            metadata = item.get("metadata") or {}
            status = item.get("status") or {}
            phase = str(status.get("phase") or "Unknown")
            namespace_name = str(metadata.get("namespace") or "")
            pod_name = str(metadata.get("name") or "")
            phase_counts[phase] += 1
            restart_total = _container_restart_total(item)
            waiting_reasons = []
            for container_state in status.get("containerStatuses") or []:
                if not isinstance(container_state, dict):
                    continue
                waiting = ((container_state.get("state") or {}).get("waiting") or {}).get("reason")
                if waiting:
                    waiting_reasons.append(str(waiting))
            if phase not in {"Running", "Succeeded", "Completed"} or waiting_reasons:
                problem_pods.append(
                    {
                        "namespace": namespace_name,
                        "name": pod_name,
                        "phase": phase,
                        "node": str(status.get("nodeName") or ""),
                        "restart_count": restart_total,
                        "reasons": waiting_reasons[:4],
                    }
                )
            elif restart_total >= 5:
                restart_heavy.append(
                    {
                        "namespace": namespace_name,
                        "name": pod_name,
                        "restart_count": restart_total,
                        "node": str(status.get("nodeName") or ""),
                    }
                )
        summary = f"Observed {sum(phase_counts.values())} pods."
        if problem_pods:
            summary += f" {len(problem_pods)} pod(s) are not healthy."
        if restart_heavy:
            summary += f" {len(restart_heavy)} additional pod(s) have elevated restarts."
        return ToolObservation(
            family=self.family,
            action="pod_overview",
            summary=summary,
            structured={
                "phase_counts": dict(phase_counts),
                "problem_pods": problem_pods[:40],
                "restart_heavy_pods": restart_heavy[:30],
            },
            commands=commands,
            raw_preview=truncate_text(str(problem_pods[:15] + restart_heavy[:15]), max_chars=2200),
        )

    def workload_overview(self, *, namespace: str | None = None, all_namespaces: bool = False) -> ToolObservation:
        kind_specs = [
            ("deployments", "deployment"),
            ("statefulsets", "statefulset"),
            ("daemonsets", "daemonset"),
        ]
        commands: list[str] = []
        unhealthy: list[dict[str, Any]] = []
        counts: dict[str, int] = {}
        for plural, label in kind_specs:
            command_list, payload = self._run_json(["get", plural, *self._namespace_args(namespace, all_namespaces=all_namespaces), "-o", "json"])
            commands.extend(command_list)
            items = payload.get("items") if isinstance(payload, dict) else []
            counts[label] = len(items)
            for item in items:
                if not isinstance(item, dict):
                    continue
                metadata = item.get("metadata") or {}
                spec = item.get("spec") or {}
                status = item.get("status") or {}
                desired = int(spec.get("replicas") or status.get("desiredNumberScheduled") or 0)
                ready = int(status.get("readyReplicas") or status.get("numberReady") or 0)
                available = int(status.get("availableReplicas") or ready)
                if desired and ready < desired:
                    unhealthy.append(
                        {
                            "kind": label,
                            "namespace": metadata.get("namespace"),
                            "name": metadata.get("name"),
                            "desired": desired,
                            "ready": ready,
                            "available": available,
                        }
                    )
        summary = (
            f"Observed {counts.get('deployment', 0)} deployments, "
            f"{counts.get('statefulset', 0)} statefulsets, and {counts.get('daemonset', 0)} daemonsets."
        )
        if unhealthy:
            summary += f" {len(unhealthy)} workload(s) are short of ready replicas."
        return ToolObservation(
            family=self.family,
            action="workload_overview",
            summary=summary,
            structured={"counts": counts, "unhealthy_workloads": unhealthy[:40]},
            commands=commands,
            raw_preview=truncate_text(str(unhealthy[:20]), max_chars=1800),
        )

    def service_overview(self, *, namespace: str | None = None, all_namespaces: bool = False) -> ToolObservation:
        commands, payload = self._run_json(["get", "svc,ingress", *self._namespace_args(namespace, all_namespaces=all_namespaces), "-o", "json"])
        items = payload.get("items") if isinstance(payload, dict) else []
        services: list[dict[str, Any]] = []
        ingresses: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            kind = str(item.get("kind") or "").lower()
            metadata = item.get("metadata") or {}
            record = {
                "namespace": metadata.get("namespace"),
                "name": metadata.get("name"),
            }
            if kind == "service":
                spec = item.get("spec") or {}
                record["type"] = spec.get("type")
                record["cluster_ip"] = spec.get("clusterIP")
                services.append(record)
            elif kind == "ingress":
                status = item.get("status") or {}
                lb = ((status.get("loadBalancer") or {}).get("ingress") or [])
                record["load_balancer"] = lb[:2]
                ingresses.append(record)
        summary = f"Observed {len(services)} services and {len(ingresses)} ingresses."
        return ToolObservation(
            family=self.family,
            action="service_overview",
            summary=summary,
            structured={"services": services[:60], "ingresses": ingresses[:60]},
            commands=commands,
            raw_preview=truncate_text(str({"services": services[:12], "ingresses": ingresses[:12]}), max_chars=1800),
        )

    def storage_overview(self, *, namespace: str | None = None, all_namespaces: bool = False) -> ToolObservation:
        commands, payload = self._run_json(["get", "pvc,pv", *self._namespace_args(namespace, all_namespaces=all_namespaces), "-o", "json"])
        items = payload.get("items") if isinstance(payload, dict) else []
        pvcs: list[dict[str, Any]] = []
        pvs: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            kind = str(item.get("kind") or "").lower()
            metadata = item.get("metadata") or {}
            status = item.get("status") or {}
            record = {
                "namespace": metadata.get("namespace"),
                "name": metadata.get("name"),
                "phase": status.get("phase"),
            }
            if kind == "persistentvolumeclaim":
                pvcs.append(record)
            elif kind == "persistentvolume":
                pvs.append(record)
        summary = f"Observed {len(pvcs)} PVCs and {len(pvs)} PVs."
        return ToolObservation(
            family=self.family,
            action="storage_overview",
            summary=summary,
            structured={"pvcs": pvcs[:80], "pvs": pvs[:80]},
            commands=commands,
            raw_preview=truncate_text(str({"pvcs": pvcs[:15], "pvs": pvs[:15]}), max_chars=1800),
        )

    def event_overview(self, *, namespace: str | None = None, all_namespaces: bool = False) -> ToolObservation:
        commands, payload = self._run_json(["get", "events", *self._namespace_args(namespace, all_namespaces=all_namespaces), "-o", "json"])
        items = payload.get("items") if isinstance(payload, dict) else []
        warnings: list[dict[str, Any]] = []
        count_by_reason: Counter[str] = Counter()
        for item in items[-200:]:
            if not isinstance(item, dict):
                continue
            event_type = str(item.get("type") or "")
            if event_type != "Warning":
                continue
            reason = str(item.get("reason") or "Unknown")
            count_by_reason[reason] += 1
            obj = item.get("involvedObject") or {}
            warnings.append(
                {
                    "namespace": item.get("metadata", {}).get("namespace"),
                    "reason": reason,
                    "message": str(item.get("message") or "")[:240],
                    "kind": obj.get("kind"),
                    "name": obj.get("name"),
                }
            )
        summary = f"Observed {sum(count_by_reason.values())} warning events."
        return ToolObservation(
            family=self.family,
            action="event_overview",
            summary=summary,
            structured={"warning_count": sum(count_by_reason.values()), "top_reasons": dict(count_by_reason.most_common(12)), "warnings": warnings[:40]},
            commands=commands,
            raw_preview=truncate_text(str(warnings[:15]), max_chars=1800),
        )

    def resource_details(self, *, kind: str, name: str, namespace: str = "") -> ToolObservation:
        tokens = ["get", kind, name, "-o", "json"]
        if namespace:
            tokens.extend(["-n", namespace])
        commands, payload = self._run_json(tokens)
        metadata = payload.get("metadata") if isinstance(payload, dict) else {}
        summary = f"Loaded {kind} `{name}`."
        if namespace:
            summary += f" Namespace: {namespace}."
        return ToolObservation(
            family=self.family,
            action="resource_details",
            summary=summary,
            structured={"kind": kind, "name": name, "namespace": namespace, "resource": payload},
            commands=commands,
            raw_preview=truncate_text(str(payload), max_chars=2200),
        )

    def namespace_overview(self, *, namespace: str) -> ToolObservation:
        pods = self.pod_overview(namespace=namespace)
        workloads = self.workload_overview(namespace=namespace)
        services = self.service_overview(namespace=namespace)
        storage = self.storage_overview(namespace=namespace)
        events = self.event_overview(namespace=namespace)
        namespace_record = {
            "namespace": namespace,
            "pods": pods.structured,
            "workloads": workloads.structured,
            "services": services.structured,
            "storage": storage.structured,
            "events": events.structured,
        }
        return ToolObservation(
            family=self.family,
            action="namespace_overview",
            summary=f"Collected a namespace-focused overview for `{namespace}`.",
            structured=namespace_record,
            commands=[*pods.commands, *workloads.commands, *services.commands, *storage.commands, *events.commands],
            raw_preview=truncate_text(str(namespace_record), max_chars=2200),
        )

    def cluster_overview(self) -> ToolObservation:
        namespaces = self.list_namespaces()
        nodes = self.node_overview()
        pods = self.pod_overview(all_namespaces=True)
        workloads = self.workload_overview(all_namespaces=True)
        services = self.service_overview(all_namespaces=True)
        storage = self.storage_overview(all_namespaces=True)
        events = self.event_overview(all_namespaces=True)
        structured = {
            "namespaces": namespaces.structured,
            "nodes": nodes.structured,
            "pods": pods.structured,
            "workloads": workloads.structured,
            "services": services.structured,
            "storage": storage.structured,
            "events": events.structured,
        }
        summary = (
            f"Cluster overview: {nodes.structured.get('ready_count', 0)}/{nodes.structured.get('node_count', 0)} ready nodes, "
            f"{len(pods.structured.get('problem_pods', []))} problematic pods, "
            f"{len(workloads.structured.get('unhealthy_workloads', []))} unhealthy workloads, "
            f"{events.structured.get('warning_count', 0)} warning events."
        )
        commands = [*namespaces.commands, *nodes.commands, *pods.commands, *workloads.commands, *services.commands, *storage.commands, *events.commands]
        return ToolObservation(
            family=self.family,
            action="cluster_overview",
            summary=summary,
            structured=structured,
            commands=commands,
            raw_preview=truncate_text(str(structured), max_chars=2800),
        )

    def raw_read(self, command: str) -> ToolObservation:
        tokens = shell_split(command)
        if tokens and tokens[0] == "kubectl":
            tokens = tokens[1:]
        verb = first_non_flag(tokens, skip_value_flags=_GLOBAL_FLAG_NAMES)
        if verb not in _READONLY_VERBS:
            raise RuntimeError(f"`{command}` is not a recognized read-only kubectl command")
        commands, payload = self._run(tokens)
        preview = truncate_text(str(payload), max_chars=Config.TOOL_STRUCTURED_OUTPUT_MAX_CHARS)
        summary = f"Executed read-only kubectl command `{command}`."
        return ToolObservation(
            family=self.family,
            action="raw_read",
            summary=summary,
            structured={"command": command, "output": payload},
            commands=commands,
            raw_preview=preview,
        )

    def execute(self, command: str) -> ToolObservation:
        tokens = shell_split(command)
        if tokens and tokens[0] == "kubectl":
            tokens = tokens[1:]
        self._ensure_binary()
        args = [*self._base_args(), *tokens]
        if Config.K8S_DRY_RUN:
            return ToolObservation(
                family=self.family,
                action="raw_write",
                summary="Kubernetes dry-run mode is enabled; no mutating command was executed.",
                structured={"command": " ".join(args), "dry_run": True},
                commands=[" ".join(args)],
                raw_preview="dry run",
            )
        result = run_subprocess(args, timeout_sec=Config.K8S_REQUEST_TIMEOUT_SEC + 20)
        if not result.ok:
            raise RuntimeError(format_error_summary(result))
        return ToolObservation(
            family=self.family,
            action="raw_write",
            summary=f"Executed mutating kubectl command `{command}` successfully.",
            structured={"command": result.command, "stdout": result.stdout, "stderr": result.stderr},
            commands=[result.command],
            raw_preview=truncate_text(result.stdout or "(no output)", max_chars=2200),
        )
