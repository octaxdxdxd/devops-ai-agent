"""Kubernetes infrastructure client wrapping kubectl."""

from __future__ import annotations

import json
import logging
import shlex
import subprocess
from decimal import Decimal, InvalidOperation

from ..config import Config

log = logging.getLogger(__name__)


_CLUSTER_SCOPED_KINDS = {
    "node", "nodes", "namespace", "namespaces", "pv", "persistentvolume", "persistentvolumes",
}
_WORKLOAD_KINDS = {
    "deployment", "deployments",
    "statefulset", "statefulsets",
    "daemonset", "daemonsets",
    "replicaset", "replicasets",
    "job", "jobs",
    "cronjob", "cronjobs",
}
_POD_KINDS = {"pod", "pods"}
_MEMORY_BINARY_UNITS = {
    "Ki": 1024,
    "Mi": 1024 ** 2,
    "Gi": 1024 ** 3,
    "Ti": 1024 ** 4,
    "Pi": 1024 ** 5,
    "Ei": 1024 ** 6,
}
_MEMORY_DECIMAL_UNITS = {
    "K": 1000,
    "M": 1000 ** 2,
    "G": 1000 ** 3,
    "T": 1000 ** 4,
    "P": 1000 ** 5,
    "E": 1000 ** 6,
}


def _safe_decimal(raw: object) -> Decimal | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        return Decimal(text)
    except (InvalidOperation, ValueError):
        return None


def _parse_cpu_to_mcpu(raw: object) -> int | float | None:
    text = str(raw or "").strip()
    if not text:
        return None
    if text.endswith("n"):
        value = _safe_decimal(text[:-1])
        if value is None:
            return None
        return round(float(value / Decimal(1_000_000)), 2)
    if text.endswith("u"):
        value = _safe_decimal(text[:-1])
        if value is None:
            return None
        return round(float(value / Decimal(1000)), 2)
    if text.endswith("m"):
        value = _safe_decimal(text[:-1])
        return int(value) if value is not None else None
    value = _safe_decimal(text)
    if value is None:
        return None
    return int(value * 1000)


def _parse_bytes(raw: object) -> int | None:
    text = str(raw or "").strip()
    if not text:
        return None
    for suffix, multiplier in _MEMORY_BINARY_UNITS.items():
        if text.endswith(suffix):
            value = _safe_decimal(text[:-len(suffix)])
            return int(value * multiplier) if value is not None else None
    for suffix, multiplier in _MEMORY_DECIMAL_UNITS.items():
        if text.endswith(suffix):
            value = _safe_decimal(text[:-len(suffix)])
            return int(value * multiplier) if value is not None else None
    value = _safe_decimal(text)
    if value is None:
        return None
    return int(value)


def _bytes_to_mib(value: int | None) -> float | None:
    if value is None:
        return None
    return round(value / (1024 ** 2), 2)


def _normalize_kind(kind: str) -> str:
    text = str(kind or "pod").strip().lower()
    aliases = {
        "deploy": "deployment",
        "deployments": "deployment",
        "statefulsets": "statefulset",
        "daemonsets": "daemonset",
        "replicasets": "replicaset",
        "jobs": "job",
        "cronjobs": "cronjob",
        "pods": "pod",
    }
    return aliases.get(text, text)


def _label_selector_from_match_labels(match_labels: dict | None) -> str:
    if not isinstance(match_labels, dict) or not match_labels:
        return ""
    parts = [
        f"{key}={value}"
        for key, value in sorted(match_labels.items())
        if str(key).strip() and str(value).strip()
    ]
    return ",".join(parts)


def _owner_ref(resource: dict) -> dict[str, str]:
    metadata = resource.get("metadata", {}) if isinstance(resource, dict) else {}
    owner_refs = metadata.get("ownerReferences")
    if isinstance(owner_refs, list) and owner_refs:
        owner = owner_refs[0] if isinstance(owner_refs[0], dict) else {}
        return {
            "kind": str(owner.get("kind", "") or ""),
            "name": str(owner.get("name", "") or ""),
        }
    return {"kind": "", "name": ""}


def _selector_for_workload(resource: dict) -> str:
    spec = resource.get("spec", {}) if isinstance(resource, dict) else {}
    kind = _normalize_kind(resource.get("kind", ""))
    if kind == "cronjob":
        template = (((spec.get("jobTemplate") or {}).get("spec") or {}).get("template") or {}).get("metadata") or {}
        return _label_selector_from_match_labels(template.get("labels"))
    selector = spec.get("selector")
    if isinstance(selector, dict):
        if isinstance(selector.get("matchLabels"), dict):
            return _label_selector_from_match_labels(selector.get("matchLabels"))
    return ""


def _container_resources(container: dict) -> dict:
    resources = container.get("resources", {}) if isinstance(container, dict) else {}
    requests = resources.get("requests", {}) if isinstance(resources.get("requests"), dict) else {}
    limits = resources.get("limits", {}) if isinstance(resources.get("limits"), dict) else {}
    return {
        "requests": {
            key: value
            for key, value in requests.items()
            if value not in (None, "")
        },
        "limits": {
            key: value
            for key, value in limits.items()
            if value not in (None, "")
        },
    }


def _quantity_summary(values: dict) -> dict:
    cpu = values.get("cpu")
    memory = values.get("memory")
    ephemeral_storage = values.get("ephemeral-storage")
    cpu_mcpu = _parse_cpu_to_mcpu(cpu)
    memory_bytes = _parse_bytes(memory)
    storage_bytes = _parse_bytes(ephemeral_storage)

    summary = {key: value for key, value in values.items() if value not in (None, "")}
    if cpu_mcpu is not None:
        summary["cpu_mcpu"] = cpu_mcpu
    if memory_bytes is not None:
        summary["memory_bytes"] = memory_bytes
        summary["memory_mib"] = _bytes_to_mib(memory_bytes)
    if storage_bytes is not None:
        summary["ephemeral_storage_bytes"] = storage_bytes
        summary["ephemeral_storage_mib"] = _bytes_to_mib(storage_bytes)
    return summary


def _percent(numerator: int | float | None, denominator: int | float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return round((numerator / denominator) * 100, 2)


def _format_cpu(value: int | float | None) -> str:
    if value is None:
        return "—"
    if isinstance(value, float) and not value.is_integer():
        return f"{value:.2f}m"
    return f"{int(value)}m"


def _format_memory_mib(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.2f}Mi"


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
        include_stderr_on_success: bool = False,
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
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            if include_stderr_on_success and stderr:
                return "\n".join(part for part in (stdout, stderr) if part)
            return stdout
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
        ns = None if kind in _CLUSTER_SCOPED_KINDS else (namespace or self.namespace)
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

    def get_pod_metrics_json(
        self,
        namespace: str | None = None,
        *,
        all_namespaces: bool = False,
    ) -> dict | list | str:
        if all_namespaces:
            path = "/apis/metrics.k8s.io/v1beta1/pods"
        else:
            ns = str(namespace or self.namespace).strip() or self.namespace
            path = f"/apis/metrics.k8s.io/v1beta1/namespaces/{ns}/pods"
        output = self.run(["get", "--raw", path], namespace=None, timeout=20)
        if output.startswith("ERROR:"):
            return output
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            return output

    def analyze_resource_usage(
        self,
        kind: str = "pod",
        namespace: str | None = None,
        name: str | None = None,
        label_selector: str | None = None,
        all_namespaces: bool = False,
        include_usage: bool = True,
        output_format: str = "auto",
    ) -> str:
        normalized_kind = _normalize_kind(kind)
        if normalized_kind not in _POD_KINDS | _WORKLOAD_KINDS:
            supported = ", ".join(sorted(_POD_KINDS | _WORKLOAD_KINDS))
            return (
                f"ERROR: Unsupported kind '{kind}'. "
                f"Supported kinds: {supported}."
            )

        target_namespace = None if all_namespaces else (namespace or self.namespace)
        pod_records: list[dict] = []
        workload_records: list[dict] = []

        if normalized_kind in _POD_KINDS:
            pod_records = self._resolve_target_pods(
                namespace=target_namespace,
                name=name,
                label_selector=label_selector,
                all_namespaces=all_namespaces,
            )
        else:
            workload_records = self._resolve_target_workloads(
                normalized_kind,
                namespace=target_namespace,
                name=name,
                label_selector=label_selector,
                all_namespaces=all_namespaces,
            )
            if isinstance(workload_records, str):
                return workload_records
            pod_records = self._pods_for_workloads(workload_records)

        if isinstance(pod_records, str):
            return pod_records

        metrics_note = ""
        metrics_map: dict[tuple[str, str], dict[str, dict[str, object]]] = {}
        if include_usage:
            metrics_map, metrics_note = self._pod_metrics_map(
                namespace=target_namespace,
                all_namespaces=all_namespaces,
            )

        summary = self._build_resource_usage_summary(
            target_kind=normalized_kind,
            requested_name=name,
            requested_namespace=target_namespace,
            requested_label_selector=label_selector,
            all_namespaces=all_namespaces,
            workloads=workload_records,
            pods=pod_records,
            metrics_map=metrics_map,
            metrics_note=metrics_note,
            include_usage=include_usage,
        )
        rendered = self._render_resource_usage_output(summary, output_format=output_format)
        return rendered

    def _resolve_target_pods(
        self,
        *,
        namespace: str | None,
        name: str | None,
        label_selector: str | None,
        all_namespaces: bool,
    ) -> list[dict] | str:
        if name and all_namespaces:
            payload = self.list_resources_json("pod", label_selector=label_selector, all_namespaces=True)
            if isinstance(payload, str):
                return payload
            items = [
                item for item in payload.get("items", [])
                if isinstance(item, dict) and item.get("metadata", {}).get("name") == name
            ]
            if not items:
                return f"ERROR: pod '{name}' not found."
            return items
        if name:
            pod = self.get_resource_json("pod", name, namespace)
            if isinstance(pod, str):
                return pod
            return [pod]
        payload = self.list_resources_json(
            "pod",
            namespace=namespace,
            label_selector=label_selector,
            all_namespaces=all_namespaces,
        )
        if isinstance(payload, str):
            return payload
        return [item for item in payload.get("items", []) if isinstance(item, dict)]

    def _resolve_target_workloads(
        self,
        kind: str,
        *,
        namespace: str | None,
        name: str | None,
        label_selector: str | None,
        all_namespaces: bool,
    ) -> list[dict] | str:
        if name and not all_namespaces:
            resource = self.get_resource_json(kind, name, namespace)
            if isinstance(resource, str):
                return resource
            return [resource]
        payload = self.list_resources_json(
            kind,
            namespace=namespace,
            label_selector=label_selector,
            all_namespaces=all_namespaces,
        )
        if isinstance(payload, str):
            return payload
        items = [item for item in payload.get("items", []) if isinstance(item, dict)]
        if name:
            items = [item for item in items if item.get("metadata", {}).get("name") == name]
            if not items:
                return f"ERROR: {kind} '{name}' not found."
        return items

    def _pods_for_workloads(self, workloads: list[dict]) -> list[dict] | str:
        pod_map: dict[tuple[str, str], dict] = {}
        for workload in workloads:
            metadata = workload.get("metadata", {}) if isinstance(workload, dict) else {}
            namespace = str(metadata.get("namespace", "") or self.namespace)
            selector = _selector_for_workload(workload)
            if not selector:
                continue
            payload = self.list_resources_json("pod", namespace=namespace, label_selector=selector, all_namespaces=False)
            if isinstance(payload, str):
                return payload
            for pod in payload.get("items", []):
                if not isinstance(pod, dict):
                    continue
                pod_meta = pod.get("metadata", {})
                key = (
                    str(pod_meta.get("namespace", "") or namespace),
                    str(pod_meta.get("name", "") or ""),
                )
                if key[1]:
                    pod_map[key] = pod
        return [pod_map[key] for key in sorted(pod_map)]

    def _pod_metrics_map(
        self,
        *,
        namespace: str | None,
        all_namespaces: bool,
    ) -> tuple[dict[tuple[str, str], dict[str, dict[str, object]]], str]:
        payload = self.get_pod_metrics_json(namespace=namespace, all_namespaces=all_namespaces)
        if isinstance(payload, str):
            return {}, payload

        metrics_map: dict[tuple[str, str], dict[str, dict[str, object]]] = {}
        for item in payload.get("items", []):
            if not isinstance(item, dict):
                continue
            metadata = item.get("metadata", {})
            pod_name = str(metadata.get("name", "") or "")
            pod_namespace = str(metadata.get("namespace", "") or namespace or self.namespace)
            if not pod_name:
                continue
            containers = item.get("containers", [])
            container_map: dict[str, dict[str, object]] = {}
            for container in containers:
                if not isinstance(container, dict):
                    continue
                container_name = str(container.get("name", "") or "")
                usage = container.get("usage", {}) if isinstance(container.get("usage"), dict) else {}
                if not container_name:
                    continue
                cpu_raw = usage.get("cpu")
                memory_raw = usage.get("memory")
                container_map[container_name] = {
                    "cpu": cpu_raw,
                    "memory": memory_raw,
                    "cpu_mcpu": _parse_cpu_to_mcpu(cpu_raw),
                    "memory_bytes": _parse_bytes(memory_raw),
                    "memory_mib": _bytes_to_mib(_parse_bytes(memory_raw)),
                }
            metrics_map[(pod_namespace, pod_name)] = container_map
        return metrics_map, ""

    def _build_resource_usage_summary(
        self,
        *,
        target_kind: str,
        requested_name: str | None,
        requested_namespace: str | None,
        requested_label_selector: str | None,
        all_namespaces: bool,
        workloads: list[dict],
        pods: list[dict],
        metrics_map: dict[tuple[str, str], dict[str, dict[str, object]]],
        metrics_note: str,
        include_usage: bool,
    ) -> dict:
        result: dict[str, object] = {
            "query": {
                "kind": target_kind,
                "name": requested_name or "",
                "namespace": requested_namespace or "",
                "label_selector": requested_label_selector or "",
                "all_namespaces": all_namespaces,
                "include_usage": include_usage,
            },
            "metrics_available": include_usage and not metrics_note,
            "metrics_note": metrics_note or "",
        }

        if workloads:
            result["workloads"] = [
                {
                    "kind": str(item.get("kind", "") or ""),
                    "name": str(item.get("metadata", {}).get("name", "") or ""),
                    "namespace": str(item.get("metadata", {}).get("namespace", "") or ""),
                    "selector": _selector_for_workload(item),
                }
                for item in workloads
            ]

        pod_entries: list[dict] = []
        total_requests_cpu = total_limits_cpu = total_usage_cpu = 0
        total_requests_memory = total_limits_memory = total_usage_memory = 0
        requests_cpu_known = limits_cpu_known = usage_cpu_known = False
        requests_memory_known = limits_memory_known = usage_memory_known = False
        containers_analyzed = 0

        for pod in sorted(
            pods,
            key=lambda item: (
                str(item.get("metadata", {}).get("namespace", "") or ""),
                str(item.get("metadata", {}).get("name", "") or ""),
            ),
        ):
            metadata = pod.get("metadata", {}) if isinstance(pod, dict) else {}
            spec = pod.get("spec", {}) if isinstance(pod, dict) else {}
            status = pod.get("status", {}) if isinstance(pod, dict) else {}
            pod_namespace = str(metadata.get("namespace", "") or requested_namespace or self.namespace)
            pod_name = str(metadata.get("name", "") or "")
            owner = _owner_ref(pod)
            pod_metrics = metrics_map.get((pod_namespace, pod_name), {})

            container_entries: list[dict] = []
            pod_requests_cpu = pod_limits_cpu = pod_usage_cpu = 0
            pod_requests_memory = pod_limits_memory = pod_usage_memory = 0
            pod_requests_cpu_known = pod_limits_cpu_known = pod_usage_cpu_known = False
            pod_requests_memory_known = pod_limits_memory_known = pod_usage_memory_known = False

            for container in spec.get("containers", []):
                if not isinstance(container, dict):
                    continue
                containers_analyzed += 1
                container_name = str(container.get("name", "") or "")
                resources = _container_resources(container)
                requests_summary = _quantity_summary(resources.get("requests", {}))
                limits_summary = _quantity_summary(resources.get("limits", {}))
                usage_summary = dict(pod_metrics.get(container_name, {}))

                requests_cpu = requests_summary.get("cpu_mcpu")
                limits_cpu = limits_summary.get("cpu_mcpu")
                usage_cpu = usage_summary.get("cpu_mcpu")
                requests_memory = requests_summary.get("memory_bytes")
                limits_memory = limits_summary.get("memory_bytes")
                usage_memory = usage_summary.get("memory_bytes")

                if requests_cpu is not None:
                    pod_requests_cpu += requests_cpu
                    total_requests_cpu += requests_cpu
                    pod_requests_cpu_known = True
                    requests_cpu_known = True
                if limits_cpu is not None:
                    pod_limits_cpu += limits_cpu
                    total_limits_cpu += limits_cpu
                    pod_limits_cpu_known = True
                    limits_cpu_known = True
                if usage_cpu is not None:
                    pod_usage_cpu += usage_cpu
                    total_usage_cpu += usage_cpu
                    pod_usage_cpu_known = True
                    usage_cpu_known = True
                if requests_memory is not None:
                    pod_requests_memory += requests_memory
                    total_requests_memory += requests_memory
                    pod_requests_memory_known = True
                    requests_memory_known = True
                if limits_memory is not None:
                    pod_limits_memory += limits_memory
                    total_limits_memory += limits_memory
                    pod_limits_memory_known = True
                    limits_memory_known = True
                if usage_memory is not None:
                    pod_usage_memory += usage_memory
                    total_usage_memory += usage_memory
                    pod_usage_memory_known = True
                    usage_memory_known = True

                container_entries.append(
                    {
                        "name": container_name,
                        "image": str(container.get("image", "") or ""),
                        "requests": requests_summary,
                        "limits": limits_summary,
                        "usage": usage_summary,
                        "usage_vs_request": {
                            "cpu_pct": _percent(usage_cpu, requests_cpu),
                            "memory_pct": _percent(usage_memory, requests_memory),
                        },
                        "usage_vs_limit": {
                            "cpu_pct": _percent(usage_cpu, limits_cpu),
                            "memory_pct": _percent(usage_memory, limits_memory),
                        },
                    }
                )

            pod_entries.append(
                {
                    "namespace": pod_namespace,
                    "name": pod_name,
                    "phase": str(status.get("phase", "") or ""),
                    "node": str(spec.get("nodeName", "") or ""),
                    "owner": owner,
                    "containers": container_entries,
                    "pod_totals": {
                        "requests": {
                            "cpu_mcpu": pod_requests_cpu if pod_requests_cpu_known else None,
                            "memory_bytes": pod_requests_memory if pod_requests_memory_known else None,
                            "memory_mib": _bytes_to_mib(pod_requests_memory) if pod_requests_memory_known else None,
                        },
                        "limits": {
                            "cpu_mcpu": pod_limits_cpu if pod_limits_cpu_known else None,
                            "memory_bytes": pod_limits_memory if pod_limits_memory_known else None,
                            "memory_mib": _bytes_to_mib(pod_limits_memory) if pod_limits_memory_known else None,
                        },
                        "usage": {
                            "cpu_mcpu": pod_usage_cpu if pod_usage_cpu_known else None,
                            "memory_bytes": pod_usage_memory if pod_usage_memory_known else None,
                            "memory_mib": _bytes_to_mib(pod_usage_memory) if pod_usage_memory_known else None,
                        },
                    },
                }
            )

        result["summary"] = {
            "pods_analyzed": len(pod_entries),
            "containers_analyzed": containers_analyzed,
            "total_requests": {
                "cpu_mcpu": total_requests_cpu if requests_cpu_known else None,
                "memory_bytes": total_requests_memory if requests_memory_known else None,
                "memory_mib": _bytes_to_mib(total_requests_memory) if requests_memory_known else None,
            },
            "total_limits": {
                "cpu_mcpu": total_limits_cpu if limits_cpu_known else None,
                "memory_bytes": total_limits_memory if limits_memory_known else None,
                "memory_mib": _bytes_to_mib(total_limits_memory) if limits_memory_known else None,
            },
            "total_usage": {
                "cpu_mcpu": total_usage_cpu if usage_cpu_known else None,
                "memory_bytes": total_usage_memory if usage_memory_known else None,
                "memory_mib": _bytes_to_mib(total_usage_memory) if usage_memory_known else None,
            },
        }
        result["pods"] = pod_entries
        return result

    def _render_resource_usage_output(self, summary: dict, *, output_format: str = "auto") -> str:
        fmt = str(output_format or "auto").strip().lower()
        pods = list(summary.get("pods") or [])
        if fmt == "auto":
            if len(pods) > 8 or bool(summary.get("query", {}).get("all_namespaces")):
                fmt = "pod_table"
            else:
                fmt = "json"
        if fmt == "json":
            return json.dumps(summary, indent=2, sort_keys=False)
        if fmt == "container_table":
            return self._render_resource_usage_container_table(summary)
        if fmt == "pod_table":
            return self._render_resource_usage_pod_table(summary)
        return json.dumps(summary, indent=2, sort_keys=False)

    def _render_resource_usage_pod_table(self, summary: dict) -> str:
        query = summary.get("query", {}) if isinstance(summary.get("query"), dict) else {}
        totals = summary.get("summary", {}) if isinstance(summary.get("summary"), dict) else {}
        total_requests = totals.get("total_requests", {}) if isinstance(totals.get("total_requests"), dict) else {}
        total_limits = totals.get("total_limits", {}) if isinstance(totals.get("total_limits"), dict) else {}
        total_usage = totals.get("total_usage", {}) if isinstance(totals.get("total_usage"), dict) else {}

        lines = [
            "Summary:",
            f"- Pods analyzed: {totals.get('pods_analyzed', 0)}",
            f"- Containers analyzed: {totals.get('containers_analyzed', 0)}",
            f"- Total requests: CPU {_format_cpu(total_requests.get('cpu_mcpu'))}, Memory {_format_memory_mib(total_requests.get('memory_mib'))}",
            f"- Total limits: CPU {_format_cpu(total_limits.get('cpu_mcpu'))}, Memory {_format_memory_mib(total_limits.get('memory_mib'))}",
            f"- Total live usage: CPU {_format_cpu(total_usage.get('cpu_mcpu'))}, Memory {_format_memory_mib(total_usage.get('memory_mib'))}",
        ]
        metrics_note = str(summary.get("metrics_note") or "").strip()
        if metrics_note:
            lines.append(f"- Metrics note: {metrics_note}")
        lines.extend(
            [
                "",
                f"Query scope: kind={query.get('kind', '') or 'pod'}, namespace={query.get('namespace', '') or '(all/default)'}, all_namespaces={bool(query.get('all_namespaces'))}",
                "",
                "| Namespace | Pod | Containers | CPU Requests | Memory Requests | CPU Limits | Memory Limits | Live CPU | Live Memory |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for pod in summary.get("pods", []):
            if not isinstance(pod, dict):
                continue
            pod_totals = pod.get("pod_totals", {}) if isinstance(pod.get("pod_totals"), dict) else {}
            requests = pod_totals.get("requests", {}) if isinstance(pod_totals.get("requests"), dict) else {}
            limits = pod_totals.get("limits", {}) if isinstance(pod_totals.get("limits"), dict) else {}
            usage = pod_totals.get("usage", {}) if isinstance(pod_totals.get("usage"), dict) else {}
            lines.append(
                "| {namespace} | {name} | {containers} | {cpu_req} | {mem_req} | {cpu_lim} | {mem_lim} | {cpu_use} | {mem_use} |".format(
                    namespace=str(pod.get("namespace", "") or ""),
                    name=str(pod.get("name", "") or ""),
                    containers=len(pod.get("containers", []) or []),
                    cpu_req=_format_cpu(requests.get("cpu_mcpu")),
                    mem_req=_format_memory_mib(requests.get("memory_mib")),
                    cpu_lim=_format_cpu(limits.get("cpu_mcpu")),
                    mem_lim=_format_memory_mib(limits.get("memory_mib")),
                    cpu_use=_format_cpu(usage.get("cpu_mcpu")),
                    mem_use=_format_memory_mib(usage.get("memory_mib")),
                )
            )
        return "\n".join(lines).strip()

    def _render_resource_usage_container_table(self, summary: dict) -> str:
        totals = summary.get("summary", {}) if isinstance(summary.get("summary"), dict) else {}
        lines = [
            "Summary:",
            f"- Pods analyzed: {totals.get('pods_analyzed', 0)}",
            f"- Containers analyzed: {totals.get('containers_analyzed', 0)}",
            "",
            "| Namespace | Pod | Container | CPU Requests | Memory Requests | CPU Limits | Memory Limits | Live CPU | Live Memory |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for pod in summary.get("pods", []):
            if not isinstance(pod, dict):
                continue
            for container in pod.get("containers", []):
                if not isinstance(container, dict):
                    continue
                requests = container.get("requests", {}) if isinstance(container.get("requests"), dict) else {}
                limits = container.get("limits", {}) if isinstance(container.get("limits"), dict) else {}
                usage = container.get("usage", {}) if isinstance(container.get("usage"), dict) else {}
                lines.append(
                    "| {namespace} | {pod} | {container} | {cpu_req} | {mem_req} | {cpu_lim} | {mem_lim} | {cpu_use} | {mem_use} |".format(
                        namespace=str(pod.get("namespace", "") or ""),
                        pod=str(pod.get("name", "") or ""),
                        container=str(container.get("name", "") or ""),
                        cpu_req=_format_cpu(requests.get("cpu_mcpu")),
                        mem_req=_format_memory_mib(requests.get("memory_mib")),
                        cpu_lim=_format_cpu(limits.get("cpu_mcpu")),
                        mem_lim=_format_memory_mib(limits.get("memory_mib")),
                        cpu_use=_format_cpu(usage.get("cpu_mcpu")),
                        mem_use=_format_memory_mib(usage.get("memory_mib")),
                    )
                )
        return "\n".join(lines).strip()

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

    def _validate_exec_target(
        self,
        pod: str,
        namespace: str | None = None,
        container: str | None = None,
    ) -> str | None:
        pod_data = self.get_resource_json("pod", pod, namespace or self.namespace)
        if isinstance(pod_data, str):
            return pod_data

        containers = [
            str(item.get("name", "") or "").strip()
            for item in pod_data.get("spec", {}).get("containers", [])
            if str(item.get("name", "") or "").strip()
        ]
        if container:
            if container not in containers:
                valid = ", ".join(containers) or "<none>"
                return (
                    f"ERROR: container '{container}' is not valid for pod '{pod}'. "
                    f"Valid containers: {valid}"
                )
        elif len(containers) > 1:
            return (
                f"ERROR: pod '{pod}' has multiple containers. "
                f"Specify one of: {', '.join(containers)}"
            )
        return None

    def exec_args(
        self,
        pod: str,
        command_args: list[str],
        namespace: str | None = None,
        container: str | None = None,
        *,
        timeout: int = 30,
        include_stderr_on_success: bool = False,
    ) -> str:
        target_error = self._validate_exec_target(pod, namespace, container)
        if target_error:
            return target_error
        args = ["exec", pod]
        if container:
            args.extend(["-c", container])
        args.append("--")
        args.extend(command_args)
        return self.run(
            args,
            namespace=namespace or self.namespace,
            timeout=timeout,
            include_stderr_on_success=include_stderr_on_success,
        )

    def exec_command(
        self,
        pod: str,
        command: str,
        namespace: str | None = None,
        container: str | None = None,
    ) -> str:
        return self.exec_args(
            pod,
            shlex.split(command),
            namespace=namespace,
            container=container,
            timeout=30,
            include_stderr_on_success=True,
        )

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
