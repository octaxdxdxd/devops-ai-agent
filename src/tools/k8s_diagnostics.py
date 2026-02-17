"""Read-only Kubernetes diagnostic tools.

These tools are intentionally non-mutating and provider-agnostic:
- EKS / AKS / GKE / on-prem
- Any environment where kubectl auth + context are configured
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

from langchain_core.tools import tool

from ..config import Config
from .k8s_common import (
    ensure_kubectl_installed,
    is_valid_k8s_name,
    kubectl_base_args,
    kubectl_not_found_msg,
    kube_access_help,
    run_kubectl,
    truncate_text,
)


def _run_readonly(sub_args: list[str]) -> str:
    if not ensure_kubectl_installed():
        return kubectl_not_found_msg()
    args = kubectl_base_args() + sub_args
    code, out, err = run_kubectl(args)
    if code != 0:
        return kube_access_help(err or out)
    return truncate_text(out or "(no output)")


def _run_json(sub_args: list[str]) -> tuple[int, dict[str, Any] | None, str]:
    if not ensure_kubectl_installed():
        return 127, None, kubectl_not_found_msg()
    args = kubectl_base_args() + sub_args + ["-o", "json"]
    code, out, err = run_kubectl(args)
    if code != 0:
        return code, None, err or out
    try:
        return 0, json.loads(out or "{}"), ""
    except json.JSONDecodeError as e:
        return 1, None, f"Failed to parse kubectl JSON output: {e}"


def _validate_namespace(namespace: str) -> str | None:
    ns = (namespace or "").strip()
    if not ns:
        return None
    if not is_valid_k8s_name(ns):
        return f"❌ Invalid namespace: {namespace!r}."
    return None


def _fmt_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "(no rows)"
    widths = [len(h) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(c))
    sep = "  "
    header = sep.join(h.ljust(widths[i]) for i, h in enumerate(headers))
    line = sep.join("-" * widths[i] for i, _h in enumerate(headers))
    body = "\n".join(sep.join(c.ljust(widths[i]) for i, c in enumerate(r)) for r in rows)
    return f"{header}\n{line}\n{body}"


def _discover_pod_namespaces(pod_name: str) -> tuple[list[str], str | None]:
    """Return namespaces containing pod_name across the cluster."""
    if not ensure_kubectl_installed():
        return [], kubectl_not_found_msg()

    args = kubectl_base_args(all_namespaces=True) + ["get", "pods", "-o", "json"]
    code, out, err = run_kubectl(args)
    if code != 0:
        return [], kube_access_help(err or out)

    try:
        data = json.loads(out or "{}")
    except json.JSONDecodeError as e:
        return [], f"❌ Failed to parse kubectl output: {e}"

    namespaces: list[str] = []
    for item in data.get("items", []):
        meta = item.get("metadata", {})
        if str(meta.get("name", "")) == pod_name:
            ns = str(meta.get("namespace", "")).strip()
            if ns:
                namespaces.append(ns)
    return namespaces, None


def _resolve_namespace_for_pod(pod_name: str, namespace: str) -> tuple[str | None, str | None]:
    """Resolve best namespace for a pod name.

    Rules:
    - If namespace is provided and specific, prefer it if pod exists there.
    - If namespace is empty/auto/all/any or pod not found there, search all namespaces.
    - If exactly one match exists, use it.
    - If none or many, return an explicit message.
    """
    requested = (namespace or "").strip()
    auto_mode = requested.lower() in {"", "auto", "any", "all"}

    # Fast path: specific namespace requested; verify pod exists there.
    if not auto_mode:
        args = kubectl_base_args(namespace=requested) + ["get", "pod", pod_name, "-o", "name"]
        code, out, err = run_kubectl(args)
        if code == 0 and (out or "").strip():
            return requested, None

    namespaces, discover_err = _discover_pod_namespaces(pod_name)
    if discover_err:
        return None, discover_err

    unique = sorted(set(namespaces))
    if not unique:
        if not auto_mode:
            return None, f"❌ Pod '{pod_name}' not found in namespace '{requested}' or any namespace."
        return None, f"❌ Pod '{pod_name}' not found in any namespace."

    if len(unique) == 1:
        return unique[0], None

    return None, (
        f"❌ Pod '{pod_name}' exists in multiple namespaces: {', '.join(unique)}. "
        "Please specify namespace explicitly."
    )


@tool
def k8s_current_context() -> str:
    """Get current kubectl context and namespace."""
    if not ensure_kubectl_installed():
        return kubectl_not_found_msg()

    code_ctx, out_ctx, err_ctx = run_kubectl(kubectl_base_args() + ["config", "current-context"])
    if code_ctx != 0:
        return kube_access_help(err_ctx or out_ctx)

    code_ns, out_ns, _err_ns = run_kubectl(
        kubectl_base_args() + ["config", "view", "--minify", "--output", "jsonpath={..namespace}"]
    )
    namespace = out_ns or Config.K8S_DEFAULT_NAMESPACE
    if code_ns != 0:
        namespace = Config.K8S_DEFAULT_NAMESPACE

    return f"Current context: {out_ctx}\nDefault namespace: {namespace}"


@tool
def k8s_list_contexts() -> str:
    """List all kubeconfig contexts and indicate current one."""
    return _run_readonly(["config", "get-contexts"])


@tool
def k8s_cluster_info() -> str:
    """Show Kubernetes cluster control plane/service endpoints."""
    return _run_readonly(["cluster-info"])


@tool
def k8s_version() -> str:
    """Show client/server Kubernetes versions."""
    return _run_readonly(["version", "--short"])


@tool
def k8s_api_resources() -> str:
    """List API resources available in this cluster."""
    return _run_readonly(["api-resources"])


@tool
def k8s_list_namespaces() -> str:
    """List namespaces in the cluster."""
    return _run_readonly(["get", "namespaces", "-o", "wide"])


@tool
def k8s_list_nodes(show_labels: bool = False) -> str:
    """List cluster nodes with status and versions."""
    args = ["get", "nodes", "-o", "wide"]
    if show_labels:
        args.append("--show-labels")
    return _run_readonly(args)


@tool
def k8s_top_nodes() -> str:
    """Show node CPU/memory usage (requires metrics-server)."""
    return _run_readonly(["top", "nodes"])


@tool
def k8s_list_pods(
    namespace: str = Config.K8S_DEFAULT_NAMESPACE,
    label_selector: str = "",
    field_selector: str = "",
    limit: int = 100,
) -> str:
    """List pods in a namespace (or all namespaces if namespace='all')."""
    ns = (namespace or "").strip()
    if ns.lower() != "all":
        err = _validate_namespace(ns)
        if err:
            return err

    if not ensure_kubectl_installed():
        return kubectl_not_found_msg()

    args = kubectl_base_args(all_namespaces=(ns.lower() == "all"), namespace=None if ns.lower() == "all" else ns)
    args += ["get", "pods", "-o", "wide"]
    if label_selector:
        args += ["-l", label_selector]
    if field_selector:
        args += ["--field-selector", field_selector]
    if limit > 0:
        args += ["--chunk-size", str(min(max(limit, 1), 500))]

    code, out, err = run_kubectl(args)
    if code != 0:
        return kube_access_help(err or out)
    return truncate_text(out or "(no pods)")


@tool
def k8s_find_pods(name_contains: str, namespace: str = "all", limit: int = 50) -> str:
    """Find pods by partial name match across one namespace or all namespaces."""
    query = (name_contains or "").strip().lower()
    if not query:
        return "❌ name_contains is required."

    ns = (namespace or "all").strip()
    if ns.lower() != "all":
        err = _validate_namespace(ns)
        if err:
            return err

    if not ensure_kubectl_installed():
        return kubectl_not_found_msg()

    args = kubectl_base_args(all_namespaces=(ns.lower() == "all"), namespace=None if ns.lower() == "all" else ns)
    args += ["get", "pods", "-o", "json"]

    code, out, err = run_kubectl(args)
    if code != 0:
        return kube_access_help(err or out)

    try:
        data = json.loads(out or "{}")
    except json.JSONDecodeError as e:
        return f"❌ Failed to parse kubectl output: {e}"

    rows: list[list[str]] = []
    for item in data.get("items", []):
        meta = item.get("metadata", {})
        status = item.get("status", {})
        pod_name = str(meta.get("name", ""))
        if query not in pod_name.lower():
            continue
        rows.append([
            str(meta.get("namespace", "")),
            pod_name,
            str(status.get("phase", "")),
            str(status.get("podIP", "")),
            str(meta.get("creationTimestamp", "")),
        ])
        if len(rows) >= max(1, min(limit, 500)):
            break

    if not rows:
        return f"No pods found containing '{name_contains}'."

    return truncate_text(_fmt_table(["NAMESPACE", "POD", "PHASE", "POD_IP", "CREATED"], rows))


@tool
def k8s_describe_pod(pod_name: str, namespace: str = Config.K8S_DEFAULT_NAMESPACE) -> str:
    """Describe one pod with events and condition details."""
    pod_name = (pod_name or "").strip()
    namespace = (namespace or Config.K8S_DEFAULT_NAMESPACE).strip() or Config.K8S_DEFAULT_NAMESPACE
    if not is_valid_k8s_name(pod_name):
        return f"❌ Invalid pod name: {pod_name!r}."

    if not ensure_kubectl_installed():
        return kubectl_not_found_msg()

    resolved_ns, resolve_err = _resolve_namespace_for_pod(pod_name, namespace)
    if resolve_err:
        return resolve_err
    if not resolved_ns:
        return f"❌ Could not resolve namespace for pod '{pod_name}'."

    err = _validate_namespace(resolved_ns)
    if err:
        return err

    args = kubectl_base_args(namespace=resolved_ns) + ["describe", "pod", pod_name]
    code, out, stderr = run_kubectl(args)
    if code != 0:
        return kube_access_help(stderr or out)
    return truncate_text(out or "(no output)")


@tool
def k8s_get_pod_logs(
    pod_name: str,
    namespace: str = Config.K8S_DEFAULT_NAMESPACE,
    container: str = "",
    tail_lines: int = 300,
    since_minutes: int = 60,
    previous: bool = False,
) -> str:
    """Get logs for a Kubernetes pod/container from cluster (not local files)."""
    pod_name = (pod_name or "").strip()
    namespace = (namespace or Config.K8S_DEFAULT_NAMESPACE).strip() or Config.K8S_DEFAULT_NAMESPACE
    container = (container or "").strip()

    if not is_valid_k8s_name(pod_name):
        return f"❌ Invalid pod name: {pod_name!r}."

    if not ensure_kubectl_installed():
        return kubectl_not_found_msg()

    resolved_ns, resolve_err = _resolve_namespace_for_pod(pod_name, namespace)
    if resolve_err:
        return resolve_err
    if not resolved_ns:
        return f"❌ Could not resolve namespace for pod '{pod_name}'."

    err = _validate_namespace(resolved_ns)
    if err:
        return err

    args = kubectl_base_args(namespace=resolved_ns) + ["logs", pod_name]
    if container:
        args += ["-c", container]
    if tail_lines > 0:
        args += ["--tail", str(min(max(tail_lines, 1), 5000))]
    if since_minutes > 0:
        args += ["--since", f"{min(max(since_minutes, 1), 10080)}m"]
    if previous:
        args.append("--previous")

    code, out, stderr = run_kubectl(args)
    if code != 0:
        return kube_access_help(stderr or out)

    header = (
        f"Pod: {pod_name}\nNamespace: {resolved_ns}\n"
        f"Container: {container or '(default)'}\nTail: {tail_lines}\nSince: {since_minutes}m\nPrevious: {previous}\n"
    )
    return truncate_text(f"{header}\n{out or '(no logs returned)'}")


@tool
def k8s_top_pods(namespace: str = Config.K8S_DEFAULT_NAMESPACE) -> str:
    """Show pod CPU/memory usage for a namespace (requires metrics-server)."""
    namespace = (namespace or Config.K8S_DEFAULT_NAMESPACE).strip() or Config.K8S_DEFAULT_NAMESPACE
    if namespace.lower() != "all":
        err = _validate_namespace(namespace)
        if err:
            return err

    if not ensure_kubectl_installed():
        return kubectl_not_found_msg()

    args = kubectl_base_args(all_namespaces=(namespace.lower() == "all"), namespace=None if namespace.lower() == "all" else namespace)
    args += ["top", "pods"]
    code, out, stderr = run_kubectl(args)
    if code != 0:
        return kube_access_help(stderr or out)
    return truncate_text(out or "(no output)")


@tool
def k8s_list_deployments(namespace: str = Config.K8S_DEFAULT_NAMESPACE, label_selector: str = "") -> str:
    """List deployments in a namespace."""
    namespace = (namespace or Config.K8S_DEFAULT_NAMESPACE).strip() or Config.K8S_DEFAULT_NAMESPACE
    if namespace.lower() != "all":
        err = _validate_namespace(namespace)
        if err:
            return err

    if not ensure_kubectl_installed():
        return kubectl_not_found_msg()

    args = kubectl_base_args(all_namespaces=(namespace.lower() == "all"), namespace=None if namespace.lower() == "all" else namespace)
    args += ["get", "deployments", "-o", "wide"]
    if label_selector:
        args += ["-l", label_selector]

    code, out, stderr = run_kubectl(args)
    if code != 0:
        return kube_access_help(stderr or out)
    return truncate_text(out or "(no deployments)")


@tool
def k8s_describe_deployment(deployment_name: str, namespace: str = Config.K8S_DEFAULT_NAMESPACE) -> str:
    """Describe one deployment."""
    deployment_name = (deployment_name or "").strip()
    namespace = (namespace or Config.K8S_DEFAULT_NAMESPACE).strip() or Config.K8S_DEFAULT_NAMESPACE
    if not is_valid_k8s_name(deployment_name):
        return f"❌ Invalid deployment name: {deployment_name!r}."
    err = _validate_namespace(namespace)
    if err:
        return err

    if not ensure_kubectl_installed():
        return kubectl_not_found_msg()

    args = kubectl_base_args(namespace=namespace) + ["describe", "deployment", deployment_name]
    code, out, stderr = run_kubectl(args)
    if code != 0:
        return kube_access_help(stderr or out)
    return truncate_text(out or "(no output)")


@tool
def k8s_list_statefulsets(namespace: str = Config.K8S_DEFAULT_NAMESPACE) -> str:
    """List statefulsets in a namespace."""
    namespace = (namespace or Config.K8S_DEFAULT_NAMESPACE).strip() or Config.K8S_DEFAULT_NAMESPACE
    if namespace.lower() != "all":
        err = _validate_namespace(namespace)
        if err:
            return err
    return _run_readonly(["get", "statefulsets", "-A" if namespace.lower() == "all" else "-n", namespace] if namespace.lower() != "all" else ["get", "statefulsets", "-A"])


@tool
def k8s_list_daemonsets(namespace: str = Config.K8S_DEFAULT_NAMESPACE) -> str:
    """List daemonsets in a namespace."""
    namespace = (namespace or Config.K8S_DEFAULT_NAMESPACE).strip() or Config.K8S_DEFAULT_NAMESPACE
    if namespace.lower() != "all":
        err = _validate_namespace(namespace)
        if err:
            return err
    return _run_readonly(["get", "daemonsets", "-A" if namespace.lower() == "all" else "-n", namespace] if namespace.lower() != "all" else ["get", "daemonsets", "-A"])


@tool
def k8s_list_services(namespace: str = Config.K8S_DEFAULT_NAMESPACE) -> str:
    """List services in a namespace."""
    namespace = (namespace or Config.K8S_DEFAULT_NAMESPACE).strip() or Config.K8S_DEFAULT_NAMESPACE
    if namespace.lower() != "all":
        err = _validate_namespace(namespace)
        if err:
            return err
    return _run_readonly(["get", "services", "-A" if namespace.lower() == "all" else "-n", namespace, "-o", "wide"] if namespace.lower() != "all" else ["get", "services", "-A", "-o", "wide"])


@tool
def k8s_list_ingresses(namespace: str = Config.K8S_DEFAULT_NAMESPACE) -> str:
    """List ingress resources."""
    namespace = (namespace or Config.K8S_DEFAULT_NAMESPACE).strip() or Config.K8S_DEFAULT_NAMESPACE
    if namespace.lower() != "all":
        err = _validate_namespace(namespace)
        if err:
            return err
    return _run_readonly(["get", "ingress", "-A" if namespace.lower() == "all" else "-n", namespace] if namespace.lower() != "all" else ["get", "ingress", "-A"])


@tool
def k8s_list_hpa(namespace: str = Config.K8S_DEFAULT_NAMESPACE) -> str:
    """List horizontal pod autoscalers."""
    namespace = (namespace or Config.K8S_DEFAULT_NAMESPACE).strip() or Config.K8S_DEFAULT_NAMESPACE
    if namespace.lower() != "all":
        err = _validate_namespace(namespace)
        if err:
            return err
    return _run_readonly(["get", "hpa", "-A" if namespace.lower() == "all" else "-n", namespace] if namespace.lower() != "all" else ["get", "hpa", "-A"])


@tool
def k8s_get_events(namespace: str = "all", since_minutes: int = 60, limit: int = 200) -> str:
    """Get recent Kubernetes events, optionally filtered by namespace and time window."""
    ns = (namespace or "all").strip()
    if ns.lower() != "all":
        err = _validate_namespace(ns)
        if err:
            return err

    code, data, err = _run_json(["get", "events", "-A" if ns.lower() == "all" else "-n", ns] if ns.lower() != "all" else ["get", "events", "-A"])
    if code != 0 or data is None:
        return kube_access_help(err)

    cutoff = datetime.now(timezone.utc) - timedelta(minutes=max(1, min(since_minutes, 10080)))
    rows: list[list[str]] = []
    for item in data.get("items", []):
        meta = item.get("metadata", {})
        involved = item.get("involvedObject", {})
        ts_raw = item.get("lastTimestamp") or item.get("eventTime") or meta.get("creationTimestamp")
        ts = ts_raw or ""

        # Parse RFC3339-ish timestamps if possible.
        include = True
        if ts_raw:
            try:
                dt = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
                include = dt >= cutoff
            except ValueError:
                include = True
        if not include:
            continue

        rows.append([
            str(meta.get("namespace", "")),
            str(item.get("type", "")),
            str(item.get("reason", "")),
            str(involved.get("kind", "")),
            str(involved.get("name", "")),
            ts,
            str(item.get("message", ""))[:120],
        ])

    rows.sort(key=lambda r: r[5], reverse=True)
    rows = rows[: max(1, min(limit, 1000))]

    if not rows:
        return "No recent events found for the given filters."

    return truncate_text(_fmt_table(["NS", "TYPE", "REASON", "KIND", "NAME", "TIMESTAMP", "MESSAGE"], rows))


@tool
def k8s_get_resource_quotas(namespace: str = Config.K8S_DEFAULT_NAMESPACE) -> str:
    """List resource quotas for a namespace."""
    namespace = (namespace or Config.K8S_DEFAULT_NAMESPACE).strip() or Config.K8S_DEFAULT_NAMESPACE
    err = _validate_namespace(namespace)
    if err:
        return err
    return _run_readonly(["get", "resourcequota", "-n", namespace])


@tool
def k8s_get_pvcs(namespace: str = Config.K8S_DEFAULT_NAMESPACE) -> str:
    """List PersistentVolumeClaims in a namespace."""
    namespace = (namespace or Config.K8S_DEFAULT_NAMESPACE).strip() or Config.K8S_DEFAULT_NAMESPACE
    if namespace.lower() != "all":
        err = _validate_namespace(namespace)
        if err:
            return err
    return _run_readonly(["get", "pvc", "-A" if namespace.lower() == "all" else "-n", namespace] if namespace.lower() != "all" else ["get", "pvc", "-A"])


@tool
def k8s_get_crashloop_pods(namespace: str = "all", limit: int = 100) -> str:
    """Find pods likely in crash/restart trouble by inspecting container statuses."""
    ns = (namespace or "all").strip()
    if ns.lower() != "all":
        err = _validate_namespace(ns)
        if err:
            return err

    if not ensure_kubectl_installed():
        return kubectl_not_found_msg()

    args = kubectl_base_args(all_namespaces=(ns.lower() == "all"), namespace=None if ns.lower() == "all" else ns)
    args += ["get", "pods", "-o", "json"]
    code, out, stderr = run_kubectl(args)
    if code != 0:
        return kube_access_help(stderr or out)

    try:
        data = json.loads(out or "{}")
    except json.JSONDecodeError as e:
        return f"❌ Failed to parse kubectl output: {e}"

    rows: list[list[str]] = []
    for item in data.get("items", []):
        meta = item.get("metadata", {})
        st = item.get("status", {})
        statuses = st.get("containerStatuses") or []
        for c in statuses:
            waiting = (c.get("state") or {}).get("waiting") or {}
            reason = waiting.get("reason") or ""
            restart_count = int(c.get("restartCount") or 0)
            if reason in {"CrashLoopBackOff", "ImagePullBackOff", "ErrImagePull"} or restart_count >= 3:
                rows.append([
                    str(meta.get("namespace", "")),
                    str(meta.get("name", "")),
                    str(c.get("name", "")),
                    reason or "(none)",
                    str(restart_count),
                    str(st.get("phase", "")),
                ])

    if not rows:
        return "No crash-loop candidates found."

    rows = rows[: max(1, min(limit, 1000))]
    return truncate_text(_fmt_table(["NS", "POD", "CONTAINER", "REASON", "RESTARTS", "PHASE"], rows))


def get_k8s_read_tools() -> list:
    """Return all read-only Kubernetes diagnostic tools."""
    return [
        k8s_current_context,
        k8s_list_contexts,
        k8s_cluster_info,
        k8s_version,
        k8s_api_resources,
        k8s_list_namespaces,
        k8s_list_nodes,
        k8s_top_nodes,
        k8s_list_pods,
        k8s_find_pods,
        k8s_describe_pod,
        k8s_get_pod_logs,
        k8s_top_pods,
        k8s_list_deployments,
        k8s_describe_deployment,
        k8s_list_statefulsets,
        k8s_list_daemonsets,
        k8s_list_services,
        k8s_list_ingresses,
        k8s_list_hpa,
        k8s_get_events,
        k8s_get_resource_quotas,
        k8s_get_pvcs,
        k8s_get_crashloop_pods,
    ]
