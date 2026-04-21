"""Kubernetes read-only LangChain tools."""

from __future__ import annotations

import json
import shlex

from langchain_core.tools import tool

from ..infra.k8s_client import K8sClient
from ..policy import guard_k8s_read_tool, parse_kubectl_command
from .output import compress_output


_GLOBAL_FLAG_NAMES = frozenset({
    "-n",
    "--namespace",
    "-A",
    "--all-namespaces",
    "--context",
    "--cluster",
    "--user",
    "--server",
    "--kubeconfig",
    "--as",
    "--as-group",
    "--token",
    "--request-timeout",
})


def _extract_exec_args(command: str) -> dict[str, str]:
    metadata = parse_kubectl_command(command)
    args = list(metadata.get("subcommand_args", []))
    namespace = str(metadata.get("namespace", "") or "")
    pod = ""
    container = ""
    command_parts: list[str] = []

    index = 0
    while index < len(args):
        part = args[index]
        if part == "--":
            command_parts = args[index + 1 :]
            break
        if part in {"-n", "--namespace"} and index + 1 < len(args):
            namespace = args[index + 1]
            index += 2
            continue
        if part.startswith("--namespace="):
            namespace = part.split("=", 1)[1]
            index += 1
            continue
        if part in _GLOBAL_FLAG_NAMES:
            index += 2 if part in {"-n", "--namespace", "--context", "--cluster", "--user", "--server", "--kubeconfig", "--as", "--as-group", "--token", "--request-timeout"} else 1
            continue
        if part in {"-c", "--container"} and index + 1 < len(args):
            container = args[index + 1]
            index += 2
            continue
        if part.startswith("--container="):
            container = part.split("=", 1)[1]
            index += 1
            continue
        if part.startswith("-"):
            index += 1
            continue
        if not pod:
            pod = part
        index += 1

    result = {
        "pod": pod,
        "command": shlex.join(command_parts) if command_parts else "",
    }
    if namespace:
        result["namespace"] = namespace
    if container:
        result["container"] = container
    return result


def _typed_write_guidance_for_kubectl(command: str) -> str:
    metadata = parse_kubectl_command(command)
    subcommand = str(metadata.get("subcommand", "") or "").strip().lower()
    if not subcommand:
        return ""
    if subcommand == "exec":
        args = _extract_exec_args(command)
        if not args.get("pod") or not args.get("command"):
            return (
                "Use the approval-gated typed write tool `k8s_exec_in_pod` instead of raw kubectl exec. "
                "Provide the pod name, namespace, optional container name, and the exact command."
            )
        return (
            "This command maps to typed write tool `k8s_exec_in_pod`. "
            f"Use `propose_action` with args {json.dumps(args, sort_keys=True)} instead of raw kubectl."
        )
    if subcommand == "scale":
        return "This command maps to typed write tool `k8s_scale`."
    if subcommand == "delete":
        return "This command maps to typed write tool `k8s_delete_resource`."
    if subcommand == "apply":
        return "This command maps to typed write tool `k8s_apply_manifest`."
    if subcommand in {"cordon", "uncordon"}:
        return "This command maps to typed write tool `k8s_cordon_node`."
    if subcommand == "drain":
        return "This command maps to typed write tool `k8s_drain_node`."
    if subcommand == "rollout":
        sub_args = list(metadata.get("subcommand_args", []))
        rollout_op = str(sub_args[0] if sub_args else "").strip().lower()
        if rollout_op == "restart":
            return "This command maps to typed write tool `k8s_restart_workload_safely` or `k8s_rollout_restart`."
        if rollout_op == "undo":
            return "This command maps to typed write tool `k8s_rollout_undo`."
    return "Use an approval-gated typed write tool instead of raw kubectl."


def create_k8s_read_tools(k8s: K8sClient) -> list:
    """Return a list of read-only K8s LangChain tools bound to the given client."""

    @tool
    def k8s_get_resources(
        kind: str,
        namespace: str = "",
        name: str = "",
        label_selector: str = "",
        all_namespaces: bool = False,
    ) -> str:
        """List or get Kubernetes resources.

        Args:
            kind: Resource type: pod, deployment, service, ingress, node, configmap, secret, pvc, job, cronjob, daemonset, statefulset, hpa, namespace, replicaset, endpoints
            namespace: K8s namespace (empty = default namespace)
            name: Specific resource name (empty = list all)
            label_selector: Label filter e.g. 'app=nginx,tier=frontend'
            all_namespaces: Set True to search across all namespaces
        """
        policy_error = guard_k8s_read_tool(
            "k8s_get_resources",
            namespace=namespace,
            all_namespaces=all_namespaces,
        )
        if policy_error:
            return f"ERROR: {policy_error}"
        return compress_output(
            k8s.get_resources(
                kind,
                namespace=namespace or None,
                name=name or None,
                label_selector=label_selector or None,
                all_namespaces=all_namespaces,
            )
        )

    @tool
    def k8s_describe_resource(kind: str, name: str, namespace: str = "") -> str:
        """Get detailed description of a Kubernetes resource including events and conditions.

        Args:
            kind: Resource type (pod, deployment, service, node, etc.)
            name: Resource name
            namespace: K8s namespace (empty = default)
        """
        policy_error = guard_k8s_read_tool("k8s_describe_resource", namespace=namespace)
        if policy_error:
            return f"ERROR: {policy_error}"
        return compress_output(
            k8s.describe(kind, name, namespace or None),
            max_lines=80,
            max_chars=5000,
        )

    @tool
    def k8s_get_pod_logs(
        pod: str,
        namespace: str = "",
        container: str = "",
        tail_lines: int = 100,
        since: str = "",
    ) -> str:
        """Get logs from a Kubernetes pod.

        Args:
            pod: Pod name
            namespace: K8s namespace (empty = default)
            container: Container name (needed for multi-container pods)
            tail_lines: Number of lines from the end (default 100)
            since: Time duration like '5m', '1h', '30s'
        """
        policy_error = guard_k8s_read_tool("k8s_get_pod_logs", namespace=namespace)
        if policy_error:
            return f"ERROR: {policy_error}"
        return compress_output(
            k8s.get_logs(pod, namespace or None, container or None, tail_lines, since or None)
        )

    @tool
    def k8s_get_events(
        namespace: str = "",
        field_selector: str = "",
        all_namespaces: bool = False,
    ) -> str:
        """Get Kubernetes events sorted by time. Useful for diagnosing issues.

        Args:
            namespace: K8s namespace (empty = default)
            field_selector: Filter e.g. 'involvedObject.name=my-pod' or 'type=Warning'
            all_namespaces: Set True to get events from all namespaces
        """
        policy_error = guard_k8s_read_tool(
            "k8s_get_events",
            namespace=namespace,
            all_namespaces=all_namespaces,
        )
        if policy_error:
            return f"ERROR: {policy_error}"
        return compress_output(
            k8s.get_events(namespace or None, field_selector or None, all_namespaces=all_namespaces)
        )

    @tool
    def k8s_get_resource_usage(resource_type: str = "pods", namespace: str = "", name: str = "") -> str:
        """Get CPU and memory usage via 'kubectl top'. Requires metrics-server.

        Args:
            resource_type: 'pods' or 'nodes'
            namespace: K8s namespace (empty = default, ignored for nodes)
            name: Specific pod or node name (empty = all)
        """
        policy_error = guard_k8s_read_tool("k8s_get_resource_usage", namespace=namespace)
        if policy_error:
            return f"ERROR: {policy_error}"
        return compress_output(k8s.top(resource_type, namespace or None, name or None))

    @tool
    def k8s_get_rollout_history(kind: str, name: str, namespace: str = "") -> str:
        """Get rollout history for a deployment or statefulset. Shows recent revisions.

        Args:
            kind: 'deployment' or 'statefulset'
            name: Resource name
            namespace: K8s namespace (empty = default)
        """
        policy_error = guard_k8s_read_tool("k8s_get_rollout_history", namespace=namespace)
        if policy_error:
            return f"ERROR: {policy_error}"
        return compress_output(k8s.rollout_history(kind, name, namespace or None))

    @tool
    def k8s_get_contexts() -> str:
        """List all available kubectl contexts and the current active context."""
        return compress_output(k8s.get_contexts())

    @tool
    def k8s_get_namespaces() -> str:
        """List all Kubernetes namespaces in the cluster."""
        return compress_output(k8s.get_namespaces())

    @tool
    def k8s_get_resource_yaml(kind: str, name: str, namespace: str = "") -> str:
        """Get the full YAML definition of a Kubernetes resource.

        Args:
            kind: Resource type
            name: Resource name
            namespace: K8s namespace (empty = default)
        """
        policy_error = guard_k8s_read_tool("k8s_get_resource_yaml", namespace=namespace)
        if policy_error:
            return f"ERROR: {policy_error}"
        return compress_output(
            k8s.get_resource_yaml(kind, name, namespace or None),
            max_lines=100,
            max_chars=6000,
        )

    # Read-only kubectl subcommands (whitelist).  Anything not listed
    # here is considered a write / mutating operation and will be blocked.
    _KUBECTL_READ_SUBCOMMANDS = frozenset({
        "get", "describe", "logs", "top", "explain", "api-resources",
        "api-versions", "version", "cluster-info", "config", "auth",
        "diff", "events", "wait",
    })

    @tool
    def k8s_run_kubectl(command: str) -> str:
        """Run an arbitrary **read-only** kubectl command for queries not covered by other tools.

        The command string should NOT include the leading 'kubectl' — it is added automatically.
        Examples: 'get crd', 'api-resources', 'get networkpolicy -A', 'auth can-i list pods'.
        WRITE operations (patch, delete, apply, create, scale, etc.) are blocked.
        To modify resources, describe what change you want and let the user request it as an action.

        Args:
            command: kubectl arguments (without the leading 'kubectl')
        """
        policy_error = guard_k8s_read_tool("k8s_run_kubectl", command=command)
        if policy_error:
            return f"ERROR: {policy_error}"
        try:
            parts = shlex.split(command.strip())
        except ValueError as exc:
            return f"ERROR: Invalid command syntax: {exc}"
        if not parts:
            return "ERROR: Empty command"
        metadata = parse_kubectl_command(command)
        subcommand = str(metadata.get("subcommand", "") or "").strip().lower()
        if not subcommand:
            return "ERROR: Could not determine kubectl subcommand from command input."
        # Safety: reject shell operators
        dangerous = ["|", ">", "<", ";", "&", "$(", "`"]
        has_dangerous_shell_operators = any(d in command for d in dangerous)
        # Safety: block mutating commands — only read-only subcommands allowed
        if subcommand not in _KUBECTL_READ_SUBCOMMANDS:
            guidance = _typed_write_guidance_for_kubectl(command)
            return (
                f"ERROR: 'kubectl {subcommand}' is a mutating operation and is blocked in read-only mode. "
                f"{guidance} Do not provide raw kubectl commands for approval when a typed write tool exists."
            )
        if has_dangerous_shell_operators:
            return "ERROR: Command contains potentially dangerous shell operators"
        return compress_output(k8s.run(parts))

    return [
        k8s_get_resources,
        k8s_describe_resource,
        k8s_get_pod_logs,
        k8s_get_events,
        k8s_get_resource_usage,
        k8s_get_rollout_history,
        k8s_get_contexts,
        k8s_get_namespaces,
        k8s_get_resource_yaml,
        k8s_run_kubectl,
    ]
