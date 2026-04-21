"""Kubernetes write (mutating) LangChain tools."""

from __future__ import annotations

from langchain_core.tools import tool

from ..infra.k8s_client import K8sClient
from ..policy import guard_k8s_write_tool


def create_k8s_write_tools(k8s: K8sClient) -> list:
    """Return mutating K8s tools. These require explicit user approval before execution."""

    @tool
    def k8s_scale(kind: str, name: str, replicas: int, namespace: str = "") -> str:
        """Scale a deployment or statefulset to a given replica count.

        Args:
            kind: 'deployment' or 'statefulset'
            name: Resource name
            replicas: Desired replica count
            namespace: K8s namespace (empty = default)
        """
        policy_error = guard_k8s_write_tool("k8s_scale", namespace=namespace)
        if policy_error:
            return f"ERROR: {policy_error}"
        return k8s.scale(kind, name, replicas, namespace or None)

    @tool
    def k8s_rollout_restart(kind: str, name: str, namespace: str = "") -> str:
        """Trigger a rolling restart of a deployment or statefulset.

        Args:
            kind: 'deployment' or 'statefulset'
            name: Resource name
            namespace: K8s namespace (empty = default)
        """
        policy_error = guard_k8s_write_tool("k8s_rollout_restart", namespace=namespace)
        if policy_error:
            return f"ERROR: {policy_error}"
        return k8s.rollout_restart(kind, name, namespace or None)

    @tool
    def k8s_rollout_undo(kind: str, name: str, namespace: str = "") -> str:
        """Roll back a deployment or statefulset to the previous revision.

        Args:
            kind: 'deployment' or 'statefulset'
            name: Resource name
            namespace: K8s namespace (empty = default)
        """
        policy_error = guard_k8s_write_tool("k8s_rollout_undo", namespace=namespace)
        if policy_error:
            return f"ERROR: {policy_error}"
        return k8s.rollout_undo(kind, name, namespace or None)

    @tool
    def k8s_delete_resource(kind: str, name: str, namespace: str = "") -> str:
        """Delete a Kubernetes resource.

        Args:
            kind: Resource type (pod, deployment, service, etc.)
            name: Resource name
            namespace: K8s namespace (empty = default)
        """
        policy_error = guard_k8s_write_tool("k8s_delete_resource", namespace=namespace)
        if policy_error:
            return f"ERROR: {policy_error}"
        return k8s.delete_resource(kind, name, namespace or None)

    @tool
    def k8s_apply_manifest(manifest_yaml: str) -> str:
        """Apply a Kubernetes manifest (YAML). Creates or updates resources.

        Args:
            manifest_yaml: Full YAML manifest content
        """
        policy_error = guard_k8s_write_tool("k8s_apply_manifest")
        if policy_error:
            return f"ERROR: {policy_error}"
        return k8s.apply_manifest(manifest_yaml)

    @tool
    def k8s_cordon_node(node: str, uncordon: bool = False) -> str:
        """Cordon (or uncordon) a node to prevent (or allow) new pod scheduling.

        Args:
            node: Node name
            uncordon: Set True to uncordon instead of cordon
        """
        policy_error = guard_k8s_write_tool("k8s_cordon_node")
        if policy_error:
            return f"ERROR: {policy_error}"
        return k8s.uncordon(node) if uncordon else k8s.cordon(node)

    @tool
    def k8s_drain_node(node: str) -> str:
        """Drain a node, evicting all pods (respects PDBs, ignores daemonsets).

        Args:
            node: Node name
        """
        policy_error = guard_k8s_write_tool("k8s_drain_node")
        if policy_error:
            return f"ERROR: {policy_error}"
        return k8s.drain(node)

    @tool
    def k8s_exec_in_pod(pod: str, command: str, namespace: str = "", container: str = "") -> str:
        """Execute a command inside a running pod.

        Args:
            pod: Pod name
            command: Shell command to run (e.g. 'cat /etc/config/app.yaml')
            namespace: K8s namespace (empty = default)
            container: Container name (needed for multi-container pods)
        """
        policy_error = guard_k8s_write_tool("k8s_exec_in_pod", namespace=namespace)
        if policy_error:
            return f"ERROR: {policy_error}"
        return k8s.exec_command(pod, command, namespace or None, container or None)

    @tool
    def k8s_restart_workload_safely(kind: str, name: str, namespace: str = "", timeout_seconds: int = 300) -> str:
        """Restart a deployment or statefulset and wait for rollout completion.

        Args:
            kind: 'deployment' or 'statefulset'
            name: Resource name
            namespace: K8s namespace (empty = default)
            timeout_seconds: Maximum rollout wait time in seconds
        """
        policy_error = guard_k8s_write_tool("k8s_restart_workload_safely", namespace=namespace)
        if policy_error:
            return f"ERROR: {policy_error}"
        return k8s.restart_workload_safely(kind, name, namespace or None, timeout_seconds=timeout_seconds)

    @tool
    def k8s_cleanup_terminated_pods(
        namespace: str = "",
        all_namespaces: bool = False,
        phases_csv: str = "Failed,Unknown,Succeeded",
    ) -> str:
        """Delete terminated pods by phase with a scoped, typed cleanup action.

        Args:
            namespace: K8s namespace (empty = default, ignored for all_namespaces)
            all_namespaces: Set True to clean across the whole cluster
            phases_csv: Comma-separated phases, e.g. 'Failed,Unknown,Succeeded'
        """
        policy_error = guard_k8s_write_tool(
            "k8s_cleanup_terminated_pods",
            namespace=namespace,
            all_namespaces=all_namespaces,
        )
        if policy_error:
            return f"ERROR: {policy_error}"
        phases = [phase.strip() for phase in phases_csv.split(",") if phase.strip()]
        return k8s.cleanup_terminated_pods(phases, namespace or None, all_namespaces=all_namespaces)

    return [
        k8s_scale,
        k8s_rollout_restart,
        k8s_rollout_undo,
        k8s_delete_resource,
        k8s_apply_manifest,
        k8s_cordon_node,
        k8s_drain_node,
        k8s_exec_in_pod,
        k8s_restart_workload_safely,
        k8s_cleanup_terminated_pods,
    ]
