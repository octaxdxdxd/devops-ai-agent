"""
Action Tools
"""
from __future__ import annotations

from langchain.tools import tool
from ..config import Config
from .k8s_common import (
    ensure_kubectl_installed,
    is_valid_k8s_name,
    kubectl_base_args,
    run_kubectl,
)


@tool
def restart_kubernetes_pod(pod_name: str, namespace: str = Config.K8S_DEFAULT_NAMESPACE, reason: str = "") -> str:
    """
    Restart a Kubernetes pod by deleting it (will be recreated by deployment/replicaset).
    IMPORTANT: Always ask for user approval before using this tool as it will cause service disruption.
    
    Args:
        pod_name: Name of the pod to restart (e.g., 'pod-java-app-7d9f8b6c5-xk2m9')
        namespace: Kubernetes namespace (default: Config.K8S_DEFAULT_NAMESPACE)
        reason: Reason for restart (e.g., 'OutOfMemoryError recovery')
    
    Returns:
        str: Success or error message
    """
    pod_name = (pod_name or "").strip()
    namespace = (namespace or Config.K8S_DEFAULT_NAMESPACE).strip() or Config.K8S_DEFAULT_NAMESPACE
    reason = (reason or "").strip()

    if not is_valid_k8s_name(pod_name):
        return f"❌ Invalid pod name: {pod_name!r}."
    if not is_valid_k8s_name(namespace):
        return f"❌ Invalid namespace: {namespace!r}."

    if not ensure_kubectl_installed():
        return "❌ `kubectl` not found in PATH. Install kubectl and configure cluster access first."

    # Optional preflight so failures are explicit (works for EKS/AKS/GKE if auth is configured)
    preflight = kubectl_base_args(namespace=None) + ["cluster-info"]
    code, _out, err = run_kubectl(preflight)
    if code != 0:
        return (
            "❌ Cannot access Kubernetes cluster with current kubectl context. "
            "For EKS/AKS/GKE, ensure kubeconfig/auth is configured (e.g., aws eks update-kubeconfig / "
            "az aks get-credentials / gcloud container clusters get-credentials). "
            f"Details: {err or 'unknown error'}"
        )

    cmd = kubectl_base_args(namespace=namespace) + [
        "delete",
        "pod",
        pod_name,
        "--wait=false",
        "--ignore-not-found=true",
    ]

    if Config.K8S_DRY_RUN:
        return (
            "🧪 DRY RUN enabled; no mutation executed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"Reason: {reason or 'n/a'}"
        )

    code, out, err = run_kubectl(cmd)
    if code != 0:
        return (
            f"❌ Failed to restart pod '{pod_name}' in namespace '{namespace}'. "
            f"kubectl exit code: {code}. Details: {err or out or 'unknown error'}"
        )

    return (
        f"✅ Successfully requested restart of pod '{pod_name}' in namespace '{namespace}'.\n"
        f"Reason: {reason or 'n/a'}\n"
        f"kubectl output: {out or '(no output)'}"
    )