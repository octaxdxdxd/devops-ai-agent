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


def _build_restart_cmd(pod_name: str, namespace: str) -> list[str]:
    return kubectl_base_args(namespace=namespace) + [
        "delete",
        "pod",
        pod_name,
        "--wait=false",
        "--ignore-not-found=true",
    ]


def _build_scale_cmd(kind: str, name: str, namespace: str, replicas: int) -> list[str]:
    return kubectl_base_args(namespace=namespace) + [
        "scale",
        kind,
        name,
        "--replicas",
        str(replicas),
    ]


def _build_rollout_restart_cmd(kind: str, name: str, namespace: str) -> list[str]:
    return kubectl_base_args(namespace=namespace) + [
        "rollout",
        "restart",
        f"{kind}/{name}",
    ]


def _validate_namespace(namespace: str) -> str | None:
    if not is_valid_k8s_name(namespace):
        return f"❌ Invalid namespace: {namespace!r}."
    return None


def _validate_pod_name(pod_name: str) -> str | None:
    if not is_valid_k8s_name(pod_name):
        return f"❌ Invalid pod name: {pod_name!r}."
    return None


def _preflight_cluster_access() -> str | None:
    preflight = kubectl_base_args(namespace=None) + ["cluster-info"]
    code, _out, err = run_kubectl(preflight)
    if code == 0:
        return None
    return (
        "❌ Cannot access Kubernetes cluster from this runtime context. "
        f"Details: {err or 'unknown error'}"
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

    pod_err = _validate_pod_name(pod_name)
    if pod_err:
        return pod_err
    ns_err = _validate_namespace(namespace)
    if ns_err:
        return ns_err

    if not ensure_kubectl_installed():
        return "❌ `kubectl` not found in PATH in this runtime. I cannot execute Kubernetes commands here until it is available."

    preflight_err = _preflight_cluster_access()
    if preflight_err:
        return preflight_err

    cmd = _build_restart_cmd(pod_name, namespace)
    cmd_text = " ".join(cmd)

    if Config.K8S_DRY_RUN:
        return (
            "🧪 DRY RUN enabled; no mutation executed.\n"
            f"Planned command: {cmd_text}\n"
            f"Reason: {reason or 'n/a'}"
        )

    code, out, err = run_kubectl(cmd)
    if code != 0:
        return (
            f"Planned command: {cmd_text}\n"
            f"❌ Failed to restart pod '{pod_name}' in namespace '{namespace}'. "
            f"kubectl exit code: {code}. Details: {err or out or 'unknown error'}"
        )

    return (
        f"Planned command: {cmd_text}\n"
        f"✅ Successfully requested restart of pod '{pod_name}' in namespace '{namespace}'.\n"
        f"Reason: {reason or 'n/a'}\n"
        f"Executed command: {cmd_text}\n"
        f"kubectl output: {out or '(no output)'}"
    )


@tool
def restart_kubernetes_pods_batch(
    pod_names: list[str],
    namespace: str = Config.K8S_DEFAULT_NAMESPACE,
    reason: str = "",
) -> str:
    """
    Restart multiple Kubernetes pods in one operation by deleting each pod.
    IMPORTANT: Always ask for user approval before using this tool as it mutates cluster state.

    Args:
        pod_names: List of pod names to restart in the same namespace.
        namespace: Kubernetes namespace.
        reason: Reason for batch restart.
    """
    namespace = (namespace or Config.K8S_DEFAULT_NAMESPACE).strip() or Config.K8S_DEFAULT_NAMESPACE
    reason = (reason or "").strip()
    pod_names = [str(p).strip() for p in (pod_names or []) if str(p).strip()]

    ns_err = _validate_namespace(namespace)
    if ns_err:
        return ns_err
    if not pod_names:
        return "❌ pod_names must contain at least one pod name."

    unique_pods: list[str] = []
    seen: set[str] = set()
    for name in pod_names:
        if name in seen:
            continue
        seen.add(name)
        unique_pods.append(name)

    invalid = [name for name in unique_pods if _validate_pod_name(name)]
    if invalid:
        return f"❌ Invalid pod names: {invalid}"

    if not ensure_kubectl_installed():
        return "❌ `kubectl` not found in PATH in this runtime. I cannot execute Kubernetes commands here until it is available."

    preflight_err = _preflight_cluster_access()
    if preflight_err:
        return preflight_err

    commands = [_build_restart_cmd(name, namespace) for name in unique_pods]
    command_lines = [" ".join(cmd) for cmd in commands]

    if Config.K8S_DRY_RUN:
        return (
            "🧪 DRY RUN enabled; no mutation executed.\n"
            "Planned commands:\n"
            + "\n".join(f"- {line}" for line in command_lines)
            + f"\nReason: {reason or 'n/a'}"
        )

    results: list[str] = []
    failures: list[str] = []

    for pod_name, cmd in zip(unique_pods, commands):
        code, out, err = run_kubectl(cmd)
        if code == 0:
            results.append(f"✅ {pod_name}: {out or '(no output)'}")
        else:
            failures.append(f"❌ {pod_name}: {err or out or 'unknown error'}")

    status = "✅ Batch restart completed." if not failures else "⚠️ Batch restart completed with failures."
    body_lines = [
        status,
        f"Namespace: {namespace}",
        f"Reason: {reason or 'n/a'}",
        "Planned commands:",
        *[f"- {line}" for line in command_lines],
        "Execution results:",
        *results,
        *failures,
    ]
    return "\n".join(body_lines)


@tool
def scale_kubernetes_deployment(
    deployment_name: str,
    namespace: str = Config.K8S_DEFAULT_NAMESPACE,
    replicas: int = 0,
    reason: str = "",
) -> str:
    """Scale a Kubernetes deployment to a target replica count (write action)."""
    deployment_name = (deployment_name or "").strip()
    namespace = (namespace or Config.K8S_DEFAULT_NAMESPACE).strip() or Config.K8S_DEFAULT_NAMESPACE
    reason = (reason or "").strip()

    if not deployment_name:
        return "❌ deployment_name is required."
    if replicas < 0:
        return "❌ replicas must be >= 0."

    name_err = _validate_pod_name(deployment_name)
    if name_err:
        return name_err.replace("pod name", "deployment name")
    ns_err = _validate_namespace(namespace)
    if ns_err:
        return ns_err

    if not ensure_kubectl_installed():
        return "❌ `kubectl` not found in PATH in this runtime. I cannot execute Kubernetes commands here until it is available."

    preflight_err = _preflight_cluster_access()
    if preflight_err:
        return preflight_err

    cmd = _build_scale_cmd("deployment", deployment_name, namespace, replicas)
    cmd_text = " ".join(cmd)

    if Config.K8S_DRY_RUN:
        return (
            "🧪 DRY RUN enabled; no mutation executed.\n"
            f"Planned command: {cmd_text}\n"
            f"Reason: {reason or 'n/a'}"
        )

    code, out, err = run_kubectl(cmd)
    if code != 0:
        return (
            f"Planned command: {cmd_text}\n"
            f"❌ Failed to scale deployment '{deployment_name}' in namespace '{namespace}'. "
            f"kubectl exit code: {code}. Details: {err or out or 'unknown error'}"
        )

    return (
        f"Planned command: {cmd_text}\n"
        f"✅ Successfully scaled deployment '{deployment_name}' in namespace '{namespace}' to replicas={replicas}.\n"
        f"Reason: {reason or 'n/a'}\n"
        f"Executed command: {cmd_text}\n"
        f"kubectl output: {out or '(no output)'}"
    )


@tool
def scale_kubernetes_statefulset(
    statefulset_name: str,
    namespace: str = Config.K8S_DEFAULT_NAMESPACE,
    replicas: int = 0,
    reason: str = "",
) -> str:
    """Scale a Kubernetes statefulset to a target replica count (write action)."""
    statefulset_name = (statefulset_name or "").strip()
    namespace = (namespace or Config.K8S_DEFAULT_NAMESPACE).strip() or Config.K8S_DEFAULT_NAMESPACE
    reason = (reason or "").strip()

    if not statefulset_name:
        return "❌ statefulset_name is required."
    if replicas < 0:
        return "❌ replicas must be >= 0."

    name_err = _validate_pod_name(statefulset_name)
    if name_err:
        return name_err.replace("pod name", "statefulset name")
    ns_err = _validate_namespace(namespace)
    if ns_err:
        return ns_err

    if not ensure_kubectl_installed():
        return "❌ `kubectl` not found in PATH in this runtime. I cannot execute Kubernetes commands here until it is available."

    preflight_err = _preflight_cluster_access()
    if preflight_err:
        return preflight_err

    cmd = _build_scale_cmd("statefulset", statefulset_name, namespace, replicas)
    cmd_text = " ".join(cmd)

    if Config.K8S_DRY_RUN:
        return (
            "🧪 DRY RUN enabled; no mutation executed.\n"
            f"Planned command: {cmd_text}\n"
            f"Reason: {reason or 'n/a'}"
        )

    code, out, err = run_kubectl(cmd)
    if code != 0:
        return (
            f"Planned command: {cmd_text}\n"
            f"❌ Failed to scale statefulset '{statefulset_name}' in namespace '{namespace}'. "
            f"kubectl exit code: {code}. Details: {err or out or 'unknown error'}"
        )

    return (
        f"Planned command: {cmd_text}\n"
        f"✅ Successfully scaled statefulset '{statefulset_name}' in namespace '{namespace}' to replicas={replicas}.\n"
        f"Reason: {reason or 'n/a'}\n"
        f"Executed command: {cmd_text}\n"
        f"kubectl output: {out or '(no output)'}"
    )


@tool
def scale_kubernetes_workloads_batch(
    namespace: str,
    changes: list[dict],
    reason: str = "",
) -> str:
    """
    Scale multiple workloads in one approved write operation.

    Each item in changes must be:
    {"kind": "deployment"|"statefulset", "name": "<resource>", "replicas": <int>}.
    """
    namespace = (namespace or Config.K8S_DEFAULT_NAMESPACE).strip() or Config.K8S_DEFAULT_NAMESPACE
    reason = (reason or "").strip()

    ns_err = _validate_namespace(namespace)
    if ns_err:
        return ns_err
    if not isinstance(changes, list) or not changes:
        return "❌ changes must be a non-empty list."

    if not ensure_kubectl_installed():
        return "❌ `kubectl` not found in PATH in this runtime. I cannot execute Kubernetes commands here until it is available."

    preflight_err = _preflight_cluster_access()
    if preflight_err:
        return preflight_err

    parsed: list[tuple[str, str, int]] = []
    for idx, item in enumerate(changes, start=1):
        if not isinstance(item, dict):
            return f"❌ changes[{idx}] must be an object."
        kind = str(item.get("kind", "")).strip().lower()
        name = str(item.get("name", "")).strip()
        replicas = item.get("replicas", 0)
        if kind not in {"deployment", "statefulset"}:
            return f"❌ changes[{idx}].kind must be 'deployment' or 'statefulset'."
        name_err = _validate_pod_name(name)
        if name_err:
            return f"❌ changes[{idx}] invalid workload name: {name!r}."
        if not isinstance(replicas, int) or replicas < 0:
            return f"❌ changes[{idx}].replicas must be integer >= 0."
        parsed.append((kind, name, replicas))

    commands = [_build_scale_cmd(kind, name, namespace, replicas) for kind, name, replicas in parsed]
    command_lines = [" ".join(cmd) for cmd in commands]

    if Config.K8S_DRY_RUN:
        return (
            "🧪 DRY RUN enabled; no mutation executed.\n"
            "Planned commands:\n"
            + "\n".join(f"- {line}" for line in command_lines)
            + f"\nReason: {reason or 'n/a'}"
        )

    successes: list[str] = []
    failures: list[str] = []
    for (kind, name, replicas), cmd in zip(parsed, commands):
        code, out, err = run_kubectl(cmd)
        label = f"{kind}/{name} -> replicas={replicas}"
        if code == 0:
            successes.append(f"✅ {label}: {out or '(no output)'}")
        else:
            failures.append(f"❌ {label}: {err or out or 'unknown error'}")

    summary = "✅ Batch scaling completed." if not failures else "⚠️ Batch scaling completed with failures."
    return "\n".join([
        summary,
        f"Namespace: {namespace}",
        f"Reason: {reason or 'n/a'}",
        "Planned commands:",
        *[f"- {line}" for line in command_lines],
        "Execution results:",
        *successes,
        *failures,
    ])


@tool
def rollout_restart_kubernetes_deployment(
    deployment_name: str,
    namespace: str = Config.K8S_DEFAULT_NAMESPACE,
    reason: str = "",
) -> str:
    """Restart a deployment using `kubectl rollout restart` (write action)."""
    deployment_name = (deployment_name or "").strip()
    namespace = (namespace or Config.K8S_DEFAULT_NAMESPACE).strip() or Config.K8S_DEFAULT_NAMESPACE
    reason = (reason or "").strip()

    if not deployment_name:
        return "❌ deployment_name is required."

    name_err = _validate_pod_name(deployment_name)
    if name_err:
        return name_err.replace("pod name", "deployment name")
    ns_err = _validate_namespace(namespace)
    if ns_err:
        return ns_err

    if not ensure_kubectl_installed():
        return "❌ `kubectl` not found in PATH in this runtime. I cannot execute Kubernetes commands here until it is available."

    preflight_err = _preflight_cluster_access()
    if preflight_err:
        return preflight_err

    cmd = _build_rollout_restart_cmd("deployment", deployment_name, namespace)
    cmd_text = " ".join(cmd)

    if Config.K8S_DRY_RUN:
        return (
            "🧪 DRY RUN enabled; no mutation executed.\n"
            f"Planned command: {cmd_text}\n"
            f"Reason: {reason or 'n/a'}"
        )

    code, out, err = run_kubectl(cmd)
    if code != 0:
        return (
            f"Planned command: {cmd_text}\n"
            f"❌ Failed to rollout restart deployment '{deployment_name}' in namespace '{namespace}'. "
            f"kubectl exit code: {code}. Details: {err or out or 'unknown error'}"
        )

    return (
        f"Planned command: {cmd_text}\n"
        f"✅ Successfully requested rollout restart of deployment '{deployment_name}' in namespace '{namespace}'.\n"
        f"Reason: {reason or 'n/a'}\n"
        f"Executed command: {cmd_text}\n"
        f"kubectl output: {out or '(no output)'}"
    )


@tool
def rollout_restart_kubernetes_statefulset(
    statefulset_name: str,
    namespace: str = Config.K8S_DEFAULT_NAMESPACE,
    reason: str = "",
) -> str:
    """Restart a statefulset using `kubectl rollout restart` (write action)."""
    statefulset_name = (statefulset_name or "").strip()
    namespace = (namespace or Config.K8S_DEFAULT_NAMESPACE).strip() or Config.K8S_DEFAULT_NAMESPACE
    reason = (reason or "").strip()

    if not statefulset_name:
        return "❌ statefulset_name is required."

    name_err = _validate_pod_name(statefulset_name)
    if name_err:
        return name_err.replace("pod name", "statefulset name")
    ns_err = _validate_namespace(namespace)
    if ns_err:
        return ns_err

    if not ensure_kubectl_installed():
        return "❌ `kubectl` not found in PATH in this runtime. I cannot execute Kubernetes commands here until it is available."

    preflight_err = _preflight_cluster_access()
    if preflight_err:
        return preflight_err

    cmd = _build_rollout_restart_cmd("statefulset", statefulset_name, namespace)
    cmd_text = " ".join(cmd)

    if Config.K8S_DRY_RUN:
        return (
            "🧪 DRY RUN enabled; no mutation executed.\n"
            f"Planned command: {cmd_text}\n"
            f"Reason: {reason or 'n/a'}"
        )

    code, out, err = run_kubectl(cmd)
    if code != 0:
        return (
            f"Planned command: {cmd_text}\n"
            f"❌ Failed to rollout restart statefulset '{statefulset_name}' in namespace '{namespace}'. "
            f"kubectl exit code: {code}. Details: {err or out or 'unknown error'}"
        )

    return (
        f"Planned command: {cmd_text}\n"
        f"✅ Successfully requested rollout restart of statefulset '{statefulset_name}' in namespace '{namespace}'.\n"
        f"Reason: {reason or 'n/a'}\n"
        f"Executed command: {cmd_text}\n"
        f"kubectl output: {out or '(no output)'}"
    )


@tool
def rollout_restart_kubernetes_daemonset(
    daemonset_name: str,
    namespace: str = Config.K8S_DEFAULT_NAMESPACE,
    reason: str = "",
) -> str:
    """Restart a daemonset using `kubectl rollout restart` (write action)."""
    daemonset_name = (daemonset_name or "").strip()
    namespace = (namespace or Config.K8S_DEFAULT_NAMESPACE).strip() or Config.K8S_DEFAULT_NAMESPACE
    reason = (reason or "").strip()

    if not daemonset_name:
        return "❌ daemonset_name is required."

    name_err = _validate_pod_name(daemonset_name)
    if name_err:
        return name_err.replace("pod name", "daemonset name")
    ns_err = _validate_namespace(namespace)
    if ns_err:
        return ns_err

    if not ensure_kubectl_installed():
        return "❌ `kubectl` not found in PATH in this runtime. I cannot execute Kubernetes commands here until it is available."

    preflight_err = _preflight_cluster_access()
    if preflight_err:
        return preflight_err

    cmd = _build_rollout_restart_cmd("daemonset", daemonset_name, namespace)
    cmd_text = " ".join(cmd)

    if Config.K8S_DRY_RUN:
        return (
            "🧪 DRY RUN enabled; no mutation executed.\n"
            f"Planned command: {cmd_text}\n"
            f"Reason: {reason or 'n/a'}"
        )

    code, out, err = run_kubectl(cmd)
    if code != 0:
        return (
            f"Planned command: {cmd_text}\n"
            f"❌ Failed to rollout restart daemonset '{daemonset_name}' in namespace '{namespace}'. "
            f"kubectl exit code: {code}. Details: {err or out or 'unknown error'}"
        )

    return (
        f"Planned command: {cmd_text}\n"
        f"✅ Successfully requested rollout restart of daemonset '{daemonset_name}' in namespace '{namespace}'.\n"
        f"Reason: {reason or 'n/a'}\n"
        f"Executed command: {cmd_text}\n"
        f"kubectl output: {out or '(no output)'}"
    )


@tool
def rollout_restart_kubernetes_workloads_batch(
    namespace: str,
    workloads: list[dict],
    reason: str = "",
) -> str:
    """
    Restart multiple workloads in one approved write operation.

    Each item in workloads must be:
    {"kind": "deployment"|"statefulset"|"daemonset", "name": "<resource>"}.
    """
    namespace = (namespace or Config.K8S_DEFAULT_NAMESPACE).strip() or Config.K8S_DEFAULT_NAMESPACE
    reason = (reason or "").strip()

    ns_err = _validate_namespace(namespace)
    if ns_err:
        return ns_err
    if not isinstance(workloads, list) or not workloads:
        return "❌ workloads must be a non-empty list."

    if not ensure_kubectl_installed():
        return "❌ `kubectl` not found in PATH in this runtime. I cannot execute Kubernetes commands here until it is available."

    preflight_err = _preflight_cluster_access()
    if preflight_err:
        return preflight_err

    parsed: list[tuple[str, str]] = []
    for idx, item in enumerate(workloads, start=1):
        if not isinstance(item, dict):
            return f"❌ workloads[{idx}] must be an object."
        kind = str(item.get("kind", "")).strip().lower()
        name = str(item.get("name", "")).strip()
        if kind not in {"deployment", "statefulset", "daemonset"}:
            return f"❌ workloads[{idx}].kind must be 'deployment', 'statefulset', or 'daemonset'."
        name_err = _validate_pod_name(name)
        if name_err:
            return f"❌ workloads[{idx}] invalid workload name: {name!r}."
        parsed.append((kind, name))

    commands = [_build_rollout_restart_cmd(kind, name, namespace) for kind, name in parsed]
    command_lines = [" ".join(cmd) for cmd in commands]

    if Config.K8S_DRY_RUN:
        return (
            "🧪 DRY RUN enabled; no mutation executed.\n"
            "Planned commands:\n"
            + "\n".join(f"- {line}" for line in command_lines)
            + f"\nReason: {reason or 'n/a'}"
        )

    successes: list[str] = []
    failures: list[str] = []
    for (kind, name), cmd in zip(parsed, commands):
        code, out, err = run_kubectl(cmd)
        label = f"{kind}/{name}"
        if code == 0:
            successes.append(f"✅ {label}: {out or '(no output)'}")
        else:
            failures.append(f"❌ {label}: {err or out or 'unknown error'}")

    summary = "✅ Batch rollout restart completed." if not failures else "⚠️ Batch rollout restart completed with failures."
    return "\n".join([
        summary,
        f"Namespace: {namespace}",
        f"Reason: {reason or 'n/a'}",
        "Planned commands:",
        *[f"- {line}" for line in command_lines],
        "Execution results:",
        *successes,
        *failures,
    ])
