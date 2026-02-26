"""Kubernetes mutation tools (approval-gated by agent policy)."""

from __future__ import annotations

import shlex

from langchain.tools import tool

from ..config import Config
from .k8s_common import (
    ensure_kubectl_installed,
    is_valid_k8s_name,
    kubectl_base_args,
    resolve_namespace_for_resource,
    run_kubectl,
    truncate_text,
)


def _kubectl_missing_msg() -> str:
    return "❌ `kubectl` not found in PATH in this runtime. I cannot execute Kubernetes commands here until it is available."


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


def _validate_name(value: str, label: str) -> str | None:
    if not is_valid_k8s_name(value):
        return f"❌ Invalid {label}: {value!r}."
    return None


def _validate_namespace_hint(namespace: str) -> str | None:
    ns = (namespace or "").strip()
    if not ns or ns.lower() in {"auto", "any", "all"}:
        return None
    if not is_valid_k8s_name(ns):
        return f"❌ Invalid namespace: {namespace!r}."
    return None


def _resolve_namespace(kind: str, name: str, namespace_hint: str) -> tuple[str | None, str | None]:
    return resolve_namespace_for_resource(kind, name, namespace_hint)


def _preflight_cluster_access() -> str | None:
    code, _out, err = run_kubectl(kubectl_base_args(namespace=None) + ["cluster-info"])
    if code == 0:
        return None
    return (
        "❌ Cannot access Kubernetes cluster from this runtime context. "
        f"Details: {err or 'unknown error'}"
    )


def _run_single_command(cmd: list[str], *, reason: str, success_text: str) -> str:
    cmd_text = shlex.join(cmd)

    if Config.K8S_DRY_RUN:
        return (
            "🧪 DRY RUN enabled; no mutation executed.\n"
            f"Planned command:\n```bash\n{cmd_text}\n```\n"
            f"Reason: {reason or 'n/a'}"
        )

    code, out, err = run_kubectl(cmd)
    if code != 0:
        return (
            f"Planned command:\n```bash\n{cmd_text}\n```\n"
            f"❌ kubectl command failed (exit={code}). "
            f"Details: {truncate_text(err or out or 'unknown error')}"
        )

    return (
        f"Planned command:\n```bash\n{cmd_text}\n```\n"
        f"✅ {success_text}\n"
        f"Reason: {reason or 'n/a'}\n"
        f"Executed command: {cmd_text}\n"
        f"kubectl output: {truncate_text(out or '(no output)')}"
    )


def _run_batch_commands(commands: list[tuple[str, list[str]]], *, reason: str, title: str) -> str:
    command_lines = [shlex.join(cmd) for _label, cmd in commands]

    if Config.K8S_DRY_RUN:
        return (
            "🧪 DRY RUN enabled; no mutation executed.\n"
            f"{title}\n"
            "Planned commands:\n"
            + "\n".join(f"- {line}" for line in command_lines)
            + f"\nReason: {reason or 'n/a'}"
        )

    successes: list[str] = []
    failures: list[str] = []

    for label, cmd in commands:
        code, out, err = run_kubectl(cmd)
        if code == 0:
            successes.append(f"✅ {label}: {truncate_text(out or '(no output)')}")
        else:
            failures.append(f"❌ {label}: {truncate_text(err or out or 'unknown error')}")

    summary = "✅ Completed." if not failures else "⚠️ Completed with failures."
    return "\n".join(
        [
            f"{title} {summary}",
            f"Reason: {reason or 'n/a'}",
            "Planned commands:",
            *[f"- {line}" for line in command_lines],
            "Execution results:",
            *successes,
            *failures,
        ]
    )


@tool
def restart_kubernetes_pod(pod_name: str, namespace: str = Config.K8S_DEFAULT_NAMESPACE, reason: str = "") -> str:
    """Restart a pod by deleting it (controller recreates it)."""
    pod_name = (pod_name or "").strip()
    namespace_hint = (namespace or "").strip()
    reason = (reason or "").strip()

    name_err = _validate_name(pod_name, "pod name")
    if name_err:
        return name_err
    ns_hint_err = _validate_namespace_hint(namespace_hint)
    if ns_hint_err:
        return ns_hint_err

    if not ensure_kubectl_installed():
        return _kubectl_missing_msg()
    preflight_err = _preflight_cluster_access()
    if preflight_err:
        return preflight_err

    resolved_ns, resolve_err = _resolve_namespace("pod", pod_name, namespace_hint)
    if resolve_err:
        return resolve_err
    if not resolved_ns:
        return f"❌ Could not resolve namespace for pod '{pod_name}'."

    cmd = _build_restart_cmd(pod_name, resolved_ns)
    return _run_single_command(
        cmd,
        reason=reason,
        success_text=f"Successfully requested restart of pod '{pod_name}' in namespace '{resolved_ns}'.",
    )


@tool
def restart_kubernetes_pods_batch(
    pod_names: list[str],
    namespace: str = Config.K8S_DEFAULT_NAMESPACE,
    reason: str = "",
) -> str:
    """Restart multiple pods (can resolve namespaces per pod)."""
    namespace_hint = (namespace or "").strip()
    reason = (reason or "").strip()
    pods = [str(p).strip() for p in (pod_names or []) if str(p).strip()]

    ns_hint_err = _validate_namespace_hint(namespace_hint)
    if ns_hint_err:
        return ns_hint_err
    if not pods:
        return "❌ pod_names must contain at least one pod name."

    unique_pods = list(dict.fromkeys(pods))
    invalid = [name for name in unique_pods if _validate_name(name, "pod name")]
    if invalid:
        return f"❌ Invalid pod names: {invalid}"

    if not ensure_kubectl_installed():
        return _kubectl_missing_msg()
    preflight_err = _preflight_cluster_access()
    if preflight_err:
        return preflight_err

    commands: list[tuple[str, list[str]]] = []
    for pod_name in unique_pods:
        resolved_ns, resolve_err = _resolve_namespace("pod", pod_name, namespace_hint)
        if resolve_err:
            return resolve_err
        if not resolved_ns:
            return f"❌ Could not resolve namespace for pod '{pod_name}'."
        cmd = _build_restart_cmd(pod_name, resolved_ns)
        commands.append((f"pod/{pod_name} (ns={resolved_ns})", cmd))

    return _run_batch_commands(commands, reason=reason, title="Batch pod restart")


@tool
def scale_kubernetes_deployment(
    deployment_name: str,
    namespace: str = Config.K8S_DEFAULT_NAMESPACE,
    replicas: int = 0,
    reason: str = "",
) -> str:
    """Scale a deployment."""
    deployment_name = (deployment_name or "").strip()
    namespace_hint = (namespace or "").strip()
    reason = (reason or "").strip()

    if not deployment_name:
        return "❌ deployment_name is required."
    if replicas < 0:
        return "❌ replicas must be >= 0."

    name_err = _validate_name(deployment_name, "deployment name")
    if name_err:
        return name_err
    ns_hint_err = _validate_namespace_hint(namespace_hint)
    if ns_hint_err:
        return ns_hint_err

    if not ensure_kubectl_installed():
        return _kubectl_missing_msg()
    preflight_err = _preflight_cluster_access()
    if preflight_err:
        return preflight_err

    resolved_ns, resolve_err = _resolve_namespace("deployment", deployment_name, namespace_hint)
    if resolve_err:
        return resolve_err
    if not resolved_ns:
        return f"❌ Could not resolve namespace for deployment '{deployment_name}'."

    cmd = _build_scale_cmd("deployment", deployment_name, resolved_ns, replicas)
    return _run_single_command(
        cmd,
        reason=reason,
        success_text=(
            f"Successfully scaled deployment '{deployment_name}' in namespace '{resolved_ns}' to replicas={replicas}."
        ),
    )


@tool
def scale_kubernetes_statefulset(
    statefulset_name: str,
    namespace: str = Config.K8S_DEFAULT_NAMESPACE,
    replicas: int = 0,
    reason: str = "",
) -> str:
    """Scale a statefulset."""
    statefulset_name = (statefulset_name or "").strip()
    namespace_hint = (namespace or "").strip()
    reason = (reason or "").strip()

    if not statefulset_name:
        return "❌ statefulset_name is required."
    if replicas < 0:
        return "❌ replicas must be >= 0."

    name_err = _validate_name(statefulset_name, "statefulset name")
    if name_err:
        return name_err
    ns_hint_err = _validate_namespace_hint(namespace_hint)
    if ns_hint_err:
        return ns_hint_err

    if not ensure_kubectl_installed():
        return _kubectl_missing_msg()
    preflight_err = _preflight_cluster_access()
    if preflight_err:
        return preflight_err

    resolved_ns, resolve_err = _resolve_namespace("statefulset", statefulset_name, namespace_hint)
    if resolve_err:
        return resolve_err
    if not resolved_ns:
        return f"❌ Could not resolve namespace for statefulset '{statefulset_name}'."

    cmd = _build_scale_cmd("statefulset", statefulset_name, resolved_ns, replicas)
    return _run_single_command(
        cmd,
        reason=reason,
        success_text=(
            f"Successfully scaled statefulset '{statefulset_name}' in namespace '{resolved_ns}' to replicas={replicas}."
        ),
    )


@tool
def scale_kubernetes_workloads_batch(
    namespace: str,
    changes: list[dict],
    reason: str = "",
) -> str:
    """Scale multiple workloads in one operation."""
    namespace_hint = (namespace or "").strip()
    reason = (reason or "").strip()

    ns_hint_err = _validate_namespace_hint(namespace_hint)
    if ns_hint_err:
        return ns_hint_err
    if not isinstance(changes, list) or not changes:
        return "❌ changes must be a non-empty list."

    if not ensure_kubectl_installed():
        return _kubectl_missing_msg()
    preflight_err = _preflight_cluster_access()
    if preflight_err:
        return preflight_err

    commands: list[tuple[str, list[str]]] = []
    for idx, item in enumerate(changes, start=1):
        if not isinstance(item, dict):
            return f"❌ changes[{idx}] must be an object."

        kind = str(item.get("kind", "")).strip().lower()
        name = str(item.get("name", "")).strip()
        replicas = item.get("replicas", 0)

        if kind not in {"deployment", "statefulset"}:
            return f"❌ changes[{idx}].kind must be 'deployment' or 'statefulset'."
        if _validate_name(name, f"changes[{idx}] workload name"):
            return f"❌ changes[{idx}] invalid workload name: {name!r}."
        if not isinstance(replicas, int) or replicas < 0:
            return f"❌ changes[{idx}].replicas must be integer >= 0."

        resolved_ns, resolve_err = _resolve_namespace(kind, name, namespace_hint)
        if resolve_err:
            return resolve_err
        if not resolved_ns:
            return f"❌ Could not resolve namespace for {kind} '{name}'."

        cmd = _build_scale_cmd(kind, name, resolved_ns, replicas)
        label = f"{kind}/{name} (ns={resolved_ns}) -> replicas={replicas}"
        commands.append((label, cmd))

    return _run_batch_commands(commands, reason=reason, title="Batch scaling")


@tool
def rollout_restart_kubernetes_deployment(
    deployment_name: str,
    namespace: str = Config.K8S_DEFAULT_NAMESPACE,
    reason: str = "",
) -> str:
    """Rollout-restart a deployment."""
    deployment_name = (deployment_name or "").strip()
    namespace_hint = (namespace or "").strip()
    reason = (reason or "").strip()

    if not deployment_name:
        return "❌ deployment_name is required."

    name_err = _validate_name(deployment_name, "deployment name")
    if name_err:
        return name_err
    ns_hint_err = _validate_namespace_hint(namespace_hint)
    if ns_hint_err:
        return ns_hint_err

    if not ensure_kubectl_installed():
        return _kubectl_missing_msg()
    preflight_err = _preflight_cluster_access()
    if preflight_err:
        return preflight_err

    resolved_ns, resolve_err = _resolve_namespace("deployment", deployment_name, namespace_hint)
    if resolve_err:
        return resolve_err
    if not resolved_ns:
        return f"❌ Could not resolve namespace for deployment '{deployment_name}'."

    cmd = _build_rollout_restart_cmd("deployment", deployment_name, resolved_ns)
    return _run_single_command(
        cmd,
        reason=reason,
        success_text=f"Successfully requested rollout restart of deployment '{deployment_name}' in namespace '{resolved_ns}'.",
    )


@tool
def rollout_restart_kubernetes_statefulset(
    statefulset_name: str,
    namespace: str = Config.K8S_DEFAULT_NAMESPACE,
    reason: str = "",
) -> str:
    """Rollout-restart a statefulset."""
    statefulset_name = (statefulset_name or "").strip()
    namespace_hint = (namespace or "").strip()
    reason = (reason or "").strip()

    if not statefulset_name:
        return "❌ statefulset_name is required."

    name_err = _validate_name(statefulset_name, "statefulset name")
    if name_err:
        return name_err
    ns_hint_err = _validate_namespace_hint(namespace_hint)
    if ns_hint_err:
        return ns_hint_err

    if not ensure_kubectl_installed():
        return _kubectl_missing_msg()
    preflight_err = _preflight_cluster_access()
    if preflight_err:
        return preflight_err

    resolved_ns, resolve_err = _resolve_namespace("statefulset", statefulset_name, namespace_hint)
    if resolve_err:
        return resolve_err
    if not resolved_ns:
        return f"❌ Could not resolve namespace for statefulset '{statefulset_name}'."

    cmd = _build_rollout_restart_cmd("statefulset", statefulset_name, resolved_ns)
    return _run_single_command(
        cmd,
        reason=reason,
        success_text=(
            f"Successfully requested rollout restart of statefulset '{statefulset_name}' in namespace '{resolved_ns}'."
        ),
    )


@tool
def rollout_restart_kubernetes_daemonset(
    daemonset_name: str,
    namespace: str = Config.K8S_DEFAULT_NAMESPACE,
    reason: str = "",
) -> str:
    """Rollout-restart a daemonset."""
    daemonset_name = (daemonset_name or "").strip()
    namespace_hint = (namespace or "").strip()
    reason = (reason or "").strip()

    if not daemonset_name:
        return "❌ daemonset_name is required."

    name_err = _validate_name(daemonset_name, "daemonset name")
    if name_err:
        return name_err
    ns_hint_err = _validate_namespace_hint(namespace_hint)
    if ns_hint_err:
        return ns_hint_err

    if not ensure_kubectl_installed():
        return _kubectl_missing_msg()
    preflight_err = _preflight_cluster_access()
    if preflight_err:
        return preflight_err

    resolved_ns, resolve_err = _resolve_namespace("daemonset", daemonset_name, namespace_hint)
    if resolve_err:
        return resolve_err
    if not resolved_ns:
        return f"❌ Could not resolve namespace for daemonset '{daemonset_name}'."

    cmd = _build_rollout_restart_cmd("daemonset", daemonset_name, resolved_ns)
    return _run_single_command(
        cmd,
        reason=reason,
        success_text=f"Successfully requested rollout restart of daemonset '{daemonset_name}' in namespace '{resolved_ns}'.",
    )


@tool
def rollout_restart_kubernetes_workloads_batch(
    namespace: str,
    workloads: list[dict],
    reason: str = "",
) -> str:
    """Rollout-restart multiple workloads in one operation."""
    namespace_hint = (namespace or "").strip()
    reason = (reason or "").strip()

    ns_hint_err = _validate_namespace_hint(namespace_hint)
    if ns_hint_err:
        return ns_hint_err
    if not isinstance(workloads, list) or not workloads:
        return "❌ workloads must be a non-empty list."

    if not ensure_kubectl_installed():
        return _kubectl_missing_msg()
    preflight_err = _preflight_cluster_access()
    if preflight_err:
        return preflight_err

    commands: list[tuple[str, list[str]]] = []
    for idx, item in enumerate(workloads, start=1):
        if not isinstance(item, dict):
            return f"❌ workloads[{idx}] must be an object."

        kind = str(item.get("kind", "")).strip().lower()
        name = str(item.get("name", "")).strip()

        if kind not in {"deployment", "statefulset", "daemonset"}:
            return f"❌ workloads[{idx}].kind must be 'deployment', 'statefulset', or 'daemonset'."
        if _validate_name(name, f"workloads[{idx}] workload name"):
            return f"❌ workloads[{idx}] invalid workload name: {name!r}."

        resolved_ns, resolve_err = _resolve_namespace(kind, name, namespace_hint)
        if resolve_err:
            return resolve_err
        if not resolved_ns:
            return f"❌ Could not resolve namespace for {kind} '{name}'."

        cmd = _build_rollout_restart_cmd(kind, name, resolved_ns)
        label = f"{kind}/{name} (ns={resolved_ns})"
        commands.append((label, cmd))

    return _run_batch_commands(commands, reason=reason, title="Batch rollout restart")
