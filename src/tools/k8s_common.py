"""Shared Kubernetes kubectl helpers (provider-agnostic: EKS/AKS/GKE/any kubeconfig)."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from typing import Tuple

from ..config import Config


_DNS_1123_LABEL = re.compile(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")
_CLUSTER_SCOPED_KINDS = {
    "namespace",
    "namespaces",
    "node",
    "nodes",
    "pv",
    "pvs",
    "persistentvolume",
    "persistentvolumes",
    "storageclass",
    "storageclasses",
    "clusterrole",
    "clusterroles",
    "clusterrolebinding",
    "clusterrolebindings",
    "customresourcedefinition",
    "customresourcedefinitions",
    "crd",
    "crds",
}


def is_valid_k8s_name(value: str) -> bool:
    if not value or len(value) > 253:
        return False
    return bool(_DNS_1123_LABEL.match(value))


def ensure_kubectl_installed() -> bool:
    return bool(shutil.which("kubectl"))


def kubectl_base_args(namespace: str | None = None, *, all_namespaces: bool = False) -> list[str]:
    # `-A/--all-namespaces` is a command-scoped flag in kubectl and should be
    # appended by callers after the verb/resource (e.g., `get pods -A`).
    args = ["kubectl"]
    if Config.K8S_KUBECONFIG:
        args.extend(["--kubeconfig", Config.K8S_KUBECONFIG])
    if Config.K8S_CONTEXT:
        args.extend(["--context", Config.K8S_CONTEXT])
    if namespace:
        args.extend(["-n", namespace])
    args.extend(["--request-timeout", f"{Config.K8S_REQUEST_TIMEOUT_SEC}s"])
    return args


def run_kubectl(args: list[str]) -> Tuple[int, str, str]:
    proc = subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=max(5, Config.K8S_REQUEST_TIMEOUT_SEC + 10),
    )
    return proc.returncode, (proc.stdout or "").strip(), (proc.stderr or "").strip()


def truncate_text(value: str, max_chars: int | None = None) -> str:
    limit = max_chars if max_chars is not None else getattr(Config, "K8S_OUTPUT_MAX_CHARS", 12000)
    if limit <= 0 or len(value) <= limit:
        return value
    return value[:limit] + "\n... [truncated]"


def kubectl_not_found_msg() -> str:
    return "❌ `kubectl` not found in PATH in this runtime. I cannot execute Kubernetes commands here until it is available."


def kube_access_help(details: str) -> str:
    return (
        "❌ Kubernetes access failed in the current runtime context. "
        f"Details: {details or 'unknown error'}"
    )


def is_cluster_scoped_kind(kind: str) -> bool:
    value = (kind or "").strip().lower()
    if "/" in value:
        value = value.split("/", 1)[0]
    if "." in value:
        value = value.split(".", 1)[0]
    return value in _CLUSTER_SCOPED_KINDS


def discover_resource_namespaces(kind: str, name: str) -> tuple[list[str], str | None]:
    """Return namespaces containing the named resource."""
    if not ensure_kubectl_installed():
        return [], kubectl_not_found_msg()

    args = kubectl_base_args() + ["get", kind, "-A", "-o", "json"]
    code, out, err = run_kubectl(args)
    if code != 0:
        return [], err or out

    try:
        data = json.loads(out or "{}")
    except json.JSONDecodeError as exc:
        return [], f"Failed to parse kubectl output: {exc}"

    namespaces: list[str] = []
    for item in data.get("items", []):
        meta = item.get("metadata", {}) or {}
        if str(meta.get("name", "")).strip() != name:
            continue
        ns = str(meta.get("namespace", "")).strip()
        if ns:
            namespaces.append(ns)
    return namespaces, None


def resolve_namespace_for_resource(kind: str, name: str, namespace_hint: str = "") -> tuple[str | None, str | None]:
    """Resolve best namespace for a namespaced resource."""
    if not ensure_kubectl_installed():
        return None, kubectl_not_found_msg()

    if is_cluster_scoped_kind(kind):
        return "", None

    requested = (namespace_hint or "").strip()
    auto_mode = requested.lower() in {"", "auto", "any", "all"}

    if not auto_mode:
        args = kubectl_base_args(namespace=requested) + ["get", kind, name, "-o", "name"]
        code, out, _err = run_kubectl(args)
        if code == 0 and (out or "").strip():
            return requested, None

    namespaces, discover_err = discover_resource_namespaces(kind, name)
    if discover_err:
        return None, kube_access_help(discover_err)

    unique = sorted(set(namespaces))
    if not unique:
        if not auto_mode:
            return None, f"❌ {kind} '{name}' not found in namespace '{requested}' or any namespace."
        return None, f"❌ {kind} '{name}' not found in any namespace."

    if len(unique) == 1:
        return unique[0], None

    return None, (
        f"❌ {kind} '{name}' exists in multiple namespaces: {', '.join(unique)}. "
        "Please specify namespace explicitly."
    )
