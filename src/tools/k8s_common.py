"""Shared Kubernetes kubectl helpers (provider-agnostic: EKS/AKS/GKE/any kubeconfig)."""

from __future__ import annotations

import re
import shutil
import subprocess
from typing import Tuple

from ..config import Config


_DNS_1123_LABEL = re.compile(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")


def is_valid_k8s_name(value: str) -> bool:
    if not value or len(value) > 253:
        return False
    return bool(_DNS_1123_LABEL.match(value))


def ensure_kubectl_installed() -> bool:
    return bool(shutil.which("kubectl"))


def kubectl_base_args(namespace: str | None = None, *, all_namespaces: bool = False) -> list[str]:
    args = ["kubectl"]
    if Config.K8S_KUBECONFIG:
        args.extend(["--kubeconfig", Config.K8S_KUBECONFIG])
    if Config.K8S_CONTEXT:
        args.extend(["--context", Config.K8S_CONTEXT])
    if all_namespaces:
        args.append("-A")
    elif namespace:
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
    return "❌ `kubectl` not found in PATH. Install kubectl and configure kubeconfig/context first."


def kube_access_help(details: str) -> str:
    return (
        "❌ Kubernetes access failed. Ensure kubeconfig/auth is configured for your cluster "
        "(EKS: aws eks update-kubeconfig, AKS: az aks get-credentials, "
        "GKE: gcloud container clusters get-credentials). "
        f"Details: {details or 'unknown error'}"
    )
