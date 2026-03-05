"""User-query intent classification for routing and tool-budget control."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


IntentMode = Literal["direct_read", "incident_rca", "command", "chat", "general"]


@dataclass(frozen=True)
class QueryIntent:
    mode: IntentMode
    resource: str = ""
    namespace: str = ""
    all_namespaces: bool = False


_CHAT_RE = re.compile(r"^(hi|hello|hey|yo|sup|good (morning|afternoon|evening))\b", re.IGNORECASE)
_INCIDENT_KEYWORDS = (
    "why",
    "issue",
    "problem",
    "incident",
    "root cause",
    "rca",
    "not working",
    "down",
    "unreachable",
    "failing",
    "failed",
    "error",
    "outage",
    "degraded",
    "latency",
    "crashloop",
    "oom",
    "pending",
    "diagnose",
    "investigate",
    "fix",
    "heal",
    "self-heal",
    "self heal",
)
_DIRECT_RESOURCES = (
    "pods",
    "nodes",
    "namespaces",
    "deployments",
    "statefulsets",
    "daemonsets",
    "services",
    "ingresses",
    "hpa",
    "events",
    "pvcs",
    "pvs",
)


def _normalize_resource(text: str) -> str:
    if "pod" in text:
        return "pods"
    if "node" in text:
        return "nodes"
    if "deployment" in text:
        return "deployments"
    if "statefulset" in text:
        return "statefulsets"
    if "daemonset" in text:
        return "daemonsets"
    if "service" in text:
        return "services"
    if "ingress" in text:
        return "ingresses"
    if "hpa" in text or "autoscaler" in text:
        return "hpa"
    if "event" in text:
        return "events"
    if "pvc" in text or "persistentvolumeclaim" in text:
        return "pvcs"
    if re.search(r"\bpv\b|persistentvolume", text):
        return "pvs"
    if "namespace" in text:
        return "namespaces"
    return ""


def _extract_namespace(text: str) -> tuple[str, bool]:
    lowered = text.lower()
    if "all namespaces" in lowered or "all ns" in lowered or "whole cluster" in lowered or "cluster" in lowered:
        return "all", True

    m = re.search(r"\b(?:in|from)\s+([a-z0-9-]+)\s+namespace\b", lowered)
    if m:
        return m.group(1), False

    m = re.search(r"\bnamespace\s+([a-z0-9-]+)\b", lowered)
    if m:
        ns = m.group(1)
        if ns not in {"all", "any", "auto"}:
            return ns, False

    return "", False


def classify_query_intent(user_input: str) -> QueryIntent:
    text = (user_input or "").strip()
    lowered = text.lower()

    if not lowered:
        return QueryIntent(mode="general")

    if _CHAT_RE.search(lowered) and len(lowered) <= 24:
        return QueryIntent(mode="chat")

    if lowered.startswith("kubectl ") or lowered.startswith("aws ") or lowered.startswith("helm "):
        return QueryIntent(mode="command")
    if " run kubectl " in f" {lowered} " or " run aws " in f" {lowered} " or " run helm " in f" {lowered} ":
        return QueryIntent(mode="command")

    if any(keyword in lowered for keyword in _INCIDENT_KEYWORDS):
        return QueryIntent(mode="incident_rca")

    read_verb = bool(re.search(r"\b(list|show|get)\b", lowered))
    resource_hit = any(token in lowered for token in _DIRECT_RESOURCES)
    advanced_read = any(token in lowered for token in {" log", "logs", "describe", "yaml", "manifest"})
    if read_verb and resource_hit and not advanced_read:
        resource = _normalize_resource(lowered)
        namespace, all_ns = _extract_namespace(lowered)
        return QueryIntent(
            mode="direct_read",
            resource=resource,
            namespace=namespace,
            all_namespaces=all_ns,
        )

    return QueryIntent(mode="general")
