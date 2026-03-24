"""User-query intent classification for routing and tool-budget control."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


IntentMode = Literal["incident_rca", "command", "chat", "general"]


@dataclass(frozen=True)
class QueryIntent:
    mode: IntentMode
    namespace: str = ""


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
_NATURAL_COMMAND_PREFIXES = (
    "restart ",
    "scale ",
    "rollout ",
    "delete ",
    "patch ",
    "apply ",
    "reboot ",
    "stop ",
    "start ",
    "terminate ",
    "cordon ",
    "uncordon ",
    "drain ",
)

def classify_query_intent(user_input: str) -> QueryIntent:
    text = (user_input or "").strip()
    lowered = text.lower()

    if not lowered:
        return QueryIntent(mode="general")

    if _CHAT_RE.search(lowered) and len(lowered) <= 24:
        return QueryIntent(mode="general")

    if lowered.startswith("kubectl ") or lowered.startswith("aws ") or lowered.startswith("helm "):
        return QueryIntent(mode="command")
    if " run kubectl " in f" {lowered} " or " run aws " in f" {lowered} " or " run helm " in f" {lowered} ":
        return QueryIntent(mode="command")
    if any(lowered.startswith(prefix) for prefix in _NATURAL_COMMAND_PREFIXES):
        return QueryIntent(mode="command")

    if any(keyword in lowered for keyword in _INCIDENT_KEYWORDS):
        return QueryIntent(mode="incident_rca")

    return QueryIntent(mode="general")
