"""Intent classifier — routes user queries to the right handler."""

from __future__ import annotations

import json
import logging
import re
import time

from langchain_core.messages import HumanMessage, SystemMessage

from ..tracing.tracer import Tracer
from .base import extract_token_usage

log = logging.getLogger(__name__)

INTENT_SYSTEM_PROMPT = """\
Classify the user's infrastructure query into exactly one category.

Categories:
- lookup: Simple data retrieval. Examples: list pods, show nodes, get logs, show events, what namespaces exist, describe deployment X, show ASGs, list instances, show EBS volumes, list load balancers, show secrets, what version, inspect certificates
- diagnose: Troubleshooting, investigation, RCA. Examples: why is X failing, pod crashing, high latency, what's wrong, investigate issue, how are certificates managed
- action: Modify infrastructure. Examples: scale X, restart X, delete X, apply manifest, update config, cordon node, drain node
- explain: Analysis, insight, planning. Examples: what does X do, monthly cost, security posture, should we use spot instances, optimize

IMPORTANT:
- Short affirmations like "yes", "do it", "go ahead", "sure" should be classified as "lookup" — the actual operation was already discussed in prior messages.
- When in doubt between lookup and explain, prefer lookup — it will fetch data first.
- Questions asking "what is" or "show me" or "list" are always lookup.

Output ONLY valid JSON:
{"intent": "<category>", "resources": ["mentioned resources"], "namespaces": ["mentioned namespaces"]}"""


class IntentResult:
    __slots__ = ("intent", "resources", "namespaces")

    def __init__(self, intent: str, resources: list[str] | None = None, namespaces: list[str] | None = None):
        self.intent = intent
        self.resources = resources or []
        self.namespaces = namespaces or []


def classify_intent(
    user_input: str,
    llm,
    model_name: str,
    tracer: Tracer,
    chat_history: list | None = None,
) -> IntentResult:
    """Classify user intent with a single, cheap LLM call (~200 tokens)."""

    messages = [
        SystemMessage(content=INTENT_SYSTEM_PROMPT),
    ]
    # Include recent chat for follow-up context (e.g. user says "yes")
    if chat_history:
        messages.extend(chat_history[-4:])
    messages.append(HumanMessage(content=user_input))

    t0 = time.monotonic()
    try:
        response = llm.invoke(messages)
    except Exception as exc:
        log.warning("Intent classification failed: %s — defaulting to 'diagnose'", exc)
        tracer.step("error", "intent", error=str(exc))
        return IntentResult(intent="diagnose")
    elapsed = int((time.monotonic() - t0) * 1000)

    tokens_in, tokens_out = extract_token_usage(response)
    raw = (response.content or "").strip()

    tracer.step(
        "intent",
        "orchestrator",
        input_summary=user_input[:200],
        output_summary=raw[:200],
        llm_model=model_name,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        duration_ms=elapsed,
    )

    return _parse_intent(raw)


def _parse_intent(raw: str) -> IntentResult:
    """Parse the classifier output, tolerant of markdown fencing and bad JSON."""
    text = raw.strip()
    # Strip markdown code fence
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    # Try JSON parse
    try:
        data = json.loads(text)
        intent = str(data.get("intent", "diagnose")).lower().strip()
        if intent not in ("lookup", "diagnose", "action", "explain"):
            intent = "diagnose"
        return IntentResult(
            intent=intent,
            resources=data.get("resources", []),
            namespaces=data.get("namespaces", []),
        )
    except (json.JSONDecodeError, AttributeError):
        pass

    # Fallback: look for keywords in the raw response
    lower = text.lower()
    if "lookup" in lower:
        return IntentResult(intent="lookup")
    if "action" in lower:
        return IntentResult(intent="action")
    if "explain" in lower:
        return IntentResult(intent="explain")
    return IntentResult(intent="diagnose")
