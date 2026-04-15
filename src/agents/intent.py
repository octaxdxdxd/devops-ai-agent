"""Intent classifier — routes user queries to the right handler."""

from __future__ import annotations

import json
import logging
import time

from langchain_core.messages import HumanMessage, SystemMessage

from ..tracing.tracer import Tracer
from .base import extract_token_usage

log = logging.getLogger(__name__)

INTENT_SYSTEM_PROMPT = """\
Classify the user's infrastructure query into exactly one category.

Categories:
- lookup: Simple data retrieval or resource inspection. Examples: list pods, show nodes, get logs, show events, what namespaces exist, describe deployment X, show ASGs, list instances, show EBS volumes, list load balancers, show secrets, what version, inspect certificates, what are the resource limits, show memory/CPU requests, what's the retention policy, what PVs exist, how many replicas, show ingresses, what domain names
- diagnose: Troubleshooting a PROBLEM, investigation, RCA. Examples: why is X failing, pod crashing, high latency, what's wrong, investigate issue, why are pods pending, why can't I connect, error analysis
- action: Modify infrastructure. Examples: scale X, restart X, delete X, apply manifest, update config, cordon node, drain node, change policy, patch resource
- explain: Analysis, insight, planning, comparison, "how to" questions. Examples: what does X do, monthly cost, security posture, should we use spot instances, optimize, difference between X and Y, how can I improve, what would happen if, cost comparison, migration planning

IMPORTANT:
- When in doubt between lookup and explain, prefer lookup — it will fetch data first.
- Questions asking "what is", "what are", "show me", "list", "how many", "do I have" are always lookup.
- Questions asking about resource properties (limits, requests, policies, versions, sizes) are lookup.
- Questions asking "why" or reporting symptoms are diagnose.
- Questions about "how to" change something or asking for recommendations are explain.
- Do not infer a concrete target, backend, or action from vague acknowledgements like "yes continue", "go on", or "check it".
- If the current user turn is too vague to route safely, set `needs_clarification` to true and provide a one-sentence `clarification_prompt` asking the user to restate the resource, problem, or next step.
- Use only the user's messages as conversation context for routing. Ignore assistant messages when deciding intent.
- Do not infer a mutating action from a short acknowledgement or confirmation. Action requires an explicit change request in the current user message.

Output ONLY valid JSON:
{"intent": "<category>", "resources": ["mentioned resources"], "namespaces": ["mentioned namespaces"], "needs_clarification": false, "clarification_prompt": ""}"""


class IntentResult:
    __slots__ = ("intent", "resources", "namespaces", "needs_clarification", "clarification_prompt")

    def __init__(
        self,
        intent: str,
        resources: list[str] | None = None,
        namespaces: list[str] | None = None,
        needs_clarification: bool = False,
        clarification_prompt: str = "",
    ):
        self.intent = intent
        self.resources = resources or []
        self.namespaces = namespaces or []
        self.needs_clarification = needs_clarification
        self.clarification_prompt = clarification_prompt or ""


def _recent_user_messages(chat_history: list | None, limit: int = 4) -> list:
    recent: list = []
    for message in chat_history or []:
        role = getattr(message, "type", "") or getattr(message, "role", "")
        if str(role).lower() not in {"human", "user"}:
            continue
        recent.append(message)
    return recent[-limit:]


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
    # Use only prior user-authored context for routing.
    if chat_history:
        messages.extend(_recent_user_messages(chat_history))
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
            needs_clarification=bool(data.get("needs_clarification", False)),
            clarification_prompt=str(data.get("clarification_prompt", "") or ""),
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
