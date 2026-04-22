"""Message-level handoff helpers for forwarding assistant output to coding tools."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any

import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage

from ..tools.command_preview import render_action_step_preview, render_tool_call_preview
from .session import get_message_content, get_message_trace_id


_CODE_BLOCK_RE = re.compile(r"```(?P<lang>[^\n`]*)\n(?P<body>[\s\S]*?)\n```")
_FENCED_JSON_RE = re.compile(r"^```(?:json)?\s*(?P<body>[\s\S]*?)\s*```$", re.IGNORECASE)
_MAX_SNIPPETS = 5
_MAX_EVIDENCE_ITEMS = 12
_MAX_TEXT_CHARS = 1600
_MAX_TOOL_PREVIEW_CHARS = 240
_MAX_RECENT_RUNS = 3
_MAX_COMMANDS_PER_TRACE = 6
_MAX_PLANNED_CHANGES_PER_TRACE = 4
_MAX_TOP_LEVEL_COMMANDS = 14
_MAX_TOP_LEVEL_CHANGES = 8
_MAX_APPROVAL_ITEMS = 12


def _trim_text(value: str, max_chars: int = _MAX_TEXT_CHARS) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _find_prior_user_request(messages: list[dict], assistant_index: int) -> str:
    for index in range(assistant_index - 1, -1, -1):
        message = messages[index]
        if str(message.get("role") or "").lower() == "user":
            return get_message_content(message)
    return ""


def _find_first_user_request(messages: list[dict], assistant_index: int) -> str:
    for index in range(0, assistant_index):
        message = messages[index]
        if str(message.get("role") or "").lower() == "user":
            return get_message_content(message)
    return ""


def _extract_code_snippets(content: str) -> list[dict[str, str]]:
    snippets: list[dict[str, str]] = []
    for match in _CODE_BLOCK_RE.finditer(str(content or "")):
        language = str(match.group("lang") or "").strip() or "text"
        body = str(match.group("body") or "").strip()
        if not body:
            continue
        snippets.append({"language": language, "content": body})
        if len(snippets) >= _MAX_SNIPPETS:
            break
    return snippets


def _build_transcript(messages: list[dict], assistant_index: int) -> list[dict[str, Any]]:
    transcript: list[dict[str, Any]] = []
    for index, message in enumerate(messages[: assistant_index + 1]):
        role = str(message.get("role") or "").lower()
        if role not in {"user", "assistant"}:
            continue
        entry = {
            "index": index,
            "role": role,
            "content": get_message_content(message),
        }
        trace_id = get_message_trace_id(message)
        if trace_id:
            entry["trace_id"] = trace_id
        transcript.append(entry)
    return transcript


def _build_conversation_summary(messages: list[dict], assistant_index: int, assistant_content: str) -> str:
    transcript = _build_transcript(messages, assistant_index)
    user_messages = [item["content"] for item in transcript if item["role"] == "user" and str(item["content"]).strip()]
    assistant_messages = [item["content"] for item in transcript if item["role"] == "assistant" and str(item["content"]).strip()]

    started_with = user_messages[0] if user_messages else ""
    latest_user = user_messages[-1] if user_messages else ""
    prior_assistant = assistant_messages[-2] if len(assistant_messages) > 1 else (assistant_messages[0] if assistant_messages else "")

    lines: list[str] = []
    if started_with:
        lines.append(f"Conversation started with the operator asking: {_trim_text(started_with, 220)}")
    if len(user_messages) > 1:
        lines.append(f"Latest operator request before handoff: {_trim_text(latest_user, 220)}")
    if prior_assistant and prior_assistant != assistant_content:
        lines.append(f"Previous assistant context: {_trim_text(prior_assistant, 220)}")
    if assistant_content:
        lines.append(f"Latest assistant response: {_trim_text(assistant_content, 320)}")
    return "\n".join(lines).strip()


def _format_timestamp(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return text
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _parse_json_response(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        return {}

    fenced_match = _FENCED_JSON_RE.match(raw)
    if fenced_match:
        raw = str(fenced_match.group("body") or "").strip()

    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(raw[start : end + 1])
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _action_steps_from_commands_json(raw_commands: Any) -> list[dict[str, Any]]:
    if raw_commands in (None, "", [], {}):
        return []

    parsed: Any = raw_commands
    if isinstance(raw_commands, str):
        text = raw_commands.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return []

    if isinstance(parsed, dict):
        items = [parsed]
    elif isinstance(parsed, list):
        items = parsed
    else:
        return []

    steps: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, str):
            command = item.strip()
            if command:
                steps.append({"command": command, "display": command})
            continue
        if not isinstance(item, dict):
            continue

        step: dict[str, Any] = {}
        command = str(item.get("command") or "").strip()
        display = str(item.get("display") or "").strip()
        tool_name = str(item.get("tool") or "").strip()
        tool_args = item.get("args") if isinstance(item.get("args"), dict) else {}

        if command:
            step["command"] = command
        if display:
            step["display"] = display
        if tool_name:
            step["tool"] = tool_name
            step["args"] = tool_args

        if step:
            steps.append(step)
    return steps


def _build_trace_action_step(tool_name: str, tool_args: Any) -> dict[str, Any] | None:
    if not isinstance(tool_args, dict):
        return None

    if tool_name == "kubectl":
        command = str(tool_args.get("command") or "").strip()
        if command:
            return {"command": command, "display": str(tool_args.get("display") or "").strip()}
        return None

    if tool_args.get("tool") or tool_args.get("command"):
        step: dict[str, Any] = {}
        if tool_args.get("tool"):
            step["tool"] = str(tool_args.get("tool") or "").strip()
            step["args"] = tool_args.get("args") if isinstance(tool_args.get("args"), dict) else {}
        elif tool_name:
            step["tool"] = tool_name
            step["args"] = tool_args.get("args") if isinstance(tool_args.get("args"), dict) else {}
        if tool_args.get("command"):
            step["command"] = str(tool_args.get("command") or "").strip()
        if tool_args.get("display"):
            step["display"] = str(tool_args.get("display") or "").strip()
        return step or None

    if tool_name and tool_name != "propose_action":
        return {"tool": tool_name, "args": tool_args}
    return None


def _render_trace_command_preview(tool_name: str, tool_args: Any) -> str:
    if tool_name == "kubectl" and isinstance(tool_args, dict):
        command = str(tool_args.get("command") or "").strip()
        if command:
            return command

    if tool_name == "propose_action":
        return ""

    action_step = _build_trace_action_step(tool_name, tool_args)
    if action_step and (action_step.get("command") or action_step.get("tool")):
        _, preview, _ = render_action_step_preview(action_step)
        if preview:
            return preview

    _, preview, _ = render_tool_call_preview(tool_name, tool_args if isinstance(tool_args, dict) else {})
    return preview


def _normalized_trace_intent(trace: dict | None) -> str:
    if not isinstance(trace, dict):
        return ""
    intent = str(trace.get("intent") or "").strip()
    if intent:
        return intent
    query = str(trace.get("query") or "").strip()
    if query.startswith("Execute action:"):
        return "action_execution"
    return ""


def _is_user_triggered_trace(trace: dict | None) -> bool:
    if not isinstance(trace, dict):
        return False
    query = str(trace.get("query") or "").strip()
    if query == "autonomous_health_scan":
        return False

    steps = [step for step in trace.get("steps", []) if isinstance(step, dict)]
    if steps and all(str(step.get("handler") or "").strip() == "health_scan" for step in steps):
        return False
    return True


def _load_trace(trace_store: Any, trace_id: str) -> dict[str, Any] | None:
    if not trace_id or trace_store is None:
        return None
    try:
        trace = trace_store.load(trace_id)
    except Exception:
        return None
    return trace if isinstance(trace, dict) else None


def _collect_message_trace_ids(messages: list[dict], assistant_index: int) -> list[str]:
    trace_ids: list[str] = []
    seen: set[str] = set()
    for index in range(assistant_index, -1, -1):
        trace_id = str(get_message_trace_id(messages[index]) or "").strip()
        if not trace_id or trace_id in seen:
            continue
        seen.add(trace_id)
        trace_ids.append(trace_id)
    return trace_ids


def _select_recent_user_traces(
    messages: list[dict],
    assistant_index: int,
    trace_store: Any = None,
    current_trace: dict | None = None,
    limit: int = _MAX_RECENT_RUNS,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_trace(trace: dict | None) -> None:
        if not isinstance(trace, dict) or not _is_user_triggered_trace(trace):
            return
        trace_id = str(trace.get("trace_id") or "").strip()
        if not trace_id or trace_id in seen:
            return
        seen.add(trace_id)
        selected.append(trace)

    add_trace(current_trace)

    for trace_id in _collect_message_trace_ids(messages, assistant_index):
        if len(selected) >= limit:
            break
        if trace_id in seen:
            continue
        add_trace(_load_trace(trace_store, trace_id))

    if trace_store is not None and len(selected) < limit:
        try:
            recent_ids = trace_store.list_recent(limit=limit * 8)
        except Exception:
            recent_ids = []
        for trace_id in recent_ids:
            if len(selected) >= limit:
                break
            trace_id = str(trace_id or "").strip()
            if not trace_id or trace_id in seen:
                continue
            add_trace(_load_trace(trace_store, trace_id))

    return selected[:limit]


def _dedupe_text_items(items: list[str], limit: int) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append(text)
        if len(deduped) >= limit:
            break
    return deduped


def _extract_trace_commands(trace: dict | None, trace_id: str, limit: int = _MAX_COMMANDS_PER_TRACE) -> list[dict[str, str]]:
    if not isinstance(trace, dict):
        return []

    commands: list[dict[str, str]] = []
    for step in trace.get("steps", []):
        if not isinstance(step, dict):
            continue
        step_type = str(step.get("step_type") or "").strip()
        if step_type not in {"tool_call", "verify"}:
            continue

        tool_name = str(step.get("tool_name") or "").strip()
        if tool_name == "propose_action":
            continue

        preview = _render_trace_command_preview(tool_name, step.get("tool_args"))
        if not preview:
            continue

        if step_type == "verify":
            status = "verification"
        elif str(step.get("handler") or "").strip() == "action":
            status = "executed"
        else:
            status = "observed"

        result = _trim_text(
            str(step.get("tool_result_preview") or step.get("output_summary") or step.get("error") or ""),
            180,
        )

        commands.append(
            {
                "trace_id": trace_id,
                "status": status,
                "tool_name": tool_name,
                "preview": preview,
                "result": result,
            }
        )
        if len(commands) >= limit:
            break
    return commands


def _extract_planned_changes(trace: dict | None, trace_id: str, limit: int = _MAX_PLANNED_CHANGES_PER_TRACE) -> list[dict[str, str]]:
    if not isinstance(trace, dict):
        return []

    planned_changes: list[dict[str, str]] = []
    for step in trace.get("steps", []):
        if not isinstance(step, dict) or str(step.get("tool_name") or "").strip() != "propose_action":
            continue

        tool_args = step.get("tool_args") if isinstance(step.get("tool_args"), dict) else {}
        description = _trim_text(str(tool_args.get("description") or ""), 260)
        risk = str(tool_args.get("risk") or "").strip()
        expected_outcome = _trim_text(str(tool_args.get("expected_outcome") or ""), 220)
        verification_command = _trim_text(str(tool_args.get("verification_command") or ""), 220)
        action_steps = _action_steps_from_commands_json(tool_args.get("commands_json"))

        if not action_steps and description:
            planned_changes.append(
                {
                    "trace_id": trace_id,
                    "display": description,
                    "preview": description,
                    "description": description,
                    "risk": risk,
                    "expected_outcome": expected_outcome,
                    "verification_command": verification_command,
                }
            )

        for action_step in action_steps:
            label, preview, _ = render_action_step_preview(action_step)
            rendered = preview or label
            if not rendered:
                continue
            planned_changes.append(
                {
                    "trace_id": trace_id,
                    "display": label or rendered,
                    "preview": rendered,
                    "description": description,
                    "risk": risk,
                    "expected_outcome": expected_outcome,
                    "verification_command": verification_command,
                }
            )
            if len(planned_changes) >= limit:
                return planned_changes
    return planned_changes[:limit]


def _extract_approvals_and_verification(trace: dict | None, trace_id: str, limit: int = 6) -> list[dict[str, str]]:
    if not isinstance(trace, dict):
        return []

    items: list[dict[str, str]] = []
    for step in trace.get("steps", []):
        if not isinstance(step, dict):
            continue
        step_type = str(step.get("step_type") or "").strip()
        if step_type == "approval":
            detail = _trim_text(str(step.get("output_summary") or ""), 260)
            if detail:
                items.append({"trace_id": trace_id, "kind": "approval", "detail": detail})
        elif step_type == "verify":
            preview = _render_trace_command_preview(str(step.get("tool_name") or ""), step.get("tool_args"))
            result = _trim_text(
                str(step.get("tool_result_preview") or step.get("output_summary") or step.get("error") or ""),
                180,
            )
            detail = preview or "verification step"
            if result:
                detail = f"{detail} -> {result}"
            items.append({"trace_id": trace_id, "kind": "verification", "detail": detail})
        if len(items) >= limit:
            break
    return items


def _extract_key_results(trace: dict | None, limit: int = 5) -> list[str]:
    if not isinstance(trace, dict):
        return []

    results: list[str] = []
    for step in trace.get("steps", []):
        if not isinstance(step, dict):
            continue

        step_type = str(step.get("step_type") or "").strip()
        if step_type not in {"tool_call", "verify", "approval", "error"}:
            continue

        tool_name = str(step.get("tool_name") or "").strip()
        preview = _trim_text(str(step.get("tool_result_preview") or ""), _MAX_TOOL_PREVIEW_CHARS)
        output_summary = _trim_text(str(step.get("output_summary") or ""), _MAX_TOOL_PREVIEW_CHARS)
        error = _trim_text(str(step.get("error") or ""), _MAX_TOOL_PREVIEW_CHARS)

        if step_type == "approval" and output_summary:
            results.append(f"approval: {output_summary}")
        elif error:
            label = tool_name or step_type
            results.append(f"{label}: {error}")
        elif preview:
            label = tool_name or step_type
            results.append(f"{label}: {preview}")
        elif output_summary:
            label = tool_name or step_type
            results.append(f"{label}: {output_summary}")

        if len(results) >= limit:
            break
    return _dedupe_text_items(results, limit)


def _summarize_last_step(trace: dict | None) -> str:
    if not isinstance(trace, dict):
        return ""

    for step in reversed(trace.get("steps", [])):
        if not isinstance(step, dict):
            continue

        step_type = str(step.get("step_type") or "").strip()
        tool_name = str(step.get("tool_name") or "").strip()
        output_summary = _trim_text(str(step.get("output_summary") or ""), 220)
        preview = _trim_text(str(step.get("tool_result_preview") or ""), 220)
        error = _trim_text(str(step.get("error") or ""), 220)

        if step_type == "approval" and output_summary:
            return f"approval queued: {output_summary}"
        if step_type == "verify":
            command = _render_trace_command_preview(tool_name, step.get("tool_args"))
            if command and preview:
                return f"verification `{command}` reported {preview}"
            if command:
                return f"verification ran `{command}`"
            if preview:
                return f"verification reported {preview}"
        if step_type == "tool_call" and tool_name == "propose_action":
            description = _trim_text(str((step.get("tool_args") or {}).get("description") or ""), 220)
            if description:
                return f"proposed change: {description}"
        if error:
            label = tool_name or step_type or "step"
            return f"{label} failed: {error}"
        if preview:
            label = tool_name or step_type or "step"
            return f"{label} returned {preview}"
        if output_summary:
            return output_summary
    return ""


def _summarize_trace_outcome(outcome: str) -> str:
    mapping = {
        "action_proposed": "ended with a pending change proposal",
        "action_executed": "executed a live change",
        "action_failed": "attempted a change but verification or execution failed",
        "answered": "finished with an answer only",
        "error": "failed before completing the run",
    }
    return mapping.get(str(outcome or "").strip(), "completed with an unknown outcome")


def _summarize_trace_focus(query: str) -> str:
    text = str(query or "").strip()
    if not text:
        return "the current issue"
    prefix = "Execute action:"
    if text.startswith(prefix):
        text = text[len(prefix) :].strip()
    return _trim_text(text, 160)


def _build_recent_run_digest(trace: dict[str, Any]) -> dict[str, Any]:
    trace_id = str(trace.get("trace_id") or "").strip()
    query = str(trace.get("query") or "").strip()
    intent = _normalized_trace_intent(trace)
    outcome = str(trace.get("outcome") or "").strip()
    last_step = _summarize_last_step(trace)
    commands_ran = _extract_trace_commands(trace, trace_id)
    planned_changes = _extract_planned_changes(trace, trace_id)
    approvals_and_verification = _extract_approvals_and_verification(trace, trace_id)
    key_results = _extract_key_results(trace)

    summary_bits = [
        f"Focused on {_summarize_trace_focus(query)}",
        _summarize_trace_outcome(outcome),
    ]
    if planned_changes:
        summary_bits.append(f"planned change: {_trim_text(planned_changes[0].get('display') or planned_changes[0].get('preview') or '', 180)}")
    elif commands_ran:
        summary_bits.append(f"key command: {_trim_text(commands_ran[0].get('preview') or '', 180)}")
    if last_step:
        summary_bits.append(f"last step: {last_step}")

    return {
        "trace_id": trace_id,
        "query": query,
        "intent": intent,
        "outcome": outcome,
        "started_at": _format_timestamp(trace.get("started_at")),
        "completed_at": _format_timestamp(trace.get("completed_at")),
        "summary": "; ".join(bit for bit in summary_bits if bit),
        "last_step": last_step,
        "commands_ran": commands_ran,
        "planned_changes": planned_changes,
        "approvals_and_verification": approvals_and_verification,
        "key_results": key_results,
    }


def _flatten_recent_run_entries(recent_runs: list[dict[str, Any]], field_name: str, limit: int) -> list[dict[str, str]]:
    flattened: list[dict[str, str]] = []
    seen: set[str] = set()
    for run in recent_runs:
        for item in run.get(field_name, []) or []:
            if not isinstance(item, dict):
                continue
            key = json.dumps(item, sort_keys=True, ensure_ascii=True)
            if key in seen:
                continue
            seen.add(key)
            flattened.append({k: str(v or "") for k, v in item.items() if v not in (None, "")})
            if len(flattened) >= limit:
                return flattened
    return flattened


def _build_recent_runs_summary(recent_runs: list[dict[str, Any]]) -> str:
    if not recent_runs:
        return "No recent trace-backed runs were available for this handoff."
    lines = []
    for index, run in enumerate(recent_runs, start=1):
        trace_id = str(run.get("trace_id") or "").strip() or "no-trace"
        outcome = str(run.get("outcome") or "unknown").strip() or "unknown"
        summary = str(run.get("summary") or "").strip()
        lines.append(f"Run {index} ({trace_id}, {outcome}): {summary}")
    return "\n".join(lines)


def _build_operator_need_summary(
    user_request: str,
    conversation_summary: str,
    recent_runs: list[dict[str, Any]],
    assistant_content: str,
) -> str:
    request = _trim_text(user_request or assistant_content, 240)
    if recent_runs:
        last_run = str(recent_runs[0].get("summary") or "").strip()
        return (
            f"Continue the operator request `{request}` with minimal re-analysis. "
            f"The latest trace-backed run concluded: {last_run}"
        )
    if conversation_summary:
        return f"Continue the operator request `{request}` using the visible conversation context: {conversation_summary}"
    return f"Continue the operator request `{request}` and turn it into concrete code or infrastructure work."


def _build_infra_change_brief(
    recent_runs: list[dict[str, Any]],
    assistant_content: str,
    snippets: list[dict[str, str]],
) -> str:
    for run in recent_runs:
        planned_changes = run.get("planned_changes") or []
        if planned_changes:
            first_change = planned_changes[0]
            change_text = str(first_change.get("description") or first_change.get("display") or first_change.get("preview") or "").strip()
            expected_outcome = str(first_change.get("expected_outcome") or "").strip()
            verification_command = str(first_change.get("verification_command") or "").strip()
            parts = [change_text] if change_text else []
            if expected_outcome:
                parts.append(f"Expected outcome: {expected_outcome}")
            if verification_command:
                parts.append(f"Suggested verification: {verification_command}")
            if parts:
                return " ".join(parts)

    if snippets:
        snippet_languages = ", ".join(snippet.get("language", "text") for snippet in snippets[:3])
        return (
            "The assistant already included implementation-oriented snippets "
            f"({snippet_languages}). Convert them into the exact repo or IaC change that should persist the fix."
        )

    if assistant_content:
        return _trim_text(
            "No explicit typed action proposal was captured. Infer the minimum durable code or infra change from the assistant response and trace evidence: "
            + assistant_content,
            320,
        )

    return "No explicit infrastructure change was proposed yet. The next agent may need to infer the repo or IaC delta from the evidence chain."


def _default_takeover_guidance(planned_changes: list[dict[str, str]], commands_ran: list[dict[str, str]]) -> list[str]:
    guidance = [
        "Reuse the recent trace-backed work before repeating investigation.",
        "If live infrastructure was already changed, reconcile the matching repo or IaC update so the fix persists.",
        "Preserve the scope of approved or proposed changes unless newer evidence contradicts them.",
        "Use the recorded verification commands and outcomes as the first validation path.",
    ]
    if not planned_changes:
        guidance.append("No clear durable change was proposed yet; identify the smallest code or infra artifact that must be updated.")
    if not commands_ran:
        guidance.append("The handoff has little command history; rely more on the structured summaries and transcript.")
    return guidance[:6]


def _build_system_prompt(package: dict[str, Any]) -> str:
    guidance = list(package.get("takeover_guidance") or [])
    lines = [
        "You are a coding agent taking over from an AIOps operator handoff.",
        "Use the structured handoff below as working memory and avoid restarting diagnosis unless the evidence is stale, contradictory, or clearly incomplete.",
        "Prefer exact code, manifest, Terraform, or script changes over generic advice.",
        "If a live change already happened, determine whether source-controlled infrastructure or code must also change to make that fix durable.",
        "State assumptions, keep scope aligned with the operator request, and validate the final state with the strongest available checks.",
    ]
    if guidance:
        lines.extend(["", "Takeover guidance:"])
        lines.extend(f"- {item}" for item in guidance)
    return "\n".join(lines).strip()


def _build_task_prompt(package: dict[str, Any]) -> str:
    lines = [
        f"Primary objective: {package.get('operator_need_summary', '')}",
        f"Infra/code change brief: {package.get('infra_change_brief', '')}",
        f"What happened last: {package.get('last_run_summary', '')}",
        "",
        "Recent runs:",
        str(package.get("recent_runs_summary") or "(not available)"),
    ]

    commands_ran = list(package.get("commands_ran") or [])
    if commands_ran:
        lines.extend(["", "Commands already run:"])
        for item in commands_ran:
            preview = str(item.get("preview") or "").strip()
            status = str(item.get("status") or "").strip()
            result = str(item.get("result") or "").strip()
            line = f"- [{status}] {preview}" if status else f"- {preview}"
            if result:
                line += f" -> {result}"
            lines.append(line)

    planned_changes = list(package.get("planned_changes") or [])
    if planned_changes:
        lines.extend(["", "Proposed or approved changes to preserve or implement:"])
        for item in planned_changes:
            text = str(item.get("display") or item.get("preview") or "").strip()
            risk = str(item.get("risk") or "").strip()
            expected = str(item.get("expected_outcome") or "").strip()
            line = f"- {text}"
            if risk:
                line += f" (risk={risk})"
            if expected:
                line += f" -> {expected}"
            lines.append(line)

    approvals = list(package.get("approvals_and_verification") or [])
    if approvals:
        lines.extend(["", "Approvals and verification:"])
        for item in approvals:
            kind = str(item.get("kind") or "").strip()
            detail = str(item.get("detail") or "").strip()
            if detail:
                lines.append(f"- [{kind}] {detail}" if kind else f"- {detail}")

    guidance = list(package.get("takeover_guidance") or [])
    if guidance:
        lines.extend(["", "Takeover guidance:"])
        lines.extend(f"- {item}" for item in guidance)

    lines.extend(["", "Acceptance criteria:"])
    lines.extend(f"- {item}" for item in package.get("acceptance_criteria", []))
    return "\n".join(lines).strip()


def _build_current_trace_context(trace: dict | None, recent_runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(trace, dict):
        return {}
    trace_id = str(trace.get("trace_id") or "").strip()
    matching_run = next((run for run in recent_runs if str(run.get("trace_id") or "").strip() == trace_id), None)
    return {
        "trace_id": trace_id,
        "query": str(trace.get("query") or "").strip(),
        "intent": _normalized_trace_intent(trace),
        "outcome": str(trace.get("outcome") or "").strip(),
        "started_at": _format_timestamp(trace.get("started_at")),
        "completed_at": _format_timestamp(trace.get("completed_at")),
        "summary": str((matching_run or {}).get("summary") or "").strip(),
        "last_step": str((matching_run or {}).get("last_step") or _summarize_last_step(trace)).strip(),
    }


def _build_trace_step_summaries(trace: dict | None) -> list[dict[str, Any]]:
    if not isinstance(trace, dict):
        return []
    summaries: list[dict[str, Any]] = []
    for step in trace.get("steps", []):
        if not isinstance(step, dict):
            continue
        entry = {
            "step_type": str(step.get("step_type") or ""),
            "handler": str(step.get("handler") or ""),
        }
        if step.get("tool_name"):
            entry["tool_name"] = str(step.get("tool_name") or "")
        if step.get("tool_args"):
            entry["tool_args"] = step.get("tool_args")
        if step.get("input_summary"):
            entry["input_summary"] = str(step.get("input_summary") or "")
        if step.get("output_summary"):
            entry["output_summary"] = str(step.get("output_summary") or "")
        if step.get("tool_result_preview"):
            entry["tool_result_preview"] = str(step.get("tool_result_preview") or "")
        if step.get("error"):
            entry["error"] = str(step.get("error") or "")
        summaries.append(entry)
    return summaries


def _build_evidence(recent_runs: list[dict[str, Any]], current_trace: dict | None) -> list[str]:
    items: list[str] = []
    if isinstance(current_trace, dict):
        items.extend(_extract_key_results(current_trace, limit=_MAX_EVIDENCE_ITEMS))
    for run in recent_runs:
        items.extend(str(item or "").strip() for item in run.get("key_results", []) or [])
    return _dedupe_text_items(items, _MAX_EVIDENCE_ITEMS)


def _apply_llm_enrichment(package: dict[str, Any], llm_enrichment: dict[str, Any], model_name: str) -> dict[str, Any]:
    updated = dict(package)
    for field_name in (
        "operator_need_summary",
        "whole_conversation_summary",
        "recent_runs_summary",
        "last_run_summary",
        "infra_change_brief",
    ):
        value = str(llm_enrichment.get(field_name) or "").strip()
        if value:
            updated[field_name] = value

    guidance = llm_enrichment.get("takeover_guidance")
    if isinstance(guidance, list):
        normalized_guidance = _dedupe_text_items([str(item or "").strip() for item in guidance], 6)
        if normalized_guidance:
            updated["takeover_guidance"] = normalized_guidance

    updated["llm_enriched"] = True
    updated["llm_model"] = model_name
    updated["llm_error"] = None
    updated["system_prompt"] = _build_system_prompt(updated)
    updated["task_prompt"] = _build_task_prompt(updated)
    return updated


def build_handoff_package(
    messages: list[dict],
    assistant_index: int,
    trace: dict | None = None,
    recent_traces: list[dict[str, Any]] | None = None,
    llm_enrichment: dict[str, Any] | None = None,
    model_name: str = "",
) -> dict[str, Any]:
    """Build a structured handoff package for one assistant message."""
    message = messages[assistant_index]
    assistant_content = get_message_content(message)
    trace_id = get_message_trace_id(message)
    original_user_request = _find_first_user_request(messages, assistant_index)
    user_request = _find_prior_user_request(messages, assistant_index)
    snippets = _extract_code_snippets(assistant_content)
    transcript = _build_transcript(messages, assistant_index)
    conversation_summary = _build_conversation_summary(messages, assistant_index, assistant_content)
    trace_steps = _build_trace_step_summaries(trace)

    if recent_traces is None:
        recent_trace_list = [trace] if isinstance(trace, dict) else []
    else:
        recent_trace_list = [item for item in recent_traces if isinstance(item, dict)]

    recent_runs = [_build_recent_run_digest(item) for item in recent_trace_list[:_MAX_RECENT_RUNS]]
    commands_ran = _flatten_recent_run_entries(recent_runs, "commands_ran", _MAX_TOP_LEVEL_COMMANDS)
    planned_changes = _flatten_recent_run_entries(recent_runs, "planned_changes", _MAX_TOP_LEVEL_CHANGES)
    approvals_and_verification = _flatten_recent_run_entries(
        recent_runs,
        "approvals_and_verification",
        _MAX_APPROVAL_ITEMS,
    )

    assistant_summary = _trim_text(assistant_content, _MAX_TEXT_CHARS)
    last_run_summary = str((recent_runs[0] or {}).get("summary") or "").strip() if recent_runs else "No recent trace-backed run was available."
    recent_runs_summary = _build_recent_runs_summary(recent_runs)
    operator_need_summary = _build_operator_need_summary(user_request, conversation_summary, recent_runs, assistant_content)
    infra_change_brief = _build_infra_change_brief(recent_runs, assistant_content, snippets)
    takeover_guidance = _default_takeover_guidance(planned_changes, commands_ran)

    title = _trim_text(user_request or assistant_content, 96) or "AIOps handoff"
    acceptance_criteria = [
        "Use the recent run history and commands to avoid repeating work that already happened.",
        "Implement or package the exact repo, manifest, or infrastructure change needed to persist the intended fix.",
        "Validate the outcome with the strongest available verification path and summarize what still needs operator follow-up.",
    ]

    package: dict[str, Any] = {
        "title": title,
        "handoff_mode": "coding_agent",
        "source_type": "assistant_message",
        "message_index": assistant_index,
        "generated_at": _now_iso(),
        "trace_id": trace_id,
        "model_name": model_name,
        "original_user_request": original_user_request,
        "user_request": user_request,
        "assistant_full_text": assistant_content,
        "assistant_summary": assistant_summary,
        "conversation_summary": conversation_summary,
        "whole_conversation_summary": conversation_summary,
        "conversation_transcript": transcript,
        "operator_need_summary": operator_need_summary,
        "recent_runs_summary": recent_runs_summary,
        "last_run_summary": last_run_summary,
        "infra_change_brief": infra_change_brief,
        "current_trace": _build_current_trace_context(trace, recent_runs),
        "recent_runs": recent_runs,
        "commands_ran": commands_ran,
        "planned_changes": planned_changes,
        "approvals_and_verification": approvals_and_verification,
        "trace_steps": trace_steps,
        "evidence": _build_evidence(recent_runs, trace),
        "snippets": snippets,
        "takeover_guidance": takeover_guidance,
        "acceptance_criteria": acceptance_criteria,
        "llm_enriched": False,
        "llm_model": "",
        "llm_error": None,
    }
    package["system_prompt"] = _build_system_prompt(package)
    package["task_prompt"] = _build_task_prompt(package)

    if isinstance(llm_enrichment, dict) and llm_enrichment:
        package = _apply_llm_enrichment(package, llm_enrichment, model_name)

    return package


def generate_handoff_llm_enrichment(package: dict[str, Any], llm: Any, model_name: str = "") -> dict[str, Any]:
    if llm is None:
        return {}

    recent_runs = []
    for run in list(package.get("recent_runs") or [])[:_MAX_RECENT_RUNS]:
        recent_runs.append(
            {
                "trace_id": run.get("trace_id"),
                "query": run.get("query"),
                "intent": run.get("intent"),
                "outcome": run.get("outcome"),
                "summary": run.get("summary"),
                "last_step": run.get("last_step"),
                "commands_ran": [item.get("preview") for item in run.get("commands_ran", [])],
                "planned_changes": [item.get("display") or item.get("preview") for item in run.get("planned_changes", [])],
                "approvals_and_verification": [item.get("detail") for item in run.get("approvals_and_verification", [])],
            }
        )

    payload = {
        "title": package.get("title"),
        "operator_request": package.get("user_request") or package.get("original_user_request"),
        "assistant_summary": package.get("assistant_summary"),
        "assistant_full_text": _trim_text(str(package.get("assistant_full_text") or ""), 1200),
        "conversation_summary": package.get("conversation_summary"),
        "current_trace": package.get("current_trace"),
        "recent_runs": recent_runs,
        "commands_ran": [item.get("preview") for item in package.get("commands_ran", [])],
        "planned_changes": [item.get("display") or item.get("preview") for item in package.get("planned_changes", [])],
        "approvals_and_verification": [item.get("detail") for item in package.get("approvals_and_verification", [])],
        "evidence": package.get("evidence"),
        "transcript_excerpt": list(package.get("conversation_transcript") or [])[-8:],
    }

    system_prompt = (
        "You summarize AIOps handoff context for a coding agent. "
        "Return strict JSON only. Do not use markdown fences. "
        "Do not invent facts beyond the provided context."
    )
    human_prompt = (
        "Produce JSON with the exact keys below:\n"
        "{\n"
        '  "operator_need_summary": "1-3 factual sentences describing what the coding agent should accomplish next",\n'
        '  "whole_conversation_summary": "1 concise paragraph covering the problem evolution",\n'
        '  "recent_runs_summary": "1 concise paragraph explaining what happened across the recent runs",\n'
        '  "last_run_summary": "1 concise paragraph focused on the latest run only",\n'
        '  "infra_change_brief": "1 concise paragraph describing the likely durable code or infrastructure delta",\n'
        '  "takeover_guidance": ["3 to 6 short imperative bullets for the next coding agent"]\n'
        "}\n\n"
        "Context JSON:\n"
        f"{json.dumps(payload, indent=2, ensure_ascii=True)}"
    )

    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]
    )
    parsed = _parse_json_response(getattr(response, "content", ""))
    if not parsed:
        return {}
    if model_name:
        parsed["llm_model"] = model_name
    return parsed


def enrich_handoff_package(package: dict[str, Any], llm: Any = None, model_name: str = "") -> dict[str, Any]:
    if llm is None:
        return package

    try:
        enrichment = generate_handoff_llm_enrichment(package, llm, model_name=model_name)
    except Exception as exc:
        failed = dict(package)
        failed["llm_error"] = str(exc)
        failed["llm_enriched"] = False
        return failed

    if not enrichment:
        failed = dict(package)
        failed["llm_error"] = "LLM returned no structured enrichment."
        failed["llm_enriched"] = False
        return failed

    return _apply_llm_enrichment(package, enrichment, model_name)


def _render_transcript(transcript: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for item in transcript:
        role = str(item.get("role") or "").strip().upper()
        index = item.get("index")
        trace_id = str(item.get("trace_id") or "").strip()
        header = f"[{index}] {role}"
        if trace_id:
            header += f" (trace {trace_id})"
        lines.extend([header, str(item.get("content") or "").strip(), ""])
    return "\n".join(lines).strip()


def _render_recent_runs_markdown(recent_runs: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    if not recent_runs:
        return ["(no recent trace-backed runs available)"]

    for index, run in enumerate(recent_runs, start=1):
        lines.extend(
            [
                f"### Run {index}",
                f"- Trace ID: {run.get('trace_id', '')}",
                f"- Intent: {run.get('intent', '') or 'unknown'}",
                f"- Outcome: {run.get('outcome', '') or 'unknown'}",
                f"- Started: {run.get('started_at', '') or 'unknown'}",
                f"- Completed: {run.get('completed_at', '') or 'unknown'}",
                f"- Summary: {run.get('summary', '') or '(not available)'}",
            ]
        )
        last_step = str(run.get("last_step") or "").strip()
        if last_step:
            lines.append(f"- Last step: {last_step}")

        commands = list(run.get("commands_ran") or [])
        if commands:
            lines.append("- Commands:")
            lines.extend(
                f"  - [{item.get('status', '') or 'unknown'}] {item.get('preview', '')}{' -> ' + item.get('result', '') if item.get('result') else ''}"
                for item in commands
            )

        planned_changes = list(run.get("planned_changes") or [])
        if planned_changes:
            lines.append("- Planned changes:")
            lines.extend(
                f"  - {item.get('display', '') or item.get('preview', '')}{' (risk=' + item.get('risk', '') + ')' if item.get('risk') else ''}"
                for item in planned_changes
            )

        approvals = list(run.get("approvals_and_verification") or [])
        if approvals:
            lines.append("- Approvals and verification:")
            lines.extend(f"  - [{item.get('kind', '')}] {item.get('detail', '')}" for item in approvals)

        key_results = list(run.get("key_results") or [])
        if key_results:
            lines.append("- Key results:")
            lines.extend(f"  - {item}" for item in key_results)

        lines.append("")
    return lines[:-1] if lines and not lines[-1] else lines


def _render_commands_markdown(commands: list[dict[str, str]]) -> list[str]:
    if not commands:
        return ["(no command history captured)"]
    lines: list[str] = []
    for item in commands:
        status = str(item.get("status") or "").strip()
        preview = str(item.get("preview") or "").strip()
        result = str(item.get("result") or "").strip()
        text = f"- [{status}] {preview}" if status else f"- {preview}"
        if result:
            text += f" -> {result}"
        lines.append(text)
    return lines


def _render_planned_changes_markdown(changes: list[dict[str, str]]) -> list[str]:
    if not changes:
        return ["(no proposed or approved change steps captured)"]
    lines: list[str] = []
    for item in changes:
        preview = str(item.get("display") or item.get("preview") or "").strip()
        risk = str(item.get("risk") or "").strip()
        expected_outcome = str(item.get("expected_outcome") or "").strip()
        verification_command = str(item.get("verification_command") or "").strip()
        line = f"- {preview}"
        if risk:
            line += f" (risk={risk})"
        if expected_outcome:
            line += f" -> {expected_outcome}"
        lines.append(line)
        if verification_command:
            lines.append(f"  verification: {verification_command}")
    return lines


def _render_approvals_markdown(items: list[dict[str, str]]) -> list[str]:
    if not items:
        return ["(no approval or verification items captured)"]
    lines: list[str] = []
    for item in items:
        kind = str(item.get("kind") or "").strip()
        detail = str(item.get("detail") or "").strip()
        lines.append(f"- [{kind}] {detail}" if kind else f"- {detail}")
    return lines


def render_agent_handoff_prompt(package: dict[str, Any]) -> str:
    """Render the primary coding-agent handoff prompt."""
    lines = [
        "# Coding Agent Handoff",
        "",
        "## System Prompt",
        str(package.get("system_prompt") or "(not available)"),
        "",
        "## Task Prompt",
        str(package.get("task_prompt") or "(not available)"),
        "",
        "## Operator Need Summary",
        str(package.get("operator_need_summary") or "(not available)"),
        "",
        "## Whole Conversation Summary",
        str(package.get("whole_conversation_summary") or "(not available)"),
        "",
        "## Infra Change Brief",
        str(package.get("infra_change_brief") or "(not available)"),
        "",
        "## Recent Runs Summary",
        str(package.get("recent_runs_summary") or "(not available)"),
        "",
        "## What Happened Last",
        str(package.get("last_run_summary") or "(not available)"),
        "",
        "## Commands Already Run",
    ]
    lines.extend(_render_commands_markdown(list(package.get("commands_ran") or [])))

    lines.extend(["", "## Proposed Or Approved Change Steps"])
    lines.extend(_render_planned_changes_markdown(list(package.get("planned_changes") or [])))

    lines.extend(["", "## Approvals And Verification"])
    lines.extend(_render_approvals_markdown(list(package.get("approvals_and_verification") or [])))

    evidence = list(package.get("evidence") or [])
    lines.extend(["", "## Evidence"])
    if evidence:
        lines.extend(f"- {item}" for item in evidence)
    else:
        lines.append("(no trace-backed evidence captured)")

    lines.extend(["", "## Recent Trace Digests"])
    lines.extend(_render_recent_runs_markdown(list(package.get("recent_runs") or [])))

    snippets = list(package.get("snippets") or [])
    if snippets:
        lines.extend(["", "## Relevant Snippets"])
        for snippet in snippets:
            language = str(snippet.get("language") or "text").strip() or "text"
            content = str(snippet.get("content") or "").strip()
            if not content:
                continue
            lines.extend(["", f"```{language}", content, "```"])

    transcript = list(package.get("conversation_transcript") or [])
    if transcript:
        lines.extend(["", "## Full Conversation Transcript", "```text", _render_transcript(transcript), "```"])

    lines.extend(["", "## Acceptance Criteria"])
    lines.extend(f"- {item}" for item in package.get("acceptance_criteria", []))

    return "\n".join(lines).strip()


def render_codex_handoff_prompt(package: dict[str, Any]) -> str:
    """Render a Codex-compatible prompt from the richer handoff package."""
    return (
        "You are taking over from the AIOps assistant in Codex.\n"
        "Use the structured handoff below as the primary context unless the repository or fresh validation proves otherwise.\n\n"
        + render_agent_handoff_prompt(package)
    ).strip()


def render_handoff_markdown(package: dict[str, Any]) -> str:
    """Render the handoff package as markdown."""
    lines = [
        f"# {package.get('title', 'AIOps handoff')}",
        "",
        f"- Handoff mode: {package.get('handoff_mode', 'coding_agent')}",
        f"- Source type: {package.get('source_type', 'assistant_message')}",
        f"- Message index: {package.get('message_index', '')}",
        f"- Generated at: {package.get('generated_at', '')}",
    ]

    trace_id = str(package.get("trace_id") or "").strip()
    if trace_id:
        lines.append(f"- Trace ID: {trace_id}")
    if package.get("llm_enriched"):
        lines.append(f"- Summary model: {package.get('llm_model', '')}")
    elif package.get("llm_error"):
        lines.append(f"- LLM enrichment: unavailable ({package.get('llm_error')})")

    lines.extend(
        [
            "",
            "## Original Conversation Opener",
            str(package.get("original_user_request") or package.get("user_request") or "(not available)"),
            "",
            "## Original Request",
            str(package.get("user_request") or "(not available)"),
            "",
            "## Operator Need Summary",
            str(package.get("operator_need_summary") or "(not available)"),
            "",
            "## Whole Conversation Summary",
            str(package.get("whole_conversation_summary") or "(not available)"),
            "",
            "## Infra Change Brief",
            str(package.get("infra_change_brief") or "(not available)"),
            "",
            "## System Prompt",
            str(package.get("system_prompt") or "(not available)"),
            "",
            "## Task Prompt",
            str(package.get("task_prompt") or "(not available)"),
            "",
            "## Recent Runs Summary",
            str(package.get("recent_runs_summary") or "(not available)"),
            "",
            "## What Happened Last",
            str(package.get("last_run_summary") or "(not available)"),
            "",
            "## Commands Already Run",
        ]
    )
    lines.extend(_render_commands_markdown(list(package.get("commands_ran") or [])))

    lines.extend(["", "## Proposed Or Approved Change Steps"])
    lines.extend(_render_planned_changes_markdown(list(package.get("planned_changes") or [])))

    lines.extend(["", "## Approvals And Verification"])
    lines.extend(_render_approvals_markdown(list(package.get("approvals_and_verification") or [])))

    evidence = list(package.get("evidence") or [])
    lines.extend(["", "## Evidence"])
    if evidence:
        lines.extend(f"- {item}" for item in evidence)
    else:
        lines.append("(no evidence captured)")

    trace_query = str((package.get("current_trace") or {}).get("query") or "").strip()
    if trace_query:
        lines.extend(["", "## Current Trace Query", trace_query])

    trace_steps = list(package.get("trace_steps") or [])
    if trace_steps:
        lines.extend(["", "## Current Trace Steps", "```json", json.dumps(trace_steps, indent=2, ensure_ascii=True), "```"])

    lines.extend(["", "## Recent Trace Digests"])
    lines.extend(_render_recent_runs_markdown(list(package.get("recent_runs") or [])))

    snippets = list(package.get("snippets") or [])
    if snippets:
        lines.extend(["", "## Snippets"])
        for snippet in snippets:
            language = str(snippet.get("language") or "text").strip() or "text"
            content = str(snippet.get("content") or "").strip()
            if not content:
                continue
            lines.extend(["", f"```{language}", content, "```"])

    transcript = list(package.get("conversation_transcript") or [])
    if transcript:
        lines.extend(["", "## Full Conversation Transcript", "```text", _render_transcript(transcript), "```"])

    criteria = list(package.get("acceptance_criteria") or [])
    if criteria:
        lines.extend(["", "## Acceptance Criteria"])
        lines.extend(f"- {item}" for item in criteria)

    return "\n".join(lines).strip() + "\n"


def _handoff_cache_key(messages: list[dict], assistant_index: int, trace_id: str, model_name: str) -> str:
    relevant_messages = []
    for message in messages[: assistant_index + 1]:
        relevant_messages.append(
            {
                "role": str(message.get("role") or "").strip(),
                "content": get_message_content(message),
                "trace_id": str(get_message_trace_id(message) or "").strip(),
            }
        )
    payload = {
        "assistant_index": assistant_index,
        "trace_id": trace_id,
        "model_name": model_name,
        "messages": relevant_messages,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()[:16]
    return f"handoff:{assistant_index}:{trace_id or 'no-trace'}:{model_name or 'no-model'}:{digest}"


def _prepare_handoff_package(
    messages: list[dict],
    assistant_index: int,
    trace_store: Any = None,
    agent: Any = None,
) -> tuple[dict[str, Any], str]:
    trace_id = str(get_message_trace_id(messages[assistant_index]) or "").strip()
    model_name = str(getattr(agent, "model_name", "") or "").strip()
    cache = st.session_state.setdefault("handoff_package_cache", {})
    cache_key = _handoff_cache_key(messages, assistant_index, trace_id, model_name)
    cached = cache.get(cache_key)
    if isinstance(cached, dict):
        return cached, cache_key

    current_trace = _load_trace(trace_store, trace_id)
    recent_traces = _select_recent_user_traces(messages, assistant_index, trace_store=trace_store, current_trace=current_trace)

    with st.spinner("Generating handoff package..."):
        package = build_handoff_package(
            messages,
            assistant_index,
            trace=current_trace,
            recent_traces=recent_traces,
            model_name=model_name,
        )

        llm = None
        if agent is not None:
            model_wrapper = getattr(agent, "_model_wrapper", None)
            if model_wrapper is not None:
                try:
                    llm = model_wrapper.get_llm()
                except Exception:
                    llm = None

        package = enrich_handoff_package(package, llm=llm, model_name=model_name)

    cache[cache_key] = package
    return package, cache_key


def render_message_handoff_panel(
    messages: list[dict],
    assistant_index: int,
    trace_store: Any = None,
    agent: Any = None,
) -> None:
    """Render a handoff destination panel for one assistant message."""
    package, cache_key = _prepare_handoff_package(messages, assistant_index, trace_store=trace_store, agent=agent)
    agent_prompt = render_agent_handoff_prompt(package)
    codex_prompt = render_codex_handoff_prompt(package)
    markdown_bundle = render_handoff_markdown(package)
    json_bundle = json.dumps(package, indent=2, ensure_ascii=True)
    file_stub = str(package.get("trace_id") or f"message-{assistant_index}")

    with st.container():
        st.markdown("**Handoff Package**")
        if package.get("trace_id"):
            run_count = len(package.get("recent_runs") or [])
            st.caption(f"Prepared from trace `{package['trace_id']}` with the last {run_count} user-triggered runs.")
        else:
            st.caption("Prepared from the visible chat message.")

        if package.get("llm_enriched"):
            st.caption(f"LLM enrichment generated with `{package.get('llm_model') or package.get('model_name')}`.")
        elif package.get("llm_error"):
            st.caption(f"LLM enrichment unavailable: {package.get('llm_error')}")

        if st.button("Regenerate Summary", key=f"handoff_regenerate_{assistant_index}"):
            st.session_state.setdefault("handoff_package_cache", {}).pop(cache_key, None)
            st.rerun()

        tab_agent, tab_codex, tab_markdown, tab_json = st.tabs(["Agent", "Codex", "Markdown", "JSON"])

        with tab_agent:
            st.caption("Primary handoff for a generic coding agent.")
            st.code(agent_prompt, language="markdown")
            st.download_button(
                "Download Agent Prompt",
                data=agent_prompt,
                file_name=f"handoff-{file_stub}-agent.md",
                mime="text/markdown",
                key=f"handoff_agent_download_{assistant_index}",
                use_container_width=True,
            )

        with tab_codex:
            st.caption("Codex-compatible variant of the same structured handoff.")
            st.code(codex_prompt, language="markdown")
            st.download_button(
                "Download Codex Prompt",
                data=codex_prompt,
                file_name=f"handoff-{file_stub}-codex.md",
                mime="text/markdown",
                key=f"handoff_codex_download_{assistant_index}",
                use_container_width=True,
            )

        with tab_markdown:
            st.code(markdown_bundle, language="markdown")
            st.download_button(
                "Download Markdown Bundle",
                data=markdown_bundle,
                file_name=f"handoff-{file_stub}.md",
                mime="text/markdown",
                key=f"handoff_markdown_download_{assistant_index}",
                use_container_width=True,
            )

        with tab_json:
            st.code(json_bundle, language="json")
            st.download_button(
                "Download JSON Bundle",
                data=json_bundle,
                file_name=f"handoff-{file_stub}.json",
                mime="application/json",
                key=f"handoff_json_download_{assistant_index}",
                use_container_width=True,
            )
