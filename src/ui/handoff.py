"""Message-level handoff helpers for forwarding assistant output to coding tools."""

from __future__ import annotations

import json
import re
from typing import Any

import streamlit as st

from .session import get_message_content, get_message_trace_id


_CODE_BLOCK_RE = re.compile(r"```(?P<lang>[^\n`]*)\n(?P<body>[\s\S]*?)\n```")
_MAX_SNIPPETS = 5
_MAX_EVIDENCE_ITEMS = 12
_MAX_TEXT_CHARS = 1600
_MAX_TOOL_PREVIEW_CHARS = 240


def _trim_text(value: str, max_chars: int = _MAX_TEXT_CHARS) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


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


def _format_tool_args(args: Any) -> str:
    if not args:
        return ""
    try:
        return json.dumps(args, sort_keys=True, ensure_ascii=True)
    except TypeError:
        return str(args)


def _build_evidence(trace: dict | None) -> list[str]:
    if not isinstance(trace, dict):
        return []

    evidence: list[str] = []
    for step in trace.get("steps", []):
        step_type = str(step.get("step_type") or "").strip()
        if step_type not in {"tool_call", "verify", "approval", "error"}:
            continue

        tool_name = str(step.get("tool_name") or "").strip()
        tool_args = _format_tool_args(step.get("tool_args"))
        preview = _trim_text(str(step.get("tool_result_preview") or ""), _MAX_TOOL_PREVIEW_CHARS)
        error = _trim_text(str(step.get("error") or ""), _MAX_TOOL_PREVIEW_CHARS)

        label_parts: list[str] = []
        if tool_name:
            label = tool_name
            if tool_args:
                label += f" {tool_args}"
            label_parts.append(label)
        if preview:
            label_parts.append(preview)
        elif error:
            label_parts.append(f"Error: {error}")
        elif step_type == "approval":
            summary = _trim_text(str(step.get("output_summary") or ""), _MAX_TOOL_PREVIEW_CHARS)
            if summary:
                label_parts.append(summary)

        if label_parts:
            evidence.append(" - ".join(label_parts))
        if len(evidence) >= _MAX_EVIDENCE_ITEMS:
            break

    return evidence


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


def _build_remediation_items(trace: dict | None, snippets: list[dict[str, str]]) -> list[str]:
    items: list[str] = []
    if isinstance(trace, dict):
        for step in trace.get("steps", []):
            if not isinstance(step, dict):
                continue
            step_type = str(step.get("step_type") or "").strip()
            if step_type not in {"approval", "verify", "tool_call"}:
                continue
            tool_name = str(step.get("tool_name") or "").strip()
            output_summary = _trim_text(str(step.get("output_summary") or ""), 240)
            preview = _trim_text(str(step.get("tool_result_preview") or ""), 240)
            if step_type == "approval" and output_summary:
                items.append(output_summary)
            elif step_type == "verify" and (output_summary or preview):
                items.append(f"Verification: {output_summary or preview}")
            elif tool_name and tool_name.startswith(("k8s_", "aws_")) and preview:
                items.append(f"{tool_name}: {preview}")
    for snippet in snippets:
        language = str(snippet.get("language") or "").strip().lower()
        content = str(snippet.get("content") or "").strip()
        if language in {"bash", "sh", "yaml", "json"} and content:
            items.append(f"{language} snippet included for remediation/reference")
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped[:10]


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


def build_handoff_package(
    messages: list[dict],
    assistant_index: int,
    trace: dict | None = None,
) -> dict[str, Any]:
    """Build a structured handoff package for one assistant message."""
    message = messages[assistant_index]
    assistant_content = get_message_content(message)
    trace_id = get_message_trace_id(message)
    original_user_request = _find_first_user_request(messages, assistant_index)
    user_request = _find_prior_user_request(messages, assistant_index)
    snippets = _extract_code_snippets(assistant_content)
    evidence = _build_evidence(trace)
    transcript = _build_transcript(messages, assistant_index)
    conversation_summary = _build_conversation_summary(messages, assistant_index, assistant_content)
    trace_steps = _build_trace_step_summaries(trace)
    remediation_items = _build_remediation_items(trace, snippets)

    title = _trim_text(user_request or assistant_content, 96) or "AIOps handoff"
    acceptance_criteria = [
        "Implement or package the recommended fix so it can be applied without re-reading the whole chat.",
        "Keep the scope aligned with the issue and call out any assumptions before making broader changes.",
        "Validate the result with the most relevant checks or commands and summarize what was verified.",
    ]

    return {
        "title": title,
        "source_type": "assistant_message",
        "message_index": assistant_index,
        "trace_id": trace_id,
        "original_user_request": original_user_request,
        "user_request": user_request,
        "assistant_full_text": assistant_content,
        "assistant_summary": _trim_text(assistant_content, _MAX_TEXT_CHARS),
        "conversation_summary": conversation_summary,
        "conversation_transcript": transcript,
        "trace_intent": str((trace or {}).get("intent") or ""),
        "trace_outcome": str((trace or {}).get("outcome") or ""),
        "trace_query": str((trace or {}).get("query") or ""),
        "trace_steps": trace_steps,
        "evidence": evidence,
        "remediation_items": remediation_items,
        "snippets": snippets,
        "acceptance_criteria": acceptance_criteria,
    }


def render_codex_handoff_prompt(package: dict[str, Any]) -> str:
    """Render a Codex-oriented prompt from a structured handoff package."""
    lines = [
        "You are taking over from the AIOps assistant.",
        "",
        "Task:",
        package.get("title", "AIOps handoff"),
        "",
        "Original operator request:",
        str(package.get("user_request") or "(not available)"),
    "",
        "AIOps summary:",
        str(package.get("assistant_summary") or "(not available)"),
    ]

    original_user_request = str(package.get("original_user_request") or "").strip()
    if original_user_request and original_user_request != str(package.get("user_request") or "").strip():
        lines[6:6] = ["", "Original conversation opener:", original_user_request]

    trace_id = str(package.get("trace_id") or "").strip()
    if trace_id:
        lines.extend(["", f"Trace ID: {trace_id}"])

    trace_intent = str(package.get("trace_intent") or "").strip()
    trace_outcome = str(package.get("trace_outcome") or "").strip()
    if trace_intent or trace_outcome:
        lines.extend(
            [
                "",
                "Trace context:",
                f"- intent: {trace_intent or 'unknown'}",
                f"- outcome: {trace_outcome or 'unknown'}",
            ]
        )

    trace_query = str(package.get("trace_query") or "").strip()
    if trace_query:
        lines.extend(["", "Trace query:", trace_query])

    conversation_summary = str(package.get("conversation_summary") or "").strip()
    if conversation_summary:
        lines.extend(["", "Conversation summary so far:", conversation_summary])

    assistant_full_text = str(package.get("assistant_full_text") or "").strip()
    if assistant_full_text:
        lines.extend(["", "Latest assistant response (full text):", assistant_full_text])

    evidence = list(package.get("evidence") or [])
    if evidence:
        lines.extend(["", "Relevant evidence:"])
        lines.extend(f"- {item}" for item in evidence)

    remediation_items = list(package.get("remediation_items") or [])
    if remediation_items:
        lines.extend(["", "Remediation and execution context:"])
        lines.extend(f"- {item}" for item in remediation_items)

    snippets = list(package.get("snippets") or [])
    if snippets:
        lines.append("")
        lines.append("Relevant snippets:")
        for snippet in snippets:
            language = str(snippet.get("language") or "text").strip() or "text"
            content = str(snippet.get("content") or "").strip()
            if not content:
                continue
            lines.extend(["", f"```{language}", content, "```"])

    transcript = list(package.get("conversation_transcript") or [])
    if transcript:
        lines.extend(["", "Full conversation transcript so far:", "", "```text", _render_transcript(transcript), "```"])

    lines.extend(
        [
            "",
            "Acceptance criteria:",
        ]
    )
    lines.extend(f"- {item}" for item in package.get("acceptance_criteria", []))

    lines.extend(
        [
            "",
            "Instructions:",
            "- Prefer repo-native changes, manifests, or scripts over one-off manual repetition when practical.",
            "- Keep the implementation scoped to the problem described above.",
            "- State assumptions, describe validation, and note any follow-up that still requires an operator.",
        ]
    )

    return "\n".join(lines).strip()


def render_handoff_markdown(package: dict[str, Any]) -> str:
    """Render the handoff package as markdown."""
    lines = [
        f"# {package.get('title', 'AIOps handoff')}",
        "",
        f"- Source type: {package.get('source_type', 'assistant_message')}",
        f"- Message index: {package.get('message_index', '')}",
    ]

    trace_id = str(package.get("trace_id") or "").strip()
    if trace_id:
        lines.append(f"- Trace ID: {trace_id}")
    trace_intent = str(package.get("trace_intent") or "").strip()
    if trace_intent:
        lines.append(f"- Trace intent: {trace_intent}")
    trace_outcome = str(package.get("trace_outcome") or "").strip()
    if trace_outcome:
        lines.append(f"- Trace outcome: {trace_outcome}")

    lines.extend(
        [
            "",
            "## Original Conversation Opener",
            str(package.get("original_user_request") or package.get("user_request") or "(not available)"),
            "",
            "## Original Request",
            str(package.get("user_request") or "(not available)"),
            "",
            "## Conversation Summary",
            str(package.get("conversation_summary") or "(not available)"),
            "",
            "## Assistant Summary",
            str(package.get("assistant_summary") or "(not available)"),
            "",
            "## Latest Assistant Response (Full Text)",
            str(package.get("assistant_full_text") or "(not available)"),
        ]
    )

    evidence = list(package.get("evidence") or [])
    if evidence:
        lines.extend(["", "## Evidence"])
        lines.extend(f"- {item}" for item in evidence)

    remediation_items = list(package.get("remediation_items") or [])
    if remediation_items:
        lines.extend(["", "## Remediation Context"])
        lines.extend(f"- {item}" for item in remediation_items)

    trace_query = str(package.get("trace_query") or "").strip()
    if trace_query:
        lines.extend(["", "## Trace Query", trace_query])

    trace_steps = list(package.get("trace_steps") or [])
    if trace_steps:
        lines.extend(["", "## Trace Steps", "```json", json.dumps(trace_steps, indent=2, ensure_ascii=True), "```"])

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


def render_message_handoff_panel(messages: list[dict], assistant_index: int, trace_store: Any = None) -> None:
    """Render a handoff destination panel for one assistant message."""
    trace = None
    trace_id = get_message_trace_id(messages[assistant_index])
    if trace_id and trace_store is not None:
        try:
            trace = trace_store.load(trace_id)
        except Exception:
            trace = None

    package = build_handoff_package(messages, assistant_index, trace=trace)
    codex_prompt = render_codex_handoff_prompt(package)
    markdown_bundle = render_handoff_markdown(package)
    json_bundle = json.dumps(package, indent=2, ensure_ascii=True)
    file_stub = str(package.get("trace_id") or f"message-{assistant_index}")

    with st.container():
        st.markdown("**Handoff Package**")
        if trace_id:
            st.caption(f"Prepared from trace `{trace_id}`.")
        else:
            st.caption("Prepared from the visible chat message.")

        tab_codex, tab_markdown, tab_json = st.tabs(["Codex", "Markdown", "JSON"])

        with tab_codex:
            st.caption("Ready to paste into Codex or another coding agent.")
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
