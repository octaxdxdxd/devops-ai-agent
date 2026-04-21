"""Message-level handoff helpers for forwarding assistant output to coding tools."""

from __future__ import annotations

import json
import re
from typing import Any

import streamlit as st

from .session import get_message_content, get_message_trace_id


_CODE_BLOCK_RE = re.compile(r"```(?P<lang>[^\n`]*)\n(?P<body>[\s\S]*?)\n```")
_MAX_SNIPPETS = 5
_MAX_EVIDENCE_ITEMS = 6
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


def build_handoff_package(
    messages: list[dict],
    assistant_index: int,
    trace: dict | None = None,
) -> dict[str, Any]:
    """Build a structured handoff package for one assistant message."""
    message = messages[assistant_index]
    assistant_content = get_message_content(message)
    trace_id = get_message_trace_id(message)
    user_request = _find_prior_user_request(messages, assistant_index)
    snippets = _extract_code_snippets(assistant_content)
    evidence = _build_evidence(trace)

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
        "user_request": user_request,
        "assistant_summary": _trim_text(assistant_content, _MAX_TEXT_CHARS),
        "trace_intent": str((trace or {}).get("intent") or ""),
        "trace_outcome": str((trace or {}).get("outcome") or ""),
        "evidence": evidence,
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

    evidence = list(package.get("evidence") or [])
    if evidence:
        lines.extend(["", "Relevant evidence:"])
        lines.extend(f"- {item}" for item in evidence)

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
            "## Original Request",
            str(package.get("user_request") or "(not available)"),
            "",
            "## Assistant Summary",
            str(package.get("assistant_summary") or "(not available)"),
        ]
    )

    evidence = list(package.get("evidence") or [])
    if evidence:
        lines.extend(["", "## Evidence"])
        lines.extend(f"- {item}" for item in evidence)

    snippets = list(package.get("snippets") or [])
    if snippets:
        lines.extend(["", "## Snippets"])
        for snippet in snippets:
            language = str(snippet.get("language") or "text").strip() or "text"
            content = str(snippet.get("content") or "").strip()
            if not content:
                continue
            lines.extend(["", f"```{language}", content, "```"])

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
