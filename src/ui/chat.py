"""Streamlit chat rendering and turn handling."""

from __future__ import annotations

import re

import streamlit as st

from ..config import Config
from .session import convert_to_langchain_messages, get_message_content, get_message_trace_id


def display_chat_messages() -> None:
    """Display all chat messages from history."""
    if not st.session_state.messages:
        st.markdown("### Start With A Targeted Prompt")
        st.markdown("- `Show current context, list namespaces, and report any crash loops`")
        st.markdown("- `Diagnose why service <name> is unreachable and propose safest remediation`")
        st.markdown("- `Correlate Kubernetes health with AWS signals for workload <name>`")
        st.markdown("---")
        return

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            content = get_message_content(message)
            if message["role"] == "assistant":
                _render_message_content(content)
            else:
                st.markdown(content)
            trace_id = get_message_trace_id(message)
            if trace_id and Config.TRACE_ENABLED:
                st.caption(f"Trace ID: `{trace_id}`")


_MARKDOWN_STRUCTURED_RE = re.compile(r"(?m)^\s*(#{1,6}\s|[-*]\s|\d+\.\s|>\s|\|.+\|)")
_CODE_BLOCK_RE = re.compile(r"```(?P<lang>[^\n`]*)\n(?P<body>[\s\S]*?)\n```")
_LARGE_BLOCK_CHARS = 4000
_LARGE_BLOCK_LINES = 120
_PREVIEW_LINES = 40


def _format_for_markdown(content: str) -> str:
    """Normalize plain multiline tool output so Streamlit renders it cleanly."""
    text = (content or "").strip()
    if not text:
        return text
    if "```" in text:
        return text
    if _MARKDOWN_STRUCTURED_RE.search(text) or "**" in text:
        return text
    if "\n" in text:
        return f"```text\n{text}\n```"
    return text


def _render_message_content(content: str) -> None:
    """Render assistant content with compact preview for very large code blocks."""
    text = (content or "").strip()
    if not text:
        st.markdown(text)
        return

    matches = list(_CODE_BLOCK_RE.finditer(text))
    if len(matches) != 1:
        st.markdown(text)
        return

    match = matches[0]
    lang = (match.group("lang") or "").strip() or "text"
    body = match.group("body") or ""
    line_count = body.count("\n") + 1
    char_count = len(body)
    is_large = char_count >= _LARGE_BLOCK_CHARS or line_count >= _LARGE_BLOCK_LINES
    if not is_large:
        st.markdown(text)
        return

    prefix = text[: match.start()].strip()
    suffix = text[match.end() :].strip()
    preview_lines = body.splitlines()[:_PREVIEW_LINES]
    preview = "\n".join(preview_lines)
    if line_count > len(preview_lines):
        preview += "\n... [preview truncated]"

    if prefix:
        st.markdown(prefix)
    st.caption(
        f"Large output detected: {line_count} lines, {char_count} chars. "
        f"Previewing first {_PREVIEW_LINES} lines."
    )
    st.code(preview, language=lang)
    with st.expander("View full output"):
        st.code(body, language=lang)
    if suffix:
        st.markdown(suffix)


def process_chat_turn() -> None:
    """Read user prompt, run the agent, and append the assistant response."""
    if prompt := st.chat_input("Ask about cluster health, pods, events, or remediation..."):
        user_message = {"role": "user", "content": prompt, "trace_id": None}
        st.session_state.messages.append(user_message)

        with st.chat_message("user"):
            st.markdown(prompt)

        response = ""
        with st.chat_message("assistant"):
            with st.status("Inspecting cluster state and diagnostics...", expanded=False) as status:
                def update_status(label: str) -> None:
                    clean_label = str(label or "").strip() or "Inspecting cluster state and diagnostics..."
                    st.session_state.agent_status_text = clean_label
                    try:
                        status.update(label=clean_label, state="running")
                    except Exception:
                        pass

                chat_history = convert_to_langchain_messages(st.session_state.messages[:-1])
                try:
                    response = st.session_state.agent.process_query(
                        user_input=prompt,
                        chat_history=chat_history,
                        status_callback=update_status,
                    )
                except Exception as exc:  # noqa: BLE001
                    response = f"Error processing query: {exc}"
                    update_status("Agent hit an error while processing the request.")
                try:
                    status.update(label="Request completed.", state="complete")
                except Exception:
                    pass

            assistant_content = (response or "").strip()
            if not assistant_content:
                assistant_content = "Agent returned an empty response. Check model/provider settings or enable tracing."
            assistant_content = _format_for_markdown(assistant_content)

            trace_id = getattr(st.session_state.agent, "last_trace_id", None)
            user_message["trace_id"] = trace_id

            _render_message_content(assistant_content)
            if trace_id and Config.TRACE_ENABLED:
                st.caption(f"Trace ID: `{trace_id}`")

        st.session_state.messages.append({"role": "assistant", "content": assistant_content, "trace_id": trace_id})
        st.rerun()
