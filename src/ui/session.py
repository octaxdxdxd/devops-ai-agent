"""Streamlit session utilities."""

from __future__ import annotations

import re

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from ..agents import LogAnalyzerAgent
from ..config import Config


_TRACE_SUFFIX_RE = re.compile(r"(?:\n\s*)*Trace ID:\s*`?([a-f0-9]{8,})`?\s*$", re.IGNORECASE)


def extract_trace_id_from_content(content: str) -> str | None:
    """Extract a legacy trace id suffix from stored assistant text."""
    text = str(content or "").strip()
    if not text:
        return None
    match = _TRACE_SUFFIX_RE.search(text)
    if not match:
        return None
    trace_id = (match.group(1) or "").strip()
    return trace_id or None


def strip_trace_suffix(content: str) -> str:
    """Remove legacy inline trace-id suffix from message content."""
    text = str(content or "")
    return _TRACE_SUFFIX_RE.sub("", text).rstrip()


def get_message_trace_id(message: dict) -> str | None:
    """Return trace id from structured metadata or legacy content suffix."""
    trace_id = str(message.get("trace_id") or "").strip()
    if trace_id:
        return trace_id
    return extract_trace_id_from_content(str(message.get("content", "")))


def get_message_content(message: dict) -> str:
    """Return message content without any legacy inline trace-id suffix."""
    return strip_trace_suffix(str(message.get("content", "")))


def initialize_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "agent" not in st.session_state:
        try:
            Config.validate()
            st.session_state.agent = LogAnalyzerAgent()
        except ValueError as exc:
            st.error(f"Configuration error: {exc}")
            st.stop()

    if "model_provider" not in st.session_state:
        st.session_state.model_provider = Config.LLM_PROVIDER
    if "model_name" not in st.session_state:
        st.session_state.model_name = Config.get_active_model_name()
    if "autonomy_last_scan" not in st.session_state:
        st.session_state.autonomy_last_scan = None
    if "agent_status_text" not in st.session_state:
        st.session_state.agent_status_text = None


def convert_to_langchain_messages(messages: list[dict[str, str]]) -> list[HumanMessage | AIMessage]:
    """Convert Streamlit chat history into LangChain messages."""
    max_messages = max(0, int(getattr(Config, "MAX_CHAT_HISTORY_MESSAGES", 14)))
    if max_messages > 0:
        messages = messages[-max_messages:]

    langchain_messages: list[HumanMessage | AIMessage] = []
    for msg in messages:
        if msg.get("role") == "user":
            langchain_messages.append(HumanMessage(content=get_message_content(msg)))
        elif msg.get("role") == "assistant":
            langchain_messages.append(AIMessage(content=get_message_content(msg)))
    return langchain_messages
