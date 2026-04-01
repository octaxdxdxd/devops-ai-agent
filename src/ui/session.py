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
            st.session_state.agent = LogAnalyzerAgent(
                model_provider=st.session_state.get("model_provider", Config.LLM_PROVIDER),
                model_name=st.session_state.get("model_name", Config.get_active_model_name()),
            )
        except ValueError as exc:
            st.error(f"Configuration error: {exc}")
            st.stop()

    if "model_provider" not in st.session_state:
        st.session_state.model_provider = Config.LLM_PROVIDER
    if "model_name" not in st.session_state:
        st.session_state.model_name = Config.get_active_model_name()
    if "agent_status_text" not in st.session_state:
        st.session_state.agent_status_text = None
    if "model_provider_draft" not in st.session_state:
        st.session_state.model_provider_draft = st.session_state.model_provider
    if "model_name_draft" not in st.session_state:
        st.session_state.model_name_draft = st.session_state.model_name
    if "model_settings_notice" not in st.session_state:
        st.session_state.model_settings_notice = None


def apply_runtime_model_selection(provider: str, model_name: str | None = None) -> tuple[str, str]:
    """Rebuild the agent with a new in-memory provider/model selection."""
    previous_provider = Config.LLM_PROVIDER
    previous_gemini_model = Config.GEMINI_MODEL
    previous_openai_model = Config.OPENAI_MODEL
    previous_openrouter_model = Config.OPENROUTER_MODEL

    selected_provider = str(provider or Config.LLM_PROVIDER or "gemini").strip().lower()
    selected_model = str(model_name or "").strip()

    try:
        applied_model = Config.set_runtime_model_selection(selected_provider, selected_model)
        Config.validate()
        st.session_state.agent = LogAnalyzerAgent(
            model_provider=selected_provider,
            model_name=applied_model,
        )
    except Exception:
        Config.LLM_PROVIDER = previous_provider
        Config.GEMINI_MODEL = previous_gemini_model
        Config.OPENAI_MODEL = previous_openai_model
        Config.OPENROUTER_MODEL = previous_openrouter_model
        raise

    st.session_state.model_provider = selected_provider
    st.session_state.model_name = applied_model
    st.session_state.model_provider_draft = selected_provider
    st.session_state.model_name_draft = applied_model
    st.session_state.agent_status_text = None
    return selected_provider, applied_model


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
