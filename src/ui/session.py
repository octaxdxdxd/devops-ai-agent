"""Streamlit session utilities."""

from __future__ import annotations

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from ..agents import LogAnalyzerAgent
from ..config import Config


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


def convert_to_langchain_messages(messages: list[dict[str, str]]) -> list[HumanMessage | AIMessage]:
    """Convert Streamlit chat history into LangChain messages."""
    max_messages = max(0, int(getattr(Config, "MAX_CHAT_HISTORY_MESSAGES", 14)))
    if max_messages > 0:
        messages = messages[-max_messages:]

    langchain_messages: list[HumanMessage | AIMessage] = []
    for msg in messages:
        if msg.get("role") == "user":
            langchain_messages.append(HumanMessage(content=msg.get("content", "")))
        elif msg.get("role") == "assistant":
            langchain_messages.append(AIMessage(content=msg.get("content", "")))
    return langchain_messages
