"""Streamlit entrypoint for the AI Ops assistant."""

from __future__ import annotations

import streamlit as st

from ..config import Config
from .chat import display_chat_messages, process_chat_turn
from .session import initialize_session_state
from .sidebar import display_sidebar


def _top_toolbar() -> None:
    """Render primary actions in main content area."""
    col1, col2 = st.columns([1, 5])

    with col1:
        if st.button("New Chat", use_container_width=True):
            st.session_state.messages = []
            try:
                st.session_state.agent.clear_history()
            except Exception:
                pass
            st.rerun()

    with col2:
        current_trace_id = getattr(st.session_state.get("agent", None), "last_trace_id", None)
        operator_intent = getattr(st.session_state.get("agent", None), "operator_intent_state", None)
        st.caption(
            f"Provider: `{st.session_state.model_provider}`  |  "
            f"Model: `{st.session_state.model_name}`  |  "
            f"K8s Context: `{Config.K8S_CONTEXT or 'active kubectl context'}`"
        )
        if current_trace_id and Config.TRACE_ENABLED:
            st.caption(f"Current Trace ID: `{current_trace_id}`")
        if operator_intent is not None:
            st.caption(
                f"Mode: `{getattr(operator_intent, 'mode', 'incident_response')}`  |  "
                f"Execution: `{getattr(operator_intent, 'execution_policy', 'approval_required')}`"
            )
            constraints = list(getattr(operator_intent, "pinned_constraints", []) or [])
            if constraints:
                st.caption("Constraints: " + " | ".join(f"`{item}`" for item in constraints[:3]))


def main() -> None:
    """Main Streamlit application routine."""
    st.set_page_config(
        page_title="AIOps Agent",
        page_icon="",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    initialize_session_state()
    display_sidebar()

    st.title("AIOps Infrastructure Agent")
    st.caption("Evidence-grounded investigation, diagnostics, and approval-gated remediation for Kubernetes & AWS.")
    _top_toolbar()
    st.markdown("---")

    display_chat_messages()
    process_chat_turn()
