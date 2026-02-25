"""Streamlit entrypoint for the AI Ops assistant."""

from __future__ import annotations

import streamlit as st

from ..config import Config
from .chat import display_chat_messages, process_chat_turn
from .session import initialize_session_state
from .sidebar import display_sidebar


def _top_toolbar() -> None:
    """Render primary actions in main content area."""
    col1, col2, col3 = st.columns([1, 1, 3])

    with col1:
        if st.button("New Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    with col2:
        if st.button("Run Health Scan", use_container_width=True):
            with st.spinner("Running Kubernetes health scan..."):
                scan = st.session_state.agent.run_autonomous_scan(send_notifications=True)
                st.session_state.autonomy_last_scan = scan

    with col3:
        st.caption(
            f"Provider: `{st.session_state.model_provider}`  |  "
            f"Model: `{st.session_state.model_name}`  |  "
            f"K8s Context: `{Config.K8S_CONTEXT or 'active kubectl context'}`"
        )


def main() -> None:
    """Main Streamlit application routine."""
    st.set_page_config(
        page_title="AI Ops K8s Assistant",
        page_icon="",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    initialize_session_state()
    display_sidebar()

    st.title("AI Ops Kubernetes Assistant")
    st.caption("Tool-first diagnostics and approval-gated remediation for Kubernetes and AWS.")
    _top_toolbar()
    st.markdown("---")

    if st.session_state.autonomy_last_scan:
        with st.expander("Latest Automated Health Scan", expanded=False):
            st.markdown(st.session_state.agent.format_autonomous_scan(st.session_state.autonomy_last_scan))

    display_chat_messages()
    process_chat_turn()
