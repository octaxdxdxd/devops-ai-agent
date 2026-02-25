"""Streamlit chat rendering and turn handling."""

from __future__ import annotations

import streamlit as st

from ..config import Config
from .session import convert_to_langchain_messages


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
            st.markdown(message["content"])


def process_chat_turn() -> None:
    """Read user prompt, run the agent, and append the assistant response."""
    if prompt := st.chat_input("Ask about cluster health, pods, events, or remediation..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Inspecting cluster state and diagnostics..."):
                chat_history = convert_to_langchain_messages(st.session_state.messages[:-1])
                response = st.session_state.agent.process_query(
                    user_input=prompt,
                    chat_history=chat_history,
                )

                if not (response or "").strip():
                    st.warning("Agent returned an empty response. Check model/provider settings or enable tracing.")
                else:
                    st.markdown(response)

                trace_id = getattr(st.session_state.agent, "last_trace_id", None)
                if trace_id and Config.TRACE_ENABLED:
                    st.caption(f"Trace ID: {trace_id}")

        st.session_state.messages.append({"role": "assistant", "content": response})
