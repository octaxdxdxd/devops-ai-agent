"""Streamlit sidebar status panel."""

from __future__ import annotations

import streamlit as st

from ..config import Config


def display_sidebar() -> None:
    """Display sidebar with runtime metadata only (no actions)."""
    with st.sidebar:
        st.title("Runtime Status")
        st.caption("Read-only status panel")
        st.markdown("---")

        st.subheader("Model")
        st.caption(f"Provider: {st.session_state.model_provider}")
        st.caption(f"Model: {st.session_state.model_name}")
        st.caption(f"Temperature: {Config.TEMPERATURE}")

        st.markdown("---")
        st.subheader("Kubernetes")
        st.caption(f"K8s Context: {Config.K8S_CONTEXT or '(active kubectl context)'}")
        st.caption(f"K8s Namespace: {Config.K8S_DEFAULT_NAMESPACE or '(auto/current context)'}")
        st.caption(f"K8s CLI Allow-All Read: {Config.K8S_CLI_ALLOW_ALL_READ}")
        st.caption(f"K8s CLI Allow-All Write: {Config.K8S_CLI_ALLOW_ALL_WRITE}")

        st.markdown("---")
        st.subheader("AWS")
        st.caption(f"AWS CLI Enabled: {Config.AWS_CLI_ENABLED}")
        st.caption(f"AWS CLI Allow-All Read: {Config.AWS_CLI_ALLOW_ALL_READ}")
        st.caption(f"AWS CLI Allow-All Write: {Config.AWS_CLI_ALLOW_ALL_WRITE}")
        st.caption(f"AWS Profile: {Config.AWS_CLI_PROFILE or '(default)'}")
        st.caption(f"AWS Region: {Config.AWS_CLI_DEFAULT_REGION or '(from aws config/env)'}")

        st.markdown("---")
        st.subheader("Tracing")
        st.caption(f"Enabled: {Config.TRACE_ENABLED}")
        st.caption(f"Dir: {Config.TRACE_DIR}")

        last_trace_id = None
        try:
            last_trace_id = getattr(st.session_state.get("agent", None), "last_trace_id", None)
        except Exception:
            last_trace_id = None

        if last_trace_id:
            st.code(last_trace_id)

        st.markdown("---")
        st.subheader("Autonomy")
        st.caption(f"Enabled: {Config.AUTONOMY_ENABLED}")
        st.caption(f"Auto-scan on each turn: {Config.AUTONOMY_SCAN_ON_USER_TURN}")
        st.caption(f"Scope: {Config.AUTONOMY_NAMESPACE}")
        st.caption(f"Event lookback: {Config.AUTONOMY_RECENT_MINUTES}m")
        st.caption(f"Pending grace: {Config.ALERT_PENDING_GRACE_MINUTES}m")
        st.caption(f"Critical event threshold: {Config.ALERT_CRITICAL_EVENT_MIN_COUNT}")
