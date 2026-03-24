"""Streamlit sidebar status panel."""

from __future__ import annotations

import streamlit as st

from ..config import Config
from .session import apply_runtime_model_selection, get_message_trace_id


def display_sidebar() -> None:
    """Display sidebar with runtime metadata and model controls."""
    with st.sidebar:
        st.title("Runtime Status")
        st.caption("Status panel and model controls")
        st.markdown("---")

        st.subheader("Model")
        st.caption(f"Provider: {st.session_state.model_provider}")
        st.caption(f"Model: {st.session_state.model_name}")
        st.caption(f"Temperature: {Config.TEMPERATURE}")
        st.caption(f"Command Safety Posture: {Config.COMMAND_SAFETY_POSTURE}")
        st.caption("Changes apply to new turns and health scans.")

        provider_options = list(Config.SUPPORTED_LLM_PROVIDERS)
        if st.session_state.get("model_provider_draft") not in provider_options:
            st.session_state.model_provider_draft = st.session_state.model_provider
        if not str(st.session_state.get("model_name_draft") or "").strip():
            st.session_state.model_name_draft = st.session_state.model_name

        with st.form("runtime_model_controls"):
            st.selectbox(
                "LLM provider",
                provider_options,
                key="model_provider_draft",
            )
            st.text_input(
                "Model name",
                key="model_name_draft",
                help="Enter any provider-supported model id.",
            )
            st.caption(
                "Suggested default: "
                f"`{Config.get_model_name_for_provider(st.session_state.model_provider_draft)}`"
            )
            col_apply, col_default = st.columns(2)
            apply_clicked = col_apply.form_submit_button("Apply", use_container_width=True)
            default_clicked = col_default.form_submit_button("Use Default", use_container_width=True)

        if default_clicked:
            st.session_state.model_name_draft = Config.get_model_name_for_provider(
                st.session_state.model_provider_draft
            )
            st.rerun()

        if apply_clicked:
            try:
                selected_provider, selected_model = apply_runtime_model_selection(
                    st.session_state.model_provider_draft,
                    st.session_state.model_name_draft,
                )
                st.session_state.model_settings_notice = (
                    f"Switched to {selected_provider}:{selected_model}"
                )
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))

        notice = str(st.session_state.get("model_settings_notice") or "").strip()
        if notice:
            st.success(notice)
            st.session_state.model_settings_notice = None

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
        st.subheader("Operator Intent")
        operator_intent = getattr(st.session_state.get("agent", None), "operator_intent_state", None)
        if operator_intent is None:
            st.caption("Mode: incident_response")
            st.caption("Execution: approval_required")
        else:
            st.caption(f"Mode: {getattr(operator_intent, 'mode', 'incident_response')}")
            st.caption(f"Execution: {getattr(operator_intent, 'execution_policy', 'approval_required')}")
            constraints = list(getattr(operator_intent, "pinned_constraints", []) or [])
            if constraints:
                for item in constraints[:4]:
                    st.caption(f"- {item}")
            last_instruction = str(getattr(operator_intent, "last_user_instruction", "") or "").strip()
            if last_instruction:
                st.caption(f"Latest: {last_instruction}")
            pending_summary = str(getattr(operator_intent, "pending_step_summary", "") or "").strip()
            pending_kind = str(getattr(operator_intent, "pending_step_kind", "") or "").strip()
            if pending_summary:
                st.caption(f"Pending Step: {pending_summary}")
            if pending_kind:
                st.caption(f"Pending Kind: {pending_kind}")

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
            st.caption("Current")
            st.code(last_trace_id)

        recent_trace_ids: list[str] = []
        seen: set[str] = set()
        for message in reversed(st.session_state.get("messages", [])):
            trace_id = get_message_trace_id(message)
            if not trace_id or trace_id in seen:
                continue
            seen.add(trace_id)
            recent_trace_ids.append(trace_id)

        if recent_trace_ids:
            st.caption("Recent")
            for trace_id in recent_trace_ids[:12]:
                st.code(trace_id)

        st.markdown("---")
        st.subheader("Autonomy")
        st.caption(f"Enabled: {Config.AUTONOMY_ENABLED}")
        st.caption(f"Auto-scan on each turn: {Config.AUTONOMY_SCAN_ON_USER_TURN}")
        st.caption(f"Scope: {Config.AUTONOMY_NAMESPACE}")
        st.caption(f"Event lookback: {Config.AUTONOMY_RECENT_MINUTES}m")
        st.caption(f"Pending grace: {Config.ALERT_PENDING_GRACE_MINUTES}m")
        st.caption(f"Critical event threshold: {Config.ALERT_CRITICAL_EVENT_MIN_COUNT}")
