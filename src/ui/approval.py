"""Streamlit approval widget for proposed infrastructure actions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from ..agents.action import PendingAction


_RISK_COLORS = {
    "LOW": "🟢",
    "MEDIUM": "🟡",
    "HIGH": "🟠",
    "CRITICAL": "🔴",
}


def render_pending_actions(actions: list[PendingAction]) -> None:
    """Render approval UI for each pending action in the current message."""
    for action in actions:
        if action.status != "pending":
            continue
        _render_single_action(action)


def _render_single_action(action: PendingAction) -> None:
    risk_icon = _RISK_COLORS.get(action.risk, "⚪")

    with st.container():
        st.markdown("---")
        st.markdown(f"### {risk_icon} Proposed Action")
        st.markdown(f"**{action.description}**")
        st.markdown(f"**Risk**: {action.risk}")
        st.markdown(f"**Expected Outcome**: {action.expected_outcome}")

        # Show commands
        if action.commands:
            st.markdown("**Commands to execute:**")
            for cmd in action.commands:
                display = cmd.get("display", str(cmd))
                st.code(display, language="bash")

        # Approval buttons
        col_approve, col_reject = st.columns(2)
        with col_approve:
            if st.button(
                "✅ Approve & Execute",
                key=f"approve_{action.id}",
                type="primary",
                use_container_width=True,
            ):
                st.session_state.execute_action_id = action.id
                st.rerun()

        with col_reject:
            if st.button(
                "❌ Reject",
                key=f"reject_{action.id}",
                use_container_width=True,
            ):
                st.session_state.reject_action_id = action.id
                st.rerun()


def render_action_status(action: PendingAction) -> None:
    """Show status badge for a non-pending action."""
    status_map = {
        "approved": "✅ Approved",
        "rejected": "❌ Rejected",
        "executed": "⚡ Executed",
        "verified": "✅ Verified",
        "failed": "❌ Failed",
    }
    label = status_map.get(action.status, action.status)
    st.caption(f"Action status: {label}")


def handle_action_approval() -> str | None:
    """Check session state for approval/rejection clicks and execute if needed.

    Returns the execution result text if an action was approved and executed,
    or None if no action was taken.
    """
    agent = st.session_state.get("agent")
    if not agent:
        return None

    # Handle approval
    action_id = st.session_state.pop("execute_action_id", None)
    if action_id:
        with st.spinner("Executing approved action..."):
            def _status_update(label: str) -> None:
                st.session_state.agent_status_text = label

            result = agent.execute_action(action_id, status_callback=_status_update)
            return result

    # Handle rejection
    reject_id = st.session_state.pop("reject_action_id", None)
    if reject_id:
        agent.reject_action(reject_id)
        return "Action rejected by operator."

    return None
