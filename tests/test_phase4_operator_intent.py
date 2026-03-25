"""Regression tests for simplified operator-intent behavior."""

from __future__ import annotations

import unittest

from src.agents.log_analyzer import LogAnalyzerAgent
from src.agents.planner import build_turn_plan
from src.agents.state import (
    IncidentState,
    OperatorIntentState,
    register_operator_follow_up,
    update_operator_intent_state,
)
from src.utils.query_intent import QueryIntent


class OperatorIntentStateTests(unittest.TestCase):
    def test_register_operator_follow_up_no_longer_infers_pending_steps_from_prose(self) -> None:
        state = OperatorIntentState()
        updated = register_operator_follow_up(
            operator_intent_state=state,
            final_text=(
                "Recommended Action: Inspect the current GitLab PVCs and Helm values before proposing a migration.\n\n"
                "Would you like me to proceed with those diagnostic reads? (yes/no)"
            ),
            turn_plan=type("Plan", (), {"focus": "storage"})(),
            approval_pending=False,
        )
        self.assertFalse(updated.awaiting_follow_up)
        self.assertEqual(updated.pending_step_summary, "")
        self.assertEqual(updated.pending_step_kind, "")

    def test_update_operator_intent_state_keeps_incident_mode_for_short_replies(self) -> None:
        state = OperatorIntentState(
            mode="follow_up_action",
            pending_step_summary="Inspect the GitLab PVCs and Helm config before proposing changes.",
            pending_step_focus="storage",
            pending_step_stage="collect",
            awaiting_follow_up=True,
        )
        updated = update_operator_intent_state(
            operator_intent_state=state,
            user_input="yes",
            turn_index=5,
            incident_state=IncidentState(active=True, namespace="gitlab", last_focus="storage"),
        )
        self.assertEqual(updated.mode, "incident_response")
        self.assertFalse(updated.awaiting_follow_up)
        self.assertEqual(updated.last_user_instruction, "yes")


class PlannerIntentTests(unittest.TestCase):
    def test_scope_reference_reuses_existing_incident_context(self) -> None:
        state = IncidentState(
            active=True,
            namespace="gitlab",
            services=["webservice"],
            pods=["gitlab-webservice-abc123"],
            last_focus="service",
        )
        plan = build_turn_plan(
            user_input="check webservice in gitlab",
            intent=QueryIntent(mode="general"),
            chat_history=[],
            incident_state=state,
            operator_intent_state=OperatorIntentState(),
        )
        self.assertTrue(plan.continue_existing)
        self.assertEqual(plan.stage, "collect")
        self.assertEqual(plan.focus, "service")

    def test_short_follow_up_words_do_not_force_verification_stage(self) -> None:
        state = IncidentState(
            active=True,
            namespace="gitlab",
            services=["webservice"],
            last_focus="service",
        )
        plan = build_turn_plan(
            user_input="is it healthy now?",
            intent=QueryIntent(mode="general"),
            chat_history=[],
            incident_state=state,
            operator_intent_state=OperatorIntentState(),
        )
        self.assertFalse(plan.continue_existing)
        self.assertNotEqual(plan.stage, "verify")


class ToolSelectionTests(unittest.TestCase):
    def test_general_turn_exposes_high_level_write_tools_without_raw_execute_tools(self) -> None:
        dummy_agent = type(
            "DummyAgent",
            (),
            {
                "tools": [
                    type("Tool", (), {"name": "k8s_list_pods"})(),
                    type("Tool", (), {"name": "kubectl_readonly"})(),
                    type("Tool", (), {"name": "kubectl_execute"})(),
                    type("Tool", (), {"name": "restart_kubernetes_pod"})(),
                    type("Tool", (), {"name": "scale_kubernetes_deployment"})(),
                    type("Tool", (), {"name": "aws_cli_readonly"})(),
                ]
            },
        )()
        selected = LogAnalyzerAgent._select_tools_for_turn(
            dummy_agent,
            intent=QueryIntent(mode="general"),
            user_input="restart gitlab webservice pod",
            turn_plan=build_turn_plan(
                user_input="restart gitlab webservice pod",
                intent=QueryIntent(mode="general"),
                chat_history=[],
                incident_state=IncidentState(),
                operator_intent_state=OperatorIntentState(),
            ),
            operator_intent_state=OperatorIntentState(),
        )
        selected_names = {tool.name for tool in selected}
        self.assertIn("restart_kubernetes_pod", selected_names)
        self.assertIn("scale_kubernetes_deployment", selected_names)
        self.assertNotIn("kubectl_execute", selected_names)

    def test_explicit_command_turn_exposes_execute_tools(self) -> None:
        dummy_agent = type(
            "DummyAgent",
            (),
            {
                "tools": [
                    type("Tool", (), {"name": "kubectl_readonly"})(),
                    type("Tool", (), {"name": "kubectl_execute"})(),
                    type("Tool", (), {"name": "aws_cli_execute"})(),
                ]
            },
        )()
        selected = LogAnalyzerAgent._select_tools_for_turn(
            dummy_agent,
            intent=QueryIntent(mode="command"),
            user_input="kubectl delete pod demo -n gitlab",
            turn_plan=build_turn_plan(
                user_input="kubectl delete pod demo -n gitlab",
                intent=QueryIntent(mode="command"),
                chat_history=[],
                incident_state=IncidentState(),
                operator_intent_state=OperatorIntentState(),
            ),
            operator_intent_state=OperatorIntentState(),
        )
        selected_names = {tool.name for tool in selected}
        self.assertIn("kubectl_execute", selected_names)


if __name__ == "__main__":
    unittest.main()
