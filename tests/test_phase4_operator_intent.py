"""Regression tests for context-driven operator follow-through behavior."""

from __future__ import annotations

import unittest

from src.agents.log_analyzer import LogAnalyzerAgent
from src.agents.planner import build_turn_plan, render_turn_plan_directive
from src.agents.state import (
    IncidentState,
    OperatorIntentState,
    register_operator_follow_up,
    update_operator_intent_state,
)
from src.utils.query_intent import QueryIntent


class OperatorIntentStateTests(unittest.TestCase):
    def test_follow_up_request_continues_previously_proposed_step(self) -> None:
        state = OperatorIntentState(
            pending_step_summary="Inspect the GitLab PVCs and Helm config before proposing changes.",
            pending_step_focus="storage",
            pending_step_stage="collect",
            awaiting_follow_up=True,
        )
        updated = update_operator_intent_state(
            operator_intent_state=state,
            user_input="implement the long term solution",
            turn_index=5,
            incident_state=IncidentState(active=True, namespace="gitlab", last_focus="storage"),
        )
        self.assertEqual(updated.mode, "follow_up_action")
        self.assertEqual(updated.execution_policy, "approval_required")
        self.assertEqual(updated.last_user_instruction, "implement the long term solution")
        self.assertIn("Continue with the previously proposed next step", updated.pinned_constraints)

    def test_question_clears_pending_follow_up(self) -> None:
        state = OperatorIntentState(
            pending_step_summary="Inspect the GitLab PVCs and Helm config before proposing changes.",
            pending_step_focus="storage",
            pending_step_stage="collect",
            awaiting_follow_up=True,
        )
        updated = update_operator_intent_state(
            operator_intent_state=state,
            user_input="why is postgres still pending?",
            turn_index=6,
            incident_state=IncidentState(active=True, namespace="gitlab", last_focus="storage"),
        )
        self.assertEqual(updated.mode, "incident_response")
        self.assertFalse(updated.awaiting_follow_up)
        self.assertEqual(updated.last_user_instruction, "why is postgres still pending?")

    def test_register_operator_follow_up_captures_next_step(self) -> None:
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
        self.assertTrue(updated.awaiting_follow_up)
        self.assertEqual(updated.pending_step_focus, "storage")
        self.assertIn("Inspect the current GitLab PVCs", updated.pending_step_summary)
        self.assertEqual(updated.pending_step_kind, "investigate")
        self.assertEqual(updated.pending_step_stage, "collect")

    def test_register_operator_follow_up_marks_implementation_steps(self) -> None:
        state = OperatorIntentState()
        updated = register_operator_follow_up(
            operator_intent_state=state,
            final_text=(
                "Recommended Action: Add topologySpreadConstraints to the PostgreSQL and Gitaly StatefulSets.\n\n"
                "Would you like me to proceed with implementing these fixes? I'll need to inspect the current manifests and then patch them."
            ),
            turn_plan=type("Plan", (), {"focus": "storage", "stage": "collect"})(),
            approval_pending=False,
        )
        self.assertTrue(updated.awaiting_follow_up)
        self.assertEqual(updated.pending_step_kind, "implementation")
        self.assertEqual(updated.pending_step_stage, "command")


class PlannerIntentTests(unittest.TestCase):
    def test_follow_up_action_forces_collect_stage_and_scope_reuse(self) -> None:
        operator_state = OperatorIntentState(
            mode="follow_up_action",
            execution_policy="approval_required",
            pinned_constraints=["Continue with the previously proposed next step"],
            last_user_instruction="implement the long term solution",
            source_turn=4,
            pending_step_summary="Inspect the GitLab PVCs and Helm config before proposing changes.",
            pending_step_focus="storage",
            pending_step_stage="collect",
            awaiting_follow_up=True,
        )
        plan = build_turn_plan(
            user_input="implement the long term solution",
            intent=QueryIntent(mode="general"),
            chat_history=[],
            incident_state=IncidentState(active=True, namespace="gitlab", last_focus="node"),
            operator_intent_state=operator_state,
        )
        rendered = render_turn_plan_directive(
            turn_plan=plan,
            incident_state=IncidentState(active=True, namespace="gitlab", last_focus="node"),
            operator_intent_state=operator_state,
        )
        self.assertTrue(plan.continue_existing)
        self.assertEqual(plan.stage, "collect")
        self.assertEqual(plan.focus, "storage")
        self.assertFalse(plan.allow_broad_discovery)
        self.assertIn("previously proposed next step", rendered.lower())

    def test_implementation_follow_up_forces_command_stage(self) -> None:
        operator_state = OperatorIntentState(
            mode="follow_up_action",
            execution_policy="approval_required",
            pending_step_summary="Patch the PostgreSQL and Gitaly StatefulSets after inspecting their manifests.",
            pending_step_focus="storage",
            pending_step_stage="command",
            pending_step_kind="implementation",
            awaiting_follow_up=True,
        )
        plan = build_turn_plan(
            user_input="yes, proceed",
            intent=QueryIntent(mode="general"),
            chat_history=[],
            incident_state=IncidentState(active=True, namespace="gitlab", last_focus="storage"),
            operator_intent_state=operator_state,
        )
        self.assertEqual(plan.stage, "command")
        self.assertEqual(plan.focus, "storage")


class ToolSelectionTests(unittest.TestCase):
    def test_follow_up_action_does_not_expose_write_tools_for_collect_stage(self) -> None:
        dummy_agent = type(
            "DummyAgent",
            (),
            {
                "tools": [
                    type("Tool", (), {"name": "k8s_get_pvcs"})(),
                    type("Tool", (), {"name": "aws_cli_readonly"})(),
                    type("Tool", (), {"name": "aws_cli_execute"})(),
                    type("Tool", (), {"name": "kubectl_execute"})(),
                ]
            },
        )()
        operator_state = OperatorIntentState(
            mode="follow_up_action",
            pending_step_focus="storage",
            pending_step_stage="collect",
            awaiting_follow_up=True,
        )
        turn_plan = build_turn_plan(
            user_input="implement the long term solution",
            intent=QueryIntent(mode="general"),
            chat_history=[],
            incident_state=IncidentState(active=True, namespace="gitlab", last_focus="storage"),
            operator_intent_state=operator_state,
        )
        selected = LogAnalyzerAgent._select_tools_for_turn(
            dummy_agent,
            intent=QueryIntent(mode="general"),
            user_input="implement the long term solution",
            turn_plan=turn_plan,
            operator_intent_state=operator_state,
        )
        selected_names = {tool.name for tool in selected}
        self.assertIn("aws_cli_readonly", selected_names)
        self.assertNotIn("aws_cli_execute", selected_names)
        self.assertNotIn("kubectl_execute", selected_names)

    def test_implementation_follow_up_exposes_write_tools(self) -> None:
        dummy_agent = type(
            "DummyAgent",
            (),
            {
                "tools": [
                    type("Tool", (), {"name": "k8s_get_resource_yaml"})(),
                    type("Tool", (), {"name": "kubectl_readonly"})(),
                    type("Tool", (), {"name": "kubectl_execute"})(),
                ]
            },
        )()
        operator_state = OperatorIntentState(
            mode="follow_up_action",
            pending_step_focus="storage",
            pending_step_stage="command",
            pending_step_kind="implementation",
            awaiting_follow_up=True,
        )
        turn_plan = build_turn_plan(
            user_input="yes, proceed",
            intent=QueryIntent(mode="general"),
            chat_history=[],
            incident_state=IncidentState(active=True, namespace="gitlab", last_focus="storage"),
            operator_intent_state=operator_state,
        )
        selected = LogAnalyzerAgent._select_tools_for_turn(
            dummy_agent,
            intent=QueryIntent(mode="general"),
            user_input="yes, proceed",
            turn_plan=turn_plan,
            operator_intent_state=operator_state,
        )
        selected_names = {tool.name for tool in selected}
        self.assertIn("kubectl_execute", selected_names)


if __name__ == "__main__":
    unittest.main()
