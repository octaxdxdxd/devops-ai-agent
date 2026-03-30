"""Regression tests for simplified operator-intent behavior."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from src.agents.log_analyzer import LogAnalyzerAgent
from src.agents.planner import build_turn_plan
from src.agents.state import (
    IncidentState,
    OperatorIntentState,
    register_operator_follow_up,
    update_operator_intent_state,
)
from src.agents.tool_loop import handle_tool_calls
from src.agents.approval import ApprovalCoordinator
from src.utils.query_intent import QueryIntent


class OperatorIntentStateTests(unittest.TestCase):
    def test_register_operator_follow_up_tracks_collect_follow_up_from_prose(self) -> None:
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
        self.assertEqual(updated.pending_step_summary, "those diagnostic reads")
        self.assertEqual(updated.pending_step_kind, "collect")
        self.assertEqual(updated.pending_step_stage, "collect")
        self.assertEqual(updated.pending_step_focus, "storage")

    def test_register_operator_follow_up_tracks_execute_follow_up_from_prose(self) -> None:
        state = OperatorIntentState(
            mode="incident_response",
        )
        updated = register_operator_follow_up(
            operator_intent_state=state,
            final_text=(
                "I recommend the following actions:\n"
                "- Drain and cordon the NotReady node.\n"
                "- Terminate its underlying EC2 instance.\n\n"
                "Would you like me to proceed with draining and cordoning the NotReady node and then terminating its underlying EC2 instance? (yes/no)"
            ),
            turn_plan=type("Plan", (), {"focus": "general"})(),
            approval_pending=False,
        )
        self.assertTrue(updated.awaiting_follow_up)
        self.assertEqual(updated.pending_step_kind, "execute")
        self.assertEqual(updated.pending_step_stage, "execute")
        self.assertEqual(updated.pending_step_focus, "node")
        self.assertIn("draining and cordoning", updated.pending_step_summary)

    def test_update_operator_intent_state_yes_activates_approved_execute_follow_up(self) -> None:
        state = OperatorIntentState(
            mode="follow_up_action",
            pending_step_summary="draining and cordoning the NotReady node and then terminating its underlying EC2 instance",
            pending_step_focus="node",
            pending_step_stage="execute",
            pending_step_kind="execute",
            awaiting_follow_up=True,
        )
        updated = update_operator_intent_state(
            operator_intent_state=state,
            user_input="yes",
            turn_index=5,
            incident_state=IncidentState(active=True, namespace="gitlab", last_focus="node"),
        )
        self.assertEqual(updated.mode, "follow_up_action")
        self.assertFalse(updated.awaiting_follow_up)
        self.assertTrue(updated.approved_proposed_plan)
        self.assertEqual(updated.execution_policy, "approved_follow_up")
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

    def test_approved_execute_follow_up_reuses_targeted_execute_stage(self) -> None:
        incident_state = IncidentState(active=True, namespace="gitlab", last_focus="node")
        operator_state = OperatorIntentState(
            mode="follow_up_action",
            execution_policy="approved_follow_up",
            pending_step_summary="draining and cordoning the NotReady node and then terminating its underlying EC2 instance",
            pending_step_focus="node",
            pending_step_stage="execute",
            pending_step_kind="execute",
            approved_proposed_plan=True,
        )
        plan = build_turn_plan(
            user_input="yes",
            intent=QueryIntent(mode="general"),
            chat_history=[],
            incident_state=incident_state,
            operator_intent_state=operator_state,
        )
        self.assertEqual(plan.stage, "execute")
        self.assertEqual(plan.focus, "node")
        self.assertTrue(plan.continue_existing)
        self.assertFalse(plan.allow_broad_discovery)
        self.assertIn("kubectl_execute", plan.preferred_tools)
        self.assertIn("aws_cli_execute", plan.preferred_tools)


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


class ApprovedFollowUpExecutionTests(unittest.TestCase):
    def test_handle_tool_calls_executes_preapproved_follow_up_without_new_approval_prompt(self) -> None:
        class DummyPrompt:
            @staticmethod
            def format_messages(*, chat_history, input):
                del chat_history, input
                return []

        class DummyLlm:
            def __init__(self, responses):
                self._responses = list(responses)

            def invoke(self, _messages):
                return self._responses.pop(0)

        class DummyTool:
            name = "kubectl_execute"

            def __init__(self):
                self.calls: list[dict] = []

            def invoke(self, args):
                self.calls.append(dict(args))
                return "Executed kubectl write."

        tool = DummyTool()
        operator_state = OperatorIntentState(
            mode="follow_up_action",
            execution_policy="approved_follow_up",
            pending_step_summary="cordon the NotReady node",
            pending_step_focus="node",
            pending_step_stage="execute",
            pending_step_kind="execute",
            approved_proposed_plan=True,
        )
        plan = build_turn_plan(
            user_input="yes",
            intent=QueryIntent(mode="general"),
            chat_history=[],
            incident_state=IncidentState(active=True, last_focus="node"),
            operator_intent_state=operator_state,
        )
        initial_response = SimpleNamespace(
            content="",
            tool_calls=[
                {
                    "id": "call-1",
                    "name": "kubectl_execute",
                    "args": {"command": "cordon ip-10-0-4-225.ec2.internal", "reason": "Replace the NotReady node"},
                }
            ],
        )
        llm_with_tools = DummyLlm([SimpleNamespace(content="Approved follow-up executed.", tool_calls=[])])

        outcome = handle_tool_calls(
            response=initial_response,
            user_input="yes",
            chat_history=[],
            prompt=DummyPrompt(),
            llm=llm_with_tools,
            llm_with_tools=llm_with_tools,
            tools=[tool],
            approval=ApprovalCoordinator(),
            turn_plan=plan,
            incident_state=IncidentState(active=True, last_focus="node"),
            operator_intent_state=operator_state,
        )

        self.assertEqual(outcome.final_text, "Approved follow-up executed.")
        self.assertEqual(len(tool.calls), 1)
        self.assertEqual(tool.calls[0]["command"], "cordon ip-10-0-4-225.ec2.internal")


if __name__ == "__main__":
    unittest.main()
