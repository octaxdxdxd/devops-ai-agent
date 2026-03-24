"""Regression tests for trace-id UI metadata and failed-command repair flow."""

from __future__ import annotations

import unittest

from src.agents.approval import ApprovalCoordinator, attempt_known_command_repair
from src.agents.log_analyzer import LogAnalyzerAgent
from src.ui.session import convert_to_langchain_messages, get_message_content, get_message_trace_id


class TraceHistoryTests(unittest.TestCase):
    def test_message_helpers_prefer_structured_trace_id_and_strip_legacy_suffix(self) -> None:
        message = {
            "role": "assistant",
            "content": "Cluster looks healthy.\n\nTrace ID: `abcdef1234567890`",
            "trace_id": "1234567890abcdef",
        }
        self.assertEqual(get_message_trace_id(message), "1234567890abcdef")
        self.assertEqual(get_message_content(message), "Cluster looks healthy.")

    def test_convert_to_langchain_messages_strips_legacy_trace_suffix(self) -> None:
        messages = [
            {"role": "user", "content": "check pods"},
            {"role": "assistant", "content": "All good.\n\nTrace ID: `abcdef1234567890`"},
        ]
        converted = convert_to_langchain_messages(messages)
        self.assertEqual(converted[0].content, "check pods")
        self.assertEqual(converted[1].content, "All good.")


class FailedCommandRepairTests(unittest.TestCase):
    def test_known_asg_mixed_instances_policy_error_is_repaired(self) -> None:
        original = (
            "autoscaling update-auto-scaling-group "
            "--auto-scaling-group-name demo-asg "
            "--mixed-instances-policy "
            "'{\"LaunchTemplate\":{\"LaunchTemplateId\":\"lt-123\",\"Version\":\"2\"},"
            "\"InstancesDistribution\":{\"OnDemandBaseCapacity\":0,"
            "\"OnDemandPercentageAboveBaseCapacity\":0,"
            "\"SpotAllocationStrategy\":\"price-capacity-optimized\"}}' "
            "--region us-east-1"
        )
        failure = (
            "❌ AWS CLI command failed (exit=252). Details: Parameter validation failed: "
            "Unknown parameter in MixedInstancesPolicy.LaunchTemplate: \"LaunchTemplateId\", "
            "must be one of: LaunchTemplateSpecification, Overrides "
            "Unknown parameter in MixedInstancesPolicy.LaunchTemplate: \"Version\", "
            "must be one of: LaunchTemplateSpecification, Overrides"
        )

        repaired = attempt_known_command_repair("aws_cli_execute", original, failure)
        self.assertIsNotNone(repaired)
        assert repaired is not None
        self.assertIn(
            "\"LaunchTemplateSpecification\":{\"LaunchTemplateId\":\"lt-123\",\"Version\":\"2\"}",
            repaired,
        )
        self.assertNotIn("\"LaunchTemplate\":{\"LaunchTemplateId\"", repaired)

    def test_pending_approval_failure_becomes_corrected_reapproval(self) -> None:
        bad_command = (
            "autoscaling update-auto-scaling-group "
            "--auto-scaling-group-name demo-asg "
            "--mixed-instances-policy "
            "'{\"LaunchTemplate\":{\"LaunchTemplateId\":\"lt-123\",\"Version\":\"2\"},"
            "\"InstancesDistribution\":{\"OnDemandBaseCapacity\":0,"
            "\"OnDemandPercentageAboveBaseCapacity\":0,"
            "\"SpotAllocationStrategy\":\"price-capacity-optimized\"}}' "
            "--region us-east-1"
        )
        failure_result = (
            "❌ AWS CLI command failed (exit=252). Details: Parameter validation failed: "
            "Unknown parameter in MixedInstancesPolicy.LaunchTemplate: \"LaunchTemplateId\", "
            "must be one of: LaunchTemplateSpecification, Overrides "
            "Unknown parameter in MixedInstancesPolicy.LaunchTemplate: \"Version\", "
            "must be one of: LaunchTemplateSpecification, Overrides"
        )

        class DummyTool:
            name = "aws_cli_execute"

            @staticmethod
            def invoke(_args: dict) -> str:
                return failure_result

        dummy_agent = type("DummyAgent", (), {})()
        dummy_agent._approval = ApprovalCoordinator()
        dummy_agent._trace_writer = None
        dummy_agent.llm = None
        dummy_agent._notify_status = LogAnalyzerAgent._notify_status
        dummy_agent._propose_repaired_command = LogAnalyzerAgent._propose_repaired_command.__get__(
            dummy_agent,
            type(dummy_agent),
        )

        dummy_agent._approval.set_pending_action(
            DummyTool(),
            {
                "command": bad_command,
                "reason": "Switch nodegroup back to spot-backed ASG behavior.",
            },
        )

        updates: list[str] = []
        response = LogAnalyzerAgent._handle_pending_approval(
            dummy_agent,
            "yes",
            "",
            "trace-123",
            status_callback=updates.append,
        )

        self.assertIsNotNone(response)
        assert response is not None
        self.assertIn("prepared a corrected command", response)
        self.assertIn("This still requires fresh approval", response)
        self.assertIsNotNone(dummy_agent._approval.pending_action)
        corrected_command = dummy_agent._approval.pending_action.args["command"]
        self.assertIn("LaunchTemplateSpecification", corrected_command)
        self.assertTrue(
            any("drafting a corrected command" in update.lower() for update in updates),
            updates,
        )
        self.assertTrue(
            any("waiting for fresh approval" in update.lower() for update in updates),
            updates,
        )


if __name__ == "__main__":
    unittest.main()
