"""Regression tests for trace-id UI metadata and failed-command repair flow."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from src.agents.approval import ApprovalCoordinator
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
    def test_generic_repair_prompt_handles_json_commands_without_template_crash(self) -> None:
        original = (
            "patch deployment gitlab-webservice-default -n gitlab --type=json "
            "'-p=[{\"op\":\"replace\",\"path\":\"/spec/template/spec/containers/0/resources/requests/memory\","
            "\"value\":\"2000M\"}]'"
        )
        failure = "The request is invalid: json patch replace operation does not apply: doc is missing path"

        class FakeLlm:
            def invoke(self, messages):
                self.messages = messages
                return SimpleNamespace(
                    content=(
                        "STATUS: repair\n"
                        "SUMMARY: Use a merge patch so the missing path can be created safely.\n"
                        "COMMAND: patch deployment gitlab-webservice-default -n gitlab --type=merge "
                        "-p '{\"spec\":{\"template\":{\"spec\":{\"containers\":[{\"name\":\"webservice\","
                        "\"resources\":{\"requests\":{\"memory\":\"2000M\"}}}]}}}}'\n"
                        "REASON: The original replace patch targeted a path that does not exist.\n"
                    )
                )

        dummy_agent = type("DummyAgent", (), {})()
        dummy_agent._trace_writer = None
        dummy_agent.llm = FakeLlm()
        dummy_agent._notify_status = LogAnalyzerAgent._notify_status
        dummy_agent._propose_repaired_command = LogAnalyzerAgent._propose_repaired_command.__get__(
            dummy_agent,
            type(dummy_agent),
        )

        repaired_args, summary = dummy_agent._propose_repaired_command(
            tool_name="kubectl_execute",
            tool_args={"command": original, "reason": "Lower memory pressure on gitlab webservice"},
            failure_text=failure,
            trace_id="trace-123",
        )

        self.assertIsNotNone(repaired_args)
        assert repaired_args is not None
        self.assertIn("--type=merge", repaired_args["command"])
        self.assertIn("missing path", summary.lower())

    def test_pending_approval_failure_becomes_corrected_reapproval(self) -> None:
        bad_command = (
            "patch deployment gitlab-webservice-default -n gitlab --type=json "
            "'-p=[{\"op\":\"replace\",\"path\":\"/spec/template/spec/containers/0/resources/requests/memory\","
            "\"value\":\"2000M\"},{\"op\":\"replace\",\"path\":\"/spec/template/spec/containers/0/resources/limits/memory\","
            "\"value\":\"2500M\"}]'"
        )
        failure_result = "❌ kubectl command failed (exit=1). Details: json patch replace operation does not apply: doc is missing path"

        class DummyTool:
            name = "kubectl_execute"

            @staticmethod
            def invoke(_args: dict) -> str:
                return failure_result

        class FakeLlm:
            def invoke(self, _messages):
                return SimpleNamespace(
                    content=(
                        "STATUS: repair\n"
                        "SUMMARY: Switch to a merge patch so memory limits can be added.\n"
                        "COMMAND: patch deployment gitlab-webservice-default -n gitlab --type=merge "
                        "-p '{\"spec\":{\"template\":{\"spec\":{\"containers\":[{\"name\":\"webservice\","
                        "\"resources\":{\"requests\":{\"memory\":\"2000M\"},\"limits\":{\"memory\":\"2500M\"}}}]}}}}'\n"
                        "REASON: The original JSON patch tried to replace a missing limits path.\n"
                    )
                )

        dummy_agent = type("DummyAgent", (), {})()
        dummy_agent._approval = ApprovalCoordinator()
        dummy_agent._trace_writer = None
        dummy_agent.llm = FakeLlm()
        dummy_agent._notify_status = LogAnalyzerAgent._notify_status
        dummy_agent._propose_repaired_command = LogAnalyzerAgent._propose_repaired_command.__get__(
            dummy_agent,
            type(dummy_agent),
        )

        dummy_agent._approval.set_pending_action(
            DummyTool(),
            {
                "command": bad_command,
                "reason": "Update memory request to 2000M and memory limit to 2500M for gitlab webservice.",
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
        self.assertIn("--type=merge", corrected_command)
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
