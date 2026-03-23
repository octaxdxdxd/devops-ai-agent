"""Regression tests for Phase 1 command-hardening changes."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from src.agents.log_analyzer import LogAnalyzerAgent
from src.agents.tool_loop import _command_guard_message, _resolve_tool_call
from src.tools.aws_cli import aws_cli_readonly
from src.tools.k8s_cli import kubectl_readonly
from src.utils.command_intent import Config
from src.utils.command_intent import classify_command_intent, target_tool_for_intent
from src.utils.query_intent import QueryIntent
from src.utils.tracing import _redact_text


class _ToolRef:
    def __init__(self, name: str):
        self.name = name


class CommandIntentTests(unittest.TestCase):
    def tearDown(self) -> None:
        Config.COMMAND_SAFETY_POSTURE = "powerful"

    def test_kubectl_safe_read_routes_to_readonly(self) -> None:
        intent = classify_command_intent("kubectl_readonly", "get pods -A")
        self.assertEqual(intent.capability, "safe_read")
        self.assertEqual(intent.verb, "get")
        self.assertEqual(target_tool_for_intent(intent), "kubectl_readonly")

    def test_kubectl_exec_routes_to_execute_in_powerful_posture(self) -> None:
        intent = classify_command_intent("kubectl_readonly", "exec my-pod -- env")
        self.assertEqual(intent.capability, "write")
        self.assertIn("execute-capable", intent.reason)
        self.assertEqual(target_tool_for_intent(intent), "kubectl_execute")

    def test_kubectl_secret_extraction_is_allowed_in_powerful_posture(self) -> None:
        intent = classify_command_intent(
            "kubectl_readonly",
            "get secret app-db -o jsonpath={.data.password}",
        )
        self.assertEqual(intent.capability, "safe_read")
        self.assertEqual(target_tool_for_intent(intent), "kubectl_readonly")

    def test_kubectl_readonly_with_literal_pipe_is_not_proactively_blocked(self) -> None:
        intent = classify_command_intent("kubectl_readonly", "get pods | head")
        self.assertEqual(intent.capability, "safe_read")
        self.assertEqual(target_tool_for_intent(intent), "kubectl_readonly")

    def test_aws_secret_read_is_allowed_in_powerful_posture(self) -> None:
        intent = classify_command_intent(
            "aws_cli_readonly",
            "secretsmanager get-secret-value --secret-id demo",
        )
        self.assertEqual(intent.capability, "safe_read")
        self.assertEqual(target_tool_for_intent(intent), "aws_cli_readonly")

    def test_aws_query_with_backticks_and_or_is_not_blocked(self) -> None:
        intent = classify_command_intent(
            "aws_cli_readonly",
            "elbv2 describe-load-balancers --region us-east-1 --query 'LoadBalancers[?contains(DNSName, `abc`) || contains(LoadBalancerName, `ingress-nginx`)]' --output json",
        )
        self.assertEqual(intent.capability, "safe_read")
        self.assertEqual(target_tool_for_intent(intent), "aws_cli_readonly")

    def test_kubectl_exec_stays_sensitive_in_strict_posture(self) -> None:
        Config.COMMAND_SAFETY_POSTURE = "strict"
        intent = classify_command_intent("kubectl_readonly", "exec my-pod -- env")
        self.assertEqual(intent.capability, "sensitive_read")
        self.assertIn("privileged/sensitive read", intent.reason)
        self.assertIsNone(target_tool_for_intent(intent))

    def test_kubectl_secret_extraction_stays_sensitive_in_strict_posture(self) -> None:
        Config.COMMAND_SAFETY_POSTURE = "strict"
        intent = classify_command_intent(
            "kubectl_readonly",
            "get secret app-db -o jsonpath={.data.password}",
        )
        self.assertEqual(intent.capability, "sensitive_read")
        self.assertIn("secret value extraction", intent.reason)


class RoutingTests(unittest.TestCase):
    def test_mutating_kubectl_reroutes_to_execute_tool(self) -> None:
        effective_name, effective_args, intent, routed = _resolve_tool_call(
            tool_name="kubectl_readonly",
            tool_args={"command": "delete pod demo -n prod"},
            tool_lookup={
                "kubectl_readonly": object(),
                "kubectl_execute": object(),
            },
        )
        self.assertEqual(effective_name, "kubectl_execute")
        self.assertTrue(routed)
        self.assertIsNotNone(intent)
        assert intent is not None
        self.assertEqual(intent.capability, "write")
        self.assertEqual(effective_args["command"], "delete pod demo -n prod")

    def test_exec_command_is_not_guard_blocked_in_powerful_posture(self) -> None:
        intent = classify_command_intent("kubectl_readonly", "exec demo -- env")
        message = _command_guard_message(requested_tool="kubectl_readonly", intent=intent)
        self.assertIsNone(message)


class CommandToolTests(unittest.TestCase):
    @patch("src.tools.k8s_cli.run_kubectl")
    @patch("src.tools.k8s_cli.ensure_kubectl_installed", return_value=True)
    def test_kubectl_readonly_blocks_exec_before_subprocess(self, _ensure, run_kubectl) -> None:
        response = kubectl_readonly.invoke({"command": "exec my-pod -- env"})
        run_kubectl.assert_not_called()
        self.assertIn("mutating and is blocked in kubectl_readonly", response)

    @patch("src.tools.k8s_cli.run_kubectl", return_value=(0, "apiVersion: v1\nkind: Secret\n", ""))
    @patch("src.tools.k8s_cli.ensure_kubectl_installed", return_value=True)
    def test_kubectl_readonly_allows_secret_extraction_in_powerful_posture(self, _ensure, run_kubectl) -> None:
        response = kubectl_readonly.invoke({"command": "get secret app-db -o jsonpath={.data.password}"})
        run_kubectl.assert_called_once()
        self.assertIn("executed successfully", response)

    @patch("src.tools.k8s_cli.run_kubectl", return_value=(0, "NAME\npod-a\n", ""))
    @patch("src.tools.k8s_cli.ensure_kubectl_installed", return_value=True)
    def test_kubectl_readonly_executes_safe_read(self, _ensure, run_kubectl) -> None:
        response = kubectl_readonly.invoke({"command": "get pods -A"})
        run_kubectl.assert_called_once()
        called_args = run_kubectl.call_args[0][0]
        self.assertIn("get", called_args)
        self.assertIn("pods", called_args)
        self.assertIn("-A", called_args)
        self.assertIn("executed successfully", response)

    @patch("src.tools.k8s_cli.run_kubectl", return_value=(0, "NAME\npod-a\n", ""))
    @patch("src.tools.k8s_cli.ensure_kubectl_installed", return_value=True)
    def test_kubectl_readonly_does_not_proactively_block_literal_pipe_tokens(self, _ensure, run_kubectl) -> None:
        response = kubectl_readonly.invoke({"command": "get pods | head"})
        run_kubectl.assert_called_once()
        self.assertIn("executed successfully", response)

    @patch("src.tools.aws_cli._execute")
    @patch("src.tools.aws_cli._ensure_aws_cli_installed", return_value=True)
    def test_aws_cli_readonly_allows_secret_reads_in_powerful_posture(self, _ensure, execute) -> None:
        execute.return_value = "ok"
        response = aws_cli_readonly.invoke({"command": "secretsmanager get-secret-value --secret-id demo"})
        execute.assert_called_once()
        self.assertEqual(response, "ok")

    @patch("src.tools.aws_cli._execute")
    @patch("src.tools.aws_cli._ensure_aws_cli_installed", return_value=True)
    def test_aws_cli_readonly_allows_trace_query_pattern(self, _ensure, execute) -> None:
        execute.return_value = "ok"
        response = aws_cli_readonly.invoke(
            {
                "command": "elbv2 describe-load-balancers --region us-east-1 --query 'LoadBalancers[?contains(DNSName, `abc`) || contains(LoadBalancerName, `ingress-nginx`)]' --output json"
            }
        )
        execute.assert_called_once()
        self.assertEqual(response, "ok")


class AgentSelectionTests(unittest.TestCase):
    def test_incident_tool_selection_includes_execute_tools(self) -> None:
        dummy_agent = type(
            "DummyAgent",
            (),
            {
                "tools": [
                    _ToolRef("k8s_list_pods"),
                    _ToolRef("kubectl_readonly"),
                    _ToolRef("kubectl_execute"),
                    _ToolRef("helm_readonly"),
                    _ToolRef("helm_execute"),
                    _ToolRef("aws_cli_readonly"),
                    _ToolRef("aws_cli_execute"),
                ]
            },
        )()
        selected = LogAnalyzerAgent._select_tools_for_turn(
            dummy_agent,
            intent=QueryIntent(mode="incident_rca"),
            user_input="diagnose why the service is down",
        )
        selected_names = {tool.name for tool in selected}
        self.assertIn("kubectl_execute", selected_names)
        self.assertIn("helm_execute", selected_names)


class TraceRedactionTests(unittest.TestCase):
    def test_trace_redacts_base64_like_secret_blobs(self) -> None:
        blob = "TjJ6V2VvS0g2R3FSSHRBMkhhRVJjNkFrbFcxOXVleWI2RmZYS05tUGFySVJiV2xTeWE3N3J5b0h6NktPQUFkRg=="
        redacted = _redact_text(f"echo {blob} | base64 -d")
        self.assertNotIn(blob, redacted)
        self.assertIn("[REDACTED_BLOB]", redacted)


if __name__ == "__main__":
    unittest.main()
