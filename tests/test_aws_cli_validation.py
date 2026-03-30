"""Regression tests for AWS CLI validation against installed help/docs."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from src.tools.aws_cli import AWSConnector
from src.tools.common import CommandExecutionResult


def _result(args: list[str], *, stdout: str = "", stderr: str = "", exit_code: int = 0) -> CommandExecutionResult:
    return CommandExecutionResult(
        command=" ".join(args),
        args=list(args),
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        duration_ms=1.0,
    )


class AWSCliValidationTests(unittest.TestCase):
    def test_raw_read_rejects_shell_syntax(self) -> None:
        connector = AWSConnector()
        with self.assertRaises(ValueError) as ctx:
            connector.raw_read("diff <(echo one) <(echo two)")
        self.assertIn("Shell syntax is not allowed", str(ctx.exception))

    def test_raw_read_rejects_invalid_latest_casing(self) -> None:
        connector = AWSConnector()

        def fake_run(args: list[str], *, timeout_sec: int) -> CommandExecutionResult:  # noqa: ARG001
            if args == ["aws", "--version"]:
                return _result(args, stderr="aws-cli/2.31.18 Python/3.12.0 Linux/6 botocore/2.0")
            if args == ["aws", "ec2", "describe-launch-template-versions", "help"]:
                return _result(
                    args,
                    stdout=(
                        "OPTIONS\n"
                        "  --launch-template-id (string)\n"
                        "  --versions (list)\n"
                        "  --region (string)\n"
                    ),
                )
            raise AssertionError(f"Unexpected subprocess call: {args}")

        with patch("src.tools.aws_cli.run_subprocess", side_effect=fake_run):
            with self.assertRaises(ValueError) as ctx:
                connector.raw_read(
                    "ec2 describe-launch-template-versions --launch-template-id lt-123 --versions $LATEST --region us-east-1",
                    all_regions=False,
                )
        self.assertIn("$Latest", str(ctx.exception))

    def test_execute_rejects_wrapped_launch_template_syntax(self) -> None:
        connector = AWSConnector()

        def fake_run(args: list[str], *, timeout_sec: int) -> CommandExecutionResult:  # noqa: ARG001
            if args == ["aws", "--version"]:
                return _result(args, stderr="aws-cli/2.31.18 Python/3.12.0 Linux/6 botocore/2.0")
            if args == ["aws", "autoscaling", "update-auto-scaling-group", "help"]:
                return _result(
                    args,
                    stdout=(
                        "OPTIONS\n"
                        "  --auto-scaling-group-name (string)\n"
                        "  --launch-template (structure)\n"
                        "  --region (string)\n"
                    ),
                )
            raise AssertionError(f"Unexpected subprocess call: {args}")

        with patch("src.tools.aws_cli.run_subprocess", side_effect=fake_run):
            with self.assertRaises(ValueError) as ctx:
                connector.execute(
                    "autoscaling update-auto-scaling-group "
                    "--auto-scaling-group-name my-asg "
                    "--launch-template LaunchTemplate={LaunchTemplateId=lt-123,Version='4'} "
                    "--region us-east-1"
                )
        self.assertIn("LaunchTemplate=", str(ctx.exception))

    def test_execute_accepts_documented_launch_template_shorthand(self) -> None:
        connector = AWSConnector()

        def fake_run(args: list[str], *, timeout_sec: int) -> CommandExecutionResult:  # noqa: ARG001
            if args == ["aws", "--version"]:
                return _result(args, stderr="aws-cli/2.31.18 Python/3.12.0 Linux/6 botocore/2.0")
            if args == ["aws", "autoscaling", "update-auto-scaling-group", "help"]:
                return _result(
                    args,
                    stdout=(
                        "OPTIONS\n"
                        "  --auto-scaling-group-name (string)\n"
                        "  --launch-template (structure)\n"
                        "  --region (string)\n"
                    ),
                )
            raise AssertionError(f"Unexpected subprocess call: {args}")

        with patch("src.tools.aws_cli.run_subprocess", side_effect=fake_run):
            with patch.object(connector, "_run", return_value=("aws autoscaling update-auto-scaling-group ...", {"ok": True})) as mock_run:
                observation = connector.execute(
                    "autoscaling update-auto-scaling-group "
                    "--auto-scaling-group-name my-asg "
                    "--launch-template LaunchTemplateId=lt-123,Version=4 "
                    "--region us-east-1"
                )
        mock_run.assert_called_once()
        self.assertTrue(observation.ok)
        self.assertIn("Executed mutating AWS command", observation.summary)


if __name__ == "__main__":
    unittest.main()
