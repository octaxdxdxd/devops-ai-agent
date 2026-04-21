"""AWS write (mutating) LangChain tools."""

from __future__ import annotations

from langchain_core.tools import tool

from ..infra.aws_client import AWSClient
from ..policy import guard_aws_write_tool


def create_aws_write_tools(aws: AWSClient) -> list:
    """Return mutating AWS tools. These require explicit user approval before execution."""

    @tool
    def aws_run_api_command(service: str, operation: str, params_json: str = "", region: str = "") -> str:
        """Execute any AWS API operation via boto3. Use for modifications not covered by other tools.

        Args:
            service: AWS service name (e.g. 'ec2', 'rds', 'ecs')
            operation: Boto3 operation name (e.g. 'stop_instances', 'reboot_db_instance')
            params_json: JSON object of API parameters e.g. '{"InstanceIds":["i-abc"]}'
            region: Optional AWS region override (e.g. 'us-east-1', 'eu-west-1'). Empty = use default region.
        """
        policy_error = guard_aws_write_tool("aws_run_api_command", region=region, service=service, operation=operation)
        if policy_error:
            return f"ERROR: {policy_error}"
        import json as _json
        params = _json.loads(params_json) if params_json.strip() else None
        return aws.run_command(service, operation, params, region=region or None)

    @tool
    def aws_update_auto_scaling(
        asg_name: str,
        min_size: int = -1,
        max_size: int = -1,
        desired: int = -1,
    ) -> str:
        """Update an Auto Scaling group's capacity settings.

        Args:
            asg_name: Auto Scaling group name
            min_size: New minimum size (-1 = don't change)
            max_size: New maximum size (-1 = don't change)
            desired: New desired capacity (-1 = don't change)
        """
        policy_error = guard_aws_write_tool("aws_update_auto_scaling", service="autoscaling", operation="update_auto_scaling_group")
        if policy_error:
            return f"ERROR: {policy_error}"
        return aws.update_auto_scaling(
            asg_name,
            min_size=min_size if min_size >= 0 else None,
            max_size=max_size if max_size >= 0 else None,
            desired=desired if desired >= 0 else None,
        )

    @tool
    def aws_resume_auto_scaling_processes(
        asg_name: str,
        processes_csv: str = "Launch,Terminate,HealthCheck,ReplaceUnhealthy,AZRebalance,ScheduledActions",
    ) -> str:
        """Resume specific suspended Auto Scaling processes with before/after verification.

        Args:
            asg_name: Auto Scaling group name
            processes_csv: Comma-separated process names to resume
        """
        policy_error = guard_aws_write_tool(
            "aws_resume_auto_scaling_processes",
            service="autoscaling",
            operation="resume_processes",
        )
        if policy_error:
            return f"ERROR: {policy_error}"
        processes = [item.strip() for item in processes_csv.split(",") if item.strip()]
        return aws.resume_auto_scaling_processes(asg_name, processes=processes or None)

    return [
        aws_run_api_command,
        aws_update_auto_scaling,
        aws_resume_auto_scaling_processes,
    ]
