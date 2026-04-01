"""AWS write (mutating) LangChain tools."""

from __future__ import annotations

from langchain_core.tools import tool

from ..infra.aws_client import AWSClient


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
        return aws.update_auto_scaling(
            asg_name,
            min_size=min_size if min_size >= 0 else None,
            max_size=max_size if max_size >= 0 else None,
            desired=desired if desired >= 0 else None,
        )

    return [
        aws_run_api_command,
        aws_update_auto_scaling,
    ]
