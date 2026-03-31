"""AWS read-only LangChain tools."""

from __future__ import annotations

from langchain_core.tools import tool

from ..infra.aws_client import AWSClient
from .output import compress_output, compress_json_output


def create_aws_read_tools(aws: AWSClient) -> list:
    """Return read-only AWS LangChain tools bound to the given client."""

    @tool
    def aws_describe_instances(filters_json: str = "", instance_ids_json: str = "", region: str = "") -> str:
        """List EC2 instances with key details (id, type, state, IPs, name).

        Args:
            filters_json: Optional JSON array of filters e.g. '[{"Name":"instance-state-name","Values":["running"]}]'
            instance_ids_json: Optional JSON array of instance IDs e.g. '["i-abc123"]'
            region: Optional AWS region override (e.g. 'us-east-1'). Empty = use default region.
        """
        import json as _json
        filters = _json.loads(filters_json) if filters_json.strip() else None
        ids = _json.loads(instance_ids_json) if instance_ids_json.strip() else None
        return compress_json_output(aws.describe_instances(filters=filters, instance_ids=ids, region=region or None))

    @tool
    def aws_describe_service(service: str, operation: str, params_json: str = "", region: str = "") -> str:
        """Call any AWS describe/list/get API. Use for services not covered by other tools.

        Args:
            service: AWS service name (e.g. 'rds', 'elbv2', 'lambda', 'ecs', 'eks', 'elasticache')
            operation: Boto3 operation name (e.g. 'describe_db_instances', 'describe_load_balancers')
            params_json: Optional JSON object of API parameters e.g. '{"DBInstanceIdentifier":"mydb"}'
            region: Optional AWS region override (e.g. 'us-east-1', 'eu-west-1'). Empty = use default region.
        """
        import json as _json
        params = _json.loads(params_json) if params_json.strip() else None
        return compress_output(aws.describe_service(service, operation, params, region=region or None))

    @tool
    def aws_get_cost(
        start_date: str = "",
        end_date: str = "",
        granularity: str = "MONTHLY",
        group_by: str = "",
    ) -> str:
        """Get AWS cost and usage data from Cost Explorer.

        Args:
            start_date: Start date YYYY-MM-DD (default: 30 days ago)
            end_date: End date YYYY-MM-DD (default: today)
            granularity: DAILY, MONTHLY, or HOURLY
            group_by: Dimension to group by: SERVICE, REGION, INSTANCE_TYPE, LINKED_ACCOUNT (empty = total)
        """
        return compress_output(
            aws.get_cost(
                start_date=start_date or None,
                end_date=end_date or None,
                granularity=granularity,
                group_by=group_by or None,
            )
        )

    @tool
    def aws_get_cloudwatch_metrics(
        namespace: str,
        metric_name: str,
        dimensions_json: str = "",
        period: int = 300,
        stat: str = "Average",
        hours: int = 1,
    ) -> str:
        """Get CloudWatch metric datapoints.

        Args:
            namespace: CloudWatch namespace (e.g. 'AWS/EC2', 'AWS/RDS', 'AWS/ELB')
            metric_name: Metric name (e.g. 'CPUUtilization', 'DatabaseConnections')
            dimensions_json: JSON array of dimensions e.g. '[{"Name":"InstanceId","Value":"i-abc123"}]'
            period: Data point interval in seconds (default 300)
            stat: Statistic: Average, Sum, Maximum, Minimum, SampleCount
            hours: How many hours of data to fetch (default 1)
        """
        import json as _json
        dims = _json.loads(dimensions_json) if dimensions_json.strip() else None
        return compress_output(
            aws.get_cloudwatch_metrics(namespace, metric_name, dims, period, stat, hours)
        )

    @tool
    def aws_get_alarms(state: str = "") -> str:
        """Get CloudWatch alarms. Useful for checking what's alerting.

        Args:
            state: Filter by state: OK, ALARM, INSUFFICIENT_DATA (empty = all)
        """
        return compress_json_output(aws.get_alarms(state=state or None))

    @tool
    def aws_describe_security_groups(vpc_id: str = "", group_ids_json: str = "") -> str:
        """Describe EC2 security groups with ingress/egress rules.

        Args:
            vpc_id: Filter by VPC ID (empty = all VPCs)
            group_ids_json: JSON array of SG IDs e.g. '["sg-abc123"]'
        """
        import json as _json
        ids = _json.loads(group_ids_json) if group_ids_json.strip() else None
        return compress_json_output(aws.describe_security_groups(vpc_id=vpc_id or None, group_ids=ids))

    @tool
    def aws_get_iam_summary() -> str:
        """Get IAM account summary showing user/role/policy counts and security settings."""
        return compress_output(aws.get_iam_summary())

    @tool
    def aws_list_resources(resource_type_filters_json: str = "") -> str:
        """List AWS resources using Resource Groups Tagging API. Good for discovering what exists.

        Args:
            resource_type_filters_json: JSON array of resource type filters e.g. '["ec2:instance","rds:db"]' (empty = all types)
        """
        import json as _json
        filters = _json.loads(resource_type_filters_json) if resource_type_filters_json.strip() else None
        return compress_json_output(aws.list_resources(resource_type_filters=filters))

    @tool
    def aws_get_caller_identity() -> str:
        """Get the current AWS identity (account, user/role ARN). Useful to verify AWS access."""
        return compress_output(aws.get_caller_identity())

    return [
        aws_describe_instances,
        aws_describe_service,
        aws_get_cost,
        aws_get_cloudwatch_metrics,
        aws_get_alarms,
        aws_describe_security_groups,
        aws_get_iam_summary,
        aws_list_resources,
        aws_get_caller_identity,
    ]
