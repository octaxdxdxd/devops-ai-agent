"""AWS read-only LangChain tools."""

from __future__ import annotations

import re

from langchain_core.tools import tool

from ..infra.aws_client import AWSClient
from ..policy import guard_aws_read_tool
from .output import compress_output, compress_json_output

_READ_ONLY_AWS_OPERATION_RE = re.compile(r"^(describe|get|list|lookup|head|batch_get)_", re.IGNORECASE)


def _is_read_only_aws_operation(operation: str) -> bool:
    return bool(_READ_ONLY_AWS_OPERATION_RE.match(str(operation or "").strip()))


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
        policy_error = guard_aws_read_tool("aws_describe_instances", region=region, service="ec2", operation="describe_instances")
        if policy_error:
            return f"ERROR: {policy_error}"
        import json as _json
        filters = _json.loads(filters_json) if filters_json.strip() else None
        ids = _json.loads(instance_ids_json) if instance_ids_json.strip() else None
        return compress_json_output(aws.describe_instances(filters=filters, instance_ids=ids, region=region or None))

    @tool
    def aws_describe_service(service: str, operation: str, params_json: str = "", region: str = "") -> str:
        """Call any AWS read-only describe/list/get style API. Use for services not covered by other tools.

        Args:
            service: AWS service name (e.g. 'rds', 'elbv2', 'lambda', 'ecs', 'eks', 'elasticache')
            operation: Boto3 operation name (e.g. 'describe_db_instances', 'describe_load_balancers')
            params_json: Optional JSON object of API parameters e.g. '{"DBInstanceIdentifier":"mydb"}'
            region: Optional AWS region override (e.g. 'us-east-1', 'eu-west-1'). Empty = use default region.
        """
        policy_error = guard_aws_read_tool("aws_describe_service", region=region, service=service, operation=operation)
        if policy_error:
            return f"ERROR: {policy_error}"
        if not _is_read_only_aws_operation(operation):
            return (
                f"ERROR: AWS read-only tool blocked non-read operation '{operation}'. "
                "Use an approval-gated write tool instead."
            )
        import json as _json
        params = _json.loads(params_json) if params_json.strip() else None
        if (
            str(service or "").strip().lower() == "events"
            and str(operation or "").strip().lower() == "list_rules"
            and isinstance(params, dict)
            and str(params.get("NamePrefix", "") or "").strip() == ""
        ):
            params = {key: value for key, value in params.items() if key != "NamePrefix"}
        return compress_output(aws.describe_service(service, operation, params, region=region or None))

    @tool
    def aws_inspect_lambda_schedules(
        name_hints_json: str = "",
        regions_json: str = "",
        lookback_days: int = 35,
        include_disabled: bool = False,
    ) -> str:
        """Inspect EventBridge scheduled rules that target Lambda functions.

        Use this for questions about how often a Lambda runs, when it last ran, and when it will run next.
        This tool inspects EventBridge schedules plus CloudWatch Logs/Metrics. It does not use CloudTrail.
        If recent invocation history is weak or inconsistent, `next_run_time` is returned as null with unknown confidence.

        Args:
            name_hints_json: JSON array of non-empty name fragments, e.g. '["kill-tagless-resources","tagless"]'
            regions_json: Optional JSON array of AWS regions to inspect, e.g. '["us-east-1","eu-central-1"]'
            lookback_days: How many days of logs/metrics to check for the last run (default 35)
            include_disabled: Whether to include disabled EventBridge rules
        """
        import json as _json

        name_hints = _json.loads(name_hints_json) if name_hints_json.strip() else []
        regions = _json.loads(regions_json) if regions_json.strip() else []
        for raw_region in regions or [""]:
            policy_error = guard_aws_read_tool(
                "aws_inspect_lambda_schedules",
                region=str(raw_region or ""),
                service="events",
                operation="list_rules",
            )
            if policy_error:
                return f"ERROR: {policy_error}"
        return compress_output(
            aws.inspect_lambda_schedules(
                name_hints=name_hints,
                regions=regions,
                lookback_days=lookback_days,
                include_disabled=include_disabled,
            )
        )

    @tool
    def aws_audit_cloudtrail(
        principal: str = "",
        event_name_exact: str = "",
        event_name_prefix: str = "",
        event_source: str = "",
        resource_type: str = "",
        resource_name: str = "",
        contains_text: str = "",
        start_time: str = "",
        end_time: str = "",
        max_events: int = 200,
        include_raw_event: bool = False,
        region: str = "",
    ) -> str:
        """Search CloudTrail event history with deterministic pagination and client-side filtering.

        Use this for CloudTrail/event-history/audit questions, especially when searching by user,
        delete-style event prefixes, or deleted resources. The tool handles username case variants,
        exact-match CloudTrail semantics, and paginates until it has enough matching events.

        Args:
            principal: Optional principal/user identifier, such as an email, username, or session name.
            event_name_exact: Optional exact CloudTrail EventName filter, e.g. 'DeleteVpc'.
            event_name_prefix: Optional client-side EventName prefix filter, e.g. 'Delete'.
            event_source: Optional exact event source, e.g. 'ec2.amazonaws.com'.
            resource_type: Optional resource type filter, e.g. 'AWS::EC2::VPC'.
            resource_name: Optional resource name/id filter, e.g. 'vpc-abc123'.
            contains_text: Optional case-insensitive text filter against the raw CloudTrail event.
            start_time: Optional start time in ISO-8601 or YYYY-MM-DD.
            end_time: Optional end time in ISO-8601 or YYYY-MM-DD.
            max_events: Maximum matched events to return (default 200, hard max 500).
            include_raw_event: Whether to include parsed raw CloudTrail event JSON in the result.
            region: Optional AWS region override. Empty uses the configured/default region.
        """
        policy_error = guard_aws_read_tool("aws_audit_cloudtrail", region=region, service="cloudtrail", operation="lookup_events")
        if policy_error:
            return f"ERROR: {policy_error}"
        return compress_output(
            aws.audit_cloudtrail(
                principal=principal,
                event_name_exact=event_name_exact,
                event_name_prefix=event_name_prefix,
                event_source=event_source,
                resource_type=resource_type,
                resource_name=resource_name,
                contains_text=contains_text,
                start_time=start_time or None,
                end_time=end_time or None,
                max_events=max_events,
                include_raw_event=include_raw_event,
                region=region or None,
            )
        )

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
            group_by: Dimension to group by. Valid values: SERVICE, REGION, INSTANCE_TYPE, LINKED_ACCOUNT, USAGE_TYPE, PLATFORM, TENANCY, PURCHASE_TYPE, OPERATING_SYSTEM, DATABASE_ENGINE (empty = total cost). Note: RESOURCE_ID is NOT a valid dimension here.
        """
        policy_error = guard_aws_read_tool("aws_get_cost", service="ce", operation="get_cost_and_usage")
        if policy_error:
            return f"ERROR: {policy_error}"
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
        policy_error = guard_aws_read_tool("aws_get_cloudwatch_metrics", service="cloudwatch", operation="get_metric_statistics")
        if policy_error:
            return f"ERROR: {policy_error}"
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
        policy_error = guard_aws_read_tool("aws_get_alarms", service="cloudwatch", operation="describe_alarms")
        if policy_error:
            return f"ERROR: {policy_error}"
        return compress_json_output(aws.get_alarms(state=state or None))

    @tool
    def aws_describe_security_groups(vpc_id: str = "", group_ids_json: str = "") -> str:
        """Describe EC2 security groups with ingress/egress rules.

        Args:
            vpc_id: Filter by VPC ID (empty = all VPCs)
            group_ids_json: JSON array of SG IDs e.g. '["sg-abc123"]'
        """
        policy_error = guard_aws_read_tool("aws_describe_security_groups", service="ec2", operation="describe_security_groups")
        if policy_error:
            return f"ERROR: {policy_error}"
        import json as _json
        ids = _json.loads(group_ids_json) if group_ids_json.strip() else None
        return compress_json_output(aws.describe_security_groups(vpc_id=vpc_id or None, group_ids=ids))

    @tool
    def aws_get_iam_summary() -> str:
        """Get IAM account summary showing user/role/policy counts and security settings."""
        policy_error = guard_aws_read_tool("aws_get_iam_summary", service="iam", operation="get_account_summary")
        if policy_error:
            return f"ERROR: {policy_error}"
        return compress_output(aws.get_iam_summary())

    @tool
    def aws_list_resources(resource_type_filters_json: str = "") -> str:
        """List AWS resources using Resource Groups Tagging API. Good for discovering what exists.

        Args:
            resource_type_filters_json: JSON array of resource type filters e.g. '["ec2:instance","rds:db"]' (empty = all types)
        """
        policy_error = guard_aws_read_tool("aws_list_resources", service="resourcegroupstaggingapi", operation="get_resources")
        if policy_error:
            return f"ERROR: {policy_error}"
        import json as _json
        filters = _json.loads(resource_type_filters_json) if resource_type_filters_json.strip() else None
        return compress_json_output(aws.list_resources(resource_type_filters=filters))

    @tool
    def aws_get_caller_identity() -> str:
        """Get the current AWS identity (account, user/role ARN). Useful to verify AWS access."""
        policy_error = guard_aws_read_tool("aws_get_caller_identity", service="sts", operation="get_caller_identity")
        if policy_error:
            return f"ERROR: {policy_error}"
        return compress_output(aws.get_caller_identity())

    return [
        aws_describe_instances,
        aws_describe_service,
        aws_inspect_lambda_schedules,
        aws_audit_cloudtrail,
        aws_get_cost,
        aws_get_cloudwatch_metrics,
        aws_get_alarms,
        aws_describe_security_groups,
        aws_get_iam_summary,
        aws_list_resources,
        aws_get_caller_identity,
    ]
