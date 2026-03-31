"""AWS infrastructure client wrapping boto3."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import boto3
from botocore.exceptions import ClientError, NoCredentialsError, BotoCoreError

from ..config import Config

log = logging.getLogger(__name__)


class AWSClient:
    """Thin wrapper around boto3 for AWS operations."""

    def __init__(self) -> None:
        self.session = self._create_session()
        self.region = Config.AWS_CLI_DEFAULT_REGION or self.session.region_name or "us-east-1"

    def _create_session(self) -> boto3.Session:
        kwargs: dict[str, Any] = {}
        if Config.AWS_CLI_PROFILE:
            kwargs["profile_name"] = Config.AWS_CLI_PROFILE
        if Config.AWS_CLI_DEFAULT_REGION:
            kwargs["region_name"] = Config.AWS_CLI_DEFAULT_REGION
        return boto3.Session(**kwargs)

    def _client(self, service: str, region: str | None = None):
        return self.session.client(service, region_name=region or self.region)

    def _safe(self, fn, *args, **kwargs) -> dict | str:
        try:
            return fn(*args, **kwargs)
        except NoCredentialsError:
            return "ERROR: AWS credentials not configured. Set AWS_PROFILE or env vars."
        except ClientError as exc:
            code = exc.response["Error"]["Code"]
            msg = exc.response["Error"]["Message"]
            return f"ERROR: AWS {code}: {msg}"
        except BotoCoreError as exc:
            return f"ERROR: {exc}"
        except Exception as exc:
            return f"ERROR: {type(exc).__name__}: {exc}"

    # ── availability ─────────────────────────────────────────────────────

    def available(self) -> bool:
        if not Config.AWS_CLI_ENABLED:
            return False
        result = self._safe(self._client("sts").get_caller_identity)
        return not isinstance(result, str)

    # ── read operations ──────────────────────────────────────────────────

    def describe_instances(self, filters: list[dict] | None = None, instance_ids: list[str] | None = None, region: str | None = None) -> str:
        ec2 = self._client("ec2", region=region)
        kwargs: dict[str, Any] = {}
        if filters:
            kwargs["Filters"] = filters
        if instance_ids:
            kwargs["InstanceIds"] = instance_ids
        result = self._safe(ec2.describe_instances, **kwargs)
        if isinstance(result, str):
            return result
        instances = []
        for reservation in result.get("Reservations", []):
            for inst in reservation.get("Instances", []):
                tags = {t["Key"]: t["Value"] for t in inst.get("Tags", [])}
                instances.append({
                    "id": inst["InstanceId"],
                    "type": inst["InstanceType"],
                    "state": inst["State"]["Name"],
                    "az": inst.get("Placement", {}).get("AvailabilityZone"),
                    "private_ip": inst.get("PrivateIpAddress"),
                    "public_ip": inst.get("PublicIpAddress"),
                    "name": tags.get("Name", ""),
                    "launch_time": str(inst.get("LaunchTime", "")),
                })
        return json.dumps(instances, indent=2, default=str)

    def describe_service(self, service: str, operation: str, params: dict | None = None, region: str | None = None) -> str:
        client = self._client(service, region=region)
        if not hasattr(client, operation):
            return f"ERROR: Operation '{operation}' not found on service '{service}'"
        fn = getattr(client, operation)
        result = self._safe(fn, **(params or {}))
        if isinstance(result, str):
            return result
        result.pop("ResponseMetadata", None)
        return json.dumps(result, indent=2, default=str)

    def get_cost(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        granularity: str = "MONTHLY",
        group_by: str | None = None,
    ) -> str:
        ce = self._client("ce")
        now = datetime.now(timezone.utc)
        if not end_date:
            end_date = now.strftime("%Y-%m-%d")
        if not start_date:
            start_date = (now - timedelta(days=30)).strftime("%Y-%m-%d")
        kwargs: dict[str, Any] = {
            "TimePeriod": {"Start": start_date, "End": end_date},
            "Granularity": granularity,
            "Metrics": ["UnblendedCost", "UsageQuantity"],
        }
        if group_by:
            kwargs["GroupBy"] = [{"Type": "DIMENSION", "Key": group_by}]
        result = self._safe(ce.get_cost_and_usage, **kwargs)
        if isinstance(result, str):
            return result
        periods = []
        for period in result.get("ResultsByTime", []):
            entry: dict[str, Any] = {
                "start": period["TimePeriod"]["Start"],
                "end": period["TimePeriod"]["End"],
            }
            if period.get("Groups"):
                entry["groups"] = [
                    {"keys": g["Keys"], "cost": g["Metrics"]["UnblendedCost"]["Amount"]}
                    for g in period["Groups"]
                ]
            else:
                entry["total_cost"] = period.get("Total", {}).get("UnblendedCost", {}).get("Amount")
            periods.append(entry)
        return json.dumps(periods, indent=2)

    def get_cloudwatch_metrics(
        self,
        namespace: str,
        metric_name: str,
        dimensions: list[dict] | None = None,
        period: int = 300,
        stat: str = "Average",
        hours: int = 1,
    ) -> str:
        cw = self._client("cloudwatch")
        now = datetime.now(timezone.utc)
        kwargs: dict[str, Any] = {
            "Namespace": namespace,
            "MetricName": metric_name,
            "StartTime": now - timedelta(hours=hours),
            "EndTime": now,
            "Period": period,
            "Statistics": [stat],
        }
        if dimensions:
            kwargs["Dimensions"] = dimensions
        result = self._safe(cw.get_metric_statistics, **kwargs)
        if isinstance(result, str):
            return result
        datapoints = sorted(result.get("Datapoints", []), key=lambda d: d.get("Timestamp", ""))
        simplified = [
            {"time": str(d["Timestamp"]), "value": d.get(stat, d.get("Average"))}
            for d in datapoints
        ]
        return json.dumps(simplified, indent=2, default=str)

    def get_alarms(self, state: str | None = None) -> str:
        cw = self._client("cloudwatch")
        kwargs: dict[str, Any] = {}
        if state:
            kwargs["StateValue"] = state
        result = self._safe(cw.describe_alarms, **kwargs)
        if isinstance(result, str):
            return result
        alarms = []
        for a in result.get("MetricAlarms", []):
            alarms.append({
                "name": a["AlarmName"],
                "state": a["StateValue"],
                "metric": a.get("MetricName"),
                "namespace": a.get("Namespace"),
                "reason": a.get("StateReason", "")[:200],
            })
        for a in result.get("CompositeAlarms", []):
            alarms.append({
                "name": a["AlarmName"],
                "state": a["StateValue"],
                "type": "composite",
                "reason": a.get("StateReason", "")[:200],
            })
        return json.dumps(alarms, indent=2)

    def describe_security_groups(self, vpc_id: str | None = None, group_ids: list[str] | None = None) -> str:
        ec2 = self._client("ec2")
        kwargs: dict[str, Any] = {}
        filters: list[dict] = []
        if vpc_id:
            filters.append({"Name": "vpc-id", "Values": [vpc_id]})
        if filters:
            kwargs["Filters"] = filters
        if group_ids:
            kwargs["GroupIds"] = group_ids
        result = self._safe(ec2.describe_security_groups, **kwargs)
        if isinstance(result, str):
            return result
        sgs = []
        for sg in result.get("SecurityGroups", []):
            sgs.append({
                "id": sg["GroupId"],
                "name": sg["GroupName"],
                "vpc": sg.get("VpcId"),
                "description": sg.get("Description", ""),
                "ingress_rules": len(sg.get("IpPermissions", [])),
                "egress_rules": len(sg.get("IpPermissionsEgress", [])),
                "ingress": [
                    {
                        "protocol": r.get("IpProtocol"),
                        "from_port": r.get("FromPort"),
                        "to_port": r.get("ToPort"),
                        "sources": [ip["CidrIp"] for ip in r.get("IpRanges", [])]
                                  + [g["GroupId"] for g in r.get("UserIdGroupPairs", [])],
                    }
                    for r in sg.get("IpPermissions", [])
                ],
            })
        return json.dumps(sgs, indent=2)

    def get_iam_summary(self) -> str:
        iam = self._client("iam")
        result = self._safe(iam.get_account_summary)
        if isinstance(result, str):
            return result
        return json.dumps(result.get("SummaryMap", {}), indent=2)

    def list_resources(self, resource_type_filters: list[str] | None = None) -> str:
        tagging = self._client("resourcegroupstaggingapi")
        kwargs: dict[str, Any] = {}
        if resource_type_filters:
            kwargs["ResourceTypeFilters"] = resource_type_filters
        result = self._safe(tagging.get_resources, **kwargs)
        if isinstance(result, str):
            return result
        resources = []
        for r in result.get("ResourceTagMappingList", [])[:100]:
            tags = {t["Key"]: t["Value"] for t in r.get("Tags", [])}
            resources.append({"arn": r["ResourceARN"], "tags": tags})
        return json.dumps(resources, indent=2)

    def get_caller_identity(self) -> str:
        sts = self._client("sts")
        result = self._safe(sts.get_caller_identity)
        if isinstance(result, str):
            return result
        result.pop("ResponseMetadata", None)
        return json.dumps(result, indent=2)

    # ── write operations ─────────────────────────────────────────────────

    def run_command(self, service: str, operation: str, params: dict | None = None, region: str | None = None) -> str:
        client = self._client(service, region=region)
        if not hasattr(client, operation):
            return f"ERROR: Operation '{operation}' not found on service '{service}'"
        fn = getattr(client, operation)
        result = self._safe(fn, **(params or {}))
        if isinstance(result, str):
            return result
        result.pop("ResponseMetadata", None)
        return json.dumps(result, indent=2, default=str)

    def update_auto_scaling(
        self,
        asg_name: str,
        min_size: int | None = None,
        max_size: int | None = None,
        desired: int | None = None,
    ) -> str:
        asg = self._client("autoscaling")
        kwargs: dict[str, Any] = {"AutoScalingGroupName": asg_name}
        if min_size is not None:
            kwargs["MinSize"] = min_size
        if max_size is not None:
            kwargs["MaxSize"] = max_size
        if desired is not None:
            kwargs["DesiredCapacity"] = desired
        result = self._safe(asg.update_auto_scaling_group, **kwargs)
        if isinstance(result, str):
            return result
        return f"Auto Scaling group '{asg_name}' updated successfully."
