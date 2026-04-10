"""AWS infrastructure client wrapping boto3."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any

import boto3
from botocore.exceptions import ClientError, NoCredentialsError, BotoCoreError

from ..config import Config

log = logging.getLogger(__name__)

_EMAIL_LOCAL_PART_RE = re.compile(r"([._-])")
_DEFAULT_CLOUDTRAIL_MAX_EVENTS = 200
_MAX_CLOUDTRAIL_MAX_EVENTS = 500
_MAX_CLOUDTRAIL_FALLBACK_SCAN = 1000


def _coerce_int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _parse_timestamp(value: object) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            try:
                dt = datetime.strptime(str(value).strip(), "%Y-%m-%d")
            except ValueError:
                return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _format_timestamp(value: object) -> str:
    dt = _parse_timestamp(value)
    if dt is None:
        return str(value or "")
    return dt.isoformat().replace("+00:00", "Z")


def _unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return unique


def _principal_variants(principal: str) -> list[str]:
    text = str(principal or "").strip()
    if not text:
        return []

    variants = [text]
    lower = text.lower()
    if lower != text:
        variants.append(lower)

    if "@" in text:
        local_part, domain = text.split("@", 1)
        titled_local = "".join(
            segment.capitalize() if segment and segment not in "._-" else segment
            for segment in _EMAIL_LOCAL_PART_RE.split(local_part)
        )
        variants.append(f"{titled_local}@{domain}")
        variants.append(f"{titled_local}@{domain.lower()}")

    return _unique_preserve_order(variants)


def _resource_value(resource: dict, *keys: str) -> str:
    for key in keys:
        value = resource.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _normalize_cloudtrail_resources(raw_event: dict, raw_payload: dict) -> list[dict[str, str]]:
    resources: list[dict[str, str]] = []

    for resource in raw_event.get("Resources", []) or []:
        resource_type = _resource_value(resource, "ResourceType", "resourceType", "type")
        resource_name = _resource_value(resource, "ResourceName", "resourceName", "name", "ARN", "arn")
        resources.append({"resource_type": resource_type, "resource_name": resource_name})

    if resources:
        return resources

    for resource in raw_payload.get("resources", []) or []:
        resource_type = _resource_value(resource, "type", "resourceType", "ResourceType")
        resource_name = _resource_value(resource, "ARN", "arn", "resourceName", "ResourceName")
        resources.append({"resource_type": resource_type, "resource_name": resource_name})

    return resources


def _normalize_cloudtrail_event(raw_event: dict, include_raw_event: bool = False) -> dict[str, Any]:
    raw_event_text = str(raw_event.get("CloudTrailEvent", "") or "")
    try:
        raw_payload = json.loads(raw_event_text) if raw_event_text else {}
    except (TypeError, ValueError):
        raw_payload = {}

    event_time_value = raw_event.get("EventTime") or raw_payload.get("eventTime")
    event_time = _parse_timestamp(event_time_value)
    identity = raw_payload.get("userIdentity", {}) if isinstance(raw_payload, dict) else {}
    session_issuer = (
        identity.get("sessionContext", {}).get("sessionIssuer", {})
        if isinstance(identity, dict)
        else {}
    )

    normalized: dict[str, Any] = {
        "event_id": str(raw_event.get("EventId", "") or raw_payload.get("eventID", "")),
        "event_name": str(raw_event.get("EventName", "") or raw_payload.get("eventName", "")),
        "event_source": str(raw_event.get("EventSource", "") or raw_payload.get("eventSource", "")),
        "event_time": event_time.isoformat().replace("+00:00", "Z") if event_time else _format_timestamp(event_time_value),
        "aws_region": str(raw_payload.get("awsRegion", "") or ""),
        "username": str(raw_event.get("Username", "") or identity.get("userName", "") or ""),
        "read_only": _coerce_bool(raw_event.get("ReadOnly", raw_payload.get("readOnly"))),
        "resources": _normalize_cloudtrail_resources(raw_event, raw_payload),
        "principal": {
            "type": str(identity.get("type", "") or ""),
            "user_name": str(identity.get("userName", "") or ""),
            "principal_id": str(identity.get("principalId", "") or ""),
            "arn": str(identity.get("arn", "") or ""),
            "session_issuer_arn": str(session_issuer.get("arn", "") or ""),
            "session_issuer_user_name": str(session_issuer.get("userName", "") or ""),
        },
        "_event_time_sort_key": event_time.timestamp() if event_time else float("-inf"),
        "_search_blob": raw_event_text.lower(),
    }
    if include_raw_event:
        normalized["raw_event"] = raw_payload or raw_event_text
    return normalized


def _build_principal_search_terms(principal: str) -> list[str]:
    variants = _principal_variants(principal)
    terms = list(variants)
    for variant in variants:
        lower = variant.lower()
        if lower not in terms:
            terms.append(lower)
    return _unique_preserve_order(terms)


def _event_matches_filters(
    event: dict[str, Any],
    *,
    principal: str = "",
    event_name_exact: str = "",
    event_name_prefix: str = "",
    event_source: str = "",
    resource_type: str = "",
    resource_name: str = "",
    contains_text: str = "",
) -> bool:
    event_name = str(event.get("event_name", "") or "")
    event_name_lower = event_name.lower()
    event_source_value = str(event.get("event_source", "") or "")
    event_source_lower = event_source_value.lower()

    if event_name_exact and event_name_lower != str(event_name_exact).strip().lower():
        return False
    if event_name_prefix and not event_name_lower.startswith(str(event_name_prefix).strip().lower()):
        return False
    if event_source and event_source_lower != str(event_source).strip().lower():
        return False

    resources = event.get("resources", []) or []
    if resource_type:
        wanted = str(resource_type).strip().lower()
        if not any(str(resource.get("resource_type", "") or "").lower() == wanted for resource in resources):
            return False
    if resource_name:
        wanted = str(resource_name).strip().lower()
        if not any(str(resource.get("resource_name", "") or "").lower() == wanted for resource in resources):
            return False

    if principal:
        terms = _build_principal_search_terms(principal)
        haystacks = [
            str(event.get("username", "") or ""),
            str(event.get("principal", {}).get("user_name", "") or ""),
            str(event.get("principal", {}).get("principal_id", "") or ""),
            str(event.get("principal", {}).get("arn", "") or ""),
            str(event.get("principal", {}).get("session_issuer_arn", "") or ""),
            str(event.get("principal", {}).get("session_issuer_user_name", "") or ""),
            str(event.get("_search_blob", "") or ""),
        ]
        lowered_haystacks = [value.lower() for value in haystacks if value]
        if not any(term.lower() in haystack for term in terms for haystack in lowered_haystacks):
            return False

    if contains_text:
        needle = str(contains_text).strip().lower()
        combined = " ".join(
            [
                event_name,
                event_source_value,
                str(event.get("username", "") or ""),
                str(event.get("_search_blob", "") or ""),
                " ".join(
                    f"{resource.get('resource_type', '')} {resource.get('resource_name', '')}"
                    for resource in resources
                ),
            ]
        ).lower()
        if needle not in combined:
            return False

    return True


def _cloudtrail_lookup_attribute(
    *,
    principal_variant: str = "",
    event_name_exact: str = "",
    event_source: str = "",
    resource_name: str = "",
    resource_type: str = "",
) -> dict[str, str] | None:
    if principal_variant:
        return {"AttributeKey": "Username", "AttributeValue": principal_variant}
    if event_name_exact:
        return {"AttributeKey": "EventName", "AttributeValue": event_name_exact}
    if event_source:
        return {"AttributeKey": "EventSource", "AttributeValue": event_source}
    if resource_name:
        return {"AttributeKey": "ResourceName", "AttributeValue": resource_name}
    if resource_type:
        return {"AttributeKey": "ResourceType", "AttributeValue": resource_type}
    return None


def _cloudtrail_fallback_scan_cap(max_events: int) -> int:
    return min(max(max_events * 10, 200), _MAX_CLOUDTRAIL_FALLBACK_SCAN)


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

    def audit_cloudtrail(
        self,
        *,
        principal: str = "",
        event_name_exact: str = "",
        event_name_prefix: str = "",
        event_source: str = "",
        resource_type: str = "",
        resource_name: str = "",
        contains_text: str = "",
        start_time: str | None = None,
        end_time: str | None = None,
        max_events: int = _DEFAULT_CLOUDTRAIL_MAX_EVENTS,
        include_raw_event: bool = False,
        region: str | None = None,
    ) -> str:
        target_region = str(region or self.region).strip() or self.region
        max_events = max(1, min(_coerce_int(max_events, _DEFAULT_CLOUDTRAIL_MAX_EVENTS), _MAX_CLOUDTRAIL_MAX_EVENTS))
        include_raw_event = _coerce_bool(include_raw_event)

        parsed_start_time = _parse_timestamp(start_time)
        if start_time and parsed_start_time is None:
            return f"ERROR: Invalid CloudTrail start_time '{start_time}'. Use ISO-8601 or YYYY-MM-DD."
        parsed_end_time = _parse_timestamp(end_time)
        if end_time and parsed_end_time is None:
            return f"ERROR: Invalid CloudTrail end_time '{end_time}'. Use ISO-8601 or YYYY-MM-DD."

        cloudtrail = self._client("cloudtrail", region=target_region)
        matched_events: list[dict[str, Any]] = []
        seen_event_ids: set[str] = set()
        lookup_attempts: list[dict[str, Any]] = []
        principal_variants_tried: list[str] = []
        used_fallback_scan = False
        scan_limited = False

        filter_kwargs = {
            "principal": principal,
            "event_name_exact": event_name_exact,
            "event_name_prefix": event_name_prefix,
            "event_source": event_source,
            "resource_type": resource_type,
            "resource_name": resource_name,
            "contains_text": contains_text,
        }

        def run_lookup(
            *,
            lookup_attribute: dict[str, str] | None,
            mode: str,
            scan_cap: int | None = None,
        ) -> str | None:
            nonlocal scan_limited

            next_token: str | None = None
            raw_events_scanned = 0
            pages = 0
            attempt = {
                "mode": mode,
                "lookup_attribute": lookup_attribute,
                "region": target_region,
                "pages": 0,
                "raw_events_scanned": 0,
                "matched_events_after_attempt": len(matched_events),
                "scan_cap": scan_cap,
            }

            while True:
                kwargs: dict[str, Any] = {"MaxResults": 50}
                if lookup_attribute:
                    kwargs["LookupAttributes"] = [lookup_attribute]
                if parsed_start_time:
                    kwargs["StartTime"] = parsed_start_time
                if parsed_end_time:
                    kwargs["EndTime"] = parsed_end_time
                if next_token:
                    kwargs["NextToken"] = next_token

                result = self._safe(cloudtrail.lookup_events, **kwargs)
                if isinstance(result, str):
                    attempt["error"] = result
                    lookup_attempts.append(attempt)
                    return result

                pages += 1
                page_events = result.get("Events", []) or []
                raw_events_scanned += len(page_events)

                for raw_event in page_events:
                    event = _normalize_cloudtrail_event(raw_event, include_raw_event=include_raw_event)
                    event_id = str(event.get("event_id", "") or "")
                    if event_id and event_id in seen_event_ids:
                        continue
                    if not _event_matches_filters(event, **filter_kwargs):
                        continue
                    if event_id:
                        seen_event_ids.add(event_id)
                    matched_events.append(event)
                    if len(matched_events) >= max_events:
                        attempt["pages"] = pages
                        attempt["raw_events_scanned"] = raw_events_scanned
                        attempt["matched_events_after_attempt"] = len(matched_events)
                        lookup_attempts.append(attempt)
                        return None

                next_token = result.get("NextToken")
                if not next_token:
                    break
                if scan_cap is not None and raw_events_scanned >= scan_cap:
                    scan_limited = True
                    break

            attempt["pages"] = pages
            attempt["raw_events_scanned"] = raw_events_scanned
            attempt["matched_events_after_attempt"] = len(matched_events)
            attempt["scan_limited"] = bool(next_token) and scan_cap is not None and raw_events_scanned >= scan_cap
            lookup_attempts.append(attempt)
            return None

        if principal:
            for variant in _principal_variants(principal):
                principal_variants_tried.append(variant)
                error = run_lookup(
                    lookup_attribute=_cloudtrail_lookup_attribute(principal_variant=variant),
                    mode="principal_lookup",
                )
                if error:
                    return error
                if matched_events:
                    break

            if not matched_events:
                used_fallback_scan = True
                error = run_lookup(
                    lookup_attribute=_cloudtrail_lookup_attribute(
                        event_name_exact=event_name_exact,
                        event_source=event_source,
                        resource_name=resource_name,
                        resource_type=resource_type,
                    ),
                    mode="fallback_scan",
                    scan_cap=_cloudtrail_fallback_scan_cap(max_events),
                )
                if error:
                    return error
        else:
            primary_lookup = _cloudtrail_lookup_attribute(
                event_name_exact=event_name_exact,
                event_source=event_source,
                resource_name=resource_name,
                resource_type=resource_type,
            )
            error = run_lookup(
                lookup_attribute=primary_lookup,
                mode="primary_lookup" if primary_lookup else "broad_scan",
                scan_cap=None if primary_lookup else _cloudtrail_fallback_scan_cap(max_events),
            )
            if error:
                return error

        matched_events.sort(key=lambda event: event.get("_event_time_sort_key", float("-inf")), reverse=True)
        for event in matched_events:
            event.pop("_event_time_sort_key", None)
            event.pop("_search_blob", None)

        payload = {
            "region": target_region,
            "filters": {
                "principal": principal,
                "event_name_exact": event_name_exact,
                "event_name_prefix": event_name_prefix,
                "event_source": event_source,
                "resource_type": resource_type,
                "resource_name": resource_name,
                "contains_text": contains_text,
                "start_time": _format_timestamp(parsed_start_time or start_time),
                "end_time": _format_timestamp(parsed_end_time or end_time),
                "max_events": max_events,
            },
            "principal_variants_tried": principal_variants_tried,
            "used_fallback_scan": used_fallback_scan,
            "scan_limited": scan_limited,
            "matched_event_count": len(matched_events),
            "lookup_attempts": lookup_attempts,
            "events": matched_events,
        }
        return json.dumps(payload, indent=2, default=str)

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
