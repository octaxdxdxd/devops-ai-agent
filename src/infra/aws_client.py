"""AWS infrastructure client wrapping boto3."""

from __future__ import annotations

import json
import logging
import re
import statistics
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError, NoCredentialsError, BotoCoreError

from ..config import Config

log = logging.getLogger(__name__)

_EMAIL_LOCAL_PART_RE = re.compile(r"([._-])")
_RATE_EXPRESSION_RE = re.compile(r"^rate\((\d+)\s+(minute|minutes|hour|hours|day|days)\)$", re.IGNORECASE)
_VALID_EVENTBRIDGE_NAME_PREFIX_RE = re.compile(r"^[.\-_A-Za-z0-9]+$")
_DEFAULT_CLOUDTRAIL_MAX_EVENTS = 200
_MAX_CLOUDTRAIL_MAX_EVENTS = 500
_MAX_CLOUDTRAIL_FALLBACK_SCAN = 1000
_CLOUDTRAIL_CONNECT_TIMEOUT_SEC = 5
_CLOUDTRAIL_READ_TIMEOUT_SEC = 15
_CLOUDTRAIL_LOOKUP_TIME_BUDGET_SEC = 30
_CLOUDTRAIL_PROGRESS_PAGE_INTERVAL = 2
_DEFAULT_SCHEDULE_LOOKBACK_DAYS = 35
_MAX_SCHEDULE_LOOKBACK_DAYS = 90
_SCHEDULE_LOGS_QUERY_LIMIT = 5
_BLOCKED_SCHEDULE_HINTS = {"owner", "discipline", "purpose"}
_PRIMARY_SCHEDULE_KEYWORD_WEIGHTS = {
    "kill": 3,
    "delete": 3,
    "cleanup": 3,
    "unused": 2,
    "tagless": 1,
}
_SECONDARY_SCHEDULE_KEYWORD_WEIGHTS = {
    "notify": 3,
    "notification": 3,
    "notific": 3,
    "alert": 2,
    "report": 2,
}


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


def _cloudtrail_primary_lookup_attribute(
    *,
    event_name_exact: str = "",
    event_source: str = "",
    resource_name: str = "",
    resource_type: str = "",
) -> dict[str, str] | None:
    if resource_name:
        return {"AttributeKey": "ResourceName", "AttributeValue": resource_name}
    if event_name_exact:
        return {"AttributeKey": "EventName", "AttributeValue": event_name_exact}
    if event_source:
        return {"AttributeKey": "EventSource", "AttributeValue": event_source}
    if resource_type:
        return {"AttributeKey": "ResourceType", "AttributeValue": resource_type}
    return None


def _cloudtrail_fallback_scan_cap(max_events: int) -> int:
    return min(max(max_events * 10, 200), _MAX_CLOUDTRAIL_FALLBACK_SCAN)


def _cloudtrail_lookup_label(lookup_attribute: dict[str, str] | None, mode: str) -> str:
    if lookup_attribute:
        key = str(lookup_attribute.get("AttributeKey", "") or "").strip() or "attribute"
        value = str(lookup_attribute.get("AttributeValue", "") or "").strip() or "value"
        return f"{key}={value}"
    if mode == "fallback_scan":
        return "fallback scan"
    return "broad scan"


def _normalize_schedule_regions(regions: list[str] | None, default_region: str) -> list[str]:
    if not regions:
        return [default_region]
    return _unique_preserve_order([str(region or "").strip() for region in regions]) or [default_region]


def _normalize_schedule_hint(value: object) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    text = re.sub(r"[\s/]+", "-", text)
    text = re.sub(r"[^.\-_A-Za-z0-9]+", "", text)
    text = re.sub(r"-{2,}", "-", text).strip("-. _")
    if not text or text in _BLOCKED_SCHEDULE_HINTS:
        return ""
    if not _VALID_EVENTBRIDGE_NAME_PREFIX_RE.match(text):
        return ""
    return text


def _normalize_name_hints(name_hints: list[str] | None) -> list[str]:
    return _unique_preserve_order(
        [_normalize_schedule_hint(raw_hint) for raw_hint in (name_hints or [])]
    )


def _matches_name_hints(*values: str, hints: list[str]) -> bool:
    if not hints:
        return True
    lowered_hints = [hint.lower() for hint in hints if hint]
    haystacks = [str(value or "").lower() for value in values if value]
    return any(hint in haystack for hint in lowered_hints for haystack in haystacks)


def _lambda_function_name_from_arn(arn: str) -> str:
    text = str(arn or "").strip()
    if ":function:" not in text:
        return text
    suffix = text.split(":function:", 1)[1]
    return suffix.split(":", 1)[0].strip()


def _timestamp_from_millis(value: object) -> str:
    try:
        millis = int(value)
    except (TypeError, ValueError):
        return ""
    dt = datetime.fromtimestamp(millis / 1000, tz=timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def _schedule_history_lookback_days(schedule_expression: str, requested_lookback_days: int) -> int:
    base = max(1, min(_coerce_int(requested_lookback_days, _DEFAULT_SCHEDULE_LOOKBACK_DAYS), _MAX_SCHEDULE_LOOKBACK_DAYS))
    text = str(schedule_expression or "").strip()
    if not text:
        return _MAX_SCHEDULE_LOOKBACK_DAYS

    rate_match = _RATE_EXPRESSION_RE.match(text)
    if rate_match:
        amount = int(rate_match.group(1))
        unit = rate_match.group(2).lower()
        if unit.startswith("day"):
            derived = amount * 4
        elif unit.startswith("hour"):
            derived = 14
        else:
            derived = 7
        return min(max(base, derived), _MAX_SCHEDULE_LOOKBACK_DAYS)

    if not text.lower().startswith("cron(") or not text.endswith(")"):
        return _MAX_SCHEDULE_LOOKBACK_DAYS

    cron_body = text[5:-1].strip()
    fields = cron_body.split()
    if len(fields) != 6:
        return _MAX_SCHEDULE_LOOKBACK_DAYS

    day_of_month = "*" if fields[2] == "?" else fields[2]
    month = fields[3]
    day_of_week = "*" if fields[4] == "?" else fields[4]

    if month != "*" or (day_of_month != "*" and day_of_week == "*"):
        return _MAX_SCHEDULE_LOOKBACK_DAYS
    return min(max(base, _DEFAULT_SCHEDULE_LOOKBACK_DAYS), _MAX_SCHEDULE_LOOKBACK_DAYS)


def _metric_period_for_lookback_days(lookback_days: int) -> int:
    raw_period = max(300, min(86400, int((lookback_days * 86400) / 400)))
    return max(300, ((raw_period + 59) // 60) * 60)


def _logs_query_field(row: list[dict[str, Any]], field_name: str) -> str:
    for item in row or []:
        if str(item.get("field", "") or "") == field_name:
            return str(item.get("value", "") or "")
    return ""


def _compute_observed_schedule(recent_run_times: list[str]) -> dict[str, Any]:
    parsed_times = [
        parsed_time
        for parsed_time in (_parse_timestamp(timestamp) for timestamp in recent_run_times)
        if parsed_time is not None
    ]
    parsed_times.sort(reverse=True)
    normalized_times = [
        timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        for timestamp in parsed_times[:_SCHEDULE_LOGS_QUERY_LIMIT]
    ]
    last_run_time = normalized_times[0] if normalized_times else ""

    if len(parsed_times) < 3:
        return {
            "recent_run_times": normalized_times,
            "last_run_time": last_run_time,
            "observed_interval_seconds": None,
            "frequency_source": "observed_history_partial" if normalized_times else "unknown",
            "next_run_confidence": "unknown",
            "next_run_time": None,
        }

    intervals = [
        int((parsed_times[index] - parsed_times[index + 1]).total_seconds())
        for index in range(len(parsed_times) - 1)
    ]
    median_interval = int(statistics.median(intervals))
    tolerance_seconds = max(600, int(median_interval * 0.05))
    is_stable = all(abs(interval - median_interval) <= tolerance_seconds for interval in intervals)

    if not is_stable:
        return {
            "recent_run_times": normalized_times,
            "last_run_time": last_run_time,
            "observed_interval_seconds": None,
            "frequency_source": "observed_history_partial",
            "next_run_confidence": "unknown",
            "next_run_time": None,
        }

    next_run_time = (parsed_times[0] + timedelta(seconds=median_interval)).astimezone(timezone.utc)
    return {
        "recent_run_times": normalized_times,
        "last_run_time": last_run_time,
        "observed_interval_seconds": median_interval,
        "frequency_source": "observed_history",
        "next_run_confidence": "high",
        "next_run_time": next_run_time.isoformat().replace("+00:00", "Z"),
    }


def _schedule_match_kind(
    rule_name: str,
    function_name: str,
    description: str,
    hints: list[str],
) -> str:
    combined = " ".join([rule_name, function_name, description]).lower()
    score = 0
    for keyword, weight in _PRIMARY_SCHEDULE_KEYWORD_WEIGHTS.items():
        if keyword in combined:
            score += weight
    for keyword, weight in _SECONDARY_SCHEDULE_KEYWORD_WEIGHTS.items():
        if keyword in combined:
            score -= weight
    if _matches_name_hints(rule_name, function_name, hints=hints):
        score += 1
    return "primary" if score > 0 else "related"


class AWSClient:
    """Thin wrapper around boto3 for AWS operations."""

    def __init__(self) -> None:
        self.session = self._create_session()
        self.region = Config.AWS_CLI_DEFAULT_REGION or self.session.region_name or "us-east-1"
        self._status_callback: Callable[[str], None] | None = None

    def _create_session(self) -> boto3.Session:
        kwargs: dict[str, Any] = {}
        if Config.AWS_CLI_PROFILE:
            kwargs["profile_name"] = Config.AWS_CLI_PROFILE
        if Config.AWS_CLI_DEFAULT_REGION:
            kwargs["region_name"] = Config.AWS_CLI_DEFAULT_REGION
        return boto3.Session(**kwargs)

    def set_status_callback(self, callback: Callable[[str], None] | None) -> None:
        self._status_callback = callback

    def clear_status_callback(self) -> None:
        self._status_callback = None

    def _emit_status(self, message: str) -> None:
        if self._status_callback:
            self._status_callback(message)

    def _client(self, service: str, region: str | None = None):
        client_config = None
        if service == "cloudtrail":
            client_config = BotoConfig(
                connect_timeout=_CLOUDTRAIL_CONNECT_TIMEOUT_SEC,
                read_timeout=_CLOUDTRAIL_READ_TIMEOUT_SEC,
                retries={"max_attempts": 2, "mode": "standard"},
            )
        return self.session.client(service, region_name=region or self.region, config=client_config)

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

    def _lambda_recent_runs_from_logs(
        self,
        function_name: str,
        region: str,
        *,
        lookback_days: int,
        limit: int = _SCHEDULE_LOGS_QUERY_LIMIT,
    ) -> list[str]:
        logs = self._client("logs", region=region)
        now = datetime.now(timezone.utc)
        result = self._safe(
            logs.start_query,
            logGroupName=f"/aws/lambda/{function_name}",
            startTime=int((now - timedelta(days=lookback_days)).timestamp()),
            endTime=int(now.timestamp()),
            queryString=(
                "fields @timestamp, @message "
                "| filter @message like /REPORT RequestId:/ "
                "| sort @timestamp desc "
                f"| limit {int(limit)}"
            ),
        )
        if isinstance(result, str):
            return []

        query_id = str(result.get("queryId", "") or "")
        if not query_id:
            return []

        for _ in range(5):
            query_result = self._safe(logs.get_query_results, queryId=query_id)
            if isinstance(query_result, str):
                return []

            status = str(query_result.get("status", "") or "").lower()
            if status == "complete":
                timestamps = [
                    _format_timestamp(_logs_query_field(row, "@timestamp"))
                    for row in query_result.get("results", []) or []
                ]
                return _unique_preserve_order([timestamp for timestamp in timestamps if timestamp])[:limit]
            if status in {"failed", "cancelled", "timeout", "unknown"}:
                return []
            time.sleep(0.2)
        return []

    def _lambda_recent_runs_from_metrics(
        self,
        function_name: str,
        region: str,
        *,
        lookback_days: int,
        limit: int = _SCHEDULE_LOGS_QUERY_LIMIT,
    ) -> list[str]:
        cloudwatch = self._client("cloudwatch", region=region)
        now = datetime.now(timezone.utc)
        result = self._safe(
            cloudwatch.get_metric_statistics,
            Namespace="AWS/Lambda",
            MetricName="Invocations",
            Dimensions=[{"Name": "FunctionName", "Value": function_name}],
            StartTime=now - timedelta(days=lookback_days),
            EndTime=now,
            Period=_metric_period_for_lookback_days(lookback_days),
            Statistics=["Sum"],
        )
        if isinstance(result, str):
            return []

        datapoints = sorted(
            result.get("Datapoints", []) or [],
            key=lambda datapoint: datapoint.get("Timestamp", ""),
            reverse=True,
        )
        timestamps: list[str] = []
        for datapoint in datapoints:
            try:
                value = float(datapoint.get("Sum", 0) or 0)
            except (TypeError, ValueError):
                value = 0
            if value <= 0:
                continue
            timestamp = _format_timestamp(datapoint.get("Timestamp"))
            if timestamp:
                timestamps.append(timestamp)
        return _unique_preserve_order(timestamps)[:limit]

    def inspect_lambda_schedules(
        self,
        *,
        name_hints: list[str] | None = None,
        regions: list[str] | None = None,
        lookback_days: int = _DEFAULT_SCHEDULE_LOOKBACK_DAYS,
        include_disabled: bool = False,
    ) -> str:
        normalized_hints = _normalize_name_hints(name_hints)
        if not normalized_hints:
            return (
                "ERROR: aws_inspect_lambda_schedules requires at least one non-empty name hint. "
                "Provide function/rule fragments such as ['kill-tagless-resources']."
            )

        lookback_days = max(1, min(_coerce_int(lookback_days, _DEFAULT_SCHEDULE_LOOKBACK_DAYS), _MAX_SCHEDULE_LOOKBACK_DAYS))
        include_disabled = _coerce_bool(include_disabled)
        target_regions = _normalize_schedule_regions(regions, self.region)

        primary_records: list[dict[str, Any]] = []
        related_records: list[dict[str, Any]] = []
        rule_keys_seen: set[tuple[str, str, str]] = set()

        for region_name in target_regions:
            events = self._client("events", region=region_name)
            candidate_rules: dict[str, dict[str, Any]] = {}

            for hint in normalized_hints:
                next_token: str | None = None
                while True:
                    kwargs: dict[str, Any] = {"NamePrefix": hint}
                    if next_token:
                        kwargs["NextToken"] = next_token
                    result = self._safe(events.list_rules, **kwargs)
                    if isinstance(result, str):
                        return result

                    for rule in result.get("Rules", []) or []:
                        schedule_expression = str(rule.get("ScheduleExpression", "") or "").strip()
                        if not schedule_expression:
                            continue
                        if not include_disabled and str(rule.get("State", "") or "").upper() == "DISABLED":
                            continue
                        rule_arn = str(rule.get("Arn", "") or "")
                        if not rule_arn:
                            continue
                        candidate_rules[rule_arn] = rule

                    next_token = result.get("NextToken")
                    if not next_token:
                        break

            for rule in candidate_rules.values():
                rule_name = str(rule.get("Name", "") or "").strip()
                targets_result = self._safe(events.list_targets_by_rule, Rule=rule_name)
                if isinstance(targets_result, str):
                    return targets_result

                for target in targets_result.get("Targets", []) or []:
                    function_arn = str(target.get("Arn", "") or "").strip()
                    if ":lambda:" not in function_arn:
                        continue
                    function_name = _lambda_function_name_from_arn(function_arn)
                    if not _matches_name_hints(rule_name, function_name, hints=normalized_hints):
                        continue

                    record_key = (region_name, str(rule.get("Arn", "") or ""), function_arn)
                    if record_key in rule_keys_seen:
                        continue
                    rule_keys_seen.add(record_key)

                    schedule_expression = str(rule.get("ScheduleExpression", "") or "")
                    history_lookback_days = _schedule_history_lookback_days(schedule_expression, lookback_days)
                    recent_run_times = self._lambda_recent_runs_from_logs(
                        function_name,
                        region_name,
                        lookback_days=history_lookback_days,
                    )
                    last_run_source = "cloudwatch_logs_insights"
                    if not recent_run_times:
                        recent_run_times = self._lambda_recent_runs_from_metrics(
                            function_name,
                            region_name,
                            lookback_days=history_lookback_days,
                        )
                        last_run_source = "cloudwatch_metrics" if recent_run_times else ""

                    observed_schedule = _compute_observed_schedule(recent_run_times)
                    match_kind = _schedule_match_kind(
                        rule_name,
                        function_name,
                        str(rule.get("Description", "") or ""),
                        normalized_hints,
                    )

                    record = {
                        "region": region_name,
                        "rule_name": rule_name,
                        "rule_arn": str(rule.get("Arn", "") or ""),
                        "state": str(rule.get("State", "") or ""),
                        "schedule_expression": schedule_expression,
                        "description": str(rule.get("Description", "") or ""),
                        "function_name": function_name,
                        "function_arn": function_arn,
                        "last_run_time": observed_schedule["last_run_time"],
                        "last_run_source": last_run_source,
                        "recent_run_times": observed_schedule["recent_run_times"],
                        "observed_interval_seconds": observed_schedule["observed_interval_seconds"],
                        "frequency_source": observed_schedule["frequency_source"],
                        "next_run_confidence": observed_schedule["next_run_confidence"],
                        "next_run_time": observed_schedule["next_run_time"],
                        "match_kind": match_kind,
                    }
                    if match_kind == "primary":
                        primary_records.append(record)
                    else:
                        related_records.append(record)

        primary_records.sort(key=lambda item: (item.get("region", ""), item.get("rule_name", ""), item.get("function_name", "")))
        related_records.sort(key=lambda item: (item.get("region", ""), item.get("rule_name", ""), item.get("function_name", "")))
        payload = {
            "regions_checked": target_regions,
            "name_hints": normalized_hints,
            "lookback_days": lookback_days,
            "include_disabled": include_disabled,
            "matched_schedule_count": len(primary_records),
            "related_schedule_count": len(related_records),
            "schedules": primary_records,
            "related_schedules": related_records,
        }
        return json.dumps(payload, indent=2)

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
        selective_scan_cap = _cloudtrail_fallback_scan_cap(max_events)

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
            attempt_started = time.monotonic()
            lookup_label = _cloudtrail_lookup_label(lookup_attribute, mode)
            attempt = {
                "mode": mode,
                "lookup_attribute": lookup_attribute,
                "region": target_region,
                "pages": 0,
                "raw_events_scanned": 0,
                "matched_events_after_attempt": len(matched_events),
                "scan_cap": scan_cap,
                "time_budget_seconds": _CLOUDTRAIL_LOOKUP_TIME_BUDGET_SEC,
                "time_limited": False,
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
                    attempt["elapsed_ms"] = int((time.monotonic() - attempt_started) * 1000)
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
                        attempt["elapsed_ms"] = int((time.monotonic() - attempt_started) * 1000)
                        lookup_attempts.append(attempt)
                        return None

                next_token = result.get("NextToken")
                elapsed_seconds = time.monotonic() - attempt_started
                if next_token and pages % _CLOUDTRAIL_PROGRESS_PAGE_INTERVAL == 0:
                    self._emit_status(
                        f"CloudTrail scan in {target_region}: {pages} pages, "
                        f"{raw_events_scanned} events scanned for {lookup_label}..."
                    )
                if not next_token:
                    break
                if elapsed_seconds >= _CLOUDTRAIL_LOOKUP_TIME_BUDGET_SEC:
                    scan_limited = True
                    attempt["time_limited"] = True
                    break
                if scan_cap is not None and raw_events_scanned >= scan_cap:
                    scan_limited = True
                    break

            attempt["pages"] = pages
            attempt["raw_events_scanned"] = raw_events_scanned
            attempt["matched_events_after_attempt"] = len(matched_events)
            attempt["elapsed_ms"] = int((time.monotonic() - attempt_started) * 1000)
            attempt["scan_limited"] = bool(next_token) and scan_cap is not None and raw_events_scanned >= scan_cap
            if next_token and attempt["time_limited"]:
                self._emit_status(
                    f"CloudTrail scan in {target_region} stopped after "
                    f"{attempt['elapsed_ms'] // 1000}s for {lookup_label}; returning partial results."
                )
            elif attempt["scan_limited"]:
                self._emit_status(
                    f"CloudTrail scan in {target_region} capped at {raw_events_scanned} events "
                    f"for {lookup_label}; returning partial results."
                )
            lookup_attempts.append(attempt)
            return None

        primary_lookup = _cloudtrail_primary_lookup_attribute(
            event_name_exact=event_name_exact,
            event_source=event_source,
            resource_name=resource_name,
            resource_type=resource_type,
        )

        if principal:
            if primary_lookup is not None:
                error = run_lookup(
                    lookup_attribute=primary_lookup,
                    mode="primary_selective_lookup",
                    scan_cap=selective_scan_cap,
                )
                if error:
                    return error

            if not matched_events:
                for variant in _principal_variants(principal):
                    principal_variants_tried.append(variant)
                    error = run_lookup(
                        lookup_attribute=_cloudtrail_lookup_attribute(principal_variant=variant),
                        mode="principal_lookup",
                        scan_cap=selective_scan_cap,
                    )
                    if error:
                        return error
                    if matched_events:
                        break

            if not matched_events and primary_lookup is None:
                used_fallback_scan = True
                error = run_lookup(
                    lookup_attribute=None,
                    mode="fallback_scan",
                    scan_cap=selective_scan_cap,
                )
                if error:
                    return error
        else:
            error = run_lookup(
                lookup_attribute=primary_lookup,
                mode="primary_lookup" if primary_lookup else "broad_scan",
                scan_cap=selective_scan_cap,
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
