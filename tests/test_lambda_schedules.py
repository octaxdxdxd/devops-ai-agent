from __future__ import annotations

import json
from datetime import datetime, timezone

from src.agents.lookup import select_lookup_tools
from src.infra.aws_client import AWSClient
from src.tools.aws_read import create_aws_read_tools
from src.tools.command_preview import render_tool_call_preview


def _schedule_rule(
    *,
    region: str,
    name: str,
    schedule_expression: str,
    description: str,
    state: str = "ENABLED",
) -> dict:
    return {
        "Name": name,
        "Arn": f"arn:aws:events:{region}:315727832121:rule/{name}",
        "State": state,
        "ScheduleExpression": schedule_expression,
        "Description": description,
    }


def _lambda_target(region: str, function_name: str) -> dict:
    return {
        "Id": function_name,
        "Arn": f"arn:aws:lambda:{region}:315727832121:function:{function_name}",
    }


def _metric_point(timestamp: str, value: float = 1.0) -> dict:
    return {
        "Timestamp": datetime.fromisoformat(timestamp.replace("Z", "+00:00")).astimezone(timezone.utc),
        "Sum": value,
    }


class _FakeEventsClient:
    def __init__(self, rules_by_prefix: dict[str, list[dict]], targets_by_rule: dict[str, list[dict]]) -> None:
        self.rules_by_prefix = rules_by_prefix
        self.targets_by_rule = targets_by_rule
        self.list_rules_calls: list[dict] = []
        self.list_targets_calls: list[dict] = []

    def list_rules(self, **kwargs):
        self.list_rules_calls.append(dict(kwargs))
        return {"Rules": list(self.rules_by_prefix.get(kwargs.get("NamePrefix", ""), []))}

    def list_targets_by_rule(self, **kwargs):
        self.list_targets_calls.append(dict(kwargs))
        return {"Targets": list(self.targets_by_rule.get(kwargs["Rule"], []))}


class _FakeLogsClient:
    def __init__(self, timestamps_by_group: dict[str, list[str]]) -> None:
        self.timestamps_by_group = timestamps_by_group
        self.start_query_calls: list[dict] = []
        self.get_query_results_calls: list[dict] = []
        self._query_results: dict[str, list[str]] = {}

    def start_query(self, **kwargs):
        self.start_query_calls.append(dict(kwargs))
        query_id = f"query-{len(self.start_query_calls)}"
        self._query_results[query_id] = list(self.timestamps_by_group.get(kwargs["logGroupName"], []))
        return {"queryId": query_id}

    def get_query_results(self, **kwargs):
        self.get_query_results_calls.append(dict(kwargs))
        timestamps = self._query_results.get(kwargs["queryId"], [])
        return {
            "status": "Complete",
            "results": [
                [
                    {"field": "@timestamp", "value": timestamp},
                    {"field": "@message", "value": "REPORT RequestId: abc Duration: 10 ms"},
                ]
                for timestamp in timestamps
            ],
        }


class _FakeCloudWatchClient:
    def __init__(self, datapoints_by_function: dict[str, list[dict]]) -> None:
        self.datapoints_by_function = datapoints_by_function
        self.calls: list[dict] = []

    def get_metric_statistics(self, **kwargs):
        self.calls.append(dict(kwargs))
        function_name = next(
            (
                dimension["Value"]
                for dimension in kwargs.get("Dimensions", [])
                if dimension.get("Name") == "FunctionName"
            ),
            "",
        )
        return {"Datapoints": list(self.datapoints_by_function.get(function_name, []))}


class _ScheduleClientFactory:
    def __init__(
        self,
        *,
        rules_by_region: dict[str, dict[str, list[dict]]],
        targets_by_region: dict[str, dict[str, list[dict]]],
        logs_by_region: dict[str, dict[str, list[str]]],
        metrics_by_region: dict[str, dict[str, list[dict]]],
    ) -> None:
        self.calls: list[tuple[str, str | None]] = []
        self.events = {
            region: _FakeEventsClient(rules_by_prefix, targets_by_region.get(region, {}))
            for region, rules_by_prefix in rules_by_region.items()
        }
        self.logs = {
            region: _FakeLogsClient(logs_by_region.get(region, {}))
            for region in set(rules_by_region) | set(logs_by_region) | set(metrics_by_region)
        }
        self.cloudwatch = {
            region: _FakeCloudWatchClient(metrics_by_region.get(region, {}))
            for region in set(rules_by_region) | set(logs_by_region) | set(metrics_by_region)
        }

    def __call__(self, service: str, region: str | None = None):
        self.calls.append((service, region))
        if service == "events":
            return self.events[region or ""]
        if service == "logs":
            return self.logs[region or ""]
        if service == "cloudwatch":
            return self.cloudwatch[region or ""]
        raise AssertionError(f"Unexpected service {service!r}")


def _build_schedule_client(
    *,
    rules_by_region: dict[str, dict[str, list[dict]]],
    targets_by_region: dict[str, dict[str, list[dict]]],
    logs_by_region: dict[str, dict[str, list[str]]],
    metrics_by_region: dict[str, dict[str, list[dict]]],
) -> tuple[AWSClient, _ScheduleClientFactory]:
    client = object.__new__(AWSClient)
    factory = _ScheduleClientFactory(
        rules_by_region=rules_by_region,
        targets_by_region=targets_by_region,
        logs_by_region=logs_by_region,
        metrics_by_region=metrics_by_region,
    )
    client.session = None
    client.region = "ap-south-1"
    client._client = factory
    client._safe = lambda fn, *args, **kwargs: fn(*args, **kwargs)
    return client, factory


class _ToolAWS:
    def describe_instances(self, **kwargs) -> str:
        return "{}"

    def describe_service(self, *args, **kwargs) -> str:
        return "{}"

    def inspect_lambda_schedules(self, *args, **kwargs) -> str:
        return "{}"

    def audit_cloudtrail(self, *args, **kwargs) -> str:
        return "{}"

    def get_cost(self, *args, **kwargs) -> str:
        return "{}"

    def get_cloudwatch_metrics(self, *args, **kwargs) -> str:
        return "{}"

    def get_alarms(self, *args, **kwargs) -> str:
        return "{}"

    def describe_security_groups(self, *args, **kwargs) -> str:
        return "{}"

    def get_iam_summary(self) -> str:
        return "{}"

    def list_resources(self, *args, **kwargs) -> str:
        return "{}"

    def get_caller_identity(self) -> str:
        return "{}"


def test_inspect_lambda_schedules_sanitizes_hints_and_uses_explicit_regions() -> None:
    client, factory = _build_schedule_client(
        rules_by_region={
            "us-east-1": {
                "kill-tagless-resources": [
                    _schedule_rule(
                        region="us-east-1",
                        name="kill-tagless-resources-rule",
                        schedule_expression="rate(14 days)",
                        description="Delete tagless resources every 14 days",
                    )
                ]
            },
            "eu-central-1": {
                "kill-tagless-resources": [
                    _schedule_rule(
                        region="eu-central-1",
                        name="kill-tagless-resources-rule",
                        schedule_expression="rate(14 days)",
                        description="Delete tagless resources every 14 days",
                    )
                ]
            },
        },
        targets_by_region={
            "us-east-1": {
                "kill-tagless-resources-rule": [
                    _lambda_target("us-east-1", "delete-resources-by-tag-DeleteResByTagFunc-GDE8H3BRZ3DT")
                ]
            },
            "eu-central-1": {
                "kill-tagless-resources-rule": [
                    _lambda_target("eu-central-1", "kill-tagless-resources-KillTaglessResourcesFunc-1S6QMJ0UUOR10")
                ]
            },
        },
        logs_by_region={
            "us-east-1": {
                "/aws/lambda/delete-resources-by-tag-DeleteResByTagFunc-GDE8H3BRZ3DT": [
                    "2026-04-17T12:50:00Z",
                    "2026-04-03T12:50:00Z",
                    "2026-03-20T12:50:00Z",
                ]
            },
            "eu-central-1": {
                "/aws/lambda/kill-tagless-resources-KillTaglessResourcesFunc-1S6QMJ0UUOR10": [
                    "2026-04-17T12:50:00Z",
                    "2026-04-03T12:50:00Z",
                    "2026-03-20T12:50:00Z",
                ]
            },
        },
        metrics_by_region={},
    )

    payload = json.loads(
        client.inspect_lambda_schedules(
            name_hints=["kill tagless resources", "Owner", "Discipline", "Purpose"],
            regions=["us-east-1", "eu-central-1"],
            lookback_days=35,
        )
    )

    assert payload["regions_checked"] == ["us-east-1", "eu-central-1"]
    assert payload["name_hints"] == ["kill-tagless-resources"]
    assert ("events", "ap-south-1") not in factory.calls
    for region in ("us-east-1", "eu-central-1"):
        assert [call["NamePrefix"] for call in factory.events[region].list_rules_calls] == [
            "kill-tagless-resources"
        ]


def test_inspect_lambda_schedules_uses_observed_history_for_stable_next_run() -> None:
    client, _ = _build_schedule_client(
        rules_by_region={
            "eu-central-1": {
                "kill-tagless-resources": [
                    _schedule_rule(
                        region="eu-central-1",
                        name="kill-tagless-resources-rule",
                        schedule_expression="rate(14 days)",
                        description="Delete untagged resources",
                    )
                ]
            }
        },
        targets_by_region={
            "eu-central-1": {
                "kill-tagless-resources-rule": [
                    _lambda_target("eu-central-1", "kill-tagless-resources-KillTaglessResourcesFunc-1S6QMJ0UUOR10")
                ]
            }
        },
        logs_by_region={
            "eu-central-1": {
                "/aws/lambda/kill-tagless-resources-KillTaglessResourcesFunc-1S6QMJ0UUOR10": [
                    "2026-04-17T12:50:00Z",
                    "2026-04-03T12:50:00Z",
                    "2026-03-20T12:50:00Z",
                    "2026-03-06T12:50:00Z",
                ]
            }
        },
        metrics_by_region={},
    )

    payload = json.loads(
        client.inspect_lambda_schedules(
            name_hints=["kill-tagless-resources"],
            regions=["eu-central-1"],
            lookback_days=35,
        )
    )

    schedule = payload["schedules"][0]
    assert schedule["last_run_source"] == "cloudwatch_logs_insights"
    assert schedule["recent_run_times"] == [
        "2026-04-17T12:50:00Z",
        "2026-04-03T12:50:00Z",
        "2026-03-20T12:50:00Z",
        "2026-03-06T12:50:00Z",
    ]
    assert schedule["observed_interval_seconds"] == 1209600
    assert schedule["frequency_source"] == "observed_history"
    assert schedule["next_run_confidence"] == "high"
    assert schedule["next_run_time"] == "2026-05-01T12:50:00Z"


def test_inspect_lambda_schedules_returns_unknown_next_run_for_inconsistent_history() -> None:
    client, _ = _build_schedule_client(
        rules_by_region={
            "eu-central-1": {
                "kill-tagless-resources": [
                    _schedule_rule(
                        region="eu-central-1",
                        name="kill-tagless-resources-rule",
                        schedule_expression="rate(14 days)",
                        description="Delete untagged resources",
                    )
                ]
            }
        },
        targets_by_region={
            "eu-central-1": {
                "kill-tagless-resources-rule": [
                    _lambda_target("eu-central-1", "kill-tagless-resources-KillTaglessResourcesFunc-1S6QMJ0UUOR10")
                ]
            }
        },
        logs_by_region={
            "eu-central-1": {
                "/aws/lambda/kill-tagless-resources-KillTaglessResourcesFunc-1S6QMJ0UUOR10": [
                    "2026-04-17T12:50:00Z",
                    "2026-04-10T12:50:00Z",
                    "2026-03-27T12:50:00Z",
                ]
            }
        },
        metrics_by_region={},
    )

    payload = json.loads(
        client.inspect_lambda_schedules(
            name_hints=["kill-tagless-resources"],
            regions=["eu-central-1"],
            lookback_days=35,
        )
    )

    schedule = payload["schedules"][0]
    assert schedule["frequency_source"] == "observed_history_partial"
    assert schedule["next_run_confidence"] == "unknown"
    assert schedule["next_run_time"] is None


def test_inspect_lambda_schedules_falls_back_to_metrics_and_requires_three_runs() -> None:
    client, _ = _build_schedule_client(
        rules_by_region={
            "us-east-1": {
                "kill-tagless-resources": [
                    _schedule_rule(
                        region="us-east-1",
                        name="kill-tagless-resources-rule",
                        schedule_expression="rate(14 days)",
                        description="Delete untagged resources",
                    )
                ]
            }
        },
        targets_by_region={
            "us-east-1": {
                "kill-tagless-resources-rule": [
                    _lambda_target("us-east-1", "delete-resources-by-tag-DeleteResByTagFunc-GDE8H3BRZ3DT")
                ]
            }
        },
        logs_by_region={"us-east-1": {}},
        metrics_by_region={
            "us-east-1": {
                "delete-resources-by-tag-DeleteResByTagFunc-GDE8H3BRZ3DT": [
                    _metric_point("2026-04-17T12:00:00Z"),
                    _metric_point("2026-04-03T12:00:00Z"),
                ]
            }
        },
    )

    payload = json.loads(
        client.inspect_lambda_schedules(
            name_hints=["kill-tagless-resources"],
            regions=["us-east-1"],
            lookback_days=35,
        )
    )

    schedule = payload["schedules"][0]
    assert schedule["last_run_source"] == "cloudwatch_metrics"
    assert schedule["recent_run_times"] == [
        "2026-04-17T12:00:00Z",
        "2026-04-03T12:00:00Z",
    ]
    assert schedule["frequency_source"] == "observed_history_partial"
    assert schedule["next_run_confidence"] == "unknown"
    assert schedule["next_run_time"] is None


def test_inspect_lambda_schedules_splits_primary_and_related_matches() -> None:
    client, _ = _build_schedule_client(
        rules_by_region={
            "eu-central-1": {
                "kill-tagless-resources": [
                    _schedule_rule(
                        region="eu-central-1",
                        name="kill-tagless-resources-rule",
                        schedule_expression="rate(14 days)",
                        description="Delete untagged resources",
                    )
                ],
                "tagless-resources": [
                    _schedule_rule(
                        region="eu-central-1",
                        name="tagless-resources-notific-rule",
                        schedule_expression="cron(0 13 ? * WED *)",
                        description="Weekly notification for tagless resources",
                    )
                ],
            }
        },
        targets_by_region={
            "eu-central-1": {
                "kill-tagless-resources-rule": [
                    _lambda_target("eu-central-1", "delete-resources-by-tag-DeleteResByTagFunc-1OG0XX92253EM")
                ],
                "tagless-resources-notific-rule": [
                    _lambda_target("eu-central-1", "tagless-resources-notific-TaglessResourceNotificat-WTA1MDLM19FV")
                ],
            }
        },
        logs_by_region={
            "eu-central-1": {
                "/aws/lambda/delete-resources-by-tag-DeleteResByTagFunc-1OG0XX92253EM": [
                    "2026-04-17T12:50:00Z",
                    "2026-04-03T12:50:00Z",
                    "2026-03-20T12:50:00Z",
                ],
                "/aws/lambda/tagless-resources-notific-TaglessResourceNotificat-WTA1MDLM19FV": [
                    "2026-04-16T13:00:00Z",
                    "2026-04-09T13:00:00Z",
                    "2026-04-02T13:00:00Z",
                ],
            }
        },
        metrics_by_region={},
    )

    payload = json.loads(
        client.inspect_lambda_schedules(
            name_hints=["kill-tagless-resources", "tagless resources"],
            regions=["eu-central-1"],
            lookback_days=35,
        )
    )

    assert payload["matched_schedule_count"] == 1
    assert payload["related_schedule_count"] == 1
    assert payload["schedules"][0]["match_kind"] == "primary"
    assert payload["related_schedules"][0]["match_kind"] == "related"
    assert payload["related_schedules"][0]["function_name"].startswith("tagless-resources-notific")


def test_schedule_lookup_tool_selection_excludes_cloudtrail() -> None:
    tools = create_aws_read_tools(_ToolAWS())
    selected = select_lookup_tools(
        (
            "i have a lambda kill tagless resources in my aws account "
            "and I want to know how frequent it runs, when did it do its last run "
            "and when will it run next"
        ),
        [],
        tools,
    )
    names = {tool.name for tool in selected}

    assert "aws_inspect_lambda_schedules" in names
    assert "aws_audit_cloudtrail" not in names


def test_schedule_tool_preview_renders_direct_schedule_inspection() -> None:
    _, preview, language = render_tool_call_preview(
        "aws_inspect_lambda_schedules",
        {
            "name_hints_json": '["kill-tagless-resources","tagless"]',
            "regions_json": '["us-east-1","eu-central-1"]',
            "lookback_days": 35,
            "include_disabled": False,
        },
    )

    assert language == "bash"
    assert preview.startswith("aws events list-rules --region us-east-1 --name-prefix kill-tagless-resources")
    assert "logs start-query/get-query-results" in preview
    assert "cloudwatch get-metric-statistics" in preview
