from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

from src.agents.lookup import select_lookup_tools
from src.infra.aws_client import AWSClient
from src.tools.aws_read import create_aws_read_tools
from src.tools.command_preview import render_tool_call_preview


class _FakeCloudTrailClient:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def lookup_events(self, **kwargs):
        self.calls.append(dict(kwargs))
        if self.responses:
            return self.responses.pop(0)
        return {"Events": []}


class _FakeClientFactory:
    def __init__(self, responses: list[dict], fallback_region: str) -> None:
        self.cloudtrail = _FakeCloudTrailClient(responses)
        self.calls: list[tuple[str, str | None]] = []
        self.fallback_region = fallback_region

    def __call__(self, service: str, region: str | None = None):
        self.calls.append((service, region))
        assert service == "cloudtrail"
        return self.cloudtrail


def _build_aws_client(
    responses: list[dict],
    *,
    region: str = "us-east-1",
) -> tuple[AWSClient, _FakeClientFactory]:
    client = object.__new__(AWSClient)
    factory = _FakeClientFactory(responses, fallback_region=region)
    client.session = None
    client.region = region
    client._status_callback = None
    client._client = factory
    client._safe = lambda fn, *args, **kwargs: fn(*args, **kwargs)
    return client, factory


def _cloudtrail_event(
    event_id: str,
    event_name: str,
    username: str,
    *,
    event_time: str,
    event_source: str = "ec2.amazonaws.com",
    aws_region: str = "us-east-1",
    resource_type: str = "AWS::EC2::VPC",
    resource_name: str = "vpc-02814c4c3b7012840",
    raw_identity: dict | None = None,
) -> dict:
    identity = raw_identity or {
        "type": "AssumedRole",
        "principalId": f"AROAUTAWYMQ46Z2KM6NHK:{username}",
        "arn": (
            "arn:aws:sts::315727832121:assumed-role/"
            f"AWSReservedSSO_315727832121-PowerUserAccess_9acad3d6e75fd059/{username}"
        ),
        "userName": username,
        "sessionContext": {
            "sessionIssuer": {
                "arn": (
                    "arn:aws:iam::315727832121:role/aws-reserved/sso.amazonaws.com/"
                    "eu-central-1/AWSReservedSSO_315727832121-PowerUserAccess_9acad3d6e75fd059"
                ),
                "userName": "AWSReservedSSO_315727832121-PowerUserAccess_9acad3d6e75fd059",
            }
        },
    }
    raw_event = {
        "eventVersion": "1.11",
        "userIdentity": identity,
        "eventTime": event_time,
        "eventSource": event_source,
        "eventName": event_name,
        "awsRegion": aws_region,
        "requestParameters": {"resourceId": resource_name},
        "readOnly": False,
    }
    return {
        "EventId": event_id,
        "EventName": event_name,
        "ReadOnly": "false",
        "EventTime": event_time,
        "EventSource": event_source,
        "Username": username,
        "Resources": [{"ResourceType": resource_type, "ResourceName": resource_name}],
        "CloudTrailEvent": json.dumps(raw_event),
    }


def test_audit_cloudtrail_matches_title_cased_username_variant() -> None:
    client, factory = _build_aws_client(
        [
            {"Events": []},
            {
                "Events": [
                    _cloudtrail_event(
                        "evt-1",
                        "DeleteVpc",
                        "Octavian.Popov@endava.com",
                        event_time="2026-04-07T15:22:12Z",
                    )
                ]
            },
        ]
    )

    payload = json.loads(
        client.audit_cloudtrail(
            principal="octavian.popov@endava.com",
            event_name_prefix="Delete",
            start_time="2026-04-04T00:00:00Z",
            end_time="2026-04-09T23:59:59Z",
            max_events=10,
            region="us-east-1",
        )
    )

    assert payload["matched_event_count"] == 1
    assert payload["events"][0]["event_name"] == "DeleteVpc"
    assert payload["events"][0]["resources"][0]["resource_name"] == "vpc-02814c4c3b7012840"
    assert payload["principal_variants_tried"] == [
        "octavian.popov@endava.com",
        "Octavian.Popov@endava.com",
    ]
    assert factory.cloudtrail.calls[0]["LookupAttributes"][0]["AttributeValue"] == "octavian.popov@endava.com"
    assert factory.cloudtrail.calls[1]["LookupAttributes"][0]["AttributeValue"] == "Octavian.Popov@endava.com"


def test_audit_cloudtrail_paginates_until_matching_delete_event_is_found() -> None:
    client, factory = _build_aws_client(
        [
            {
                "Events": [
                    _cloudtrail_event(
                        "evt-1",
                        "DescribeVpcs",
                        "Octavian.Popov@endava.com",
                        event_time="2026-04-09T15:17:49Z",
                    )
                ],
                "NextToken": "next-page",
            },
            {
                "Events": [
                    _cloudtrail_event(
                        "evt-2",
                        "DeleteVpc",
                        "Octavian.Popov@endava.com",
                        event_time="2026-04-07T15:22:12Z",
                    )
                ]
            },
        ]
    )

    payload = json.loads(
        client.audit_cloudtrail(
            principal="Octavian.Popov@endava.com",
            event_name_prefix="Delete",
            max_events=10,
            region="us-east-1",
        )
    )

    assert payload["matched_event_count"] == 1
    assert payload["events"][0]["event_id"] == "evt-2"
    assert len(factory.cloudtrail.calls) == 2
    assert factory.cloudtrail.calls[1]["NextToken"] == "next-page"


def test_audit_cloudtrail_filters_delete_prefix_client_side() -> None:
    client, _ = _build_aws_client(
        [
            {
                "Events": [
                    _cloudtrail_event(
                        "evt-1",
                        "DeleteSubnet",
                        "Octavian.Popov@endava.com",
                        event_time="2026-04-07T15:22:13Z",
                        resource_type="AWS::EC2::Subnet",
                        resource_name="subnet-04f925bcc7a8c52ee",
                    ),
                    _cloudtrail_event(
                        "evt-2",
                        "DeleteVpc",
                        "Octavian.Popov@endava.com",
                        event_time="2026-04-07T15:22:12Z",
                    ),
                    _cloudtrail_event(
                        "evt-3",
                        "CreateVpc",
                        "Octavian.Popov@endava.com",
                        event_time="2026-04-07T15:21:59Z",
                    ),
                ]
            }
        ]
    )

    payload = json.loads(
        client.audit_cloudtrail(
            principal="Octavian.Popov@endava.com",
            event_name_prefix="Delete",
            max_events=10,
            region="us-east-1",
        )
    )

    assert [event["event_name"] for event in payload["events"]] == ["DeleteSubnet", "DeleteVpc"]


def test_audit_cloudtrail_filters_exact_event_name() -> None:
    client, _ = _build_aws_client(
        [
            {
                "Events": [
                    _cloudtrail_event(
                        "evt-1",
                        "DeleteSubnet",
                        "Octavian.Popov@endava.com",
                        event_time="2026-04-07T15:22:13Z",
                        resource_type="AWS::EC2::Subnet",
                        resource_name="subnet-04f925bcc7a8c52ee",
                    ),
                    _cloudtrail_event(
                        "evt-2",
                        "DeleteVpc",
                        "Octavian.Popov@endava.com",
                        event_time="2026-04-07T15:22:12Z",
                    ),
                ]
            }
        ]
    )

    payload = json.loads(
        client.audit_cloudtrail(
            principal="Octavian.Popov@endava.com",
            event_name_exact="DeleteVpc",
            max_events=10,
            region="us-east-1",
        )
    )

    assert payload["matched_event_count"] == 1
    assert payload["events"][0]["event_name"] == "DeleteVpc"


def test_audit_cloudtrail_prefers_selective_lookup_before_principal_scan() -> None:
    client, factory = _build_aws_client(
        [
            {"Events": []},
            {
                "Events": [
                    _cloudtrail_event(
                        "evt-1",
                        "PutRule",
                        "devops.interns",
                        event_time="2026-04-10T06:00:00Z",
                        event_source="events.amazonaws.com",
                        resource_type="AWS::Events::Rule",
                        resource_name="kill-tagless-resources-rule",
                    )
                ]
            },
        ]
    )

    payload = json.loads(
        client.audit_cloudtrail(
            principal="devops.interns",
            event_name_exact="PutRule",
            event_source="events.amazonaws.com",
            resource_name="kill-tagless-resources-rule",
            max_events=10,
            region="us-east-1",
        )
    )

    assert payload["matched_event_count"] == 1
    assert payload["lookup_attempts"][0]["mode"] == "primary_selective_lookup"
    assert factory.cloudtrail.calls[0]["LookupAttributes"][0] == {
        "AttributeKey": "ResourceName",
        "AttributeValue": "kill-tagless-resources-rule",
    }


def test_audit_cloudtrail_caps_principal_lookup_pagination() -> None:
    responses = [{"Events": []}]
    for page in range(4):
        page_events = [
            _cloudtrail_event(
                f"evt-{page}-{index}",
                "DescribeRules",
                "someone-else",
                event_time="2026-04-10T06:00:00Z",
                event_source="events.amazonaws.com",
                resource_type="AWS::Events::Rule",
                resource_name="kill-tagless-resources-rule",
            )
            for index in range(50)
        ]
        response = {"Events": page_events}
        if page < 3:
            response["NextToken"] = f"token-{page + 1}"
        else:
            response["NextToken"] = "token-stop"
        responses.append(response)

    client, factory = _build_aws_client(responses)

    payload = json.loads(
        client.audit_cloudtrail(
            principal="devops.interns",
            resource_name="kill-tagless-resources-rule",
            max_events=20,
            region="us-east-1",
        )
    )

    principal_attempt = next(
        attempt for attempt in payload["lookup_attempts"] if attempt["mode"] == "principal_lookup"
    )

    assert factory.cloudtrail.calls[0]["LookupAttributes"][0] == {
        "AttributeKey": "ResourceName",
        "AttributeValue": "kill-tagless-resources-rule",
    }
    assert principal_attempt["raw_events_scanned"] == 200
    assert principal_attempt["scan_limited"] is True
    assert len(factory.cloudtrail.calls) == 5


def test_audit_cloudtrail_caps_broad_event_source_scans_with_client_side_filters() -> None:
    responses: list[dict] = []
    for page in range(5):
        page_events = [
            _cloudtrail_event(
                f"evt-{page}-{index}",
                "DeleteBucket",
                "someone-else",
                event_time="2026-04-10T06:00:00Z",
                event_source="s3.amazonaws.com",
                resource_type="AWS::S3::Bucket",
                resource_name=f"bucket-{page}-{index}",
            )
            for index in range(50)
        ]
        response = {"Events": page_events}
        if page < 4:
            response["NextToken"] = f"token-{page + 1}"
        responses.append(response)

    client, factory = _build_aws_client(responses)
    updates: list[str] = []
    client._status_callback = updates.append

    payload = json.loads(
        client.audit_cloudtrail(
            event_source="s3.amazonaws.com",
            event_name_prefix="Delete",
            contains_text="aiops",
            max_events=20,
            region="us-east-1",
        )
    )

    attempt = payload["lookup_attempts"][0]
    assert attempt["mode"] == "primary_lookup"
    assert attempt["scan_limited"] is True
    assert attempt["raw_events_scanned"] == 200
    assert len(factory.cloudtrail.calls) == 4
    assert any("CloudTrail scan in us-east-1" in update for update in updates)


def test_audit_cloudtrail_time_limits_slow_scans() -> None:
    client, factory = _build_aws_client(
        [
            {
                "Events": [
                    _cloudtrail_event(
                        "evt-1",
                        "DeleteBucket",
                        "someone-else",
                        event_time="2026-04-10T06:00:00Z",
                        event_source="s3.amazonaws.com",
                        resource_type="AWS::S3::Bucket",
                        resource_name="bucket-1",
                    )
                ],
                "NextToken": "token-2",
            }
        ]
    )

    with patch("src.infra.aws_client.time.monotonic", side_effect=[0.0, 31.0, 31.0]):
        payload = json.loads(
            client.audit_cloudtrail(
                event_source="s3.amazonaws.com",
                max_events=20,
                region="us-east-1",
            )
        )

    attempt = payload["lookup_attempts"][0]
    assert attempt["time_limited"] is True
    assert attempt["pages"] == 1
    assert len(factory.cloudtrail.calls) == 1


def test_audit_cloudtrail_falls_back_to_unfiltered_scan_for_raw_identity_match() -> None:
    client, factory = _build_aws_client(
        [
            {"Events": []},
            {"Events": []},
            {
                "Events": [
                    _cloudtrail_event(
                        "evt-raw",
                        "DeleteVpc",
                        "",
                        event_time="2026-04-07T15:22:12Z",
                        raw_identity={
                            "type": "AssumedRole",
                            "principalId": "AROAUTAWYMQ46Z2KM6NHK:Octavian.Popov@endava.com",
                            "arn": (
                                "arn:aws:sts::315727832121:assumed-role/"
                                "AWSReservedSSO_315727832121-PowerUserAccess_9acad3d6e75fd059/"
                                "Octavian.Popov@endava.com"
                            ),
                            "userName": "",
                            "sessionContext": {
                                "sessionIssuer": {
                                    "arn": "arn:aws:iam::315727832121:role/PowerUser",
                                    "userName": "PowerUser",
                                }
                            },
                        },
                    )
                ]
            },
        ]
    )

    payload = json.loads(
        client.audit_cloudtrail(
            principal="octavian.popov@endava.com",
            event_name_prefix="Delete",
            max_events=10,
            region="us-east-1",
        )
    )

    assert payload["used_fallback_scan"] is True
    assert payload["matched_event_count"] == 1
    assert payload["events"][0]["event_id"] == "evt-raw"
    assert "LookupAttributes" not in factory.cloudtrail.calls[2]


def test_audit_cloudtrail_uses_default_region_when_none_is_provided() -> None:
    client, factory = _build_aws_client([{"Events": []}], region="eu-central-1")

    json.loads(client.audit_cloudtrail(principal="Octavian.Popov@endava.com"))

    assert factory.calls == [("cloudtrail", "eu-central-1")]


def test_aws_audit_cloudtrail_preview_renders_cli_equivalent_command() -> None:
    label, preview, language = render_tool_call_preview(
        "aws_audit_cloudtrail",
        {
            "principal": "octavian.popov@endava.com",
            "event_name_prefix": "Delete",
            "start_time": "2026-04-04T00:00:00Z",
            "end_time": "2026-04-09T23:59:59Z",
            "max_events": 200,
            "region": "us-east-1",
        },
    )

    assert label == ""
    assert preview.startswith("aws cloudtrail lookup-events")
    assert "--region us-east-1" in preview
    assert "AttributeKey=Username,AttributeValue=octavian.popov@endava.com" in preview
    assert "client-side filters: event_name_prefix=Delete" in preview
    assert language == "bash"


def test_cloudtrail_lookup_queries_use_the_dedicated_audit_toolset() -> None:
    tools = [
        SimpleNamespace(name="aws_audit_cloudtrail"),
        SimpleNamespace(name="aws_get_caller_identity"),
        SimpleNamespace(name="aws_describe_service"),
        SimpleNamespace(name="aws_get_cost"),
    ]

    selected = select_lookup_tools(
        "show me all aws resources deleted by octavian.popov@endava.com in the last 5 days",
        [],
        tools,
    )

    assert [tool.name for tool in selected] == ["aws_audit_cloudtrail", "aws_get_caller_identity"]


def test_trace_regression_april_9_delete_query_returns_deleted_vpc() -> None:
    client, _ = _build_aws_client(
        [
            {"Events": []},
            {
                "Events": [
                    _cloudtrail_event(
                        "evt-delete-vpc",
                        "DeleteVpc",
                        "Octavian.Popov@endava.com",
                        event_time="2026-04-07T15:22:12Z",
                    )
                ]
            },
        ]
    )
    tools = {tool.name: tool for tool in create_aws_read_tools(client)}

    result = json.loads(
        tools["aws_audit_cloudtrail"].invoke(
            {
                "principal": "octavian.popov@endava.com",
                "event_name_prefix": "Delete",
                "start_time": "2026-04-04T00:00:00Z",
                "end_time": "2026-04-09T23:59:59Z",
                "max_events": 50,
                "include_raw_event": False,
                "region": "us-east-1",
            }
        )
    )

    assert result["matched_event_count"] == 1
    assert result["events"][0]["event_name"] == "DeleteVpc"
    assert result["events"][0]["resources"][0]["resource_name"] == "vpc-02814c4c3b7012840"
