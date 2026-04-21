from __future__ import annotations

import json

from src.agents.action import _validate_proposed_steps
from src.config import Config
from src.tools.aws_write import create_aws_write_tools
from src.tools.command_preview import render_tool_call_preview
from src.tools.correlation_read import create_correlation_read_tools


class _FakeAWSWrite:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, object, object]] = []

    def run_command(self, service, operation, params=None, region=None) -> str:
        self.calls.append((service, operation, params, region))
        return "{}"

    def update_auto_scaling(self, *args, **kwargs) -> str:  # pragma: no cover - not used here
        return "ok"

    def resume_auto_scaling_processes(self, *args, **kwargs) -> str:  # pragma: no cover - not used here
        return "ok"


class _FakeK8sCorrelation:
    def get_resource_json(self, kind: str, name: str, namespace: str | None = None):
        data = {
            ("pod", "api-123", "prod"): {
                "metadata": {
                    "name": "api-123",
                    "namespace": "prod",
                    "uid": "pod-uid",
                    "ownerReferences": [{"kind": "ReplicaSet", "name": "api-rs"}],
                },
                "spec": {
                    "nodeName": "ip-10-0-0-234.ec2.internal",
                    "volumes": [{"persistentVolumeClaim": {"claimName": "data-api-0"}}],
                },
                "status": {
                    "phase": "Running",
                    "podIP": "10.0.0.10",
                    "containerStatuses": [{"restartCount": 2}],
                },
            },
            ("node", "ip-10-0-0-234.ec2.internal", None): {
                "metadata": {
                    "name": "ip-10-0-0-234.ec2.internal",
                    "labels": {
                        "topology.kubernetes.io/zone": "us-east-1a",
                        "node.kubernetes.io/instance-type": "t3.large",
                        "eks.amazonaws.com/capacityType": "ON_DEMAND",
                        "eks.amazonaws.com/nodegroup": "ng-a",
                    },
                },
                "spec": {"providerID": "aws:///us-east-1a/i-abc123"},
                "status": {"addresses": [{"type": "InternalIP", "address": "10.0.0.234"}]},
            },
            ("pvc", "data-api-0", "prod"): {
                "spec": {"volumeName": "pvc-123", "storageClassName": "gp3"},
                "status": {"accessModes": ["ReadWriteOnce"], "phase": "Bound"},
            },
            ("pv", "pvc-123", None): {
                "spec": {"csi": {"volumeHandle": "vol-123"}},
            },
        }
        key = (kind, name, namespace)
        return data.get(key, f"ERROR: missing {key}")

    def list_resources_json(self, kind: str, namespace: str | None = None, label_selector: str | None = None, all_namespaces: bool = False):
        return {"items": []}


class _FakeAWSCorrelation:
    def get_instance_details(self, *, instance_id: str = "", private_dns_name: str = "", region: str | None = None):
        assert instance_id == "i-abc123"
        return {
            "instance_id": "i-abc123",
            "state": "running",
            "instance_type": "t3.large",
            "private_dns_name": "ip-10-0-0-234.ec2.internal",
            "private_ip": "10.0.0.234",
            "availability_zone": "us-east-1a",
            "launch_time": "2026-04-20T10:00:00Z",
            "autoscaling_group_name": "asg-a",
            "tags": {"aws:autoscaling:groupName": "asg-a"},
        }

    def get_auto_scaling_group_details(self, asg_name: str):
        assert asg_name == "asg-a"
        return {
            "name": "asg-a",
            "desired_capacity": 2,
            "min_size": 1,
            "max_size": 3,
            "availability_zones": ["us-east-1a"],
            "suspended_processes": ["Launch"],
            "instance_ids": ["i-abc123"],
        }


def test_policy_blocks_raw_kubectl_write_commands_when_allow_all_write_disabled() -> None:
    previous_allow = Config.K8S_CLI_ALLOW_ALL_WRITE
    previous_posture = Config.COMMAND_SAFETY_POSTURE
    try:
        Config.K8S_CLI_ALLOW_ALL_WRITE = False
        Config.COMMAND_SAFETY_POSTURE = "approval_required"
        error = _validate_proposed_steps(
            [{"command": "kubectl delete pod api-123 -n prod", "display": "Delete pod"}],
            {"k8s_delete_resource"},
        )
    finally:
        Config.K8S_CLI_ALLOW_ALL_WRITE = previous_allow
        Config.COMMAND_SAFETY_POSTURE = previous_posture

    assert error == "step 1: Policy blocked raw kubectl write commands because K8S_CLI_ALLOW_ALL_WRITE is disabled."


def test_policy_blocks_generic_aws_write_tool_when_allow_all_write_disabled() -> None:
    previous_allow = Config.AWS_CLI_ALLOW_ALL_WRITE
    previous_posture = Config.COMMAND_SAFETY_POSTURE
    try:
        Config.AWS_CLI_ALLOW_ALL_WRITE = False
        Config.COMMAND_SAFETY_POSTURE = "approval_required"
        fake = _FakeAWSWrite()
        tools = {tool.name: tool for tool in create_aws_write_tools(fake)}
        result = tools["aws_run_api_command"].invoke(
            {
                "service": "autoscaling",
                "operation": "resume_processes",
                "params_json": '{"AutoScalingGroupName":"asg-a"}',
                "region": "us-east-1",
            }
        )
    finally:
        Config.AWS_CLI_ALLOW_ALL_WRITE = previous_allow
        Config.COMMAND_SAFETY_POSTURE = previous_posture

    assert result == "ERROR: Policy blocked generic AWS write API calls because AWS_CLI_ALLOW_ALL_WRITE is disabled."
    assert fake.calls == []


def test_correlation_tool_joins_k8s_and_aws_context_for_pod() -> None:
    tools = create_correlation_read_tools(_FakeK8sCorrelation(), _FakeAWSCorrelation())
    result = tools[0].invoke({"kind": "pod", "name": "api-123", "namespace": "prod"})
    data = json.loads(result)

    assert data["resource"]["name"] == "api-123"
    assert data["pod"]["node"] == "ip-10-0-0-234.ec2.internal"
    assert data["storage"][0]["volume_handle"] == "vol-123"
    assert data["node_correlation"]["ec2"]["instance_id"] == "i-abc123"
    assert data["node_correlation"]["autoscaling_group"]["name"] == "asg-a"


def test_command_preview_renders_typed_restart_workflow() -> None:
    _, preview, language = render_tool_call_preview(
        "k8s_restart_workload_safely",
        {"kind": "deployment", "name": "api", "namespace": "prod", "timeout_seconds": 180},
    )

    assert "kubectl rollout restart deployment/api -n prod" in preview
    assert "kubectl rollout status deployment/api -n prod --timeout=180s" in preview
    assert language == "bash"
