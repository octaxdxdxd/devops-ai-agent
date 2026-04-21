from __future__ import annotations

import json

from src.agents.action import _validate_proposed_steps
from src.config import Config
from src.infra.k8s_client import K8sClient
from src.tools.aws_write import create_aws_write_tools
from src.tools.command_preview import render_tool_call_preview
from src.tools.correlation_read import create_correlation_read_tools
from src.tools.k8s_read import create_k8s_read_tools


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


class _FakeK8sReadOnly:
    def run(self, args):  # pragma: no cover - should not be called for blocked writes
        raise AssertionError("read-only kubectl wrapper should not execute blocked write commands")


class _FakeK8sExecClient(K8sClient):
    def __init__(self) -> None:
        pass

    def get_resource_json(self, kind: str, name: str, namespace: str | None = None):
        assert kind == "pod"
        return {
            "spec": {
                "containers": [
                    {"name": "ingress-controller"},
                    {"name": "proxy"},
                ]
            }
        }

    def run(self, args, namespace: str | None = None, timeout: int = 30) -> str:  # pragma: no cover - should not be called on invalid container
        raise AssertionError("kubectl exec should not run when the container name is invalid")


class _FakeK8sExecStdErrClient(K8sClient):
    def __init__(self) -> None:
        pass

    def get_resource_json(self, kind: str, name: str, namespace: str | None = None):
        assert kind == "pod"
        return {"spec": {"containers": [{"name": "nexus-repository-manager"}]}}

    def run(
        self,
        args,
        namespace: str | None = None,
        timeout: int = 30,
        include_stderr_on_success: bool = False,
    ) -> str:
        assert include_stderr_on_success is True
        return "sh: line 1: wget: command not found"


class _FakeK8sUsageClient(K8sClient):
    def __init__(self) -> None:
        pass

    def get_resource_json(self, kind: str, name: str, namespace: str | None = None):
        if kind == "deployment":
            return {
                "kind": "Deployment",
                "metadata": {"name": "gitlab-webservice-default", "namespace": "gitlab"},
                "spec": {
                    "selector": {
                        "matchLabels": {
                            "app": "webservice",
                            "component": "webservice",
                        }
                    }
                },
            }
        return f"ERROR: unexpected get_resource_json call: {kind}/{name}"

    def list_resources_json(
        self,
        kind: str,
        namespace: str | None = None,
        label_selector: str | None = None,
        all_namespaces: bool = False,
    ):
        if kind == "pod":
            assert namespace == "gitlab"
            assert label_selector == "app=webservice,component=webservice"
            return {
                "items": [
                    {
                        "metadata": {
                            "name": "gitlab-webservice-default-abc",
                            "namespace": "gitlab",
                            "ownerReferences": [{"kind": "ReplicaSet", "name": "gitlab-webservice-default-rs"}],
                        },
                        "spec": {
                            "nodeName": "node-a",
                            "containers": [
                                {
                                    "name": "webservice",
                                    "image": "gitlab/webservice:latest",
                                    "resources": {
                                        "requests": {"cpu": "500m", "memory": "2048Mi"},
                                        "limits": {"cpu": "1", "memory": "3072Mi"},
                                    },
                                },
                                {
                                    "name": "workhorse",
                                    "image": "gitlab/workhorse:latest",
                                    "resources": {
                                        "requests": {"cpu": "100m", "memory": "256Mi"},
                                        "limits": {"cpu": "300m", "memory": "512Mi"},
                                    },
                                },
                            ],
                        },
                        "status": {"phase": "Running"},
                    }
                ]
            }
        return {"items": []}

    def get_pod_metrics_json(self, namespace: str | None = None, *, all_namespaces: bool = False):
        assert namespace == "gitlab"
        assert all_namespaces is False
        return {
            "items": [
                {
                    "metadata": {"name": "gitlab-webservice-default-abc", "namespace": "gitlab"},
                    "containers": [
                        {"name": "webservice", "usage": {"cpu": "17m", "memory": "2206Mi"}},
                        {"name": "workhorse", "usage": {"cpu": "4m", "memory": "82Mi"}},
                    ],
                }
            ]
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


def test_k8s_run_kubectl_reports_exec_subcommand_and_typed_tool_guidance() -> None:
    previous_allow = Config.K8S_CLI_ALLOW_ALL_READ
    try:
        Config.K8S_CLI_ALLOW_ALL_READ = True
        tools = {tool.name: tool for tool in create_k8s_read_tools(_FakeK8sReadOnly())}
        result = tools["k8s_run_kubectl"].invoke(
            {
                "command": "-n kong exec kong-kong-abc123 -c proxy -- curl -fsS http://127.0.0.1:8100/status/ready",
            }
        )
    finally:
        Config.K8S_CLI_ALLOW_ALL_READ = previous_allow

    assert "'kubectl exec' is a mutating operation" in result
    assert "typed write tool `k8s_exec_in_pod`" in result
    assert '"namespace": "kong"' in result
    assert '"pod": "kong-kong-abc123"' in result
    assert '"container": "proxy"' in result
    assert '"command": "curl -fsS http://127.0.0.1:8100/status/ready"' in result


def test_k8s_run_kubectl_exec_with_shell_operators_still_returns_typed_exec_guidance() -> None:
    previous_allow = Config.K8S_CLI_ALLOW_ALL_READ
    try:
        Config.K8S_CLI_ALLOW_ALL_READ = True
        tools = {tool.name: tool for tool in create_k8s_read_tools(_FakeK8sReadOnly())}
        result = tools["k8s_run_kubectl"].invoke(
            {
                "command": "exec -n nexus nexus-nexus-repository-manager-5fc4cdf4c7-krks4 -- sh -lc 'wget -qO- http://127.0.0.1:8081/ >/dev/null && echo OK || (echo FAIL; exit 1)'",
            }
        )
    finally:
        Config.K8S_CLI_ALLOW_ALL_READ = previous_allow

    assert "'kubectl exec' is a mutating operation" in result
    assert "typed write tool `k8s_exec_in_pod`" in result
    assert '"namespace": "nexus"' in result
    assert '"pod": "nexus-nexus-repository-manager-5fc4cdf4c7-krks4"' in result
    assert '"command": "sh -lc' in result
    assert "wget -qO- http://127.0.0.1:8081/" in result


def test_k8s_exec_preflights_container_name_before_running() -> None:
    client = _FakeK8sExecClient()

    result = client.exec_command(
        "kong-kong-abc123",
        "curl -fsS http://127.0.0.1:8100/status/ready",
        namespace="kong",
        container="kong",
    )

    assert "container 'kong' is not valid for pod 'kong-kong-abc123'" in result
    assert "ingress-controller, proxy" in result


def test_k8s_exec_command_preserves_stderr_on_success_path() -> None:
    client = _FakeK8sExecStdErrClient()

    result = client.exec_command(
        "nexus-nexus-repository-manager-5fc4cdf4c7-krks4",
        "sh -lc 'wget -qO- http://127.0.0.1:8081/service/rest/v1/status || curl -fsS http://127.0.0.1:8081/service/rest/v1/status'",
        namespace="nexus",
        container="nexus-repository-manager",
    )

    assert "wget: command not found" in result


def test_k8s_analyze_resource_usage_returns_exact_per_container_requests_limits_and_usage() -> None:
    client = _FakeK8sUsageClient()

    result = client.analyze_resource_usage(
        kind="deployment",
        namespace="gitlab",
        name="gitlab-webservice-default",
    )
    data = json.loads(result)

    assert data["query"]["kind"] == "deployment"
    assert data["metrics_available"] is True
    assert data["summary"]["pods_analyzed"] == 1
    assert data["summary"]["containers_analyzed"] == 2
    assert data["summary"]["total_requests"]["cpu_mcpu"] == 600
    assert data["summary"]["total_limits"]["cpu_mcpu"] == 1300
    assert data["summary"]["total_usage"]["cpu_mcpu"] == 21

    pod = data["pods"][0]
    assert pod["name"] == "gitlab-webservice-default-abc"
    container = {item["name"]: item for item in pod["containers"]}
    assert container["webservice"]["requests"]["cpu"] == "500m"
    assert container["webservice"]["limits"]["memory"] == "3072Mi"
    assert container["webservice"]["usage"]["cpu"] == "17m"
    assert container["webservice"]["usage_vs_request"]["memory_pct"] == 107.71
    assert container["workhorse"]["requests"]["memory"] == "256Mi"
    assert container["workhorse"]["usage"]["memory"] == "82Mi"


def test_k8s_read_tool_exposes_structured_resource_usage_analysis() -> None:
    tools = {tool.name: tool for tool in create_k8s_read_tools(_FakeK8sUsageClient())}

    result = tools["k8s_analyze_resource_usage"].invoke(
        {
            "kind": "deployment",
            "namespace": "gitlab",
            "name": "gitlab-webservice-default",
            "include_usage": True,
        }
    )
    data = json.loads(result)

    assert data["query"]["name"] == "gitlab-webservice-default"
    assert data["pods"][0]["containers"][0]["requests"]["cpu_mcpu"] == 500
