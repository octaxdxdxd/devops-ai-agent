import json

from src.agents.action import _create_propose_action_tool, _parse_proposed_steps, format_action_step_preview


def test_format_action_step_preview_uses_real_command() -> None:
    label, preview, language = format_action_step_preview(
        {
            "command": "kubectl get nodes -o wide",
            "display": "Confirm current node state before scaling",
        }
    )

    assert label == "Confirm current node state before scaling"
    assert preview == "kubectl get nodes -o wide"
    assert language == "bash"


def test_propose_action_rejects_placeholder_steps() -> None:
    captured: list[dict] = []
    propose_action = _create_propose_action_tool(captured, {"aws_update_auto_scaling"})

    result = propose_action.invoke(
        {
            "description": "Scale the cluster",
            "commands_json": '[{"command":"kubectl scale nodegroup <nodegroup-name> --replicas=3","display":"Scale node group"}]',
            "risk": "MEDIUM",
            "expected_outcome": "More capacity",
            "verification_command": "kubectl get nodes -o wide",
        }
    )

    assert result == "Proposal rejected: step 1 contains a placeholder. Discover the real identifier first."
    assert captured == []


def test_propose_action_accepts_write_tool_steps() -> None:
    captured: list[dict] = []
    propose_action = _create_propose_action_tool(captured, {"aws_update_auto_scaling"})

    result = propose_action.invoke(
        {
            "description": "Scale the backing ASG",
            "commands_json": '[{"tool":"aws_update_auto_scaling","args":{"asg_name":"eks-general-123","desired":3},"display":"Scale ASG to 3"}]',
            "risk": "MEDIUM",
            "expected_outcome": "Two more nodes join the cluster",
            "verification_command": "kubectl get nodes -o wide",
        }
    )

    assert result.startswith("Action proposal captured.")
    assert captured == [
        {
            "description": "Scale the backing ASG",
            "commands": [
                {
                    "tool": "aws_update_auto_scaling",
                    "args": {"asg_name": "eks-general-123", "desired": 3},
                    "display": "Scale ASG to 3",
                }
            ],
            "risk": "MEDIUM",
            "expected_outcome": "Two more nodes join the cluster",
            "verification": {"command": "kubectl get nodes -o wide"},
        }
    ]


def test_propose_action_rejects_write_tool_step_with_empty_required_arg() -> None:
    captured: list[dict] = []
    propose_action = _create_propose_action_tool(captured, {"aws_update_auto_scaling"})

    result = propose_action.invoke(
        {
            "description": "Scale the backing ASG",
            "commands_json": '[{"tool":"aws_update_auto_scaling","args":{"asg_name":"","desired":3},"display":"Scale ASG to 3"}]',
            "risk": "MEDIUM",
            "expected_outcome": "One more node joins the cluster",
            "verification_command": "kubectl get nodes -o wide",
        }
    )

    assert result == (
        "Proposal rejected: step 1 tool 'aws_update_auto_scaling' has empty required arg 'asg_name'."
    )
    assert captured == []


def test_propose_action_rejects_read_only_aws_run_api_command_step() -> None:
    captured: list[dict] = []
    propose_action = _create_propose_action_tool(captured, {"aws_run_api_command", "aws_update_auto_scaling"})

    result = propose_action.invoke(
        {
            "description": "Scale the backing ASG",
            "commands_json": (
                '[{"tool":"aws_run_api_command","args":{"service":"autoscaling","operation":"describe_auto_scaling_groups","params_json":"{}","region":""},'
                '"display":"Inspect the worker Auto Scaling group to identify the current node group and capacity settings"},'
                '{"tool":"aws_update_auto_scaling","args":{"asg_name":"eks-general-123","desired":3},"display":"Scale ASG to 3"}]'
            ),
            "risk": "MEDIUM",
            "expected_outcome": "One more node joins the cluster",
            "verification_command": "kubectl get nodes -o wide",
        }
    )

    assert result == (
        "Proposal rejected: step 1 tool 'aws_run_api_command' cannot be used for read-only AWS operation "
        "'describe_auto_scaling_groups'. Resolve identifiers with read tools before proposing the action."
    )
    assert captured == []


def test_propose_action_accepts_mutating_aws_run_api_command_step() -> None:
    captured: list[dict] = []
    propose_action = _create_propose_action_tool(captured, {"aws_run_api_command"})
    commands_json = json.dumps(
        [
            {
                "tool": "aws_run_api_command",
                "args": {
                    "service": "autoscaling",
                    "operation": "resume_processes",
                    "params_json": '{"AutoScalingGroupName":"eks-general-123","ScalingProcesses":["Launch"]}',
                    "region": "us-east-1",
                },
                "display": "Resume Launch on the worker ASG",
            }
        ]
    )

    result = propose_action.invoke(
        {
            "description": "Resume ASG processes",
            "commands_json": commands_json,
            "risk": "MEDIUM",
            "expected_outcome": "Launch resumes on the ASG",
            "verification_command": "kubectl get nodes -o wide",
        }
    )

    assert result.startswith("Action proposal captured.")
    assert captured == [
        {
            "description": "Resume ASG processes",
            "commands": [
                {
                    "tool": "aws_run_api_command",
                    "args": {
                        "service": "autoscaling",
                        "operation": "resume_processes",
                        "params_json": '{"AutoScalingGroupName":"eks-general-123","ScalingProcesses":["Launch"]}',
                        "region": "us-east-1",
                    },
                    "display": "Resume Launch on the worker ASG",
                }
            ],
            "risk": "MEDIUM",
            "expected_outcome": "Launch resumes on the ASG",
            "verification": {"command": "kubectl get nodes -o wide"},
        }
    ]


def test_propose_action_rejects_read_tools_as_action_steps() -> None:
    captured: list[dict] = []
    propose_action = _create_propose_action_tool(captured, {"aws_update_auto_scaling"})

    result = propose_action.invoke(
        {
            "description": "Scale the cluster",
            "commands_json": '[{"tool":"aws_describe_service","args":{"service":"eks"},"display":"Inspect node group"}]',
            "risk": "MEDIUM",
            "expected_outcome": "More capacity",
            "verification_command": "kubectl get nodes -o wide",
        }
    )

    assert result == (
        "Proposal rejected: step 1 tool 'aws_describe_service' is not an allowed write tool. "
        "Allowed tools: aws_update_auto_scaling."
    )
    assert captured == []


def test_parse_proposed_steps_salvages_malformed_typed_tool_array() -> None:
    steps = _parse_proposed_steps(
        '[{"tool":"k8s_exec_in_pod","args":{"pod":"gitlab-webservice-default-586d7cb8d-58j65","namespace":"gitlab","command":"whoami"},"display":"verify pod user"},{"tool":"k8s_exec_in_pod","args":{"pod":"gitlab-webservice-default-586d7cb8d-58j65","namespace":"gitlab","command":"curl -fsS http://127.0.0.1:8080/-/health","display":"run local healthcheck"}]'
    )

    assert steps == [
        {
            "tool": "k8s_exec_in_pod",
            "args": {
                "pod": "gitlab-webservice-default-586d7cb8d-58j65",
                "namespace": "gitlab",
                "command": "whoami",
            },
            "display": "verify pod user",
        },
        {
            "tool": "k8s_exec_in_pod",
            "args": {
                "pod": "gitlab-webservice-default-586d7cb8d-58j65",
                "namespace": "gitlab",
                "command": "curl -fsS http://127.0.0.1:8080/-/health",
            },
            "display": "run local healthcheck",
        },
    ]


def test_propose_action_accepts_salvaged_typed_exec_steps() -> None:
    captured: list[dict] = []
    propose_action = _create_propose_action_tool(captured, {"k8s_exec_in_pod"})

    result = propose_action.invoke(
        {
            "description": "Run health checks from inside the GitLab webservice pod",
            "commands_json": '[{"tool":"k8s_exec_in_pod","args":{"pod":"gitlab-webservice-default-586d7cb8d-58j65","namespace":"gitlab","command":"whoami"},"display":"verify pod user"},{"tool":"k8s_exec_in_pod","args":{"pod":"gitlab-webservice-default-586d7cb8d-58j65","namespace":"gitlab","command":"curl -fsS http://127.0.0.1:8080/-/health","display":"run local healthcheck"}]',
            "risk": "LOW",
            "expected_outcome": "Local pod checks succeed",
            "verification_command": "",
        }
    )

    assert result.startswith("Action proposal captured.")
    assert captured == [
        {
            "description": "Run health checks from inside the GitLab webservice pod",
            "commands": [
                {
                    "tool": "k8s_exec_in_pod",
                    "args": {
                        "pod": "gitlab-webservice-default-586d7cb8d-58j65",
                        "namespace": "gitlab",
                        "command": "whoami",
                    },
                    "display": "verify pod user",
                },
                {
                    "tool": "k8s_exec_in_pod",
                    "args": {
                        "pod": "gitlab-webservice-default-586d7cb8d-58j65",
                        "namespace": "gitlab",
                        "command": "curl -fsS http://127.0.0.1:8080/-/health",
                    },
                    "display": "run local healthcheck",
                },
            ],
            "risk": "LOW",
            "expected_outcome": "Local pod checks succeed",
            "verification": None,
        }
    ]


def test_propose_action_rejects_exec_step_with_empty_command() -> None:
    captured: list[dict] = []
    propose_action = _create_propose_action_tool(captured, {"k8s_exec_in_pod"})

    result = propose_action.invoke(
        {
            "description": "Exec into pod",
            "commands_json": '[{"tool":"k8s_exec_in_pod","args":{"pod":"gitlab-webservice-default-586d7cb8d-58j65","namespace":"gitlab","command":""},"display":"Run healthcheck"}]',
            "risk": "LOW",
            "expected_outcome": "Healthcheck succeeds",
            "verification_command": "",
        }
    )

    assert result == (
        "Proposal rejected: step 1 tool 'k8s_exec_in_pod' has empty required arg 'command'."
    )
    assert captured == []
