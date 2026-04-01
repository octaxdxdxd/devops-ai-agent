from src.agents.action import _create_propose_action_tool, format_action_step_preview


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
