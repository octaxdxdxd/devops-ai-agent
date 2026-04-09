from __future__ import annotations

from types import SimpleNamespace

from langchain_core.messages import HumanMessage

from src.agents.action import PendingAction, _create_propose_action_tool, format_action_step_preview
from src.agents.base import run_tool_loop
from src.agents.orchestrator import AIOpsAgent
from src.tools.aws_read import create_aws_read_tools
from src.tracing.tracer import Tracer


class _FakeAWS:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, object, object]] = []

    def describe_service(self, service, operation, params=None, region=None) -> str:
        self.calls.append((service, operation, params, region))
        return "{}"


class _FakeResponse:
    def __init__(self, content: str = "", tool_calls: list[dict] | None = None) -> None:
        self.content = content
        self.tool_calls = tool_calls or []


class _FakeLLM:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self.responses = list(responses)

    def invoke(self, messages):
        return self.responses.pop(0)


class _CountingTool:
    def __init__(self, result: str) -> None:
        self.result = result
        self.calls: list[dict] = []

    def invoke(self, args: dict) -> str:
        self.calls.append(dict(args))
        return self.result


class _DummyStore:
    def __init__(self) -> None:
        self.saved = []

    def save(self, trace) -> None:
        self.saved.append(trace)


class _DummyTools:
    def __init__(self, result: str) -> None:
        self.result = result
        self.calls: list[tuple[str, dict]] = []

    def execute(self, tool_name: str, args: dict) -> str:
        self.calls.append((tool_name, dict(args)))
        return self.result


def test_aws_describe_service_blocks_mutating_operations_in_read_tool() -> None:
    fake = _FakeAWS()
    tools = {tool.name: tool for tool in create_aws_read_tools(fake)}

    result = tools["aws_describe_service"].invoke(
        {
            "service": "autoscaling",
            "operation": "resume_processes",
            "params_json": '{"AutoScalingGroupName":"asg-1"}',
            "region": "us-east-1",
        }
    )

    assert result == (
        "ERROR: AWS read-only tool blocked non-read operation 'resume_processes'. "
        "Use an approval-gated write tool instead."
    )
    assert fake.calls == []


def test_propose_action_rejects_chained_verification_command() -> None:
    captured: list[dict] = []
    propose_action = _create_propose_action_tool(captured, {"aws_update_auto_scaling"})

    result = propose_action.invoke(
        {
            "description": "Scale the cluster",
            "commands_json": '[{"tool":"aws_update_auto_scaling","args":{"asg_name":"eks-general-123","desired":2},"display":"Scale ASG"}]',
            "risk": "MEDIUM",
            "expected_outcome": "More capacity",
            "verification_command": "kubectl get nodes && kubectl get pods -A",
        }
    )

    assert result == (
        "Proposal rejected: verification_command must be a single kubectl command without shell operators."
    )
    assert captured == []


def test_format_action_step_preview_renders_cli_for_write_tool() -> None:
    label, preview, language = format_action_step_preview(
        {
            "tool": "aws_update_auto_scaling",
            "args": {"asg_name": "eks-general-123", "desired": 3},
            "display": "Scale ASG to 3",
        }
    )

    assert label == "Scale ASG to 3"
    assert preview == (
        "aws autoscaling update-auto-scaling-group "
        "--auto-scaling-group-name eks-general-123 --desired-capacity 3"
    )
    assert language == "bash"


def test_run_tool_loop_dedupes_identical_read_calls_within_a_turn() -> None:
    fake_tool = _CountingTool("namespace-a")
    llm = _FakeLLM(
        [
            _FakeResponse(tool_calls=[{"name": "k8s_get_namespaces", "args": {}, "id": "call-1"}]),
            _FakeResponse(tool_calls=[{"name": "k8s_get_namespaces", "args": {}, "id": "call-2"}]),
            _FakeResponse(content="done"),
        ]
    )

    result = run_tool_loop(
        messages=[HumanMessage(content="list namespaces")],
        llm_with_tools=llm,
        tool_map={"k8s_get_namespaces": fake_tool},
        max_steps=3,
        handler_name="lookup",
        model_name="test-model",
        tracer=Tracer(),
    )

    assert result == "done"
    assert fake_tool.calls == [{}]


def test_execute_action_marks_failed_and_uses_failed_trace_outcome() -> None:
    action = PendingAction(
        id="a1",
        description="Scale the ASG",
        commands=[
            {
                "tool": "aws_update_auto_scaling",
                "args": {"asg_name": "eks-general-123", "desired": 2},
                "display": "Scale ASG to 2",
            }
        ],
        risk="MEDIUM",
        expected_outcome="Two nodes",
        verification={"command": "kubectl get nodes -o wide"},
    )
    store = _DummyStore()
    agent = object.__new__(AIOpsAgent)
    agent.pending_actions = [action]
    agent.tracer = Tracer()
    agent.trace_store = store
    agent.last_trace_id = None
    agent.tools = _DummyTools("ERROR: boom")
    agent.k8s = SimpleNamespace(run=lambda args: "NAME\nnode-1")
    agent.operator_intent_state = SimpleNamespace(pending_step_summary="pending", pending_step_kind="action")
    agent._post_remediation_analysis = lambda action, output: ""

    updates: list[str] = []
    result = agent.execute_action("a1", status_callback=updates.append)

    assert action.status == "failed"
    assert updates == [
        "aws autoscaling update-auto-scaling-group --auto-scaling-group-name eks-general-123 --desired-capacity 2",
        "kubectl get nodes -o wide",
        "Analyzing results...",
    ]
    assert "ERROR: boom" in result
    assert store.saved[0].outcome == "action_failed"
