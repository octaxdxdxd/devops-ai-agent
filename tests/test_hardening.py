from __future__ import annotations

from types import SimpleNamespace

from langchain_core.messages import HumanMessage

import src.agents.orchestrator as orchestrator_module
from src.agents.action import PendingAction, _create_propose_action_tool, format_action_step_preview
from src.agents.base import run_tool_loop
from src.agents.diagnose import handle_diagnose
from src.agents.explain import handle_explain
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


class _StaticLLM:
    def __init__(self, content: str) -> None:
        self._content = content

    def invoke(self, messages):
        return _FakeResponse(self._content)


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


class _DummyAWSStatus:
    def set_status_callback(self, cb) -> None:
        self.callback = cb

    def clear_status_callback(self) -> None:
        self.callback = None


class _DummyModelWrapper:
    def __init__(self, llm) -> None:
        self.llm = llm

    def get_llm(self):
        return self.llm

    def get_llm_with_tools(self, tools):
        return self.llm


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


def test_process_query_requests_clarification_for_ambiguous_acknowledgement() -> None:
    class _ModelWrapper:
        def __init__(self):
            self.llm = _StaticLLM(
                '{"intent":"lookup","resources":[],"namespaces":[],"needs_clarification":true,'
                '"clarification_prompt":"Please restate the resource, problem, or next step you want me to continue with."}'
            )

        def get_llm(self):
            return self.llm

        def get_llm_with_tools(self, tools):  # pragma: no cover
            raise AssertionError("Handler/tool selection should not run for ambiguous acknowledgements")

    store = _DummyStore()
    agent = object.__new__(AIOpsAgent)
    agent._model_wrapper = _ModelWrapper()
    agent.model_name = "test-model"
    agent.pending_actions = []
    agent.tracer = Tracer()
    agent.trace_store = store
    agent.last_trace_id = None
    agent.aws = _DummyAWSStatus()
    agent.operator_intent_state = SimpleNamespace(
        last_user_instruction="",
        pending_step_kind="",
        pending_step_summary="",
    )

    result = agent.process_query("yes continue", chat_history=[], status_callback=lambda _: None)

    assert "Please restate" in result
    assert store.saved[0].outcome == "answered"
    selector_steps = [step for step in store.saved[0].steps if step.step_type == "selector"]
    assert len(selector_steps) == 1
    assert selector_steps[0].output_summary == "blocked_for_clarification"


def test_run_tool_loop_retries_when_read_handler_answers_without_tools() -> None:
    fake_tool = _CountingTool("statefulset yaml")
    llm = _FakeLLM(
        [
            _FakeResponse(content="I cannot inspect Kubernetes from here."),
            _FakeResponse(
                tool_calls=[
                    {
                        "name": "k8s_get_resource_yaml",
                        "args": {"kind": "statefulset", "name": "jenkins", "namespace": "jenkins"},
                        "id": "call-1",
                    }
                ]
            ),
            _FakeResponse(content="No explicit nodeAffinity is present."),
        ]
    )

    result = run_tool_loop(
        messages=[HumanMessage(content="what node affinity does jenkins have")],
        llm_with_tools=llm,
        tool_map={"k8s_get_resource_yaml": fake_tool},
        max_steps=4,
        handler_name="lookup",
        model_name="test-model",
        tracer=Tracer(),
        original_query="what node affinity does jenkins have",
        require_relevant_tool_call_before_answer=True,
        available_capability_families=["Kubernetes"],
        insufficient_tool_names={"aws_get_caller_identity"},
    )

    assert result == "No explicit nodeAffinity is present."
    assert fake_tool.calls == [{"kind": "statefulset", "name": "jenkins", "namespace": "jenkins"}]


def test_run_tool_loop_does_not_count_aws_identity_as_sufficient_for_k8s_lookup() -> None:
    fake_identity_tool = _CountingTool('{"Arn":"arn:aws:iam::123456789012:user/test"}')
    fake_k8s_tool = _CountingTool("statefulset yaml")
    llm = _FakeLLM(
        [
            _FakeResponse(tool_calls=[{"name": "aws_get_caller_identity", "args": {}, "id": "call-1"}]),
            _FakeResponse(content="I only have AWS tools here."),
            _FakeResponse(
                tool_calls=[
                    {
                        "name": "k8s_get_resource_yaml",
                        "args": {"kind": "statefulset", "name": "jenkins", "namespace": "jenkins"},
                        "id": "call-2",
                    }
                ]
            ),
            _FakeResponse(content="No explicit nodeAffinity is present."),
        ]
    )

    result = run_tool_loop(
        messages=[HumanMessage(content="what node affinity does jenkins have")],
        llm_with_tools=llm,
        tool_map={
            "aws_get_caller_identity": fake_identity_tool,
            "k8s_get_resource_yaml": fake_k8s_tool,
        },
        max_steps=5,
        handler_name="lookup",
        model_name="test-model",
        tracer=Tracer(),
        original_query="what node affinity does jenkins have",
        require_relevant_tool_call_before_answer=True,
        available_capability_families=["Kubernetes"],
        insufficient_tool_names={"aws_get_caller_identity"},
    )

    assert result == "No explicit nodeAffinity is present."
    assert fake_identity_tool.calls == [{}]
    assert fake_k8s_tool.calls == [{"kind": "statefulset", "name": "jenkins", "namespace": "jenkins"}]


def test_handle_diagnose_inherits_tool_first_enforcement() -> None:
    fake_tool = _CountingTool("pod output")
    llm = _FakeLLM(
        [
            _FakeResponse(content="I cannot inspect Kubernetes from here."),
            _FakeResponse(
                tool_calls=[
                    {
                        "name": "k8s_get_resources",
                        "args": {"kind": "pod", "namespace": "jenkins", "name": "", "label_selector": "", "all_namespaces": False},
                        "id": "call-1",
                    }
                ]
            ),
            _FakeResponse(content="Jenkins is pending."),
        ]
    )

    result = handle_diagnose(
        "why is jenkins pending?",
        [],
        llm,
        {"k8s_get_resources": fake_tool},
        "test-model",
        Tracer(),
        capability_prompt="Capabilities in this turn:\n- Kubernetes read tools available.",
        require_live_inspection=True,
        available_capability_families=["Kubernetes"],
        insufficient_tool_names={"aws_get_caller_identity"},
    )

    assert result == "Jenkins is pending."
    assert fake_tool.calls == [{"kind": "pod", "namespace": "jenkins", "name": "", "label_selector": "", "all_namespaces": False}]


def test_handle_explain_inherits_tool_first_enforcement() -> None:
    fake_tool = _CountingTool("usage output")
    llm = _FakeLLM(
        [
            _FakeResponse(content="I only have AWS tools available."),
            _FakeResponse(
                tool_calls=[
                    {
                        "name": "k8s_get_resource_usage",
                        "args": {"resource_type": "pods", "namespace": "jenkins", "name": ""},
                        "id": "call-1",
                    }
                ]
            ),
            _FakeResponse(content="Current usage is available now."),
        ]
    )

    result = handle_explain(
        "show me current pod usage in jenkins",
        [],
        llm,
        {"k8s_get_resource_usage": fake_tool},
        "test-model",
        Tracer(),
        capability_prompt="Capabilities in this turn:\n- Kubernetes read tools available.",
        require_live_inspection=True,
        available_capability_families=["Kubernetes"],
        insufficient_tool_names={"aws_get_caller_identity"},
    )

    assert result == "Current usage is available now."
    assert fake_tool.calls == [{"resource_type": "pods", "namespace": "jenkins", "name": ""}]


def test_run_tool_loop_reopens_incomplete_investigation_before_returning() -> None:
    fake_tool = _CountingTool("resource details")
    llm = _FakeLLM(
        [
            _FakeResponse(tool_calls=[{"name": "k8s_get_namespaces", "args": {}, "id": "call-1"}]),
            _FakeResponse(content="## Root Cause Analysis\n**Symptom**: pods are failing"),
            _FakeResponse(tool_calls=[{"name": "k8s_get_namespaces", "args": {"all_namespaces": True}, "id": "call-2"}]),
            _FakeResponse(
                content=(
                    "## Root Cause Analysis\n"
                    "**Symptom**: pods are failing\n"
                    "**Root Cause**: node churn\n"
                    "**Evidence Chain**: confirmed by tool output\n"
                    "**Recommended Next Steps**: restart the workload\n"
                    "**Exact Fix**: use a typed restart action"
                )
            ),
        ]
    )

    result = run_tool_loop(
        messages=[HumanMessage(content="why are pods failing")],
        llm_with_tools=llm,
        tool_map={"k8s_get_namespaces": fake_tool},
        max_steps=5,
        handler_name="diagnose",
        model_name="test-model",
        tracer=Tracer(),
        final_answer_feedback=lambda text, meta: (
            "Continue investigating until you include an Exact Fix."
            if "exact fix" not in text.lower()
            else None
        ),
    )

    assert "**Exact Fix**: use a typed restart action" in result
    assert fake_tool.calls == [{}, {"all_namespaces": True}]


def test_maybe_plan_fix_from_diagnosis_creates_pending_action() -> None:
    original_handle_action = orchestrator_module.handle_action
    captured_statuses: list[str] = []

    try:
        orchestrator_module.handle_action = lambda *args, **kwargs: (
            "Planned remediation",
            [
                PendingAction(
                    id="p1",
                    description="Restart deployment safely",
                    commands=[
                        {
                            "tool": "k8s_restart_workload_safely",
                            "args": {"kind": "deployment", "name": "api", "namespace": "prod"},
                        }
                    ],
                    risk="MEDIUM",
                    expected_outcome="Deployment recovers",
                )
            ],
        )

        agent = object.__new__(AIOpsAgent)
        agent.tools = SimpleNamespace(
            read_tools=[SimpleNamespace(name="k8s_get_resources")],
            write_tools=[SimpleNamespace(name="k8s_restart_workload_safely")],
        )
        agent._model_wrapper = _DummyModelWrapper(object())
        agent.model_name = "test-model"
        agent.tracer = Tracer()
        agent.pending_actions = []
        agent.operator_intent_state = SimpleNamespace(pending_step_summary="", pending_step_kind="")

        diagnosis = (
            "## Root Cause Analysis\n"
            "**Symptom**: pods are failing\n"
            "**Root Cause**: bad rollout\n"
            "**Evidence Chain**: rollout history and events\n"
            "**Recommended Next Steps**: restart the deployment safely\n"
            "**Exact Fix**: restart deployment api in prod"
        )

        note = agent._maybe_plan_fix_from_diagnosis(
            "why is api failing",
            [],
            diagnosis,
            captured_statuses.append,
        )
    finally:
        orchestrator_module.handle_action = original_handle_action

    assert "Proposed Fix" in note
    assert agent.pending_actions[0].description == "Restart deployment safely"
    assert captured_statuses == ["Planning remediation from RCA..."]
