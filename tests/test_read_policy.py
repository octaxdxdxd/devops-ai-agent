from __future__ import annotations

from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

from src.agents.read_policy import ReadScopeResult, classify_read_scope, select_read_tools
from src.tracing.tracer import Tracer


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _StaticLLM:
    def __init__(self, content: str) -> None:
        self._content = content

    def invoke(self, messages):
        return _FakeResponse(self._content)


def _tool(name: str):
    return SimpleNamespace(name=name)


def _sample_k8s_tools() -> list:
    return [
        _tool("k8s_get_resources"),
        _tool("k8s_get_resource_yaml"),
        _tool("k8s_get_resource_usage"),
        _tool("k8s_analyze_resource_usage"),
    ]


def _sample_aws_tools() -> list:
    return [
        _tool("aws_get_caller_identity"),
        _tool("aws_describe_service"),
        _tool("aws_audit_cloudtrail"),
        _tool("aws_inspect_lambda_schedules"),
    ]


def test_trace_regression_8f1fca82_remains_k8s_scoped() -> None:
    selection = select_read_tools(
        "what node affinity does jenkins have",
        [],
        _sample_k8s_tools(),
        _sample_aws_tools(),
        scope=ReadScopeResult(backend="k8s", specialization="none", confidence="high"),
    )

    names = [tool.name for tool in selection.tools]
    assert selection.backend == "k8s"
    assert names == ["k8s_get_resources", "k8s_get_resource_yaml", "k8s_get_resource_usage", "k8s_analyze_resource_usage"]
    assert "aws_get_caller_identity" not in names


def test_trace_regression_1bdd0a70_cpu_memory_query_remains_k8s_scoped() -> None:
    selection = select_read_tools(
        "show me all cpu/memory request for each workload vs how much theyre currently using",
        [],
        _sample_k8s_tools(),
        _sample_aws_tools(),
        scope=ReadScopeResult(backend="k8s", specialization="none", confidence="high"),
    )

    assert selection.backend == "k8s"
    assert all(tool.name.startswith("k8s_") for tool in selection.tools)


def test_schedule_specialization_selects_only_schedule_tool() -> None:
    selection = select_read_tools(
        "i have a lambda kill tagless resources in my aws account and i want to know how frequent it runs",
        [],
        _sample_k8s_tools(),
        _sample_aws_tools(),
        scope=ReadScopeResult(backend="aws", specialization="schedule", confidence="high"),
    )

    assert [tool.name for tool in selection.tools] == ["aws_inspect_lambda_schedules"]


def test_cloudtrail_specialization_selects_only_audit_tool() -> None:
    selection = select_read_tools(
        "show me all aws resources deleted by octavian.popov@endava.com in the last 5 days",
        [],
        _sample_k8s_tools(),
        _sample_aws_tools(),
        scope=ReadScopeResult(backend="aws", specialization="cloudtrail", confidence="high"),
    )

    assert [tool.name for tool in selection.tools] == ["aws_audit_cloudtrail"]


def test_low_confidence_scope_binds_broad_safe_read_tools() -> None:
    selection = select_read_tools(
        "maybe check it",
        [],
        _sample_k8s_tools(),
        _sample_aws_tools(),
        scope=ReadScopeResult(backend="mixed", specialization="none", confidence="low"),
    )

    names = {tool.name for tool in selection.tools}
    assert names == {
        "k8s_get_resources",
        "k8s_get_resource_yaml",
        "k8s_get_resource_usage",
        "k8s_analyze_resource_usage",
        "aws_describe_service",
        "aws_audit_cloudtrail",
        "aws_inspect_lambda_schedules",
    }


def test_classifier_ignores_assistant_schedule_language_when_user_context_is_k8s() -> None:
    history = [
        HumanMessage(content="analyze why gitlab isnt running successfully"),
        AIMessage(content="I can continue by checking Lambda schedules and next run times if needed."),
    ]

    scope = classify_read_scope(
        "check the gitlab webservice pod logs",
        history,
        _StaticLLM('{"backend":"k8s","specialization":"none","confidence":"high"}'),
        "test-model",
        Tracer(),
    )

    assert scope.backend == "k8s"
    assert scope.specialization == "none"


def test_classifier_emits_selector_trace_step() -> None:
    tracer = Tracer()
    tracer.start("classify scope")

    scope = classify_read_scope(
        "show me all aws resources deleted by octavian.popov@endava.com in the last 5 days",
        [],
        _StaticLLM('{"backend":"aws","specialization":"cloudtrail","confidence":"high"}'),
        "test-model",
        tracer,
    )

    assert scope.backend == "aws"
    selector_steps = [step for step in tracer.current_trace.steps if step.step_type == "selector"]
    assert len(selector_steps) == 1
    assert "backend=aws" in selector_steps[0].output_summary
    assert "specialization=cloudtrail" in selector_steps[0].output_summary
