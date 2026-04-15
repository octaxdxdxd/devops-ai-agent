from src.agents.intent import classify_intent
from src.tracing.tracer import Tracer


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _StaticLLM:
    def __init__(self, content: str) -> None:
        self._content = content

    def invoke(self, messages):
        return _FakeResponse(self._content)


def test_ambiguous_acknowledgement_requests_clarification() -> None:
    result = classify_intent(
        user_input="yeah do that",
        llm=_StaticLLM(
            '{"intent":"lookup","resources":[],"namespaces":[],"needs_clarification":true,'
            '"clarification_prompt":"Please restate the resource, problem, or next step you want me to continue with."}'
        ),
        model_name="test-model",
        tracer=Tracer(),
        chat_history=[],
    )

    assert result.intent == "lookup"
    assert result.needs_clarification is True
    assert "Please restate" in result.clarification_prompt


def test_explicit_action_still_uses_classifier() -> None:
    result = classify_intent(
        user_input="restart nexus pod",
        llm=_StaticLLM(
            '{"intent":"action","resources":["pod"],"namespaces":["nexus"],'
            '"needs_clarification":false,"clarification_prompt":""}'
        ),
        model_name="test-model",
        tracer=Tracer(),
        chat_history=[],
    )

    assert result.intent == "action"
    assert result.resources == ["pod"]
    assert result.namespaces == ["nexus"]
