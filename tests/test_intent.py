from src.agents.intent import classify_intent
from src.tracing.tracer import Tracer


class _FailIfCalledLLM:
    def invoke(self, messages):  # pragma: no cover
        raise AssertionError("LLM should not be called for confirmation-only inputs")


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _StaticLLM:
    def __init__(self, content: str) -> None:
        self._content = content

    def invoke(self, messages):
        return _FakeResponse(self._content)


def test_confirmation_only_input_defaults_to_lookup_without_llm() -> None:
    result = classify_intent(
        user_input="yeah do that",
        llm=_FailIfCalledLLM(),
        model_name="test-model",
        tracer=Tracer(),
        chat_history=[],
    )

    assert result.intent == "lookup"
    assert result.resources == []
    assert result.namespaces == []


def test_explicit_action_still_uses_classifier() -> None:
    result = classify_intent(
        user_input="restart nexus pod",
        llm=_StaticLLM('{"intent":"action","resources":["pod"],"namespaces":["nexus"]}'),
        model_name="test-model",
        tracer=Tracer(),
        chat_history=[],
    )

    assert result.intent == "action"
    assert result.resources == ["pod"]
    assert result.namespaces == ["nexus"]
