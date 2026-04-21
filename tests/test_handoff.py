from src.ui.handoff import (
    build_handoff_package,
    render_codex_handoff_prompt,
    render_handoff_markdown,
)


def test_build_handoff_package_uses_prior_user_trace_and_snippets() -> None:
    messages = [
        {"role": "user", "content": "fix the stale pod cleanup path"},
        {
            "role": "assistant",
            "content": (
                "Use a cleanup CronJob and validate it.\n\n"
                "```yaml\napiVersion: batch/v1\nkind: CronJob\nmetadata:\n  name: pod-gc\n```\n"
            ),
            "trace_id": "abc123trace",
        },
    ]
    trace = {
        "trace_id": "abc123trace",
        "intent": "diagnose",
        "outcome": "answered",
        "steps": [
            {
                "step_type": "tool_call",
                "tool_name": "k8s_get_resources",
                "tool_args": {"kind": "pod", "all_namespaces": True},
                "tool_result_preview": "Found failed pods across namespaces.",
            }
        ],
    }

    package = build_handoff_package(messages, 1, trace=trace)

    assert package["trace_id"] == "abc123trace"
    assert package["user_request"] == "fix the stale pod cleanup path"
    assert package["trace_intent"] == "diagnose"
    assert package["trace_outcome"] == "answered"
    assert len(package["snippets"]) == 1
    assert package["snippets"][0]["language"] == "yaml"
    assert "k8s_get_resources" in package["evidence"][0]
    assert "Found failed pods" in package["evidence"][0]


def test_render_handoff_outputs_include_core_context() -> None:
    package = {
        "title": "Clean up stale pods",
        "source_type": "assistant_message",
        "message_index": 2,
        "trace_id": "551d9edbacdd",
        "user_request": "why are dead pods piling up after instance restarts",
        "assistant_summary": "Deploy a cleanup CronJob and review the stop/start flow.",
        "trace_intent": "diagnose",
        "trace_outcome": "answered",
        "evidence": ["k8s_get_resources - Failed pods remain after node churn."],
        "snippets": [{"language": "bash", "content": "kubectl delete pod -A --field-selector=status.phase=Failed"}],
        "acceptance_criteria": ["Verify stale pods are cleaned up safely."],
    }

    codex_prompt = render_codex_handoff_prompt(package)
    markdown_bundle = render_handoff_markdown(package)

    assert "Trace ID: 551d9edbacdd" in codex_prompt
    assert "Original operator request:" in codex_prompt
    assert "```bash" in codex_prompt
    assert "## Assistant Summary" in markdown_bundle
    assert "Verify stale pods are cleaned up safely." in markdown_bundle
