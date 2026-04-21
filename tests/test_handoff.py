from src.ui.handoff import (
    build_handoff_package,
    render_codex_handoff_prompt,
    render_handoff_markdown,
)


def test_build_handoff_package_uses_prior_user_trace_and_snippets() -> None:
    messages = [
        {"role": "user", "content": "fix the stale pod cleanup path"},
        {"role": "assistant", "content": "I investigated the failed pods and found node churn."},
        {"role": "user", "content": "package the exact fix and cleanup steps"},
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
        "query": "package the exact fix and cleanup steps",
        "intent": "diagnose",
        "outcome": "answered",
        "steps": [
            {
                "step_type": "tool_call",
                "tool_name": "k8s_get_resources",
                "tool_args": {"kind": "pod", "all_namespaces": True},
                "tool_result_preview": "Found failed pods across namespaces.",
            },
            {
                "step_type": "approval",
                "output_summary": "Proposed a cleanup CronJob rollout for approval.",
            }
        ],
    }

    package = build_handoff_package(messages, 3, trace=trace)

    assert package["trace_id"] == "abc123trace"
    assert package["original_user_request"] == "fix the stale pod cleanup path"
    assert package["user_request"] == "package the exact fix and cleanup steps"
    assert package["trace_intent"] == "diagnose"
    assert package["trace_outcome"] == "answered"
    assert package["trace_query"] == "package the exact fix and cleanup steps"
    assert "cleanup CronJob" in package["assistant_full_text"]
    assert "Conversation started with the operator asking" in package["conversation_summary"]
    assert len(package["conversation_transcript"]) == 4
    assert package["conversation_transcript"][-1]["role"] == "assistant"
    assert len(package["snippets"]) == 1
    assert package["snippets"][0]["language"] == "yaml"
    assert "k8s_get_resources" in package["evidence"][0]
    assert "Found failed pods" in package["evidence"][0]
    assert any("cleanup CronJob rollout" in item for item in package["remediation_items"])
    assert len(package["trace_steps"]) == 2


def test_render_handoff_outputs_include_core_context() -> None:
    package = {
        "title": "Clean up stale pods",
        "source_type": "assistant_message",
        "message_index": 2,
        "trace_id": "551d9edbacdd",
        "original_user_request": "why are dead pods piling up after instance restarts",
        "user_request": "why are dead pods piling up after instance restarts",
        "assistant_full_text": "Deploy a cleanup CronJob, run the deletes, and validate the node stop/start flow.",
        "assistant_summary": "Deploy a cleanup CronJob and review the stop/start flow.",
        "conversation_summary": "Conversation started with the operator asking about dead pods.",
        "conversation_transcript": [
            {"index": 0, "role": "user", "content": "why are dead pods piling up after instance restarts"},
            {"index": 1, "role": "assistant", "content": "Deploy a cleanup CronJob and review the stop/start flow.", "trace_id": "551d9edbacdd"},
        ],
        "trace_intent": "diagnose",
        "trace_outcome": "answered",
        "trace_query": "why are dead pods piling up after instance restarts",
        "trace_steps": [{"step_type": "tool_call", "tool_name": "k8s_get_resources"}],
        "evidence": ["k8s_get_resources - Failed pods remain after node churn."],
        "remediation_items": ["Proposed cleanup CronJob rollout for approval."],
        "snippets": [{"language": "bash", "content": "kubectl delete pod -A --field-selector=status.phase=Failed"}],
        "acceptance_criteria": ["Verify stale pods are cleaned up safely."],
    }

    codex_prompt = render_codex_handoff_prompt(package)
    markdown_bundle = render_handoff_markdown(package)

    assert "Trace ID: 551d9edbacdd" in codex_prompt
    assert "Original operator request:" in codex_prompt
    assert "Conversation summary so far:" in codex_prompt
    assert "Latest assistant response (full text):" in codex_prompt
    assert "Full conversation transcript so far:" in codex_prompt
    assert "```bash" in codex_prompt
    assert "Remediation and execution context:" in codex_prompt
    assert "## Assistant Summary" in markdown_bundle
    assert "## Latest Assistant Response (Full Text)" in markdown_bundle
    assert "## Full Conversation Transcript" in markdown_bundle
    assert "## Remediation Context" in markdown_bundle
    assert "Verify stale pods are cleaned up safely." in markdown_bundle
