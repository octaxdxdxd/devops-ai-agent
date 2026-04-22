from langchain_core.messages import AIMessage

from src.ui.handoff import (
    build_handoff_package,
    enrich_handoff_package,
    render_agent_handoff_prompt,
    render_codex_handoff_prompt,
    render_handoff_markdown,
)


def _sample_messages() -> list[dict]:
    return [
        {"role": "user", "content": "investigate why the cleanup CronJob keeps pulling the wrong image"},
        {"role": "assistant", "content": "I inspected the stuck job and the failing pods."},
        {"role": "user", "content": "package the exact durable fix"},
        {
            "role": "assistant",
            "content": (
                "Update the CronJob image and preserve the verification path.\n\n"
                "```yaml\napiVersion: batch/v1\nkind: CronJob\nmetadata:\n  name: terminated-pod-gc\n```\n"
            ),
            "trace_id": "trace-current",
        },
    ]


def _sample_recent_traces() -> list[dict]:
    return [
        {
            "trace_id": "trace-current",
            "query": "package the exact durable fix",
            "intent": "diagnose",
            "outcome": "action_proposed",
            "started_at": "2026-04-22T00:49:07+00:00",
            "completed_at": "2026-04-22T00:49:54+00:00",
            "steps": [
                {
                    "step_type": "tool_call",
                    "handler": "action",
                    "tool_name": "k8s_get_resource_yaml",
                    "tool_args": {"kind": "cronjob", "name": "terminated-pod-gc", "namespace": "kube-system"},
                    "tool_result_preview": "apiVersion: batch/v1\nkind: CronJob",
                },
                {
                    "step_type": "tool_call",
                    "handler": "action",
                    "tool_name": "propose_action",
                    "tool_args": {
                        "description": "Update the kube-system terminated-pod-gc CronJob to use a known-good kubectl image tag.",
                        "commands_json": (
                            "[{\"tool\":\"k8s_apply_manifest\",\"args\":{\"manifest_yaml\":\"apiVersion: batch/v1\\nkind: CronJob\"},"
                            "\"display\":\"Apply CronJob image fix\"}]"
                        ),
                        "risk": "MEDIUM",
                        "expected_outcome": "New Jobs pull the corrected image.",
                        "verification_command": "kubectl get cronjob terminated-pod-gc -n kube-system -o=jsonpath='{.spec.jobTemplate.spec.template.spec.containers[0].image}'",
                    },
                    "tool_result_preview": "Action proposal captured.",
                },
                {
                    "step_type": "approval",
                    "handler": "action",
                    "output_summary": "Proposed: Update the kube-system terminated-pod-gc CronJob to use a known-good kubectl image tag. (risk=MEDIUM)",
                },
            ],
        },
        {
            "trace_id": "trace-executed",
            "query": "Execute action: Update the kube-system terminated-pod-gc CronJob to use a known-good kubectl image tag.",
            "intent": "",
            "outcome": "action_executed",
            "started_at": "2026-04-22T00:50:35+00:00",
            "completed_at": "2026-04-22T00:50:44+00:00",
            "steps": [
                {
                    "step_type": "tool_call",
                    "handler": "action",
                    "tool_name": "k8s_apply_manifest",
                    "tool_args": {
                        "tool": "k8s_apply_manifest",
                        "display": "Apply CronJob image fix",
                        "args": {"manifest_yaml": "apiVersion: batch/v1\nkind: CronJob"},
                    },
                    "tool_result_preview": "cronjob.batch/terminated-pod-gc configured",
                },
                {
                    "step_type": "verify",
                    "handler": "action",
                    "tool_name": "kubectl",
                    "tool_args": {
                        "command": "kubectl get cronjob terminated-pod-gc -n kube-system -o=jsonpath='{.spec.jobTemplate.spec.template.spec.containers[0].image}'"
                    },
                    "tool_result_preview": "bitnami/kubectl:1.34.3",
                },
            ],
        },
        {
            "trace_id": "trace-investigate",
            "query": "why is the cleanup job still imagepullbackoff",
            "intent": "diagnose",
            "outcome": "answered",
            "started_at": "2026-04-22T00:45:22+00:00",
            "completed_at": "2026-04-22T00:46:19+00:00",
            "steps": [
                {
                    "step_type": "tool_call",
                    "handler": "diagnose",
                    "tool_name": "k8s_run_kubectl",
                    "tool_args": {"command": "-n kube-system get pod -l job-name=terminated-pod-gc-29612880 -o wide"},
                    "tool_result_preview": "terminated-pod-gc-29612880-474dd   0/1   ImagePullBackOff",
                },
                {
                    "step_type": "tool_call",
                    "handler": "diagnose",
                    "tool_name": "k8s_get_events",
                    "tool_args": {"namespace": "kube-system", "field_selector": "type=Warning", "all_namespaces": False},
                    "tool_result_preview": "Failed to pull image docker.io/bitnami/kubectl:1.35.0",
                },
            ],
        },
    ]


def test_build_handoff_package_collects_recent_runs_commands_and_changes() -> None:
    messages = _sample_messages()
    recent_traces = _sample_recent_traces()

    package = build_handoff_package(
        messages,
        3,
        trace=recent_traces[0],
        recent_traces=recent_traces,
        model_name="openai/gpt-5.4-mini",
    )

    assert package["trace_id"] == "trace-current"
    assert package["original_user_request"] == "investigate why the cleanup CronJob keeps pulling the wrong image"
    assert package["user_request"] == "package the exact durable fix"
    assert len(package["conversation_transcript"]) == 4
    assert package["conversation_transcript"][-1]["role"] == "assistant"
    assert len(package["snippets"]) == 1
    assert package["snippets"][0]["language"] == "yaml"
    assert len(package["recent_runs"]) == 3
    assert package["current_trace"]["trace_id"] == "trace-current"
    assert "Run 1" in package["recent_runs_summary"]
    assert "package the exact durable fix" in package["operator_need_summary"]
    assert any(item["display"] == "Apply CronJob image fix" for item in package["planned_changes"])
    assert any(item["status"] == "verification" and "kubectl get cronjob" in item["preview"] for item in package["commands_ran"])
    assert any(item["kind"] == "approval" for item in package["approvals_and_verification"])
    assert "Expected outcome: New Jobs pull the corrected image." in package["infra_change_brief"]
    assert "If a live change already happened" in package["system_prompt"]
    assert "Commands already run:" in package["task_prompt"]
    assert any("cronjob.batch/terminated-pod-gc configured" in item for item in package["evidence"])


class _FakeLLM:
    def __init__(self, content: str) -> None:
        self._content = content

    def invoke(self, _messages: list) -> AIMessage:
        return AIMessage(content=self._content)


def test_enrich_handoff_package_applies_llm_json_summary() -> None:
    package = build_handoff_package(
        _sample_messages(),
        3,
        trace=_sample_recent_traces()[0],
        recent_traces=_sample_recent_traces()[:2],
        model_name="openai/gpt-5.4-mini",
    )
    llm = _FakeLLM(
        """```json
        {
          "operator_need_summary": "Package the durable CronJob image fix and keep the prior verification path intact.",
          "whole_conversation_summary": "The operator traced an ImagePullBackOff to a bad CronJob image and now wants the durable fix packaged for a coding agent.",
          "recent_runs_summary": "Recent runs identified the bad image, proposed an apply-manifest fix, and verified the corrected image after execution.",
          "last_run_summary": "The latest run ended with a proposed CronJob image fix waiting on durable packaging.",
          "infra_change_brief": "Reflect the live CronJob image correction in source-controlled manifests or IaC and keep the existing verification command.",
          "takeover_guidance": [
            "Do not repeat the image RCA unless the repository conflicts with the trace evidence.",
            "Preserve the verified kubectl check as the primary validation step.",
            "Treat the live change as a signal that repo state may still need reconciliation."
          ]
        }
        ```"""
    )

    enriched = enrich_handoff_package(package, llm=llm, model_name="openai/gpt-5.4-mini")

    assert enriched["llm_enriched"] is True
    assert enriched["llm_model"] == "openai/gpt-5.4-mini"
    assert enriched["operator_need_summary"] == "Package the durable CronJob image fix and keep the prior verification path intact."
    assert "bad CronJob image" in enriched["whole_conversation_summary"]
    assert enriched["takeover_guidance"][0].startswith("Do not repeat")
    assert "Treat the live change as a signal" in enriched["task_prompt"]


def test_render_handoff_outputs_include_new_sections() -> None:
    package = build_handoff_package(
        _sample_messages(),
        3,
        trace=_sample_recent_traces()[0],
        recent_traces=_sample_recent_traces(),
        model_name="openai/gpt-5.4-mini",
    )

    agent_prompt = render_agent_handoff_prompt(package)
    codex_prompt = render_codex_handoff_prompt(package)
    markdown_bundle = render_handoff_markdown(package)

    assert "## System Prompt" in agent_prompt
    assert "## Commands Already Run" in agent_prompt
    assert "## Recent Trace Digests" in agent_prompt
    assert "## Acceptance Criteria" in agent_prompt
    assert codex_prompt.startswith("You are taking over from the AIOps assistant in Codex.")
    assert "## Operator Need Summary" in markdown_bundle
    assert "## Proposed Or Approved Change Steps" in markdown_bundle
    assert "## Recent Trace Digests" in markdown_bundle
    assert "## Full Conversation Transcript" in markdown_bundle
    assert "```yaml" in markdown_bundle
