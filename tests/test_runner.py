"""Smoke tests for the rebuilt investigation runner."""

from __future__ import annotations

import json
import unittest

from src.agents.runner import InvestigationRunner
from src.agents.state import OperatorIntentState, TurnContext
from src.tools.common import ToolObservation


class _FakeResponse:
    def __init__(self, content: str):
        self.content = content


class _QueuedLLM:
    def __init__(self, responses: list[str]):
        self._responses = list(responses)

    def invoke(self, messages):  # noqa: ANN001
        if not self._responses:
            raise AssertionError("LLM was invoked more times than expected")
        return _FakeResponse(self._responses.pop(0))


class _FakeK8s:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def cluster_overview(self) -> ToolObservation:
        self.calls.append("cluster_overview")
        return ToolObservation(
            family="k8s",
            action="cluster_overview",
            summary="Cluster has 0 ready nodes and 12 pending pods.",
            structured={"nodes": {"ready_count": 0, "node_count": 0}, "pods": {"problem_pods": [{"name": "api-123", "phase": "Pending"}]}},
            commands=["kubectl get nodes -o json", "kubectl get pods -A -o json"],
            raw_preview="0 ready nodes; pending pods present",
        )

    def raw_read(self, command: str) -> ToolObservation:
        self.calls.append(f"raw_read:{command}")
        raise RuntimeError("not read-only")

    def execute(self, command: str) -> ToolObservation:
        self.calls.append(f"execute:{command}")
        return ToolObservation(
            family="k8s",
            action="raw_write",
            summary=f"Executed `{command}`.",
            structured={"command": command, "changed": True},
            commands=[command],
            raw_preview="scaled deployment",
        )


class _FakeAWS:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def compute_backing_overview(self) -> ToolObservation:
        self.calls.append("compute_backing_overview")
        return ToolObservation(
            family="aws",
            action="compute_backing_overview",
            summary="ASG desired capacity is 0 and no worker instances are running.",
            structured={"asg": {"groups": [{"name": "workers", "desired": 0}]}, "ec2": {"instances": []}},
            commands=["aws autoscaling describe-auto-scaling-groups", "aws ec2 describe-instances"],
            raw_preview="desired=0; no worker instances",
        )

    def raw_read(self, command: str, all_regions: bool = True) -> ToolObservation:  # noqa: ARG002
        self.calls.append(f"raw_read:{command}")
        return ToolObservation(
            family="aws",
            action="raw_read",
            summary=f"Executed `{command}`.",
            structured={"command": command},
            commands=[command],
            raw_preview="ok",
        )


class _FakeHelm:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def release_overview(self, all_namespaces: bool = True) -> ToolObservation:  # noqa: ARG002
        self.calls.append("release_overview")
        return ToolObservation(
            family="helm",
            action="release_overview",
            summary="All Helm releases are deployed.",
            structured={"releases": [{"name": "gitlab", "status": "deployed"}]},
            commands=["helm list -A -o json"],
            raw_preview="gitlab deployed",
        )

    def raw_read(self, command: str) -> ToolObservation:
        self.calls.append(f"raw_read:{command}")
        return ToolObservation(
            family="helm",
            action="raw_read",
            summary=f"Executed `{command}`.",
            structured={"command": command},
            commands=[command],
            raw_preview="ok",
        )


class _Connectors:
    def __init__(self) -> None:
        self.kubernetes = _FakeK8s()
        self.aws = _FakeAWS()
        self.helm = _FakeHelm()


class InvestigationRunnerTests(unittest.TestCase):
    def test_runner_can_complete_multi_step_case_without_follow_up(self) -> None:
        llm = _QueuedLLM(
            [
                json.dumps(
                    {
                        "decision": "start_new_case",
                        "goal": "Restore workload scheduling",
                        "desired_outcome": "Pods can run again",
                        "profile": "restore_workloads",
                        "domains": ["k8s", "aws", "helm"],
                        "notes": [],
                    }
                ),
                json.dumps(
                    {
                        "assistant_status": "Checking cluster scheduling and cloud backing.",
                        "phase": "observe",
                        "working_summary": "Workloads are not scheduling because compute backing may be missing.",
                        "hypotheses": ["Worker capacity may be absent."],
                        "gaps": ["Need to confirm cluster and cloud backing state."],
                        "actions": [
                            {"family": "k8s", "mode": "read", "action": "cluster_overview", "params": {}, "reason": "Check cluster health", "expected_outcome": "Node and pod state"},
                            {"family": "aws", "mode": "read", "action": "compute_backing_overview", "params": {}, "reason": "Check compute backing", "expected_outcome": "ASG and EC2 state"},
                        ],
                        "stop": False,
                        "stop_reason": "",
                        "answer": "",
                        "approval_summary": "",
                    }
                ),
                json.dumps(
                    {
                        "summary": "The cluster has no ready worker capacity because the worker ASG is scaled to zero.",
                        "phase": "verify",
                        "entities": [
                            {"kind": "autoscaling_group", "name": "workers", "namespace": "", "scope": "aws", "provider_id": "", "attrs": {"desired": 0}},
                            {"kind": "cluster", "name": "default", "namespace": "", "scope": "k8s", "provider_id": "", "attrs": {"ready_nodes": 0}},
                        ],
                        "findings": [
                            {"claim": "The Kubernetes cluster has no ready nodes.", "confidence": 95, "verified": True, "entity_refs": ["cluster::default"]},
                            {"claim": "The worker Auto Scaling Group is configured with desired capacity 0.", "confidence": 95, "verified": True, "entity_refs": ["autoscaling_group::workers"]},
                        ],
                        "hypotheses": ["Scaling the worker group back up should restore scheduling."],
                        "gaps": [],
                    }
                ),
                json.dumps(
                    {
                        "assistant_status": "I have enough evidence to answer.",
                        "phase": "synthesize",
                        "working_summary": "The worker ASG is scaled to zero, leaving the cluster with no nodes.",
                        "hypotheses": [],
                        "gaps": [],
                        "actions": [],
                        "stop": True,
                        "stop_reason": "enough_evidence",
                        "answer": "**Bottom Line:** The cluster cannot run workloads because its worker Auto Scaling Group is at desired capacity 0.\n\n**What I Found:**\n- Kubernetes has no ready nodes.\n- AWS shows the worker group desired capacity is 0.\n\n**Recommended Next Step:** Scale the worker group back up or approve a change to do that.",
                        "approval_summary": "",
                    }
                ),
            ]
        )
        connectors = _Connectors()
        runner = InvestigationRunner(llm=llm, connectors=connectors, system_prompt="system prompt")

        result = runner.run_turn(
            context=TurnContext(user_input="make my workloads run", chat_history=[], trace_id="trace-1"),
            case=None,
            operator_state=OperatorIntentState(),
        )

        self.assertIn("worker Auto Scaling Group", result.response_text)
        self.assertEqual(connectors.kubernetes.calls, ["cluster_overview"])
        self.assertEqual(connectors.aws.calls, ["compute_backing_overview"])
        self.assertEqual(result.case_state.profile, "restore_workloads")
        self.assertFalse(result.operator_intent_state.awaiting_follow_up)

    def test_direct_write_command_requires_approval_then_executes(self) -> None:
        llm = _QueuedLLM(
            [
                json.dumps(
                    {
                        "summary": "Scaled the deployment after approval.",
                        "phase": "remediate",
                        "entities": [{"kind": "deployment", "name": "api", "namespace": "default", "scope": "k8s", "provider_id": "", "attrs": {"replicas": 3}}],
                        "findings": [{"claim": "The requested kubectl scale command executed successfully.", "confidence": 90, "verified": True, "entity_refs": ["deployment:default:api"]}],
                        "hypotheses": [],
                        "gaps": [],
                    }
                ),
                "**Bottom Line:** The requested scale operation completed.\n\n**What I Found:**\n- The approved kubectl scale command executed successfully.\n\n**Recommended Next Step:** Verify the deployment reaches the desired ready replica count.",
            ]
        )
        connectors = _Connectors()
        runner = InvestigationRunner(llm=llm, connectors=connectors, system_prompt="system prompt")
        operator_state = OperatorIntentState()

        first = runner.run_turn(
            context=TurnContext(
                user_input="kubectl scale deployment api --replicas=3 -n default",
                chat_history=[],
                trace_id="trace-2",
            ),
            case=None,
            operator_state=operator_state,
        )
        self.assertIn("Would you like me to proceed", first.response_text)
        self.assertTrue(first.operator_intent_state.awaiting_follow_up)

        second = runner.run_turn(
            context=TurnContext(user_input="yes", chat_history=[], trace_id="trace-3"),
            case=first.case_state,
            operator_state=first.operator_intent_state,
        )
        self.assertIn("scale operation completed", second.response_text)
        self.assertIn("execute:kubectl scale deployment api --replicas=3 -n default", connectors.kubernetes.calls)
        self.assertFalse(second.operator_intent_state.awaiting_follow_up)


if __name__ == "__main__":
    unittest.main()
