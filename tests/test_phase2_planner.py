"""Regression tests for Phase 2 planner/state behavior."""

from __future__ import annotations

import unittest

from src.agents.planner import build_turn_plan, render_turn_plan_directive
from src.agents.state import IncidentState, OperatorIntentState, ToolExecutionRecord, ToolLoopOutcome, apply_turn_outcome_to_state
from src.agents.tool_loop import (
    _missing_requested_aspects,
    _plan_has_enough_evidence,
    _response_requests_readonly_confirmation,
    _semantic_tool_signature,
)
from src.utils.query_intent import QueryIntent


class PlannerTests(unittest.TestCase):
    def test_workload_inventory_request_gets_workload_focus_and_collect_stage(self) -> None:
        plan = build_turn_plan(
            user_input="list all workloads in my cluster",
            intent=QueryIntent(mode="general"),
            chat_history=[],
            incident_state=IncidentState(),
            operator_intent_state=OperatorIntentState(),
        )

        self.assertEqual(plan.focus, "workload")
        self.assertEqual(plan.stage, "collect")
        self.assertIn("inventory", plan.requested_aspects)
        self.assertIn("workload_health", plan.required_categories)
        self.assertEqual(len(plan.objectives), 1)
        self.assertEqual(plan.objectives[0].key, "inventory")
        self.assertEqual(plan.objectives[0].focus, "workload")

    def test_capacity_and_optimization_question_tracks_multiple_aspects(self) -> None:
        plan = build_turn_plan(
            user_input="what ec2 instances is my cluster running and how much cpu/ram do they have in total, and is this optimized for my workloads?",
            intent=QueryIntent(mode="general"),
            chat_history=[],
            incident_state=IncidentState(),
            operator_intent_state=OperatorIntentState(),
        )

        self.assertEqual(plan.focus, "node")
        self.assertIn("capacity", plan.requested_aspects)
        self.assertIn("optimization", plan.requested_aspects)
        self.assertIn("node_health", plan.required_categories)
        self.assertIn("pod_health", plan.required_categories)
        objective_keys = [objective.key for objective in plan.objectives]
        self.assertEqual(objective_keys, ["inventory", "capacity", "optimization"])
        self.assertEqual(plan.objectives[0].focus, "aws")
        self.assertEqual(plan.objectives[1].focus, "node")
        self.assertEqual(plan.objectives[2].focus, "node")

    def test_follow_up_verification_continues_existing_incident(self) -> None:
        state = IncidentState(
            active=True,
            namespace="gitlab",
            services=["webservice"],
            pods=["gitlab-webservice-7cdb9f"],
            last_focus="service",
            summary="GitLab webservice was unreachable through ingress.",
        )

        plan = build_turn_plan(
            user_input="is it healthy now?",
            intent=QueryIntent(mode="general"),
            chat_history=[],
            incident_state=state,
            operator_intent_state=OperatorIntentState(),
        )

        self.assertTrue(plan.continue_existing)
        self.assertEqual(plan.stage, "verify")
        self.assertTrue(plan.prefer_fresh_reads)
        self.assertFalse(plan.allow_broad_discovery)
        self.assertEqual(plan.focus, "service")
        self.assertIn("service_network", plan.required_categories)

    def test_follow_up_collect_prefers_cached_reads(self) -> None:
        state = IncidentState(
            active=True,
            namespace="gitlab",
            services=["webservice"],
            pods=["gitlab-webservice-7cdb9f"],
            last_focus="service",
        )

        plan = build_turn_plan(
            user_input="what else do we know about webservice?",
            intent=QueryIntent(mode="general"),
            chat_history=[],
            incident_state=state,
            operator_intent_state=OperatorIntentState(),
        )

        self.assertTrue(plan.continue_existing)
        self.assertEqual(plan.stage, "collect")
        self.assertTrue(plan.prefer_cached_reads)
        self.assertFalse(plan.prefer_fresh_reads)
        self.assertFalse(plan.allow_broad_discovery)

    def test_new_incident_resets_existing_context_when_scope_changes(self) -> None:
        state = IncidentState(
            active=True,
            namespace="gitlab",
            services=["webservice"],
            pods=["gitlab-webservice-7cdb9f"],
            last_focus="service",
        )

        plan = build_turn_plan(
            user_input="diagnose why the payments service is timing out",
            intent=QueryIntent(mode="incident_rca"),
            chat_history=[],
            incident_state=state,
            operator_intent_state=OperatorIntentState(),
        )

        self.assertFalse(plan.continue_existing)
        self.assertTrue(plan.reset_existing_context)
        self.assertEqual(plan.focus, "service")

    def test_render_turn_plan_directive_includes_scope_and_evidence(self) -> None:
        state = IncidentState(
            active=True,
            namespace="gitlab",
            services=["webservice"],
            pods=["gitlab-webservice-7cdb9f"],
            evidence_notes=["k8s_describe_pod showed repeated readiness probe failures."],
            summary="GitLab webservice is failing readiness checks.",
        )
        plan = build_turn_plan(
            user_input="check again",
            intent=QueryIntent(mode="general"),
            chat_history=[],
            incident_state=state,
            operator_intent_state=OperatorIntentState(),
        )

        rendered = render_turn_plan_directive(
            turn_plan=plan,
            incident_state=state,
            operator_intent_state=OperatorIntentState(),
        )
        self.assertIn("Known namespace: gitlab", rendered)
        self.assertIn("Known services: webservice", rendered)
        self.assertIn("Prior evidence", rendered)
        self.assertIn("Target evidence categories", rendered)
        self.assertIn("Objective", rendered)


class ToolLoopPlannerTests(unittest.TestCase):
    def test_semantic_signature_ignores_limit_for_pod_listing(self) -> None:
        sig_a = _semantic_tool_signature("k8s_list_pods", {"namespace": "all", "limit": 100})
        sig_b = _semantic_tool_signature("k8s_list_pods", {"namespace": "all", "limit": 250})
        self.assertEqual(sig_a, sig_b)

    def test_plan_completion_requires_fresh_verification_evidence(self) -> None:
        state = IncidentState(
            active=True,
            namespace="gitlab",
            services=["webservice"],
            last_focus="service",
        )
        plan = build_turn_plan(
            user_input="is it healthy now?",
            intent=QueryIntent(mode="general"),
            chat_history=[],
            incident_state=state,
            operator_intent_state=OperatorIntentState(),
        )

        cached_only_records = [
            ToolExecutionRecord(
                tool_name="k8s_list_services",
                requested_tool="k8s_list_services",
                args={"namespace": "gitlab"},
                semantic_key="svc:list",
                success=True,
                from_cache=True,
                summary="Cached service state",
                evidence_categories=("service_network",),
            ),
            ToolExecutionRecord(
                tool_name="k8s_describe_pod",
                requested_tool="k8s_describe_pod",
                args={"namespace": "gitlab", "pod_name": "gitlab-webservice-7cdb9f"},
                semantic_key="pod:describe",
                success=True,
                from_cache=True,
                summary="Cached pod state",
                evidence_categories=("pod_health",),
            ),
        ]
        self.assertFalse(_plan_has_enough_evidence(plan, cached_only_records))

        fresh_records = [
            ToolExecutionRecord(
                tool_name="k8s_list_services",
                requested_tool="k8s_list_services",
                args={"namespace": "gitlab"},
                semantic_key="svc:list",
                success=True,
                summary="Service still points at ready endpoints.",
                evidence_categories=("service_network",),
            ),
            ToolExecutionRecord(
                tool_name="k8s_describe_pod",
                requested_tool="k8s_describe_pod",
                args={"namespace": "gitlab", "pod_name": "gitlab-webservice-7cdb9f"},
                semantic_key="pod:describe",
                success=True,
                summary="Pod is Ready with no recent restarts.",
                evidence_categories=("pod_health",),
            ),
        ]
        self.assertTrue(_plan_has_enough_evidence(plan, fresh_records))

    def test_capacity_optimization_plan_needs_more_than_two_successful_reads(self) -> None:
        plan = build_turn_plan(
            user_input="what ec2 instances is my cluster running and how much cpu/ram do they have in total, and is this optimized for my workloads?",
            intent=QueryIntent(mode="general"),
            chat_history=[],
            incident_state=IncidentState(),
            operator_intent_state=OperatorIntentState(),
        )

        partial_records = [
            ToolExecutionRecord(
                tool_name="aws_cli_readonly",
                requested_tool="aws_cli_readonly",
                args={"command": "ec2 describe-instances"},
                semantic_key="aws:instances",
                success=True,
                summary="Found instance types for worker nodes.",
                evidence_categories=("aws", "node_health"),
            ),
            ToolExecutionRecord(
                tool_name="k8s_list_nodes",
                requested_tool="k8s_list_nodes",
                args={},
                semantic_key="k8s:nodes",
                success=True,
                summary="Listed cluster nodes and allocatable resources.",
                evidence_categories=("node_health",),
            ),
        ]
        self.assertFalse(_plan_has_enough_evidence(plan, partial_records))

        complete_records = partial_records + [
            ToolExecutionRecord(
                tool_name="k8s_list_pods",
                requested_tool="k8s_list_pods",
                args={"namespace": "all"},
                semantic_key="k8s:pods",
                success=True,
                summary="Listed running workloads and requests for optimization review.",
                evidence_categories=("pod_health",),
            ),
        ]
        self.assertTrue(_plan_has_enough_evidence(plan, complete_records))

    def test_readonly_confirmation_prompt_is_detected(self) -> None:
        text = "Would you like me to proceed with these read-only diagnostic checks? (yes/no)"
        self.assertTrue(_response_requests_readonly_confirmation(text))

    def test_missing_requested_aspects_flags_partial_capacity_answer(self) -> None:
        plan = build_turn_plan(
            user_input="what ec2 instances is my cluster running and how much cpu/ram do they have in total, and is this optimized for my workloads?",
            intent=QueryIntent(mode="general"),
            chat_history=[],
            incident_state=IncidentState(),
            operator_intent_state=OperatorIntentState(),
        )

        missing = _missing_requested_aspects(
            plan,
            "The cluster uses t3.large and t3.medium instances. I can list them for you.",
        )
        self.assertIn("capacity", missing)
        self.assertIn("optimization", missing)


class IncidentStateTests(unittest.TestCase):
    def test_state_update_merges_scope_and_caches_tool_results(self) -> None:
        state = IncidentState()
        plan = build_turn_plan(
            user_input="diagnose why service webservice is unreachable in namespace gitlab",
            intent=QueryIntent(mode="incident_rca", namespace="gitlab"),
            chat_history=[],
            incident_state=state,
            operator_intent_state=OperatorIntentState(),
        )

        records = [
            ToolExecutionRecord(
                tool_name="k8s_find_pods",
                requested_tool="k8s_find_pods",
                args={"name_contains": "webservice", "namespace": "all"},
                semantic_key="find:webservice",
                success=True,
                result_excerpt="gitlab webservice-abc123",
                summary="Found pod webservice-abc123 in namespace gitlab.",
                evidence_categories=("discovery_cluster", "pod_health"),
            ),
            ToolExecutionRecord(
                tool_name="k8s_describe_pod",
                requested_tool="k8s_describe_pod",
                args={"pod_name": "webservice-abc123", "namespace": "gitlab"},
                semantic_key="describe:webservice-abc123",
                success=True,
                result_excerpt="Warning  Unhealthy  readiness probe failed",
                summary="Pod webservice-abc123 is failing readiness probes in namespace gitlab.",
                evidence_categories=("pod_health", "events"),
            ),
        ]
        outcome = ToolLoopOutcome(
            final_text=(
                "**Issue Summary:** webservice is failing readiness probes\n"
                "**Severity:** P1\n"
                "**Confidence Score:** 87"
            ),
            records=records,
        )

        updated = apply_turn_outcome_to_state(
            incident_state=state,
            user_input="diagnose why service webservice is unreachable in namespace gitlab",
            intent_mode="incident_rca",
            turn_plan=plan,
            outcome=outcome,
            final_text=outcome.final_text,
            turn_index=3,
        )

        self.assertTrue(updated.active)
        self.assertEqual(updated.namespace, "gitlab")
        self.assertIn("webservice-abc123", updated.pods)
        self.assertIn("webservice", updated.services)
        self.assertEqual(updated.summary, "webservice is failing readiness probes")
        self.assertEqual(updated.severity, "P1")
        self.assertEqual(updated.confidence_score, 87)
        self.assertIn("find:webservice", updated.cached_tool_results)
        self.assertIn("describe:webservice-abc123", updated.cached_tool_results)
        self.assertTrue(updated.evidence_notes)


if __name__ == "__main__":
    unittest.main()
