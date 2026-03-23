"""Regression tests for Phase 2 planner/state behavior."""

from __future__ import annotations

import unittest

from src.agents.planner import build_turn_plan, render_turn_plan_directive
from src.agents.state import IncidentState, ToolExecutionRecord, ToolLoopOutcome, apply_turn_outcome_to_state
from src.agents.tool_loop import _plan_has_enough_evidence, _semantic_tool_signature
from src.utils.query_intent import QueryIntent


class PlannerTests(unittest.TestCase):
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
        )

        rendered = render_turn_plan_directive(turn_plan=plan, incident_state=state)
        self.assertIn("Known namespace: gitlab", rendered)
        self.assertIn("Known services: webservice", rendered)
        self.assertIn("Prior evidence", rendered)
        self.assertIn("Target evidence categories", rendered)


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


class IncidentStateTests(unittest.TestCase):
    def test_state_update_merges_scope_and_caches_tool_results(self) -> None:
        state = IncidentState()
        plan = build_turn_plan(
            user_input="diagnose why service webservice is unreachable in namespace gitlab",
            intent=QueryIntent(mode="incident_rca", namespace="gitlab"),
            chat_history=[],
            incident_state=state,
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
