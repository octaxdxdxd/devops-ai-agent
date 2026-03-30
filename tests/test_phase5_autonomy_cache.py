"""Regression tests for cached/background autonomy scan behavior."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest

from src.agents.approval import ApprovalCoordinator
from src.agents.log_analyzer import LogAnalyzerAgent
from src.agents.state import OperatorIntentState
from src.config import Config
from src.ui.session import autonomy_scan_is_due


class _DummyAutonomy:
    def __init__(self, result: dict):
        self.result = result
        self.calls: list[tuple[str | None, bool]] = []

    def run_scan(self, namespace: str | None = None, *, send_notifications: bool = True) -> dict:
        self.calls.append((namespace, send_notifications))
        return dict(self.result)


class CachedAutonomyScanTests(unittest.TestCase):
    def test_run_autonomous_scan_caches_completed_timestamp(self) -> None:
        dummy_agent = type("DummyAgent", (), {})()
        dummy_agent._autonomy = _DummyAutonomy(
            {
                "ok": True,
                "incident": {
                    "should_alert": True,
                    "fingerprint": "incident-1",
                    "severity": "P2",
                    "confidence_score": 82,
                    "impact_score": 71,
                    "issue_summary": "GitLab webservice is degraded.",
                },
                "notifications": {"slack": "ok"},
            }
        )
        dummy_agent.last_autonomous_scan = None
        dummy_agent.capture_autonomous_scan = LogAnalyzerAgent.capture_autonomous_scan.__get__(
            dummy_agent,
            type(dummy_agent),
        )
        dummy_agent._cache_autonomous_scan = LogAnalyzerAgent._cache_autonomous_scan.__get__(
            dummy_agent,
            type(dummy_agent),
        )
        dummy_agent.run_autonomous_scan = LogAnalyzerAgent.run_autonomous_scan.__get__(
            dummy_agent,
            type(dummy_agent),
        )

        result = dummy_agent.run_autonomous_scan(send_notifications=False)

        self.assertEqual(dummy_agent._autonomy.calls, [(None, False)])
        self.assertIn("completed_at", result)
        self.assertEqual(dummy_agent.last_autonomous_scan, result)

    def test_get_cached_autonomous_scan_respects_max_age(self) -> None:
        now = datetime.now(timezone.utc)
        dummy_agent = type("DummyAgent", (), {})()
        dummy_agent.last_autonomous_scan = {
            "ok": True,
            "completed_at": (now - timedelta(seconds=45)).isoformat().replace("+00:00", "Z"),
            "incident": {"should_alert": False},
            "notifications": {},
        }
        dummy_agent.get_cached_autonomous_scan = LogAnalyzerAgent.get_cached_autonomous_scan.__get__(
            dummy_agent,
            type(dummy_agent),
        )

        fresh = dummy_agent.get_cached_autonomous_scan(max_age_sec=60)
        stale = dummy_agent.get_cached_autonomous_scan(max_age_sec=10)

        self.assertIsNotNone(fresh)
        self.assertIsNone(stale)

    def test_build_alert_prefix_uses_cached_scan_without_triggering_new_scan(self) -> None:
        original_enabled = Config.AUTONOMY_SCAN_ON_USER_TURN
        try:
            Config.AUTONOMY_SCAN_ON_USER_TURN = True
            now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            dummy_agent = type("DummyAgent", (), {})()
            dummy_agent._autonomy = object()
            dummy_agent._approval = ApprovalCoordinator()
            dummy_agent._last_announced_incident_fingerprint = None
            dummy_agent.last_autonomous_scan = {
                "ok": True,
                "completed_at": now,
                "incident": {
                    "should_alert": True,
                    "fingerprint": "incident-42",
                    "severity": "P1",
                    "confidence_score": 96,
                    "impact_score": 91,
                    "issue_summary": "The GitLab ingress is failing health checks.",
                },
                "notifications": {"slack": "ok"},
            }
            dummy_agent.get_cached_autonomous_scan = LogAnalyzerAgent.get_cached_autonomous_scan.__get__(
                dummy_agent,
                type(dummy_agent),
            )
            dummy_agent._build_alert_prefix = LogAnalyzerAgent._build_alert_prefix.__get__(
                dummy_agent,
                type(dummy_agent),
            )

            def fail_run_scan(*_args, **_kwargs):
                raise AssertionError("user turns should not synchronously run autonomy scans")

            dummy_agent.run_autonomous_scan = fail_run_scan

            prefix = dummy_agent._build_alert_prefix(OperatorIntentState())
            repeated = dummy_agent._build_alert_prefix(OperatorIntentState())

            self.assertIn("Autonomous alert monitor had already detected an incident", prefix)
            self.assertIn("Observed at:", prefix)
            self.assertEqual(repeated, "")
        finally:
            Config.AUTONOMY_SCAN_ON_USER_TURN = original_enabled


class AutonomyScanSchedulingTests(unittest.TestCase):
    def test_autonomy_scan_is_due_for_missing_or_stale_cache(self) -> None:
        original_enabled = Config.AUTONOMY_ENABLED
        original_scan_on_turn = Config.AUTONOMY_SCAN_ON_USER_TURN
        original_interval = Config.AUTONOMY_BACKGROUND_SCAN_INTERVAL_SEC
        try:
            Config.AUTONOMY_ENABLED = True
            Config.AUTONOMY_SCAN_ON_USER_TURN = True
            Config.AUTONOMY_BACKGROUND_SCAN_INTERVAL_SEC = 120

            now = datetime(2026, 3, 25, 12, 0, tzinfo=timezone.utc)
            fresh_scan = {
                "completed_at": (now - timedelta(seconds=30)).isoformat().replace("+00:00", "Z"),
            }
            stale_scan = {
                "completed_at": (now - timedelta(seconds=180)).isoformat().replace("+00:00", "Z"),
            }

            self.assertTrue(autonomy_scan_is_due(None, now=now))
            self.assertFalse(autonomy_scan_is_due(fresh_scan, now=now))
            self.assertTrue(autonomy_scan_is_due(stale_scan, now=now))
        finally:
            Config.AUTONOMY_ENABLED = original_enabled
            Config.AUTONOMY_SCAN_ON_USER_TURN = original_scan_on_turn
            Config.AUTONOMY_BACKGROUND_SCAN_INTERVAL_SEC = original_interval


if __name__ == "__main__":
    unittest.main()
