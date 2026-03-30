"""Focused tests for the rebuilt case state."""

from __future__ import annotations

import unittest

from src.agents.state import CaseEntity, CaseFinding, CaseState, normalize_yes_no


class CaseStateTests(unittest.TestCase):
    def test_normalize_yes_no_handles_common_variants(self) -> None:
        self.assertEqual(normalize_yes_no("yes"), "yes")
        self.assertEqual(normalize_yes_no("Go ahead"), "yes")
        self.assertEqual(normalize_yes_no("don't"), "no")
        self.assertEqual(normalize_yes_no("maybe later"), "")

    def test_case_state_merges_entities_and_findings(self) -> None:
        case = CaseState.create(
            goal="restore workloads",
            desired_outcome="workloads are healthy",
            profile="restore_workloads",
            domains=["k8s", "aws"],
            initial_message="make workloads run",
        )

        case.merge_entity(CaseEntity(kind="deployment", name="web", namespace="gitlab", attrs={"ready": 1}))
        case.merge_entity(CaseEntity(kind="deployment", name="web", namespace="gitlab", attrs={"desired": 3}))
        case.merge_finding(CaseFinding(claim="web deployment is short on replicas", confidence=65, verified=False))
        case.merge_finding(CaseFinding(claim="web deployment is short on replicas", confidence=85, verified=True))

        self.assertEqual(len(case.entities), 1)
        self.assertEqual(case.entities[0].attrs["ready"], 1)
        self.assertEqual(case.entities[0].attrs["desired"], 3)
        self.assertEqual(len(case.findings), 1)
        self.assertEqual(case.findings[0].confidence, 85)
        self.assertTrue(case.findings[0].verified)


if __name__ == "__main__":
    unittest.main()
