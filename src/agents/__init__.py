"""Agent package — exposes the main AIOps orchestrator."""

from .orchestrator import AIOpsAgent

# Backward-compatible alias used by existing session.py
LogAnalyzerAgent = AIOpsAgent

__all__ = ["AIOpsAgent", "LogAnalyzerAgent"]
