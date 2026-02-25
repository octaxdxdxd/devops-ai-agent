"""Autonomy scan formatting helpers."""

from __future__ import annotations


def format_autonomous_scan(scan: dict) -> str:
    """Render autonomous scan output in markdown."""
    if not scan.get("ok"):
        return f"Autonomous scan failed: {scan.get('error', 'unknown error')}"

    incident = scan.get("incident", {}) or {}
    notifications = scan.get("notifications", {}) or {}
    lines = [
        "**Autonomous Alert Scan**",
        f"- Severity: {incident.get('severity', 'unknown')}",
        f"- Confidence: {incident.get('confidence_score', 0)}/100",
        f"- Impact: {incident.get('impact_score', 0)}/100",
        f"- Should alert: {incident.get('should_alert', False)}",
        f"- Summary: {incident.get('issue_summary', '(none)')}",
    ]

    anomalies = incident.get("anomalies", []) or []
    if anomalies:
        lines.append("- Anomalies:")
        for item in anomalies[:5]:
            lines.append(f"  • {item}")

    evidence = incident.get("evidence", []) or []
    if evidence:
        lines.append("- Evidence:")
        for item in evidence[:5]:
            lines.append(f"  • {item}")

    if notifications:
        lines.append(f"- Notifications: {notifications}")

    return "\n".join(lines)


def notification_was_sent(notifications: dict) -> bool:
    """Return True if at least one notification channel reported success."""
    if not isinstance(notifications, dict) or not notifications:
        return False
    return any(str(status).strip().lower() == "ok" for status in notifications.values())
