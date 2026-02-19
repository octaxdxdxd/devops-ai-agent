"""Simple, production-oriented Kubernetes autonomous alerting.

Design goals:
- Prefer hard health signals over noisy heuristics.
- Keep logic explicit and auditable.
- Produce clear evidence with concrete affected resources.
"""

from __future__ import annotations

import hashlib
import json
import smtplib
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any

import requests

from ..config import Config
from ..tools.k8s_common import ensure_kubectl_installed, kubectl_base_args, run_kubectl


@dataclass
class AlertIncident:
    detected_at: str
    namespace_scope: str
    severity: str
    confidence_score: int
    impact_score: int
    should_alert: bool
    fingerprint: str
    issue_summary: str
    evidence: list[str]
    anomalies: list[str]
    impacted_resources: list[str]
    recommended_action: str


class StateStore:
    def __init__(self, state_file: str):
        self.path = Path(state_file)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"fingerprints": {}, "active_incidents": {}}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {"fingerprints": {}, "active_incidents": {}}

    def _save(self, value: dict[str, Any]) -> None:
        self.path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

    def should_send(self, fingerprint: str, cooldown_minutes: int) -> bool:
        state = self._load()
        ts_raw = (state.get("fingerprints") or {}).get(fingerprint)
        if not ts_raw:
            return True
        try:
            ts = datetime.fromisoformat(str(ts_raw))
        except ValueError:
            return True
        return datetime.now(timezone.utc) - ts >= timedelta(minutes=max(1, cooldown_minutes))

    def mark_sent(self, fingerprint: str) -> None:
        state = self._load()
        state.setdefault("fingerprints", {})[fingerprint] = datetime.now(timezone.utc).isoformat()
        self._save(state)

    @staticmethod
    def _severity_rank(severity: str) -> int:
        ranks = {"info": 0, "P3": 1, "P2": 2, "P1": 3}
        return ranks.get(str(severity), 0)

    def should_send_active_incident(
        self,
        scope: str,
        fingerprint: str,
        severity: str,
        *,
        repeat_minutes: int,
    ) -> bool:
        state = self._load()
        active = state.setdefault("active_incidents", {})
        record = active.get(scope)
        if not record:
            return True

        prev_fp = str(record.get("fingerprint") or "")
        prev_sev = str(record.get("severity") or "info")
        last_sent_raw = str(record.get("last_sent_at") or "")

        if prev_fp != fingerprint:
            return True
        if self._severity_rank(severity) > self._severity_rank(prev_sev):
            return True

        try:
            last_sent = datetime.fromisoformat(last_sent_raw)
        except ValueError:
            return True

        return datetime.now(timezone.utc) - last_sent >= timedelta(minutes=max(1, repeat_minutes))

    def mark_active_incident_sent(self, scope: str, fingerprint: str, severity: str) -> None:
        state = self._load()
        active = state.setdefault("active_incidents", {})
        active[scope] = {
            "fingerprint": fingerprint,
            "severity": severity,
            "last_sent_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save(state)

    def clear_active_incident(self, scope: str) -> None:
        state = self._load()
        active = state.setdefault("active_incidents", {})
        if scope in active:
            del active[scope]
            self._save(state)


class NotificationRouter:
    def __init__(self):
        self.slack_webhook = Config.ALERT_SLACK_WEBHOOK
        self.teams_webhook = Config.ALERT_TEAMS_WEBHOOK

    def send(self, incident: AlertIncident) -> dict[str, str]:
        out: dict[str, str] = {}
        if self.slack_webhook:
            out["slack"] = self._post_webhook(self.slack_webhook, self._render_text(incident))
        if self.teams_webhook:
            out["teams"] = self._post_webhook(self.teams_webhook, self._render_text(incident))
        if Config.ALERT_EMAIL_ENABLED:
            out["email"] = self._send_email(incident)
        if not out:
            out["none"] = "No channels configured"
        return out

    @staticmethod
    def _post_webhook(url: str, text: str) -> str:
        try:
            response = requests.post(url, json={"text": text}, timeout=10)
            return "ok" if 200 <= response.status_code < 300 else f"http_{response.status_code}"
        except Exception as exc:
            return f"error:{exc}"

    @staticmethod
    def _render_text(incident: AlertIncident) -> str:
        sev_emoji = {
            "P1": "🔴",
            "P2": "🟠",
            "P3": "🟡",
            "info": "🟢",
        }.get(incident.severity, "🚨")

        lines = [
            f"{sev_emoji} *Kubernetes automated incident alert*",
            f"*Severity:* {incident.severity}",
            f"*Confidence:* {incident.confidence_score}/100",
            f"*Impact:* {incident.impact_score}/100",
            f"*Summary:* {incident.issue_summary}",
        ]
        if incident.evidence:
            lines.append("*Evidence (active signals only):*")
            lines.extend([f"• {item}" for item in incident.evidence[:10]])
        if incident.impacted_resources:
            lines.append("*Impacted resources (sample):*")
            lines.extend([f"• {item}" for item in incident.impacted_resources[:8]])
        lines.append("*Recommended action:*")
        lines.append(f"• {incident.recommended_action}")
        return "\n".join(lines)

    @staticmethod
    def _send_email(incident: AlertIncident) -> str:
        recipients = [x.strip() for x in Config.ALERT_EMAIL_TO.split(",") if x.strip()]
        if not recipients:
            return "skipped:no_recipients"
        if not Config.ALERT_SMTP_HOST or not Config.ALERT_EMAIL_FROM:
            return "skipped:smtp_not_configured"

        message = EmailMessage()
        message["Subject"] = f"[{incident.severity}] Kubernetes incident"
        message["From"] = Config.ALERT_EMAIL_FROM
        message["To"] = ", ".join(recipients)
        message.set_content(NotificationRouter._render_text(incident))

        try:
            with smtplib.SMTP(Config.ALERT_SMTP_HOST, Config.ALERT_SMTP_PORT, timeout=20) as server:
                if Config.ALERT_SMTP_USE_TLS:
                    server.starttls()
                if Config.ALERT_SMTP_USER:
                    server.login(Config.ALERT_SMTP_USER, Config.ALERT_SMTP_PASSWORD)
                server.send_message(message)
            return "ok"
        except Exception as exc:
            return f"error:{exc}"


class SimpleAutonomyEngine:
    """Simple alert engine with hard Kubernetes-health triggers."""

    CRITICAL_EVENT_REASONS = {
        "FailedScheduling",
        "FailedMount",
        "Unhealthy",
        "NodeNotReady",
        "FailedAttachVolume",
        "Evicted",
        "BackOff",
        "ImagePullBackOff",
        "ErrImagePull",
    }

    def __init__(self):
        self.state = StateStore(Config.AUTONOMY_STATE_FILE)
        self.notifier = NotificationRouter()

    def run_scan(
        self,
        namespace: str | None = None,
        *,
        send_notifications: bool = True,
        workload_filter: str = "",
    ) -> dict[str, Any]:
        namespace_scope = (namespace or Config.AUTONOMY_NAMESPACE or "all").strip() or "all"
        if not ensure_kubectl_installed():
            return {"ok": False, "error": "kubectl not installed", "incident": None, "notifications": {}}

        snap = self._collect(namespace_scope)
        if not snap["ok"]:
            return {"ok": False, "error": snap["error"], "incident": None, "notifications": {}}

        incident = self._analyze(snap["data"], namespace_scope, workload_filter=workload_filter)
        notifications: dict[str, str] = {}
        if incident.should_alert and send_notifications:
            if self.state.should_send_active_incident(
                namespace_scope,
                incident.fingerprint,
                incident.severity,
                repeat_minutes=Config.ALERT_REPEAT_MINUTES,
            ) and self.state.should_send(incident.fingerprint, Config.ALERT_COOLDOWN_MINUTES):
                notifications = self.notifier.send(incident)
                self.state.mark_sent(incident.fingerprint)
                self.state.mark_active_incident_sent(namespace_scope, incident.fingerprint, incident.severity)
            else:
                notifications = {"suppressed": "active_incident_repeat"}
        elif not incident.should_alert:
            self.state.clear_active_incident(namespace_scope)

        return {
            "ok": True,
            "incident": asdict(incident),
            "notifications": notifications,
            "detected": incident.should_alert,
        }

    def _collect(self, namespace: str) -> dict[str, Any]:
        all_ns = namespace.lower() == "all"
        suffix = ["-A"] if all_ns else ["-n", namespace]
        data: dict[str, Any] = {}

        for key, args in [
            ("pods", ["get", "pods", *suffix, "-o", "json"]),
            ("deployments", ["get", "deployments", *suffix, "-o", "json"]),
            ("statefulsets", ["get", "statefulsets", *suffix, "-o", "json"]),
            ("events", ["get", "events", *suffix, "-o", "json"]),
        ]:
            code, out, err = run_kubectl(kubectl_base_args() + args)
            if code != 0:
                return {"ok": False, "error": err or out}
            try:
                data[key] = json.loads(out or "{}")
            except json.JSONDecodeError as exc:
                return {"ok": False, "error": f"Failed parsing {key}: {exc}"}

        data["collected_at"] = datetime.now(timezone.utc).isoformat()
        return {"ok": True, "data": data}

    def _analyze(self, data: dict[str, Any], namespace: str, *, workload_filter: str = "") -> AlertIncident:
        now = datetime.now(timezone.utc)
        lookback_minutes = max(5, Config.AUTONOMY_RECENT_MINUTES)
        cutoff = now - timedelta(minutes=lookback_minutes)
        pending_grace = max(3, Config.ALERT_PENDING_GRACE_MINUTES)

        def match(name: str, ns: str) -> bool:
            q = (workload_filter or "").strip().lower()
            if not q:
                return True
            return q in name.lower() or q in f"{ns}/{name}".lower()

        not_ready_pods: list[str] = []
        pending_pods_over_grace: list[str] = []
        crashloop_pods: list[str] = []
        oom_pods: list[str] = []

        for pod in data.get("pods", {}).get("items", []):
            meta = pod.get("metadata", {})
            status = pod.get("status", {})
            pod_ns = str(meta.get("namespace", ""))
            pod_name = str(meta.get("name", ""))
            if not match(pod_name, pod_ns):
                continue

            phase = str(status.get("phase", ""))
            pod_ref = f"{pod_ns}/{pod_name}"

            ready_status = "unknown"
            for cond in status.get("conditions") or []:
                if str(cond.get("type", "")) == "Ready":
                    ready_status = str(cond.get("status", "")).lower()
                    break

            if phase in {"Pending", "Unknown", "Failed"} or ready_status == "false":
                not_ready_pods.append(f"{pod_ref} (phase={phase}, ready={ready_status})")

            created_raw = str(meta.get("creationTimestamp", ""))
            created_dt = self._parse_ts(created_raw)
            age_minutes = int((now - created_dt).total_seconds() / 60) if created_dt else 0
            if phase == "Pending" and age_minutes >= pending_grace:
                pending_pods_over_grace.append(f"{pod_ref} (pending_for={age_minutes}m)")

            for cs in status.get("containerStatuses") or []:
                waiting_reason = str(((cs.get("state") or {}).get("waiting") or {}).get("reason") or "")
                term_reason = str((((cs.get("lastState") or {}).get("terminated") or {}).get("reason") or ""))
                if waiting_reason in {"CrashLoopBackOff", "ImagePullBackOff", "ErrImagePull"}:
                    crashloop_pods.append(f"{pod_ref}:{str(cs.get('name', 'container'))} ({waiting_reason})")
                if term_reason == "OOMKilled":
                    oom_pods.append(f"{pod_ref}:{str(cs.get('name', 'container'))}")

        unavailable_deployments: list[str] = []
        for dep in data.get("deployments", {}).get("items", []):
            meta = dep.get("metadata", {})
            spec = dep.get("spec", {})
            st = dep.get("status", {})
            ns = str(meta.get("namespace", ""))
            name = str(meta.get("name", ""))
            if not match(name, ns):
                continue
            desired = int(spec.get("replicas") or 1)
            available = int(st.get("availableReplicas") or 0)
            if desired > 0 and available < desired:
                unavailable_deployments.append(f"{ns}/{name} ({available}/{desired} available)")

        unavailable_statefulsets: list[str] = []
        for sts in data.get("statefulsets", {}).get("items", []):
            meta = sts.get("metadata", {})
            spec = sts.get("spec", {})
            st = sts.get("status", {})
            ns = str(meta.get("namespace", ""))
            name = str(meta.get("name", ""))
            if not match(name, ns):
                continue
            desired = int(spec.get("replicas") or 1)
            ready = int(st.get("readyReplicas") or 0)
            if desired > 0 and ready < desired:
                unavailable_statefulsets.append(f"{ns}/{name} ({ready}/{desired} ready)")

        critical_events: list[str] = []
        for ev in data.get("events", {}).get("items", []):
            meta = ev.get("metadata", {})
            involved = ev.get("involvedObject", {})
            ns = str(meta.get("namespace", ""))
            name = str(involved.get("name", ""))
            if not match(name, ns):
                continue
            ts = self._parse_ts(ev.get("lastTimestamp") or ev.get("eventTime") or meta.get("creationTimestamp"))
            if ts and ts < cutoff:
                continue
            reason = str(ev.get("reason", ""))
            if reason in self.CRITICAL_EVENT_REASONS:
                kind = str(involved.get("kind", ""))
                msg = str(ev.get("message", ""))[:120]
                critical_events.append(f"{ns}/{kind}/{name}: {reason} - {msg}")

        critical_hard_signal = bool(
            pending_pods_over_grace
            or crashloop_pods
            or oom_pods
            or unavailable_deployments
            or unavailable_statefulsets
            or len(critical_events) >= Config.ALERT_CRITICAL_EVENT_MIN_COUNT
        )

        impacted = sorted(
            set(
                [item.split(" (")[0] for item in unavailable_deployments]
                + [item.split(" (")[0] for item in unavailable_statefulsets]
                + [item.split(":")[0] for item in crashloop_pods]
                + [item.split(":")[0] for item in oom_pods]
                + [item.split(" (")[0] for item in pending_pods_over_grace]
            )
        )

        severity = self._severity(
            pending_pods_over_grace=pending_pods_over_grace,
            crashloop_pods=crashloop_pods,
            oom_pods=oom_pods,
            unavailable_deployments=unavailable_deployments,
            unavailable_statefulsets=unavailable_statefulsets,
            critical_events=critical_events,
        )
        impact_score = self._impact_score(
            unavailable_deployments=unavailable_deployments,
            unavailable_statefulsets=unavailable_statefulsets,
            pending_pods_over_grace=pending_pods_over_grace,
            crashloop_pods=crashloop_pods,
            oom_pods=oom_pods,
            critical_events=critical_events,
        )
        confidence = self._confidence_score(severity=severity, critical_hard_signal=critical_hard_signal)
        should_alert = critical_hard_signal and severity in {"P1", "P2"} and confidence >= Config.ALERT_MIN_CONFIDENCE

        evidence: list[str] = []
        if unavailable_deployments:
            evidence.append(f"Unavailable deployments: {len(unavailable_deployments)}")
            evidence.append(f"Unavailable deployment sample: {unavailable_deployments[:4]}")
        if unavailable_statefulsets:
            evidence.append(f"Unavailable statefulsets: {len(unavailable_statefulsets)}")
            evidence.append(f"Unavailable statefulset sample: {unavailable_statefulsets[:4]}")
        if pending_pods_over_grace:
            evidence.append(f"Pending pods over grace ({pending_grace}m): {len(pending_pods_over_grace)}")
            evidence.append(f"Pending pod sample: {pending_pods_over_grace[:4]}")
        if crashloop_pods:
            evidence.append(f"CrashLoop/ImagePull pods: {len(crashloop_pods)}")
            evidence.append(f"CrashLoop sample: {crashloop_pods[:4]}")
        if oom_pods:
            evidence.append(f"OOMKilled containers: {len(oom_pods)}")
            evidence.append(f"OOMKilled sample: {oom_pods[:4]}")
        if len(critical_events) >= Config.ALERT_CRITICAL_EVENT_MIN_COUNT:
            evidence.append(f"Critical events ({lookback_minutes}m): {len(critical_events)}")
            evidence.append(f"Critical event sample: {critical_events[:4]}")

        anomalies = []
        if not critical_hard_signal and not_ready_pods:
            anomalies.append("Not-ready pods detected but no hard critical signal thresholds met")

        summary = self._summary(severity, unavailable_deployments, unavailable_statefulsets, pending_pods_over_grace, crashloop_pods, oom_pods)
        recommended = self._recommendation(severity, unavailable_deployments, unavailable_statefulsets, pending_pods_over_grace, crashloop_pods, oom_pods)

        fingerprint = self._fingerprint(
            namespace=namespace,
            severity=severity,
            unavailable_deployments=unavailable_deployments,
            unavailable_statefulsets=unavailable_statefulsets,
            pending_pods_over_grace=pending_pods_over_grace,
            crashloop_pods=crashloop_pods,
            oom_pods=oom_pods,
            critical_events=critical_events,
        )

        return AlertIncident(
            detected_at=data.get("collected_at", datetime.now(timezone.utc).isoformat()),
            namespace_scope=namespace,
            severity=severity,
            confidence_score=confidence,
            impact_score=impact_score,
            should_alert=should_alert,
            fingerprint=fingerprint,
            issue_summary=summary,
            evidence=evidence,
            anomalies=anomalies,
            impacted_resources=impacted[:20],
            recommended_action=recommended,
        )

    @staticmethod
    def _severity(
        *,
        pending_pods_over_grace: list[str],
        crashloop_pods: list[str],
        oom_pods: list[str],
        unavailable_deployments: list[str],
        unavailable_statefulsets: list[str],
        critical_events: list[str],
    ) -> str:
        if len(unavailable_deployments) >= 2 or len(unavailable_statefulsets) >= 1:
            return "P1"
        if len(crashloop_pods) >= 2 or len(pending_pods_over_grace) >= 2:
            return "P1"
        if unavailable_deployments or pending_pods_over_grace or crashloop_pods or oom_pods:
            return "P2"
        if len(critical_events) >= Config.ALERT_CRITICAL_EVENT_MIN_COUNT:
            return "P2"
        return "info"

    @staticmethod
    def _impact_score(
        *,
        unavailable_deployments: list[str],
        unavailable_statefulsets: list[str],
        pending_pods_over_grace: list[str],
        crashloop_pods: list[str],
        oom_pods: list[str],
        critical_events: list[str],
    ) -> int:
        score = 0
        score += min(40, len(unavailable_deployments) * 15)
        score += min(30, len(unavailable_statefulsets) * 20)
        score += min(15, len(pending_pods_over_grace) * 5)
        score += min(20, len(crashloop_pods) * 5)
        score += min(10, len(oom_pods) * 3)
        score += min(10, len(critical_events) * 2)
        return max(1, min(100, score))

    @staticmethod
    def _confidence_score(*, severity: str, critical_hard_signal: bool) -> int:
        if severity == "P1":
            return 92 if critical_hard_signal else 70
        if severity == "P2":
            return 82 if critical_hard_signal else 65
        return 40

    @staticmethod
    def _summary(
        severity: str,
        unavailable_deployments: list[str],
        unavailable_statefulsets: list[str],
        pending_pods_over_grace: list[str],
        crashloop_pods: list[str],
        oom_pods: list[str],
    ) -> str:
        parts: list[str] = []
        if unavailable_deployments:
            parts.append(f"deployments_unavailable={len(unavailable_deployments)}")
        if unavailable_statefulsets:
            parts.append(f"statefulsets_unavailable={len(unavailable_statefulsets)}")
        if pending_pods_over_grace:
            parts.append(f"pending_over_grace={len(pending_pods_over_grace)}")
        if crashloop_pods:
            parts.append(f"crashloops={len(crashloop_pods)}")
        if oom_pods:
            parts.append(f"OOMKilled={len(oom_pods)}")
        detail = ", ".join(parts) if parts else "no active hard-failure signals"

        if severity == "P1":
            return f"Critical Kubernetes availability issue detected: {detail}."
        if severity == "P2":
            return f"High-severity Kubernetes health issue detected: {detail}."
        return "No critical Kubernetes health issue detected."

    @staticmethod
    def _recommendation(
        severity: str,
        unavailable_deployments: list[str],
        unavailable_statefulsets: list[str],
        pending_pods_over_grace: list[str],
        crashloop_pods: list[str],
        oom_pods: list[str],
    ) -> str:
        if unavailable_deployments:
            return f"Prioritize restoring deployment availability: {unavailable_deployments[:3]}."
        if unavailable_statefulsets:
            return f"Investigate statefulset readiness immediately: {unavailable_statefulsets[:3]}."
        if crashloop_pods:
            return f"Inspect pod logs/events for crashloop containers: {crashloop_pods[:3]}."
        if pending_pods_over_grace:
            return f"Investigate scheduling/resource constraints for pending pods: {pending_pods_over_grace[:3]}."
        if oom_pods:
            return f"Check memory requests/limits for OOMKilled containers: {oom_pods[:3]}."
        if severity == "info":
            return "Continue monitoring; no alert condition met."
        return "Run targeted kubectl describe/log checks on affected workloads."

    @staticmethod
    def _parse_ts(raw: str | None) -> datetime | None:
        if not raw:
            return None
        try:
            return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        except ValueError:
            return None

    @staticmethod
    def _fingerprint(
        *,
        namespace: str,
        severity: str,
        unavailable_deployments: list[str],
        unavailable_statefulsets: list[str],
        pending_pods_over_grace: list[str],
        crashloop_pods: list[str],
        oom_pods: list[str],
        critical_events: list[str],
    ) -> str:
        def _stable_pending(values: list[str]) -> list[str]:
            out = []
            for item in values:
                out.append(str(item).split(" (")[0])
            return sorted(set(out))

        def _stable_crash(values: list[str]) -> list[str]:
            out = []
            for item in values:
                out.append(str(item).split(" (")[0])
            return sorted(set(out))

        def _stable_events(values: list[str]) -> list[str]:
            out = []
            for item in values:
                left = str(item).split(" - ")[0]
                out.append(left)
            return sorted(set(out))

        key = {
            "ns": namespace,
            "severity": severity,
            "dep": sorted(unavailable_deployments)[:8],
            "sts": sorted(unavailable_statefulsets)[:8],
            "pending": _stable_pending(pending_pods_over_grace)[:8],
            "crash": _stable_crash(crashloop_pods)[:8],
            "oom": sorted(oom_pods)[:8],
            "events": _stable_events(critical_events)[:8],
        }
        raw = json.dumps(key, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
