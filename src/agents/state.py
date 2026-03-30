"""Typed case state for the rebuilt AI Ops agent."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
import uuid


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _coerce_text(value: Any) -> str:
    return str(value or "").strip()


def _key_text(value: Any) -> str:
    return _coerce_text(value).lower()


def normalize_yes_no(text: str) -> str:
    normalized = " ".join(_coerce_text(text).lower().rstrip(".!?").split())
    if normalized in {"yes", "y", "ok", "okay", "sure", "do it", "run it", "go ahead", "approve", "approved"}:
        return "yes"
    if normalized in {"no", "n", "cancel", "stop", "do not", "don't"}:
        return "no"
    return ""


@dataclass
class CaseEntity:
    kind: str
    name: str
    namespace: str = ""
    scope: str = ""
    provider_id: str = ""
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (
            _key_text(self.kind),
            _key_text(self.namespace),
            _key_text(self.name),
            _key_text(self.provider_id),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CaseFinding:
    claim: str
    confidence: int = 50
    verified: bool = False
    entity_refs: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        return _key_text(self.claim)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ActionSpec:
    family: str
    mode: str
    action: str
    params: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    expected_outcome: str = ""

    def label(self) -> str:
        command = _coerce_text(self.params.get("command"))
        if command:
            return command
        param_text = ", ".join(f"{key}={value}" for key, value in sorted(self.params.items()) if value not in {"", None, []})
        if param_text:
            return f"{self.family}.{self.action}({param_text})"
        return f"{self.family}.{self.action}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceRecord:
    evidence_id: str
    family: str
    action: str
    summary: str
    structured: dict[str, Any] = field(default_factory=dict)
    commands: list[str] = field(default_factory=list)
    raw_preview: str = ""
    ok: bool = True
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ApprovalRequest:
    request_id: str
    summary: str
    rationale: str
    commands: list[str]
    actions: list[ActionSpec]
    expected_outcome: str = ""
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["actions"] = [action.to_dict() for action in self.actions]
        return payload


@dataclass
class CaseState:
    case_id: str
    goal: str
    desired_outcome: str
    profile: str
    domains: list[str] = field(default_factory=list)
    status: str = "running"
    phase: str = "observe"
    summary: str = ""
    user_messages: list[str] = field(default_factory=list)
    hypotheses: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    entities: list[CaseEntity] = field(default_factory=list)
    findings: list[CaseFinding] = field(default_factory=list)
    evidence: list[EvidenceRecord] = field(default_factory=list)
    action_history: list[ActionSpec] = field(default_factory=list)
    pending_approval: ApprovalRequest | None = None
    final_response: str = ""
    opened_at: str = field(default_factory=utc_now_iso)
    last_updated_at: str = field(default_factory=utc_now_iso)

    @classmethod
    def create(
        cls,
        *,
        goal: str,
        desired_outcome: str,
        profile: str,
        domains: list[str] | None = None,
        initial_message: str = "",
    ) -> "CaseState":
        case = cls(
            case_id=uuid.uuid4().hex,
            goal=_coerce_text(goal),
            desired_outcome=_coerce_text(desired_outcome),
            profile=_coerce_text(profile) or "general_investigation",
            domains=list(domains or []),
        )
        if initial_message:
            case.user_messages.append(_coerce_text(initial_message))
        return case

    def touch(self) -> None:
        self.last_updated_at = utc_now_iso()

    def add_user_message(self, text: str) -> None:
        clean = _coerce_text(text)
        if clean:
            self.user_messages.append(clean)
            self.touch()

    def add_action(self, action: ActionSpec) -> None:
        self.action_history.append(action)
        self.touch()

    def add_evidence(self, record: EvidenceRecord, *, limit: int = 24) -> None:
        self.evidence.append(record)
        if len(self.evidence) > limit:
            self.evidence = self.evidence[-limit:]
        self.touch()

    def merge_entity(self, entity: CaseEntity, *, limit: int = 64) -> None:
        for existing in self.entities:
            if existing.key != entity.key:
                continue
            existing.attrs.update(entity.attrs)
            if entity.scope:
                existing.scope = entity.scope
            if entity.provider_id:
                existing.provider_id = entity.provider_id
            if entity.namespace:
                existing.namespace = entity.namespace
            self.touch()
            return
        self.entities.append(entity)
        if len(self.entities) > limit:
            self.entities = self.entities[-limit:]
        self.touch()

    def merge_finding(self, finding: CaseFinding, *, limit: int = 64) -> None:
        for existing in self.findings:
            if existing.key != finding.key:
                continue
            existing.confidence = max(existing.confidence, finding.confidence)
            existing.verified = existing.verified or finding.verified
            for ref in finding.entity_refs:
                if ref not in existing.entity_refs:
                    existing.entity_refs.append(ref)
            for ref in finding.evidence_refs:
                if ref not in existing.evidence_refs:
                    existing.evidence_refs.append(ref)
            self.touch()
            return
        self.findings.append(finding)
        if len(self.findings) > limit:
            self.findings = self.findings[-limit:]
        self.touch()

    def replace_hypotheses(self, values: list[str], *, limit: int = 8) -> None:
        cleaned = [_coerce_text(item) for item in values if _coerce_text(item)]
        self.hypotheses = cleaned[:limit]
        self.touch()

    def replace_gaps(self, values: list[str], *, limit: int = 8) -> None:
        cleaned = [_coerce_text(item) for item in values if _coerce_text(item)]
        self.gaps = cleaned[:limit]
        self.touch()

    def snapshot(self, *, evidence_limit: int = 10, finding_limit: int = 10, entity_limit: int = 12) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "goal": self.goal,
            "desired_outcome": self.desired_outcome,
            "profile": self.profile,
            "domains": list(self.domains),
            "status": self.status,
            "phase": self.phase,
            "summary": self.summary,
            "user_messages": self.user_messages[-6:],
            "hypotheses": self.hypotheses[:8],
            "gaps": self.gaps[:8],
            "entities": [entity.to_dict() for entity in self.entities[:entity_limit]],
            "findings": [finding.to_dict() for finding in self.findings[:finding_limit]],
            "evidence": [record.to_dict() for record in self.evidence[-evidence_limit:]],
            "action_history": [action.to_dict() for action in self.action_history[-12:]],
            "pending_approval": self.pending_approval.to_dict() if self.pending_approval else None,
            "opened_at": self.opened_at,
            "last_updated_at": self.last_updated_at,
        }


@dataclass
class OperatorIntentState:
    """UI-facing execution state."""

    mode: str = "incident_response"
    execution_policy: str = "approval_required"
    pinned_constraints: list[str] = field(default_factory=list)
    last_user_instruction: str = ""
    pending_step_summary: str = ""
    pending_step_kind: str = ""
    awaiting_follow_up: bool = False
    approved_proposed_plan: bool = False

    def clear(self) -> None:
        self.mode = "incident_response"
        self.execution_policy = "approval_required"
        self.pinned_constraints = []
        self.last_user_instruction = ""
        self.pending_step_summary = ""
        self.pending_step_kind = ""
        self.awaiting_follow_up = False
        self.approved_proposed_plan = False


@dataclass
class TurnContext:
    user_input: str
    chat_history: list[Any]
    trace_id: str


def make_evidence_record(
    *,
    family: str,
    action: str,
    summary: str,
    structured: dict[str, Any],
    commands: list[str],
    raw_preview: str,
    ok: bool = True,
) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=uuid.uuid4().hex,
        family=family,
        action=action,
        summary=_coerce_text(summary),
        structured=dict(structured or {}),
        commands=list(commands or []),
        raw_preview=_coerce_text(raw_preview),
        ok=ok,
    )
