"""Autonomous case runner for the rebuilt AI Ops agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from langchain_core.messages import HumanMessage

from ..config import Config
from ..tools import ConnectorSuite
from ..tools.common import ToolObservation
from ..utils.llm_retry import invoke_with_retries
from ..utils.response import extract_response_text
from .parsing import parse_json_response, safe_json_dumps, truncate_text
from .prompts import ACTION_CATALOG, INVESTIGATION_PROFILES, build_integration_prompt, build_planning_prompt, build_synthesis_prompt, build_turn_analysis_prompt
from .state import ActionSpec, ApprovalRequest, CaseEntity, CaseFinding, CaseState, OperatorIntentState, TurnContext, make_evidence_record, normalize_yes_no


@dataclass
class TurnResult:
    response_text: str
    case_state: CaseState | None
    operator_intent_state: OperatorIntentState


class InvestigationRunner:
    def __init__(self, *, llm: Any, connectors: ConnectorSuite, system_prompt: str, trace_writer: Any = None):
        self.llm = llm
        self.connectors = connectors
        self.system_prompt = system_prompt
        self.trace_writer = trace_writer

    def _emit(self, trace_id: str | None, payload: dict[str, Any]) -> None:
        if self.trace_writer and trace_id:
            self.trace_writer.emit({"trace_id": trace_id, **payload})

    def _notify(self, callback: Callable[[str], None] | None, text: str) -> None:
        if callable(callback):
            try:
                callback(text)
            except Exception:
                return

    def _invoke_json(self, prompt: str, *, trace_id: str, event: str, default: dict[str, Any]) -> dict[str, Any]:
        response = invoke_with_retries(
            self.llm,
            [HumanMessage(content=prompt)],
            trace_writer=self.trace_writer,
            trace_id=trace_id,
            event=event,
        )
        text = extract_response_text(response)
        parsed = parse_json_response(text, default=default)
        return parsed if isinstance(parsed, dict) else default

    def _invoke_text(self, prompt: str, *, trace_id: str, event: str) -> str:
        response = invoke_with_retries(
            self.llm,
            [HumanMessage(content=prompt)],
            trace_writer=self.trace_writer,
            trace_id=trace_id,
            event=event,
        )
        return extract_response_text(response)

    def _default_turn_analysis(self, user_input: str, *, has_active_case: bool) -> dict[str, Any]:
        return {
            "decision": "continue_case" if has_active_case else "start_new_case",
            "goal": user_input,
            "desired_outcome": user_input,
            "profile": "general_investigation",
            "domains": ["k8s", "aws", "helm"],
            "notes": [],
        }

    def _profile_openers(self, profile: str) -> list[ActionSpec]:
        openers: dict[str, list[ActionSpec]] = {
            "restore_workloads": [
                ActionSpec(family="k8s", mode="read", action="cluster_overview", reason="Check whether workloads can schedule and whether nodes exist."),
                ActionSpec(family="aws", mode="read", action="compute_backing_overview", reason="Check the cloud compute that should back cluster nodes."),
                ActionSpec(family="helm", mode="read", action="release_overview", reason="Check whether Helm release failures correlate with workload issues."),
            ],
            "cluster_health": [
                ActionSpec(family="k8s", mode="read", action="cluster_overview", reason="Start with broad cluster health."),
                ActionSpec(family="aws", mode="read", action="identity", reason="Confirm the AWS account context."),
            ],
            "service_outage": [
                ActionSpec(family="k8s", mode="read", action="cluster_overview", reason="Check service/backing workload health across the cluster."),
                ActionSpec(family="helm", mode="read", action="release_overview", reason="Correlate release status with the outage."),
            ],
            "inventory": [
                ActionSpec(family="k8s", mode="read", action="cluster_overview", reason="Collect broad inventory context first."),
                ActionSpec(family="aws", mode="read", action="compute_backing_overview", reason="Collect AWS inventory alongside cluster state."),
                ActionSpec(family="helm", mode="read", action="release_overview", reason="Collect Helm inventory."),
            ],
            "capacity": [
                ActionSpec(family="k8s", mode="read", action="node_overview", reason="Check current node capacity and pressure."),
                ActionSpec(family="k8s", mode="read", action="pod_overview", reason="Check current pod scheduling and pressure signals."),
                ActionSpec(family="aws", mode="read", action="compute_backing_overview", reason="Check backing cloud compute shape and availability."),
            ],
            "helm_release": [
                ActionSpec(family="helm", mode="read", action="release_overview", reason="Start with Helm releases."),
                ActionSpec(family="k8s", mode="read", action="cluster_overview", reason="Correlate Helm with cluster state."),
            ],
            "direct_command": [],
            "general_investigation": [
                ActionSpec(family="k8s", mode="read", action="cluster_overview", reason="Start broad when the request is ambiguous."),
                ActionSpec(family="aws", mode="read", action="compute_backing_overview", reason="Correlate backing cloud state early."),
            ],
        }
        return openers.get(profile, openers["general_investigation"])

    def _action_seen(self, case: CaseState, action: ActionSpec) -> bool:
        target = (action.family, action.mode, action.action, safe_json_dumps(action.params))
        for existing in case.action_history:
            if (existing.family, existing.mode, existing.action, safe_json_dumps(existing.params)) == target:
                return True
        return False

    def _default_plan(self, case: CaseState) -> dict[str, Any]:
        actions = []
        for action in self._profile_openers(case.profile):
            if self._action_seen(case, action):
                continue
            actions.append(action.to_dict())
        if actions:
            return {
                "assistant_status": "Gathering high-signal infrastructure evidence...",
                "phase": case.phase or "observe",
                "working_summary": case.summary or case.goal,
                "hypotheses": case.hypotheses[:4],
                "gaps": case.gaps[:4],
                "actions": actions[: max(1, int(Config.AGENT_MAX_ACTIONS_PER_STEP))],
                "stop": False,
                "stop_reason": "",
                "answer": "",
                "approval_summary": "",
            }
        return {
            "assistant_status": "Synthesizing the evidence already collected...",
            "phase": "synthesize",
            "working_summary": case.summary or case.goal,
            "hypotheses": case.hypotheses[:4],
            "gaps": case.gaps[:4],
            "actions": [],
            "stop": True,
            "stop_reason": "enough_evidence",
            "answer": "",
            "approval_summary": "",
        }

    def _coerce_action(self, payload: dict[str, Any]) -> ActionSpec | None:
        family = str(payload.get("family") or "").strip().lower()
        mode = str(payload.get("mode") or "read").strip().lower()
        action_name = str(payload.get("action") or "").strip()
        params = payload.get("params") if isinstance(payload.get("params"), dict) else {}
        if family not in ACTION_CATALOG or action_name not in ACTION_CATALOG[family]:
            return None
        if mode not in {"read", "write"}:
            mode = "read"
        return ActionSpec(
            family=family,
            mode=mode,
            action=action_name,
            params=dict(params),
            reason=str(payload.get("reason") or "").strip(),
            expected_outcome=str(payload.get("expected_outcome") or "").strip(),
        )

    def _execute_action(self, action: ActionSpec) -> ToolObservation:
        if action.family == "k8s":
            connector = self.connectors.kubernetes
            if action.mode == "write":
                return connector.execute(str(action.params.get("command") or ""))
            if action.action == "cluster_overview":
                return connector.cluster_overview()
            if action.action == "namespace_overview":
                return connector.namespace_overview(namespace=str(action.params.get("namespace") or ""))
            if action.action == "node_overview":
                return connector.node_overview()
            if action.action == "pod_overview":
                return connector.pod_overview(
                    namespace=str(action.params.get("namespace") or ""),
                    all_namespaces=bool(action.params.get("all_namespaces", not action.params.get("namespace"))),
                )
            if action.action == "workload_overview":
                return connector.workload_overview(
                    namespace=str(action.params.get("namespace") or ""),
                    all_namespaces=bool(action.params.get("all_namespaces", not action.params.get("namespace"))),
                )
            if action.action == "service_overview":
                return connector.service_overview(
                    namespace=str(action.params.get("namespace") or ""),
                    all_namespaces=bool(action.params.get("all_namespaces", not action.params.get("namespace"))),
                )
            if action.action == "storage_overview":
                return connector.storage_overview(
                    namespace=str(action.params.get("namespace") or ""),
                    all_namespaces=bool(action.params.get("all_namespaces", not action.params.get("namespace"))),
                )
            if action.action == "event_overview":
                return connector.event_overview(
                    namespace=str(action.params.get("namespace") or ""),
                    all_namespaces=bool(action.params.get("all_namespaces", not action.params.get("namespace"))),
                )
            if action.action == "resource_details":
                return connector.resource_details(
                    kind=str(action.params.get("kind") or ""),
                    name=str(action.params.get("name") or ""),
                    namespace=str(action.params.get("namespace") or ""),
                )
            if action.action == "raw_read":
                return connector.raw_read(str(action.params.get("command") or ""))
        if action.family == "aws":
            connector = self.connectors.aws
            if action.mode == "write":
                return connector.execute(str(action.params.get("command") or ""))
            if action.action == "identity":
                return connector.identity()
            if action.action == "regions":
                return connector.regions()
            if action.action == "eks_overview":
                return connector.eks_overview(all_regions=bool(action.params.get("all_regions", True)))
            if action.action == "ec2_overview":
                return connector.ec2_overview(
                    states=list(action.params.get("states") or []),
                    regions=list(action.params.get("regions") or []),
                    all_regions=bool(action.params.get("all_regions", True)),
                )
            if action.action == "asg_overview":
                return connector.asg_overview(all_regions=bool(action.params.get("all_regions", True)))
            if action.action == "compute_backing_overview":
                return connector.compute_backing_overview()
            if action.action == "raw_read":
                return connector.raw_read(
                    str(action.params.get("command") or ""),
                    all_regions=bool(action.params.get("all_regions", True)),
                )
        if action.family == "helm":
            connector = self.connectors.helm
            if action.mode == "write":
                return connector.execute(str(action.params.get("command") or ""))
            if action.action == "release_overview":
                return connector.release_overview(all_namespaces=bool(action.params.get("all_namespaces", True)))
            if action.action == "release_details":
                return connector.release_details(
                    release_name=str(action.params.get("release_name") or ""),
                    namespace=str(action.params.get("namespace") or ""),
                )
            if action.action == "raw_read":
                return connector.raw_read(str(action.params.get("command") or ""))
        raise RuntimeError(f"Unsupported action: {action.family}.{action.action}")

    def _approval_text(self, case: CaseState, approval: ApprovalRequest) -> str:
        commands = "\n".join(approval.commands) if approval.commands else "(no commands prepared)"
        summary = case.summary or case.goal
        return (
            f"**Bottom Line:** {summary}\n\n"
            f"**What I Found:**\n"
            f"- {approval.rationale}\n\n"
            f"**Planned Change:** {approval.summary}\n\n"
            f"```bash\n{commands}\n```\n\n"
            f"Would you like me to proceed with {approval.summary}? (yes/no)"
        )

    def _fallback_synthesis(self, case: CaseState) -> str:
        findings = [finding.claim for finding in case.findings[:5]] or [record.summary for record in case.evidence[-5:]]
        finding_lines = "\n".join(f"- {item}" for item in findings[:5]) if findings else "- I do not have enough grounded findings yet."
        next_step = case.gaps[0] if case.gaps else "Continue targeted investigation or approve the proposed change if one is pending."
        bottom_line = case.summary or case.goal
        return (
            f"**Bottom Line:** {bottom_line}\n\n"
            f"**What I Found:**\n{finding_lines}\n\n"
            f"**Recommended Next Step:** {next_step}"
        )

    def _apply_integration(self, case: CaseState, integration: dict[str, Any], observations: list[ToolObservation]) -> None:
        case.summary = str(integration.get("summary") or case.summary or case.goal).strip()
        case.phase = str(integration.get("phase") or case.phase or "observe").strip() or "observe"
        entities = integration.get("entities") if isinstance(integration.get("entities"), list) else []
        for payload in entities:
            if not isinstance(payload, dict):
                continue
            kind = str(payload.get("kind") or "").strip()
            name = str(payload.get("name") or "").strip()
            if not kind or not name:
                continue
            case.merge_entity(
                CaseEntity(
                    kind=kind,
                    name=name,
                    namespace=str(payload.get("namespace") or "").strip(),
                    scope=str(payload.get("scope") or "").strip(),
                    provider_id=str(payload.get("provider_id") or "").strip(),
                    attrs=dict(payload.get("attrs") or {}),
                )
            )
        findings = integration.get("findings") if isinstance(integration.get("findings"), list) else []
        recent_evidence_ids = [record.evidence_id for record in case.evidence[-len(observations) :]]
        for payload in findings:
            if not isinstance(payload, dict):
                continue
            claim = str(payload.get("claim") or "").strip()
            if not claim:
                continue
            case.merge_finding(
                CaseFinding(
                    claim=claim,
                    confidence=max(0, min(100, int(payload.get("confidence") or 0))),
                    verified=bool(payload.get("verified")),
                    entity_refs=[str(item) for item in payload.get("entity_refs") or [] if str(item).strip()],
                    evidence_refs=recent_evidence_ids[:],
                )
            )
        case.replace_hypotheses([str(item) for item in integration.get("hypotheses") or [] if str(item).strip()])
        case.replace_gaps([str(item) for item in integration.get("gaps") or [] if str(item).strip()])

    def _plan_case_transition(self, context: TurnContext, case: CaseState | None) -> dict[str, Any]:
        stripped = context.user_input.strip().lower()
        if stripped.startswith(("kubectl ", "aws ", "helm ")):
            return {
                "decision": "start_new_case" if case is None else "continue_case",
                "goal": context.user_input,
                "desired_outcome": "Execute the explicitly requested infrastructure command or return its read-only output.",
                "profile": "direct_command",
                "domains": ["k8s" if stripped.startswith("kubectl ") else "aws" if stripped.startswith("aws ") else "helm"],
                "notes": ["The user provided a direct CLI command."],
            }
        active_case = case.snapshot() if case is not None else None
        default = self._default_turn_analysis(context.user_input, has_active_case=case is not None)
        prompt = build_turn_analysis_prompt(
            system_prompt=self.system_prompt,
            user_input=context.user_input,
            active_case=active_case,
            chat_history=context.chat_history,
        )
        return self._invoke_json(prompt, trace_id=context.trace_id, event="llm.turn_analysis", default=default)

    def _update_operator_from_case(self, operator_state: OperatorIntentState, case: CaseState | None) -> None:
        if case is None:
            operator_state.clear()
            return
        operator_state.mode = "approval_pending" if case.pending_approval else "incident_response"
        operator_state.execution_policy = "awaiting_approval" if case.pending_approval else "approval_required"
        operator_state.pending_step_summary = case.pending_approval.summary if case.pending_approval else ""
        operator_state.pending_step_kind = "write" if case.pending_approval else ""
        operator_state.awaiting_follow_up = case.pending_approval is not None
        operator_state.approved_proposed_plan = False
        operator_state.pinned_constraints = (
            [f"Writes require approval before execution: {case.pending_approval.summary}"]
            if case.pending_approval
            else []
        )

    def _handle_approval_reply(
        self,
        *,
        case: CaseState,
        operator_state: OperatorIntentState,
        user_input: str,
        trace_id: str,
        status_callback: Callable[[str], None] | None,
    ) -> TurnResult | None:
        if case.pending_approval is None:
            return None
        decision = normalize_yes_no(user_input)
        if decision == "":
            return TurnResult(
                response_text=f"I have a pending change waiting for approval: {case.pending_approval.summary}. Please reply with `yes` or `no`.",
                case_state=case,
                operator_intent_state=operator_state,
            )
        if decision == "no":
            case.pending_approval = None
            case.status = "running"
            case.gaps = []
            self._update_operator_from_case(operator_state, case)
            return TurnResult(
                response_text="I did not make any changes. The investigation remains valid, and I can continue with additional read-only checks if you want.",
                case_state=case,
                operator_intent_state=operator_state,
            )

        approval = case.pending_approval
        case.pending_approval = None
        case.status = "executing"
        self._notify(status_callback, "Executing the approved infrastructure change...")
        observations: list[ToolObservation] = []
        for action in approval.actions:
            case.add_action(action)
            self._emit(trace_id, {"event": "approval.execute_action", "action": action.to_dict()})
            try:
                observation = self._execute_action(action)
            except Exception as exc:  # noqa: BLE001
                observation = ToolObservation(
                    family=action.family,
                    action=action.action,
                    summary=f"Execution failed for {action.label()}: {exc}",
                    structured={"error": str(exc)},
                    commands=[action.label()],
                    raw_preview=str(exc),
                    ok=False,
                )
            observations.append(observation)
            case.add_evidence(
                make_evidence_record(
                    family=observation.family,
                    action=observation.action,
                    summary=observation.summary,
                    structured=observation.structured,
                    commands=observation.commands,
                    raw_preview=observation.raw_preview,
                    ok=observation.ok,
                )
            )
        integration_default = {
            "summary": case.summary or case.goal,
            "phase": "remediate",
            "entities": [],
            "findings": [],
            "hypotheses": case.hypotheses[:],
            "gaps": [],
        }
        integration = self._invoke_json(
            build_integration_prompt(
                system_prompt=self.system_prompt,
                case_snapshot=case.snapshot(),
                executed_actions=[action.to_dict() for action in approval.actions],
                observations=[item.to_dict() for item in observations],
            ),
            trace_id=trace_id,
            event="llm.integrate_execute",
            default=integration_default,
        )
        self._apply_integration(case, integration, observations)
        case.status = "completed"
        case.phase = "synthesize"
        prompt = build_synthesis_prompt(system_prompt=self.system_prompt, case_snapshot=case.snapshot())
        response_text = self._invoke_text(prompt, trace_id=trace_id, event="llm.synthesize_execute").strip() or self._fallback_synthesis(case)
        self._update_operator_from_case(operator_state, case)
        return TurnResult(response_text=response_text, case_state=case, operator_intent_state=operator_state)

    def run_turn(
        self,
        *,
        context: TurnContext,
        case: CaseState | None,
        operator_state: OperatorIntentState,
        status_callback: Callable[[str], None] | None = None,
    ) -> TurnResult:
        operator_state.last_user_instruction = context.user_input

        if case is not None and case.pending_approval is not None:
            approval_result = self._handle_approval_reply(
                case=case,
                operator_state=operator_state,
                user_input=context.user_input,
                trace_id=context.trace_id,
                status_callback=status_callback,
            )
            if approval_result is not None:
                return approval_result

        self._notify(status_callback, "Understanding your goal and deciding whether to continue the current case...")
        transition = self._plan_case_transition(context, case)
        self._emit(context.trace_id, {"event": "case.transition", "decision": transition})

        if case is None or str(transition.get("decision") or "start_new_case") == "start_new_case":
            case = CaseState.create(
                goal=str(transition.get("goal") or context.user_input),
                desired_outcome=str(transition.get("desired_outcome") or context.user_input),
                profile=str(transition.get("profile") or "general_investigation"),
                domains=[str(item) for item in transition.get("domains") or [] if str(item).strip()],
                initial_message=context.user_input,
            )
        else:
            case.goal = str(transition.get("goal") or case.goal).strip() or case.goal
            case.desired_outcome = str(transition.get("desired_outcome") or case.desired_outcome).strip() or case.desired_outcome
            case.profile = str(transition.get("profile") or case.profile).strip() or case.profile
            proposed_domains = [str(item) for item in transition.get("domains") or [] if str(item).strip()]
            if proposed_domains:
                case.domains = list(dict.fromkeys([*case.domains, *proposed_domains]))
            case.add_user_message(context.user_input)

        self._update_operator_from_case(operator_state, case)

        if case.profile == "direct_command":
            direct = self._handle_direct_command(case, operator_state)
            if direct is not None:
                return direct

        max_steps = max(1, int(Config.AGENT_MAX_STEPS))
        for step in range(max_steps):
            self._notify(status_callback, f"Investigating {case.profile.replace('_', ' ')}: step {step + 1}/{max_steps}...")
            default_plan = self._default_plan(case)
            planning = self._invoke_json(
                build_planning_prompt(
                    system_prompt=self.system_prompt,
                    case_snapshot=case.snapshot(),
                    latest_user_message=context.user_input,
                ),
                trace_id=context.trace_id,
                event="llm.plan_step",
                default=default_plan,
            )

            assistant_status = str(planning.get("assistant_status") or "").strip()
            if assistant_status:
                self._notify(status_callback, assistant_status)

            case.phase = str(planning.get("phase") or case.phase or "observe").strip() or "observe"
            if str(planning.get("working_summary") or "").strip():
                case.summary = str(planning.get("working_summary")).strip()
            case.replace_hypotheses([str(item) for item in planning.get("hypotheses") or [] if str(item).strip()])
            case.replace_gaps([str(item) for item in planning.get("gaps") or [] if str(item).strip()])

            actions: list[ActionSpec] = []
            for payload in planning.get("actions") or []:
                if not isinstance(payload, dict):
                    continue
                action = self._coerce_action(payload)
                if action is not None:
                    actions.append(action)
            stop = bool(planning.get("stop"))
            stop_reason = str(planning.get("stop_reason") or "").strip()
            if not actions and not stop:
                for payload in default_plan.get("actions") or []:
                    action = self._coerce_action(payload if isinstance(payload, dict) else {})
                    if action is not None:
                        actions.append(action)
            actions = actions[: max(1, int(Config.AGENT_MAX_ACTIONS_PER_STEP))]

            if any(action.mode == "write" for action in actions) or stop_reason == "needs_approval":
                write_actions = [action for action in actions if action.mode == "write"]
                if not write_actions:
                    write_actions = [ActionSpec(family="k8s", mode="write", action="raw_write", params={"command": ""}, reason="", expected_outcome="")]
                approval = ApprovalRequest(
                    request_id=case.case_id + "-approval",
                    summary=str(planning.get("approval_summary") or "the proposed infrastructure change").strip(),
                    rationale=case.summary or case.goal,
                    commands=[action.label() for action in write_actions],
                    actions=write_actions,
                    expected_outcome=", ".join(filter(None, [action.expected_outcome for action in write_actions])),
                )
                case.pending_approval = approval
                case.status = "awaiting_approval"
                self._update_operator_from_case(operator_state, case)
                return TurnResult(
                    response_text=self._approval_text(case, approval),
                    case_state=case,
                    operator_intent_state=operator_state,
                )

            if stop and not actions:
                prompt = build_synthesis_prompt(system_prompt=self.system_prompt, case_snapshot=case.snapshot())
                response_text = str(planning.get("answer") or "").strip()
                if not response_text:
                    response_text = self._invoke_text(prompt, trace_id=context.trace_id, event="llm.synthesize").strip()
                response_text = response_text or self._fallback_synthesis(case)
                case.status = "completed"
                case.final_response = response_text
                self._update_operator_from_case(operator_state, case)
                return TurnResult(response_text=response_text, case_state=case, operator_intent_state=operator_state)

            observations: list[ToolObservation] = []
            for action in actions:
                case.add_action(action)
                self._emit(context.trace_id, {"event": "case.action", "action": action.to_dict()})
                try:
                    observation = self._execute_action(action)
                except Exception as exc:  # noqa: BLE001
                    observation = ToolObservation(
                        family=action.family,
                        action=action.action,
                        summary=f"{action.label()} failed: {exc}",
                        structured={"error": str(exc), "action": action.to_dict()},
                        commands=[action.label()],
                        raw_preview=truncate_text(str(exc), max_chars=1600),
                        ok=False,
                    )
                observations.append(observation)
                case.add_evidence(
                    make_evidence_record(
                        family=observation.family,
                        action=observation.action,
                        summary=observation.summary,
                        structured=observation.structured,
                        commands=observation.commands,
                        raw_preview=observation.raw_preview,
                        ok=observation.ok,
                    )
                )

            integration_default = {
                "summary": case.summary or case.goal,
                "phase": case.phase,
                "entities": [],
                "findings": [{"claim": item.summary, "confidence": 60 if item.ok else 40, "verified": bool(item.ok), "entity_refs": []} for item in observations[:5]],
                "hypotheses": case.hypotheses[:],
                "gaps": case.gaps[:],
            }
            integration = self._invoke_json(
                build_integration_prompt(
                    system_prompt=self.system_prompt,
                    case_snapshot=case.snapshot(),
                    executed_actions=[action.to_dict() for action in actions],
                    observations=[item.to_dict() for item in observations],
                ),
                trace_id=context.trace_id,
                event="llm.integrate_step",
                default=integration_default,
            )
            self._apply_integration(case, integration, observations)

        response_text = self._invoke_text(
            build_synthesis_prompt(system_prompt=self.system_prompt, case_snapshot=case.snapshot()),
            trace_id=context.trace_id,
            event="llm.synthesize_fallback",
        ).strip() or self._fallback_synthesis(case)
        case.status = "completed"
        case.final_response = response_text
        self._update_operator_from_case(operator_state, case)
        return TurnResult(response_text=response_text, case_state=case, operator_intent_state=operator_state)

    def _handle_direct_command(self, case: CaseState, operator_state: OperatorIntentState) -> TurnResult | None:
        latest = case.user_messages[-1] if case.user_messages else case.goal
        stripped = latest.strip()
        if stripped.startswith("kubectl "):
            family = "k8s"
            connector = self.connectors.kubernetes
        elif stripped.startswith("aws "):
            family = "aws"
            connector = self.connectors.aws
        elif stripped.startswith("helm "):
            family = "helm"
            connector = self.connectors.helm
        else:
            return None

        if family == "k8s":
            try:
                observation = connector.raw_read(stripped)
                case.add_evidence(make_evidence_record(family=observation.family, action=observation.action, summary=observation.summary, structured=observation.structured, commands=observation.commands, raw_preview=observation.raw_preview, ok=observation.ok))
                case.summary = observation.summary
                case.final_response = truncate_text(str(observation.raw_preview or observation.structured), max_chars=Config.LLM_MAX_RESPONSE_CHARS)
                self._update_operator_from_case(operator_state, case)
                return TurnResult(response_text=case.final_response, case_state=case, operator_intent_state=operator_state)
            except Exception:
                approval = ApprovalRequest(
                    request_id=case.case_id + "-approval",
                    summary="the requested kubectl command",
                    rationale="The user explicitly requested a mutating kubectl command.",
                    commands=[stripped],
                    actions=[ActionSpec(family="k8s", mode="write", action="raw_write", params={"command": stripped}, reason="Direct user request")],
                )
        elif family == "aws":
            try:
                observation = connector.raw_read(stripped, all_regions=True)
                case.add_evidence(make_evidence_record(family=observation.family, action=observation.action, summary=observation.summary, structured=observation.structured, commands=observation.commands, raw_preview=observation.raw_preview, ok=observation.ok))
                case.summary = observation.summary
                case.final_response = truncate_text(str(observation.raw_preview or observation.structured), max_chars=Config.LLM_MAX_RESPONSE_CHARS)
                self._update_operator_from_case(operator_state, case)
                return TurnResult(response_text=case.final_response, case_state=case, operator_intent_state=operator_state)
            except Exception:
                approval = ApprovalRequest(
                    request_id=case.case_id + "-approval",
                    summary="the requested AWS command",
                    rationale="The user explicitly requested a mutating AWS command.",
                    commands=[stripped],
                    actions=[ActionSpec(family="aws", mode="write", action="raw_write", params={"command": stripped}, reason="Direct user request")],
                )
        else:
            try:
                observation = connector.raw_read(stripped)
                case.add_evidence(make_evidence_record(family=observation.family, action=observation.action, summary=observation.summary, structured=observation.structured, commands=observation.commands, raw_preview=observation.raw_preview, ok=observation.ok))
                case.summary = observation.summary
                case.final_response = truncate_text(str(observation.raw_preview or observation.structured), max_chars=Config.LLM_MAX_RESPONSE_CHARS)
                self._update_operator_from_case(operator_state, case)
                return TurnResult(response_text=case.final_response, case_state=case, operator_intent_state=operator_state)
            except Exception:
                approval = ApprovalRequest(
                    request_id=case.case_id + "-approval",
                    summary="the requested Helm command",
                    rationale="The user explicitly requested a mutating Helm command.",
                    commands=[stripped],
                    actions=[ActionSpec(family="helm", mode="write", action="raw_write", params={"command": stripped}, reason="Direct user request")],
                )

        case.pending_approval = approval
        case.status = "awaiting_approval"
        self._update_operator_from_case(operator_state, case)
        return TurnResult(response_text=self._approval_text(case, approval), case_state=case, operator_intent_state=operator_state)

    def run_health_scan(self, *, trace_id: str, status_callback: Callable[[str], None] | None = None) -> tuple[dict[str, Any], CaseState]:
        case = CaseState.create(
            goal="Run a broad cluster and cloud health scan.",
            desired_outcome="Know whether the environment looks healthy and what should be investigated next.",
            profile="cluster_health",
            domains=["k8s", "aws", "helm"],
            initial_message="Run a broad cluster and cloud health scan.",
        )
        operator_state = OperatorIntentState()
        result = self.run_turn(
            context=TurnContext(user_input="Run a broad cluster and cloud health scan.", chat_history=[], trace_id=trace_id),
            case=case,
            operator_state=operator_state,
            status_callback=status_callback,
        )
        case = result.case_state or case
        severity = "info"
        confidence = 60
        summary = case.summary or "No scan summary available."
        finding_text = "\n".join(f"- {finding.claim}" for finding in case.findings[:6]) or "- No major findings were recorded."
        if any("not ready" in finding.claim.lower() or "unhealthy" in finding.claim.lower() or "warning" in finding.claim.lower() for finding in case.findings):
            severity = "P2"
            confidence = 80
        if any("no nodes" in finding.claim.lower() or "outage" in finding.claim.lower() for finding in case.findings):
            severity = "P1"
            confidence = 90
        scan = {
            "ok": True,
            "completed_at": case.last_updated_at,
            "incident": {
                "should_alert": severity in {"P1", "P2"},
                "severity": severity,
                "confidence_score": confidence,
                "impact_score": 85 if severity == "P1" else 55 if severity == "P2" else 15,
                "issue_summary": summary,
                "details_markdown": f"**Bottom Line:** {summary}\n\n**Findings:**\n{finding_text}",
            },
            "notifications": {},
        }
        return scan, case
