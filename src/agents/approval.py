"""Approval state and write-command preview helpers for the agent."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

@dataclass
class PendingAction:
    """Pending write tool invocation awaiting explicit approval."""

    tool: Any
    args: dict


class ApprovalCoordinator:
    """Tracks pending write actions and batch-restart promotion context."""

    def __init__(self) -> None:
        self.pending_action: PendingAction | None = None
        self.pending_actions: list[PendingAction] = []
        self.recent_restart_candidates: list[str] = []
        self.recent_restart_namespace: str | None = None
        self.recent_restart_reason: str = ""

    def clear(self) -> None:
        self.pending_action = None
        self.pending_actions = []
        self.recent_restart_candidates = []
        self.recent_restart_namespace = None
        self.recent_restart_reason = ""

    def set_pending_action(self, tool: Any, args: dict) -> None:
        self.pending_actions = []
        self.pending_action = PendingAction(tool=tool, args=args)

    def set_pending_actions(self, actions: list[PendingAction]) -> None:
        self.pending_action = None
        self.pending_actions = list(actions)

    def has_pending(self) -> bool:
        return self.pending_action is not None or bool(self.pending_actions)

    def record_restart_context(self, tool_args: dict, model_response_text: str) -> None:
        namespace = str(tool_args.get("namespace") or "").strip() or "auto"
        reason = str(tool_args.get("reason") or "")
        candidates = extract_pod_candidates_from_text(model_response_text)

        requested = str(tool_args.get("pod_name") or "").strip()
        if requested and requested not in candidates:
            candidates.insert(0, requested)

        self.recent_restart_candidates = candidates[:20]
        self.recent_restart_namespace = namespace
        self.recent_restart_reason = reason

    def should_offer_batch_prompt(self, tool_name: str, tool_args: dict) -> bool:
        if tool_name != "restart_kubernetes_pod":
            return False

        requested = str(tool_args.get("pod_name") or "").strip()
        candidates = [str(p).strip() for p in self.recent_restart_candidates if str(p).strip()]
        if requested and requested not in candidates:
            candidates.insert(0, requested)

        unique_candidates = list(dict.fromkeys(candidates))
        return len(unique_candidates) > 1

    def should_promote_to_batch(self, decision: str) -> bool:
        if not self.pending_action:
            return False
        return self.pending_action.tool.name == "restart_kubernetes_pod" and is_batch_intent(decision)

    def promote_pending_to_batch(self, tools: list[Any]) -> bool:
        if not self.pending_action:
            return False
        if self.pending_action.tool.name != "restart_kubernetes_pod":
            return False

        candidates = list(self.recent_restart_candidates or [])
        if len(candidates) <= 1:
            return False

        batch_tool = next((tool for tool in tools if tool.name == "restart_kubernetes_pods_batch"), None)
        if batch_tool is None:
            return False

        # Batch restarts should default to namespace auto-resolution per pod.
        namespace = "auto"
        reason = self.recent_restart_reason or self.pending_action.args.get("reason") or "Batch restart requested by user"

        self.pending_action = PendingAction(
            tool=batch_tool,
            args={
                "pod_names": candidates,
                "namespace": namespace,
                "reason": reason,
            },
        )
        self.pending_actions = []
        return True


def extract_pod_candidates_from_text(text: str) -> list[str]:
    """Extract likely pod identifiers from free text."""
    tokens = re.findall(r"\b[a-z0-9]+(?:-[a-z0-9]+){2,}\b", text or "")
    out: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out[:20]


def is_batch_intent(decision: str) -> bool:
    """Heuristic to detect user intent to restart all suggested pods at once."""
    text = (decision or "").lower()
    return ("all" in text and ("restart" in text or "do" in text)) or "at once" in text or "batch" in text


def format_command_preview(tool_name: str, tool_args: dict) -> str:
    """Render planned kubectl/aws command(s) for approval prompts."""
    namespace_raw = str(tool_args.get("namespace") or "").strip()
    auto_namespace = namespace_raw.lower() in {"", "default", "auto", "any", "all"}
    if auto_namespace:
        namespace = "<auto-resolve>"
    else:
        namespace = namespace_raw
    base = ["kubectl", "-n", namespace, "delete", "pod"]

    if tool_name == "restart_kubernetes_pod":
        pod_name = str(tool_args.get("pod_name") or "<pod>")
        return "- " + " ".join(base + [pod_name, "--wait=false", "--ignore-not-found=true"])

    if tool_name == "restart_kubernetes_pods_batch":
        pod_names = [str(p) for p in (tool_args.get("pod_names") or []) if str(p).strip()]
        if not pod_names:
            return "- (no pods provided)"
        ns_for_batch = "<auto-resolve-per-pod>" if auto_namespace else namespace
        batch_base = ["kubectl", "-n", ns_for_batch, "delete", "pod"]
        return "\n".join(
            "- " + " ".join(batch_base + [pod, "--wait=false", "--ignore-not-found=true"])
            for pod in pod_names
        )

    if tool_name == "scale_kubernetes_deployment":
        name = str(tool_args.get("deployment_name") or "<deployment>")
        replicas = str(tool_args.get("replicas") if tool_args.get("replicas") is not None else "<replicas>")
        return "- " + " ".join(["kubectl", "-n", namespace, "scale", "deployment", name, "--replicas", replicas])

    if tool_name == "scale_kubernetes_statefulset":
        name = str(tool_args.get("statefulset_name") or "<statefulset>")
        replicas = str(tool_args.get("replicas") if tool_args.get("replicas") is not None else "<replicas>")
        return "- " + " ".join(["kubectl", "-n", namespace, "scale", "statefulset", name, "--replicas", replicas])

    if tool_name == "scale_kubernetes_workloads_batch":
        changes = tool_args.get("changes") or []
        previews: list[str] = []
        for item in changes:
            if not isinstance(item, dict):
                continue
            kind = str(item.get("kind") or "<kind>")
            name = str(item.get("name") or "<name>")
            replicas = str(item.get("replicas") if item.get("replicas") is not None else "<replicas>")
            previews.append("- " + " ".join(["kubectl", "-n", namespace, "scale", kind, name, "--replicas", replicas]))
        return "\n".join(previews) if previews else "- (no workload changes provided)"

    if tool_name == "rollout_restart_kubernetes_deployment":
        name = str(tool_args.get("deployment_name") or "<deployment>")
        return "- " + " ".join(["kubectl", "-n", namespace, "rollout", "restart", f"deployment/{name}"])

    if tool_name == "rollout_restart_kubernetes_statefulset":
        name = str(tool_args.get("statefulset_name") or "<statefulset>")
        return "- " + " ".join(["kubectl", "-n", namespace, "rollout", "restart", f"statefulset/{name}"])

    if tool_name == "rollout_restart_kubernetes_daemonset":
        name = str(tool_args.get("daemonset_name") or "<daemonset>")
        return "- " + " ".join(["kubectl", "-n", namespace, "rollout", "restart", f"daemonset/{name}"])

    if tool_name == "rollout_restart_kubernetes_workloads_batch":
        workloads = tool_args.get("workloads") or []
        previews: list[str] = []
        for item in workloads:
            if not isinstance(item, dict):
                continue
            kind = str(item.get("kind") or "<kind>")
            name = str(item.get("name") or "<name>")
            previews.append("- " + " ".join(["kubectl", "-n", namespace, "rollout", "restart", f"{kind}/{name}"]))
        return "\n".join(previews) if previews else "- (no workload restarts provided)"

    if tool_name == "aws_cli_execute":
        command = str(tool_args.get("command") or "").strip()
        if not command:
            return "- aws <service> <operation> [args]"
        if command.lower().startswith("aws "):
            return "- " + command
        return "- aws " + command

    if tool_name == "kubectl_execute":
        command = str(tool_args.get("command") or "").strip()
        if not command:
            return "- kubectl <verb> [args]"
        if command.lower().startswith("kubectl "):
            return "- " + command
        return "- kubectl " + command

    if tool_name == "helm_execute":
        command = str(tool_args.get("command") or "").strip()
        if not command:
            return "- helm <verb> [args]"
        if command.lower().startswith("helm "):
            return "- " + command
        return "- helm " + command

    return "- command preview unavailable for this write tool"


def commands_code_block(command_preview: str) -> str:
    """Format command preview text as a bash code block."""
    lines: list[str] = []
    for raw in (command_preview or "").splitlines():
        line = raw.strip()
        if line.startswith("- "):
            line = line[2:]
        if line:
            lines.append(line)

    if not lines:
        lines = ["(no command preview)"]

    return "```bash\n" + "\n".join(lines) + "\n```"
