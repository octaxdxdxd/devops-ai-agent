"""Approval state and write-command preview helpers for the agent."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any


REPAIRABLE_COMMAND_TOOLS = {
    "aws_cli_execute": "aws",
    "kubectl_execute": "kubectl",
    "helm_execute": "helm",
}


@dataclass
class PendingAction:
    """Pending write tool invocation awaiting explicit approval."""

    tool: Any
    args: dict


class ApprovalCoordinator:
    """Tracks pending write actions awaiting explicit approval."""

    def __init__(self) -> None:
        self.pending_action: PendingAction | None = None
        self.pending_actions: list[PendingAction] = []

    def clear(self) -> None:
        self.pending_action = None
        self.pending_actions = []

    def set_pending_action(self, tool: Any, args: dict) -> None:
        self.pending_actions = []
        self.pending_action = PendingAction(tool=tool, args=args)

    def set_pending_actions(self, actions: list[PendingAction]) -> None:
        self.pending_action = None
        self.pending_actions = list(actions)

    def has_pending(self) -> bool:
        return self.pending_action is not None or bool(self.pending_actions)


def is_repairable_command_tool(tool_name: str) -> bool:
    """Return True when a failed approved command can be syntax-repaired and re-approved."""
    return tool_name in REPAIRABLE_COMMAND_TOOLS


def tool_result_indicates_failure(result: Any) -> bool:
    """Detect failure-style tool results returned as strings instead of exceptions."""
    text = str(result or "").strip()
    if not text:
        return False
    return text.startswith("❌") or "\n❌ " in text or " failed (exit=" in text.lower()


def normalize_command_for_tool(tool_name: str, command: str) -> str:
    """Strip any repeated tool binary prefix from a repaired command body."""
    text = str(command or "").strip()
    if not text:
        return ""
    prefix = REPAIRABLE_COMMAND_TOOLS.get(tool_name)
    if prefix and text.lower().startswith(prefix + " "):
        return text[len(prefix) + 1 :].strip()
    return text


def parse_command_repair_response(text: str) -> dict[str, str] | None:
    """Parse the compact command-repair format emitted by the LLM."""
    raw = str(text or "").strip()
    if not raw:
        return None

    parsed: dict[str, str] = {}
    for line in raw.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        parsed[key.strip().lower()] = value.strip()

    status = parsed.get("status", "").lower()
    if status not in {"repair", "cannot_repair"}:
        return None
    return {
        "status": status,
        "summary": parsed.get("summary", ""),
        "command": parsed.get("command", ""),
        "reason": parsed.get("reason", ""),
    }


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
