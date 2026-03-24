"""
Tools package for the AI agent
"""
from .k8s_diagnostics import (
    get_k8s_read_tools,
    k8s_current_context,
    k8s_list_contexts,
    k8s_cluster_info,
    k8s_version,
    k8s_api_resources,
    k8s_list_namespaces,
    k8s_list_nodes,
    k8s_top_nodes,
    k8s_list_pods,
    k8s_find_pods,
    k8s_describe_pod,
    k8s_get_pod_logs,
    k8s_top_pods,
    k8s_list_deployments,
    k8s_describe_deployment,
    k8s_list_statefulsets,
    k8s_list_daemonsets,
    k8s_list_services,
    k8s_list_secrets,
    k8s_list_ingresses,
    k8s_list_hpa,
    k8s_get_events,
    k8s_get_resource_quotas,
    k8s_get_pvcs,
    k8s_list_pvs,
    k8s_describe_pvc,
    k8s_describe_pv,
    k8s_describe_node,
    k8s_get_resource_yaml,
    k8s_get_pod_scheduling_report,
    k8s_get_crashloop_pods,
)
from .actions import (
    restart_kubernetes_pod,
    restart_kubernetes_pods_batch,
    scale_kubernetes_deployment,
    scale_kubernetes_statefulset,
    scale_kubernetes_workloads_batch,
    rollout_restart_kubernetes_deployment,
    rollout_restart_kubernetes_statefulset,
    rollout_restart_kubernetes_daemonset,
    rollout_restart_kubernetes_workloads_batch,
)
from .aws_cli import (
    get_aws_tools,
    aws_cli_readonly,
    aws_cli_execute,
)
from .k8s_cli import (
    get_k8s_cli_tools,
    kubectl_readonly,
    kubectl_execute,
)
from .helm_cli import (
    get_helm_tools,
    helm_readonly,
    helm_execute,
)


# Tool policy
# Keep it simple: anything that can change infra/state must be explicitly approved.
WRITE_TOOL_NAMES = {
    "restart_kubernetes_pod",
    "restart_kubernetes_pods_batch",
    "scale_kubernetes_deployment",
    "scale_kubernetes_statefulset",
    "scale_kubernetes_workloads_batch",
    "rollout_restart_kubernetes_deployment",
    "rollout_restart_kubernetes_statefulset",
    "rollout_restart_kubernetes_daemonset",
    "rollout_restart_kubernetes_workloads_batch",
    "kubectl_execute",
    "helm_execute",
    "aws_cli_execute",
}


def is_write_tool(tool_name: str) -> bool:
    """Return True if tool mutates state and requires user approval."""
    return tool_name in WRITE_TOOL_NAMES


_WRITE_TOOLS = [
    restart_kubernetes_pod,
    restart_kubernetes_pods_batch,
    scale_kubernetes_deployment,
    scale_kubernetes_statefulset,
    scale_kubernetes_workloads_batch,
    rollout_restart_kubernetes_deployment,
    rollout_restart_kubernetes_statefulset,
    rollout_restart_kubernetes_daemonset,
    rollout_restart_kubernetes_workloads_batch,
]


def get_all_tools():
    """Get all available tools for the agent"""
    return [*get_k8s_read_tools(), *get_k8s_cli_tools(), *get_helm_tools(), *get_aws_tools(), *_WRITE_TOOLS]


__all__ = [
    'k8s_current_context',
    'k8s_list_contexts',
    'k8s_cluster_info',
    'k8s_version',
    'k8s_api_resources',
    'k8s_list_namespaces',
    'k8s_list_nodes',
    'k8s_top_nodes',
    'k8s_list_pods',
    'k8s_find_pods',
    'k8s_describe_pod',
    'k8s_get_pod_logs',
    'k8s_top_pods',
    'k8s_list_deployments',
    'k8s_describe_deployment',
    'k8s_list_statefulsets',
    'k8s_list_daemonsets',
    'k8s_list_services',
    'k8s_list_secrets',
    'k8s_list_ingresses',
    'k8s_list_hpa',
    'k8s_get_events',
    'k8s_get_resource_quotas',
    'k8s_get_pvcs',
    'k8s_list_pvs',
    'k8s_describe_pvc',
    'k8s_describe_pv',
    'k8s_describe_node',
    'k8s_get_resource_yaml',
    'k8s_get_pod_scheduling_report',
    'k8s_get_crashloop_pods',
    'kubectl_readonly',
    'kubectl_execute',
    'get_k8s_cli_tools',
    'helm_readonly',
    'helm_execute',
    'get_helm_tools',
    'get_k8s_read_tools',
    'restart_kubernetes_pod',
    'restart_kubernetes_pods_batch',
    'scale_kubernetes_deployment',
    'scale_kubernetes_statefulset',
    'scale_kubernetes_workloads_batch',
    'rollout_restart_kubernetes_deployment',
    'rollout_restart_kubernetes_statefulset',
    'rollout_restart_kubernetes_daemonset',
    'rollout_restart_kubernetes_workloads_batch',
    'aws_cli_readonly',
    'aws_cli_execute',
    'get_aws_tools',
    'WRITE_TOOL_NAMES',
    'is_write_tool',
    'get_all_tools'
]
