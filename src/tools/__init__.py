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
    k8s_list_ingresses,
    k8s_list_hpa,
    k8s_get_events,
    k8s_get_resource_quotas,
    k8s_get_pvcs,
    k8s_get_crashloop_pods,
)
from .actions import restart_kubernetes_pod


# Tool policy
# Keep it simple: anything that can change infra/state must be explicitly approved.
WRITE_TOOL_NAMES = {
    "restart_kubernetes_pod",
}


def is_write_tool(tool_name: str) -> bool:
    """Return True if tool mutates state and requires user approval."""
    return tool_name in WRITE_TOOL_NAMES


def get_all_tools():
    """Get all available tools for the agent"""
    tools = get_k8s_read_tools()
    tools.append(restart_kubernetes_pod)
    return tools


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
    'k8s_list_ingresses',
    'k8s_list_hpa',
    'k8s_get_events',
    'k8s_get_resource_quotas',
    'k8s_get_pvcs',
    'k8s_get_crashloop_pods',
    'get_k8s_read_tools',
    'restart_kubernetes_pod',
    'WRITE_TOOL_NAMES',
    'is_write_tool',
    'get_all_tools'
]