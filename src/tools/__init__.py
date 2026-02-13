"""
Tools package for the AI agent
"""
from .log_reader import read_log_file, list_log_files, search_logs, get_log_tools
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
    tools = get_log_tools()
    tools.append(restart_kubernetes_pod)
    return tools


__all__ = [
    'read_log_file', 
    'list_log_files', 
    'search_logs', 
    'get_log_tools',
    'restart_kubernetes_pod',
    'WRITE_TOOL_NAMES',
    'is_write_tool',
    'get_all_tools'
]