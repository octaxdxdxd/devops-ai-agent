"""Central tool registry that creates and organises all LangChain tools."""

from __future__ import annotations

import logging

from ..config import Config
from ..infra.k8s_client import K8sClient
from ..infra.aws_client import AWSClient
from .correlation_read import create_correlation_read_tools
from .k8s_read import create_k8s_read_tools
from .k8s_write import create_k8s_write_tools
from .aws_read import create_aws_read_tools
from .aws_write import create_aws_write_tools

log = logging.getLogger(__name__)


class ToolRegistry:
    """Holds all available tools, separated by permission level."""

    def __init__(self, k8s: K8sClient, aws: AWSClient) -> None:
        self.k8s = k8s
        self.aws = aws

        # Build tools for available backends
        self.k8s_read_tools: list = []
        self.k8s_write_tools: list = []
        self.aws_read_tools: list = []
        self.aws_write_tools: list = []
        self.correlation_read_tools: list = []

        if k8s.available():
            self.k8s_read_tools = create_k8s_read_tools(k8s)
            self.k8s_write_tools = create_k8s_write_tools(k8s)
            log.info("K8s tools loaded: %d read, %d write", len(self.k8s_read_tools), len(self.k8s_write_tools))
        else:
            log.warning("kubectl not available — K8s tools disabled")

        if aws.available():
            self.aws_read_tools = create_aws_read_tools(aws)
            self.aws_write_tools = create_aws_write_tools(aws)
            log.info("AWS tools loaded: %d read, %d write", len(self.aws_read_tools), len(self.aws_write_tools))
        else:
            log.warning("AWS credentials not available — AWS tools disabled")

        if self.k8s_read_tools and self.aws_read_tools:
            self.correlation_read_tools = create_correlation_read_tools(k8s, aws)
            log.info("Correlation tools loaded: %d read", len(self.correlation_read_tools))

        # Aggregated views
        self.read_tools = self.k8s_read_tools + self.aws_read_tools + self.correlation_read_tools
        self.write_tools = self.k8s_write_tools + self.aws_write_tools
        self.all_tools = self.read_tools + self.write_tools

        # Name → tool for direct execution
        self.tool_map: dict[str, object] = {t.name: t for t in self.all_tools}

    def execute(self, tool_name: str, args: dict, *, approved: bool = False) -> str:
        tool = self.tool_map.get(tool_name)
        if not tool:
            return f"ERROR: Unknown tool '{tool_name}'"
        if self.is_write_tool(tool_name) and not approved:
            return (
                f"ERROR: Tool '{tool_name}' requires explicit approval before execution."
            )
        try:
            return str(tool.invoke(args))
        except Exception as exc:
            return f"ERROR executing {tool_name}: {exc}"

    def is_write_tool(self, tool_name: str) -> bool:
        write_names = {t.name for t in self.write_tools}
        return tool_name in write_names

    def get_tool_names(self) -> list[str]:
        return list(self.tool_map.keys())
