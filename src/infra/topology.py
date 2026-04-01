"""Infrastructure topology discovery and graph."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from ..config import Config
from .k8s_client import K8sClient
from .aws_client import AWSClient

log = logging.getLogger(__name__)


@dataclass
class InfraNode:
    kind: str
    name: str
    namespace: str | None = None
    provider: str = "k8s"
    metadata: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        parts = [self.provider, self.kind]
        if self.namespace:
            parts.append(self.namespace)
        parts.append(self.name)
        return "/".join(parts)


@dataclass
class InfraEdge:
    source_id: str
    target_id: str
    relationship: str


@dataclass
class TopologyGraph:
    nodes: dict[str, InfraNode] = field(default_factory=dict)
    edges: list[InfraEdge] = field(default_factory=list)

    def add_node(self, node: InfraNode) -> None:
        self.nodes[node.id] = node

    def add_edge(self, source_id: str, target_id: str, relationship: str) -> None:
        if source_id in self.nodes and target_id in self.nodes:
            self.edges.append(InfraEdge(source_id, target_id, relationship))

    def to_summary(self, max_nodes: int = 60, max_edges: int = 40) -> str:
        lines = [f"Infrastructure Topology: {len(self.nodes)} resources, {len(self.edges)} relationships\n"]
        by_kind: dict[str, list[InfraNode]] = {}
        for node in self.nodes.values():
            key = f"{node.provider}/{node.kind}"
            by_kind.setdefault(key, []).append(node)
        for kind, nodes in sorted(by_kind.items()):
            names = [n.name for n in nodes[:15]]
            suffix = f" (+{len(nodes)-15} more)" if len(nodes) > 15 else ""
            lines.append(f"  {kind}: {', '.join(names)}{suffix}")
        if self.edges:
            lines.append("\nRelationships:")
            for edge in self.edges[:max_edges]:
                lines.append(f"  {edge.source_id} --[{edge.relationship}]--> {edge.target_id}")
            if len(self.edges) > max_edges:
                lines.append(f"  ... and {len(self.edges) - max_edges} more")
        return "\n".join(lines)

    def get_neighbors(self, node_id: str) -> list[tuple[str, str, str]]:
        neighbors: list[tuple[str, str, str]] = []
        for edge in self.edges:
            if edge.source_id == node_id:
                neighbors.append((edge.target_id, edge.relationship, "outgoing"))
            elif edge.target_id == node_id:
                neighbors.append((edge.source_id, edge.relationship, "incoming"))
        return neighbors

    def find_nodes(self, kind: str | None = None, name_contains: str | None = None) -> list[InfraNode]:
        results = []
        for node in self.nodes.values():
            if kind and node.kind != kind:
                continue
            if name_contains and name_contains.lower() not in node.name.lower():
                continue
            results.append(node)
        return results


class TopologyBuilder:
    """Crawls Kubernetes and AWS to build an infrastructure graph."""

    def __init__(self, k8s: K8sClient, aws: AWSClient) -> None:
        self.k8s = k8s
        self.aws = aws

    def build(self, namespace: str | None = None) -> TopologyGraph:
        graph = TopologyGraph()
        self._build_k8s(graph, namespace)
        if Config.AWS_CLI_ENABLED:
            try:
                self._build_aws(graph)
                self._map_k8s_to_aws(graph)
            except Exception as exc:
                log.warning("AWS topology build failed: %s", exc)
        return graph

    def _build_k8s(self, graph: TopologyGraph, namespace: str | None = None) -> None:
        ns = namespace or self.k8s.namespace

        # Deployments
        data = self.k8s.run_json(["get", "deployments"], namespace=ns)
        if isinstance(data, dict):
            for item in data.get("items", []):
                meta = item.get("metadata", {})
                spec = item.get("spec", {})
                status = item.get("status", {})
                graph.add_node(InfraNode(
                    kind="deployment",
                    name=meta.get("name", ""),
                    namespace=meta.get("namespace"),
                    metadata={
                        "replicas": spec.get("replicas"),
                        "ready": status.get("readyReplicas", 0),
                        "updated": status.get("updatedReplicas", 0),
                        "selector": spec.get("selector", {}).get("matchLabels", {}),
                    },
                ))

        # Services
        data = self.k8s.run_json(["get", "services"], namespace=ns)
        if isinstance(data, dict):
            for item in data.get("items", []):
                meta = item.get("metadata", {})
                spec = item.get("spec", {})
                svc_node = InfraNode(
                    kind="service",
                    name=meta.get("name", ""),
                    namespace=meta.get("namespace"),
                    metadata={
                        "type": spec.get("type"),
                        "cluster_ip": spec.get("clusterIP"),
                        "selector": spec.get("selector", {}),
                        "ports": [
                            {"port": p.get("port"), "target": p.get("targetPort"), "protocol": p.get("protocol")}
                            for p in spec.get("ports", [])
                        ],
                    },
                )
                graph.add_node(svc_node)
                selector = spec.get("selector") or {}
                if selector:
                    for dep in graph.nodes.values():
                        if dep.kind == "deployment" and dep.namespace == meta.get("namespace"):
                            dep_sel = dep.metadata.get("selector", {})
                            if dep_sel and all(dep_sel.get(k) == v for k, v in selector.items()):
                                graph.add_edge(svc_node.id, dep.id, "routes_to")

        # Pods
        data = self.k8s.run_json(["get", "pods"], namespace=ns)
        if isinstance(data, dict):
            for item in data.get("items", []):
                meta = item.get("metadata", {})
                status = item.get("status", {})
                pod_node = InfraNode(
                    kind="pod",
                    name=meta.get("name", ""),
                    namespace=meta.get("namespace"),
                    metadata={
                        "phase": status.get("phase"),
                        "node_name": item.get("spec", {}).get("nodeName"),
                        "restart_count": sum(
                            cs.get("restartCount", 0)
                            for cs in status.get("containerStatuses", [])
                        ),
                        "ready": all(
                            cs.get("ready", False)
                            for cs in status.get("containerStatuses", [])
                        ) if status.get("containerStatuses") else False,
                    },
                )
                graph.add_node(pod_node)
                # Link pod → deployment via name prefix
                for dep in list(graph.nodes.values()):
                    if dep.kind == "deployment" and dep.namespace == meta.get("namespace"):
                        if meta.get("name", "").startswith(dep.name + "-"):
                            graph.add_edge(dep.id, pod_node.id, "manages")
                            break

        # Ingresses
        data = self.k8s.run_json(["get", "ingresses"], namespace=ns)
        if isinstance(data, dict):
            for item in data.get("items", []):
                meta = item.get("metadata", {})
                spec = item.get("spec", {})
                ing_node = InfraNode(
                    kind="ingress",
                    name=meta.get("name", ""),
                    namespace=meta.get("namespace"),
                    metadata={
                        "hosts": [r.get("host") for r in spec.get("rules", [])],
                    },
                )
                graph.add_node(ing_node)
                for rule in spec.get("rules", []):
                    for path in rule.get("http", {}).get("paths", []):
                        svc_name = path.get("backend", {}).get("service", {}).get("name")
                        if svc_name:
                            svc_id = f"k8s/service/{meta.get('namespace')}/{svc_name}"
                            graph.add_edge(ing_node.id, svc_id, "routes_to")

        # Nodes
        data = self.k8s.run_json(["get", "nodes"])
        if isinstance(data, dict):
            for item in data.get("items", []):
                meta = item.get("metadata", {})
                status = item.get("status", {})
                conditions = {c["type"]: c["status"] for c in status.get("conditions", [])}
                labels = meta.get("labels", {})
                k8s_node = InfraNode(
                    kind="node",
                    name=meta.get("name", ""),
                    metadata={
                        "ready": conditions.get("Ready") == "True",
                        "disk_pressure": conditions.get("DiskPressure") == "True",
                        "memory_pressure": conditions.get("MemoryPressure") == "True",
                        "instance_type": labels.get("node.kubernetes.io/instance-type"),
                        "zone": labels.get("topology.kubernetes.io/zone"),
                        "provider_id": item.get("spec", {}).get("providerID", ""),
                    },
                )
                graph.add_node(k8s_node)
                # Link pods to their nodes
                for pod in graph.nodes.values():
                    if pod.kind == "pod" and pod.metadata.get("node_name") == meta.get("name"):
                        graph.add_edge(k8s_node.id, pod.id, "hosts")

    def _build_aws(self, graph: TopologyGraph) -> None:
        result = self.aws.describe_instances()
        if not result.startswith("ERROR"):
            try:
                for inst in json.loads(result):
                    graph.add_node(InfraNode(
                        kind="ec2",
                        name=inst.get("name") or inst["id"],
                        provider="aws",
                        metadata=inst,
                    ))
            except (json.JSONDecodeError, KeyError):
                pass

    def _map_k8s_to_aws(self, graph: TopologyGraph) -> None:
        aws_ec2 = [n for n in graph.nodes.values() if n.provider == "aws" and n.kind == "ec2"]
        for k8s_node in graph.nodes.values():
            if k8s_node.kind != "node" or k8s_node.provider != "k8s":
                continue
            provider_id = k8s_node.metadata.get("provider_id", "")
            for ec2 in aws_ec2:
                ec2_id = ec2.metadata.get("id", "")
                ec2_ip = ec2.metadata.get("private_ip", "")
                if ec2_id and ec2_id in provider_id:
                    graph.add_edge(k8s_node.id, ec2.id, "runs_on")
                elif ec2_ip and ec2_ip in k8s_node.name:
                    graph.add_edge(k8s_node.id, ec2.id, "runs_on")


class TopologyCache:
    """Lazy-loading, time-limited topology cache."""

    def __init__(self, builder: TopologyBuilder, ttl_seconds: int = 300) -> None:
        self._builder = builder
        self._ttl = ttl_seconds
        self._graph: TopologyGraph | None = None
        self._built_at: float = 0.0

    def get(self, namespace: str | None = None, force_refresh: bool = False) -> TopologyGraph:
        now = time.monotonic()
        if force_refresh or self._graph is None or (now - self._built_at) > self._ttl:
            self._graph = self._builder.build(namespace)
            self._built_at = now
        return self._graph

    def invalidate(self) -> None:
        self._graph = None
        self._built_at = 0.0
