"""Connector factory for the rebuilt AI Ops backend."""

from __future__ import annotations

from dataclasses import dataclass

from .aws_cli import AWSConnector
from .helm_cli import HelmConnector
from .k8s_cli import KubernetesConnector


@dataclass
class ConnectorSuite:
    kubernetes: KubernetesConnector
    aws: AWSConnector
    helm: HelmConnector


def build_connectors() -> ConnectorSuite:
    return ConnectorSuite(
        kubernetes=KubernetesConnector(),
        aws=AWSConnector(),
        helm=HelmConnector(),
    )


__all__ = ["AWSConnector", "HelmConnector", "KubernetesConnector", "ConnectorSuite", "build_connectors"]
