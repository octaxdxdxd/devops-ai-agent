"""Cross-layer correlation tools that join Kubernetes state with AWS infrastructure."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import tool

from ..infra.aws_client import AWSClient
from ..infra.k8s_client import K8sClient
from ..policy import guard_k8s_read_tool
from .output import compress_json_output


def _parse_instance_id(provider_id: str) -> str:
    text = str(provider_id or "").strip()
    if not text:
        return ""
    if "/" in text:
        text = text.rsplit("/", 1)[-1]
    return text if text.startswith("i-") else ""


def _owner_refs(metadata: dict[str, Any]) -> list[dict[str, str]]:
    owners: list[dict[str, str]] = []
    for owner in metadata.get("ownerReferences", []) or []:
        owners.append({
            "kind": str(owner.get("kind", "") or ""),
            "name": str(owner.get("name", "") or ""),
        })
    return owners


def _selector_from_labels(match_labels: dict[str, Any]) -> str:
    pairs = []
    for key, value in (match_labels or {}).items():
        key_text = str(key or "").strip()
        value_text = str(value or "").strip()
        if key_text and value_text:
            pairs.append(f"{key_text}={value_text}")
    return ",".join(pairs)


def _pod_storage_details(k8s: K8sClient, pod: dict[str, Any], namespace: str) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for volume in pod.get("spec", {}).get("volumes", []) or []:
        claim_name = str(volume.get("persistentVolumeClaim", {}).get("claimName", "") or "")
        if not claim_name:
            continue

        pvc_obj = k8s.get_resource_json("pvc", claim_name, namespace)
        pvc_data = pvc_obj if isinstance(pvc_obj, dict) else {}
        pv_name = str(pvc_data.get("spec", {}).get("volumeName", "") or "")
        pv_data: dict[str, Any] = {}
        if pv_name:
            pv_obj = k8s.get_resource_json("pv", pv_name, None)
            if isinstance(pv_obj, dict):
                pv_data = pv_obj

        csi = pv_data.get("spec", {}).get("csi", {}) if pv_data else {}
        aws_ebs = pv_data.get("spec", {}).get("awsElasticBlockStore", {}) if pv_data else {}
        volume_handle = str(csi.get("volumeHandle", "") or aws_ebs.get("volumeID", "") or "")

        details.append({
            "claim_name": claim_name,
            "persistent_volume": pv_name,
            "storage_class": str(pvc_data.get("spec", {}).get("storageClassName", "") or ""),
            "volume_handle": volume_handle,
            "access_modes": list(pvc_data.get("status", {}).get("accessModes", []) or []),
            "phase": str(pvc_data.get("status", {}).get("phase", "") or ""),
        })
    return details


def _correlate_node(k8s: K8sClient, aws: AWSClient, node_name: str) -> dict[str, Any]:
    node_obj = k8s.get_resource_json("node", node_name, None)
    if isinstance(node_obj, str):
        return {"error": node_obj, "node_name": node_name}

    metadata = node_obj.get("metadata", {})
    labels = metadata.get("labels", {}) or {}
    spec = node_obj.get("spec", {}) or {}
    status = node_obj.get("status", {}) or {}
    provider_id = str(spec.get("providerID", "") or "")
    instance_id = _parse_instance_id(provider_id)

    result: dict[str, Any] = {
        "node_name": node_name,
        "provider_id": provider_id,
        "instance_id": instance_id,
        "zone": str(labels.get("topology.kubernetes.io/zone", "") or ""),
        "instance_type": str(labels.get("node.kubernetes.io/instance-type", "") or ""),
        "capacity_type": str(labels.get("eks.amazonaws.com/capacityType", "") or ""),
        "node_group": str(labels.get("eks.amazonaws.com/nodegroup", "") or ""),
        "internal_ips": [
            str(address.get("address", "") or "")
            for address in status.get("addresses", []) or []
            if str(address.get("type", "") or "") == "InternalIP"
        ],
    }

    aws_instance = aws.get_instance_details(
        instance_id=instance_id,
        private_dns_name="" if instance_id else node_name,
    )
    if isinstance(aws_instance, str):
        result["aws_error"] = aws_instance
        return result
    if not aws_instance:
        return result

    result["ec2"] = aws_instance
    asg_name = str(aws_instance.get("autoscaling_group_name", "") or "")
    if asg_name:
        asg_details = aws.get_auto_scaling_group_details(asg_name)
        if isinstance(asg_details, str):
            result["autoscaling_error"] = asg_details
        elif asg_details:
            result["autoscaling_group"] = asg_details
    return result


def _pod_summary(pod: dict[str, Any]) -> dict[str, Any]:
    statuses = pod.get("status", {}).get("containerStatuses", []) or []
    return {
        "name": str(pod.get("metadata", {}).get("name", "") or ""),
        "phase": str(pod.get("status", {}).get("phase", "") or ""),
        "node": str(pod.get("spec", {}).get("nodeName", "") or ""),
        "pod_ip": str(pod.get("status", {}).get("podIP", "") or ""),
        "restarts": sum(int(status.get("restartCount", 0) or 0) for status in statuses),
    }


def create_correlation_read_tools(k8s: K8sClient, aws: AWSClient) -> list:
    """Return read-only tools that correlate Kubernetes resources with AWS backing infrastructure."""

    @tool
    def infra_correlate_k8s_resource(kind: str, name: str, namespace: str = "") -> str:
        """Correlate a Kubernetes pod, deployment, statefulset, or node with its backing AWS infrastructure.

        Args:
            kind: One of 'pod', 'deployment', 'statefulset', or 'node'
            name: Resource name
            namespace: Namespace for namespaced resources
        """
        normalized_kind = str(kind or "").strip().lower()
        namespaced = normalized_kind in {"pod", "deployment", "statefulset"}
        policy_error = guard_k8s_read_tool(
            "infra_correlate_k8s_resource",
            namespace=namespace if namespaced else "",
        )
        if policy_error:
            return f"ERROR: {policy_error}"

        if normalized_kind == "node":
            payload = {"resource": {"kind": "node", "name": name}, "correlation": _correlate_node(k8s, aws, name)}
            return compress_json_output(json.dumps(payload, indent=2, default=str))

        resource_obj = k8s.get_resource_json(normalized_kind, name, namespace or None)
        if isinstance(resource_obj, str):
            return resource_obj

        metadata = resource_obj.get("metadata", {})
        base_payload: dict[str, Any] = {
            "resource": {
                "kind": normalized_kind,
                "name": str(metadata.get("name", "") or name),
                "namespace": str(metadata.get("namespace", "") or namespace or ""),
                "uid": str(metadata.get("uid", "") or ""),
                "owners": _owner_refs(metadata),
            }
        }

        if normalized_kind == "pod":
            node_name = str(resource_obj.get("spec", {}).get("nodeName", "") or "")
            resource_namespace = str(metadata.get("namespace", "") or namespace or "")
            base_payload["pod"] = _pod_summary(resource_obj)
            base_payload["storage"] = _pod_storage_details(k8s, resource_obj, resource_namespace)
            if node_name:
                base_payload["node_correlation"] = _correlate_node(k8s, aws, node_name)
            return compress_json_output(json.dumps(base_payload, indent=2, default=str))

        selector = _selector_from_labels(resource_obj.get("spec", {}).get("selector", {}).get("matchLabels", {}) or {})
        pods_obj = (
            k8s.list_resources_json("pods", namespace or None, label_selector=selector)
            if selector
            else {"items": []}
        )
        pods = pods_obj.get("items", []) if isinstance(pods_obj, dict) else []
        node_names = []
        for pod in pods:
            node_name = str(pod.get("spec", {}).get("nodeName", "") or "")
            if node_name and node_name not in node_names:
                node_names.append(node_name)

        base_payload["workload"] = {
            "replicas": resource_obj.get("spec", {}).get("replicas"),
            "available_replicas": resource_obj.get("status", {}).get("availableReplicas"),
            "selector": selector,
            "pods": [_pod_summary(pod) for pod in pods],
        }
        base_payload["node_correlation"] = [_correlate_node(k8s, aws, node_name) for node_name in node_names]
        return compress_json_output(json.dumps(base_payload, indent=2, default=str))

    return [infra_correlate_k8s_resource]
