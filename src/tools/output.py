"""Output compression utilities to keep tool results token-efficient.

Strategy:
- Prefer structure-aware compression over blind truncation.
- Preserve important middle fields in Kubernetes manifests.
- Preserve named sections in command output before falling back to head/tail.
"""

from __future__ import annotations

import json as _json
import re as _re

import yaml as _yaml


_K8S_WORKLOAD_KINDS = frozenset({
    "Pod",
    "Deployment",
    "StatefulSet",
    "DaemonSet",
    "ReplicaSet",
    "Job",
    "CronJob",
})
_SECTION_HEADER_RE = _re.compile(r"^(?:--- .+ ---|== .+ ==|#{1,6}\s.+)$")


def compress_output(
    output: str,
    max_lines: int = 200,
    max_chars: int = 12000,
    *,
    format_hint: str = "",
) -> str:
    """Trim tool output to fit LLM context with structure-aware compression first."""
    text = str(output or "").strip()
    if not text:
        return "(empty output)"
    if len(text) <= max_chars and text.count("\n") + 1 <= max_lines:
        return text

    hint = str(format_hint or "").strip().lower()

    # Kubernetes manifests need smarter handling than head/tail truncation because
    # the most important fields are often in the middle of the spec.
    if hint == "k8s_manifest" or _looks_like_k8s_manifest_text(text):
        manifest_summary = _summarize_k8s_manifest_text(text)
        if manifest_summary:
            if len(manifest_summary) <= max_chars and manifest_summary.count("\n") + 1 <= max_lines:
                return manifest_summary
            text = manifest_summary

    # Try JSON-aware compression before plain text truncation.
    try:
        data = _json.loads(text)
        compressed = _compress_json_value(data, depth=0)
        result = _json.dumps(compressed, indent=2, default=str)
        if len(result) <= max_chars and result.count("\n") + 1 <= max_lines:
            return result
        text = result
    except (ValueError, TypeError):
        pass

    # Preserve named sections in exec/diagnostic output so middle sections survive.
    if hint == "sectioned_text" or _looks_like_sectioned_text(text):
        sectioned = _compress_sectioned_text(text, max_lines=max_lines, max_chars=max_chars)
        if sectioned:
            if len(sectioned) <= max_chars and sectioned.count("\n") + 1 <= max_lines:
                return sectioned
            text = sectioned

    lines = text.splitlines()
    if len(lines) <= max_lines and len(text) <= max_chars:
        return text

    # Generic line-based truncation preserving head + tail.
    if len(lines) > max_lines:
        half = max_lines // 2
        head = lines[:half]
        tail = lines[-half:]
        omitted = len(lines) - max_lines
        text = (
            "\n".join(head)
            + f"\n\n... [{omitted} lines omitted] ...\n\n"
            + "\n".join(tail)
        )

    if len(text) > max_chars:
        head_budget = int(max_chars * 0.8)
        tail_budget = max(max_chars - head_budget - 60, 0)
        text = (
            text[:head_budget]
            + "\n\n... [output truncated] ...\n\n"
            + (text[-tail_budget:] if tail_budget else "")
        )
    return text


# ── JSON-aware compression ───────────────────────────────────────────────

_VERBOSE_KEYS = frozenset({
    "ResponseMetadata", "NextToken", "NextMarker",
    "IpPermissionsEgress",
})

_TRIM_STRING_KEYS = frozenset({
    "UserData", "PolicyDocument", "AssumeRolePolicyDocument",
    "LaunchConfigurationARN", "NotificationConfigurations",
})


def _compress_json_value(obj, depth: int = 0):
    """Recursively slim down a parsed JSON value while preserving all items."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in _VERBOSE_KEYS:
                continue
            if k in _TRIM_STRING_KEYS and isinstance(v, str) and len(v) > 200:
                out[k] = v[:200] + "...[trimmed]"
                continue
            out[k] = _compress_json_value(v, depth + 1)
        return out
    if isinstance(obj, list):
        return [_compress_json_value(item, depth + 1) for item in obj]
    if isinstance(obj, str) and len(obj) > 500 and depth > 2:
        return obj[:500] + "...[trimmed]"
    return obj


def compress_json_output(output: str, max_items: int = 50) -> str:
    """For JSON array output, keep items but note if list was very long."""
    text = str(output or "").strip()
    try:
        data = _json.loads(text)
    except (_json.JSONDecodeError, ValueError):
        return compress_output(text)

    if isinstance(data, list):
        if len(data) > max_items:
            truncated = data[:max_items]
            truncated.append({"_note": f"{len(data) - max_items} more items not shown (total: {len(data)})"})
            data = truncated
        compressed = [_compress_json_value(item) for item in data]
        return _json.dumps(compressed, indent=2, default=str)

    if isinstance(data, dict):
        compressed = _compress_json_value(data)
        return _json.dumps(compressed, indent=2, default=str)

    return compress_output(text)


# ── Kubernetes manifest summarization ────────────────────────────────────


def _looks_like_k8s_manifest_text(text: str) -> bool:
    sample = "\n".join(str(text or "").splitlines()[:20])
    return "apiVersion:" in sample and "kind:" in sample and "metadata:" in sample


def _summarize_k8s_manifest_text(text: str) -> str:
    try:
        docs = [doc for doc in _yaml.safe_load_all(text) if isinstance(doc, dict)]
    except _yaml.YAMLError:
        return ""
    if not docs:
        return ""
    if not all(_is_k8s_manifest_doc(doc) for doc in docs):
        return ""

    payload = [_summarize_k8s_manifest_doc(doc) for doc in docs]
    if len(payload) == 1:
        payload = payload[0]
    return _yaml.safe_dump(payload, sort_keys=False, default_flow_style=False).strip()


def _is_k8s_manifest_doc(doc: dict) -> bool:
    return bool(doc.get("apiVersion")) and bool(doc.get("kind")) and isinstance(doc.get("metadata"), dict)


def _summarize_k8s_manifest_doc(doc: dict) -> dict:
    kind = str(doc.get("kind", "") or "")
    summary = {
        "_note": "Summarized from the full Kubernetes manifest to preserve key spec fields across the entire resource.",
        "apiVersion": doc.get("apiVersion"),
        "kind": kind,
        "metadata": _summarize_manifest_metadata(doc.get("metadata", {})),
    }

    spec = doc.get("spec")
    if isinstance(spec, dict):
        spec_summary = _summarize_manifest_spec(kind, spec)
        if spec_summary:
            summary["spec_summary"] = spec_summary

    status = doc.get("status")
    if isinstance(status, dict):
        status_summary = _summarize_manifest_status(kind, status)
        if status_summary:
            summary["status_summary"] = status_summary
    return summary


def _summarize_manifest_metadata(metadata: dict) -> dict:
    out: dict[str, object] = {}
    for key in ("name", "namespace", "generateName"):
        value = metadata.get(key)
        if value:
            out[key] = value
    labels = metadata.get("labels")
    if isinstance(labels, dict) and labels:
        out["labels"] = labels
    annotations = metadata.get("annotations")
    if isinstance(annotations, dict) and annotations:
        out["annotation_keys"] = sorted(annotations.keys())
    owner_refs = metadata.get("ownerReferences")
    if isinstance(owner_refs, list) and owner_refs:
        out["owners"] = [
            {
                "kind": item.get("kind"),
                "name": item.get("name"),
            }
            for item in owner_refs
            if isinstance(item, dict)
        ]
    return out


def _summarize_manifest_spec(kind: str, spec: dict) -> dict:
    if kind in _K8S_WORKLOAD_KINDS:
        return _summarize_workload_spec(kind, spec)
    if kind == "Service":
        return _summarize_service_spec(spec)
    if kind == "Ingress":
        return _summarize_ingress_spec(spec)
    if kind == "PersistentVolumeClaim":
        return _summarize_pvc_spec(spec)
    if kind == "PersistentVolume":
        return _summarize_pv_spec(spec)
    if kind == "Node":
        return _summarize_node_spec(spec)
    if kind == "HorizontalPodAutoscaler":
        return _summarize_hpa_spec(spec)
    return _compress_json_value(spec)


def _summarize_manifest_status(kind: str, status: dict) -> dict:
    summary: dict[str, object] = {}
    for key in ("phase", "podIP", "hostIP", "replicas", "readyReplicas", "availableReplicas", "updatedReplicas"):
        value = status.get(key)
        if value not in (None, "", []):
            summary[key] = value

    conditions = status.get("conditions")
    if isinstance(conditions, list) and conditions:
        summary["conditions"] = [
            {
                "type": item.get("type"),
                "status": item.get("status"),
                "reason": item.get("reason"),
            }
            for item in conditions
            if isinstance(item, dict)
        ]

    if kind == "Node":
        for key in ("capacity", "allocatable", "nodeInfo"):
            value = status.get(key)
            if isinstance(value, dict) and value:
                summary[key] = value
    return summary


def _summarize_workload_spec(kind: str, spec: dict) -> dict:
    out: dict[str, object] = {}
    for key in ("replicas", "serviceName", "parallelism", "completions", "suspend"):
        value = spec.get(key)
        if value not in (None, "", []):
            out[key] = value

    if kind == "CronJob":
        for key in ("schedule", "concurrencyPolicy", "successfulJobsHistoryLimit", "failedJobsHistoryLimit"):
            value = spec.get(key)
            if value not in (None, "", []):
                out[key] = value
        job_spec = (((spec.get("jobTemplate") or {}).get("spec") or {}).get("template") or {}).get("spec")
        if isinstance(job_spec, dict):
            out["pod_template"] = _summarize_pod_spec(job_spec)
        return out

    if kind in {"Deployment", "StatefulSet", "DaemonSet", "ReplicaSet", "Job"}:
        selector = spec.get("selector")
        if selector:
            out["selector"] = selector
        strategy = spec.get("strategy") or spec.get("updateStrategy")
        if strategy:
            out["strategy"] = strategy
        template_spec = ((spec.get("template") or {}).get("spec"))
        if isinstance(template_spec, dict):
            out["pod_template"] = _summarize_pod_spec(template_spec)
        return out

    # Pod
    return _summarize_pod_spec(spec)


def _summarize_pod_spec(spec: dict) -> dict:
    out: dict[str, object] = {}
    for key in ("serviceAccountName", "restartPolicy", "priorityClassName", "nodeName", "hostNetwork", "dnsPolicy"):
        value = spec.get(key)
        if value not in (None, "", []):
            out[key] = value

    if spec.get("nodeSelector"):
        out["nodeSelector"] = spec.get("nodeSelector")
    if spec.get("tolerations"):
        out["tolerations"] = spec.get("tolerations")
    if spec.get("affinity"):
        out["affinity"] = _compress_json_value(spec.get("affinity"))
    if spec.get("securityContext"):
        out["podSecurityContext"] = _compress_json_value(spec.get("securityContext"))

    containers = spec.get("containers")
    if isinstance(containers, list) and containers:
        out["containers"] = [_summarize_container(item) for item in containers if isinstance(item, dict)]

    init_containers = spec.get("initContainers")
    if isinstance(init_containers, list) and init_containers:
        out["initContainers"] = [_summarize_container(item) for item in init_containers if isinstance(item, dict)]

    volumes = spec.get("volumes")
    if isinstance(volumes, list) and volumes:
        out["volumes"] = [_summarize_volume(item) for item in volumes if isinstance(item, dict)]
    return out


def _summarize_container(container: dict) -> dict:
    out: dict[str, object] = {}
    for key in ("name", "image", "imagePullPolicy", "workingDir"):
        value = container.get(key)
        if value not in (None, "", []):
            out[key] = value
    if container.get("command"):
        out["command"] = container.get("command")
    if container.get("args"):
        out["args"] = container.get("args")
    if container.get("ports"):
        out["ports"] = [
            {
                "name": port.get("name"),
                "containerPort": port.get("containerPort"),
                "protocol": port.get("protocol"),
            }
            for port in container.get("ports", [])
            if isinstance(port, dict)
        ]
    resources = container.get("resources")
    if isinstance(resources, dict) and resources:
        out["resources"] = {
            key: value
            for key, value in resources.items()
            if value not in (None, "", {})
        }
    env = container.get("env")
    if isinstance(env, list) and env:
        out["env"] = [_summarize_env_var(item) for item in env if isinstance(item, dict)]
    env_from = container.get("envFrom")
    if isinstance(env_from, list) and env_from:
        out["envFrom"] = [_compress_json_value(item) for item in env_from if isinstance(item, dict)]
    for probe_key in ("startupProbe", "readinessProbe", "livenessProbe"):
        probe = container.get(probe_key)
        if isinstance(probe, dict) and probe:
            out[probe_key] = _summarize_probe(probe)
    mounts = container.get("volumeMounts")
    if isinstance(mounts, list) and mounts:
        out["volumeMounts"] = [
            {
                "name": mount.get("name"),
                "mountPath": mount.get("mountPath"),
                "readOnly": mount.get("readOnly", False),
                "subPath": mount.get("subPath"),
            }
            for mount in mounts
            if isinstance(mount, dict)
        ]
    if container.get("securityContext"):
        out["securityContext"] = _compress_json_value(container.get("securityContext"))
    return out


def _summarize_env_var(env_var: dict) -> dict:
    out = {"name": env_var.get("name")}
    if "value" in env_var:
        value = str(env_var.get("value", "") or "")
        out["value"] = value if len(value) <= 120 else value[:120] + "...[trimmed]"
    elif "valueFrom" in env_var:
        out["valueFrom"] = _compress_json_value(env_var.get("valueFrom"))
    return out


def _summarize_probe(probe: dict) -> dict:
    out: dict[str, object] = {}
    for key in ("initialDelaySeconds", "periodSeconds", "timeoutSeconds", "failureThreshold", "successThreshold"):
        value = probe.get(key)
        if value not in (None, ""):
            out[key] = value
    for key in ("httpGet", "exec", "tcpSocket", "grpc"):
        value = probe.get(key)
        if isinstance(value, dict) and value:
            out[key] = _compress_json_value(value)
    return out


def _summarize_volume(volume: dict) -> dict:
    out = {"name": volume.get("name")}
    for key in (
        "persistentVolumeClaim",
        "configMap",
        "secret",
        "emptyDir",
        "projected",
        "hostPath",
        "csi",
        "downwardAPI",
    ):
        value = volume.get(key)
        if isinstance(value, dict) and value:
            out[key] = _compress_json_value(value)
            break
    return out


def _summarize_service_spec(spec: dict) -> dict:
    out: dict[str, object] = {}
    for key in ("type", "clusterIP", "clusterIPs", "sessionAffinity", "externalTrafficPolicy"):
        value = spec.get(key)
        if value not in (None, "", []):
            out[key] = value
    if spec.get("selector"):
        out["selector"] = spec.get("selector")
    if spec.get("ports"):
        out["ports"] = [
            {
                "name": item.get("name"),
                "port": item.get("port"),
                "targetPort": item.get("targetPort"),
                "protocol": item.get("protocol"),
            }
            for item in spec.get("ports", [])
            if isinstance(item, dict)
        ]
    return out


def _summarize_ingress_spec(spec: dict) -> dict:
    out: dict[str, object] = {}
    if spec.get("ingressClassName"):
        out["ingressClassName"] = spec.get("ingressClassName")
    if spec.get("tls"):
        out["tls"] = _compress_json_value(spec.get("tls"))
    if spec.get("rules"):
        out["rules"] = _compress_json_value(spec.get("rules"))
    if spec.get("defaultBackend"):
        out["defaultBackend"] = _compress_json_value(spec.get("defaultBackend"))
    return out


def _summarize_pvc_spec(spec: dict) -> dict:
    out: dict[str, object] = {}
    for key in ("storageClassName", "volumeName"):
        value = spec.get(key)
        if value not in (None, "", []):
            out[key] = value
    if spec.get("accessModes"):
        out["accessModes"] = spec.get("accessModes")
    if spec.get("resources"):
        out["resources"] = _compress_json_value(spec.get("resources"))
    return out


def _summarize_pv_spec(spec: dict) -> dict:
    out: dict[str, object] = {}
    for key in ("storageClassName", "persistentVolumeReclaimPolicy", "volumeMode"):
        value = spec.get(key)
        if value not in (None, "", []):
            out[key] = value
    if spec.get("capacity"):
        out["capacity"] = spec.get("capacity")
    if spec.get("accessModes"):
        out["accessModes"] = spec.get("accessModes")
    for key in ("claimRef", "csi", "awsElasticBlockStore", "hostPath", "local", "nodeAffinity"):
        value = spec.get(key)
        if isinstance(value, dict) and value:
            out[key] = _compress_json_value(value)
    return out


def _summarize_node_spec(spec: dict) -> dict:
    out: dict[str, object] = {}
    if spec.get("providerID"):
        out["providerID"] = spec.get("providerID")
    if spec.get("taints"):
        out["taints"] = spec.get("taints")
    if spec.get("unschedulable") is not None:
        out["unschedulable"] = spec.get("unschedulable")
    return out


def _summarize_hpa_spec(spec: dict) -> dict:
    out: dict[str, object] = {}
    for key in ("minReplicas", "maxReplicas", "behavior"):
        value = spec.get(key)
        if value not in (None, "", []):
            out[key] = _compress_json_value(value)
    if spec.get("scaleTargetRef"):
        out["scaleTargetRef"] = _compress_json_value(spec.get("scaleTargetRef"))
    if spec.get("metrics"):
        out["metrics"] = _compress_json_value(spec.get("metrics"))
    return out


# ── Section-aware text compression ───────────────────────────────────────


def _looks_like_sectioned_text(text: str) -> bool:
    lines = str(text or "").splitlines()
    return sum(1 for line in lines if _SECTION_HEADER_RE.match(line.strip())) >= 2


def _compress_sectioned_text(text: str, *, max_lines: int, max_chars: int) -> str:
    sections = _split_named_sections(text)
    if len(sections) < 2:
        return ""

    line_budget = max(max_lines - len(sections), len(sections) * 3)
    body_budget = max(2, line_budget // len(sections) - 1)
    rendered: list[str] = []

    for header, body in sections:
        if header:
            rendered.append(header)
        keep = body if len(body) <= body_budget else body[:body_budget]
        rendered.extend(keep)
        omitted = len(body) - len(keep)
        if omitted > 0:
            rendered.append(f"... [{omitted} lines omitted in section] ...")

    result = "\n".join(rendered).strip()
    if len(result) > max_chars:
        head_budget = int(max_chars * 0.85)
        tail_budget = max(max_chars - head_budget - 60, 0)
        result = (
            result[:head_budget]
            + "\n\n... [section output truncated] ...\n\n"
            + (result[-tail_budget:] if tail_budget else "")
        )
    return result


def _split_named_sections(text: str) -> list[tuple[str, list[str]]]:
    sections: list[tuple[str, list[str]]] = []
    current_header = ""
    current_body: list[str] = []

    for raw_line in str(text or "").splitlines():
        line = raw_line.rstrip()
        if _SECTION_HEADER_RE.match(line.strip()):
            if current_header or current_body:
                sections.append((current_header, current_body))
            current_header = line
            current_body = []
        else:
            current_body.append(line)
    if current_header or current_body:
        sections.append((current_header, current_body))
    return sections
