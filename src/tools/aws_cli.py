"""AWS connector for semantic investigations and approval-gated writes."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import os
import re
import shutil
from typing import Any

from ..config import Config
from .common import ToolObservation, format_error_summary, parse_json_output, run_subprocess, shell_split, truncate_text


_GLOBAL_SERVICES = {"iam", "route53", "organizations", "cloudfront", "account", "sts"}
_READONLY_OPERATION_EXCEPTIONS = {"start-query", "get-query-results", "filter-log-events"}
_MUTATING_OPERATION_PREFIXES = (
    "add",
    "associate",
    "attach",
    "cancel",
    "copy",
    "create",
    "delete",
    "deregister",
    "detach",
    "disable",
    "disassociate",
    "enable",
    "execute",
    "import",
    "modify",
    "patch",
    "put",
    "reboot",
    "register",
    "remove",
    "replace",
    "reset",
    "restore",
    "resume",
    "revoke",
    "run",
    "send",
    "set",
    "start",
    "stop",
    "suspend",
    "tag",
    "terminate",
    "untag",
    "update",
)
_GLOBAL_AWS_FLAGS_WITH_VALUE = {
    "--endpoint-url",
    "--output",
    "--query",
    "--profile",
    "--region",
    "--ca-bundle",
    "--cli-read-timeout",
    "--cli-connect-timeout",
    "--cli-binary-format",
    "--cli-input-json",
    "--cli-input-yaml",
    "--color",
}
_GLOBAL_AWS_STANDALONE_FLAGS = {
    "--debug",
    "--no-verify-ssl",
    "--no-paginate",
    "--version",
    "--no-sign-request",
    "--no-cli-pager",
    "--cli-auto-prompt",
    "--no-cli-auto-prompt",
}
_ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-9;]*[A-Za-z]")
_HELP_OPTION_RE = re.compile(r"(--[a-z0-9][a-z0-9-]*)", re.IGNORECASE)
_LAUNCH_TEMPLATE_ALLOWED_KEYS = {"LaunchTemplateId", "LaunchTemplateName", "Version"}


@dataclass
class AWSOperationSpec:
    cli_version: str
    service: str
    operation: str
    options: set[str]
    help_text: str


def _tag_name(tags: list[dict[str, Any]] | None) -> str:
    for tag in tags or []:
        if not isinstance(tag, dict):
            continue
        if str(tag.get("Key") or "") == "Name":
            return str(tag.get("Value") or "")
    return ""


class AWSConnector:
    def __init__(self) -> None:
        self.family = "aws"
        self._cli_version_cache: str | None = None
        self._operation_spec_cache: dict[tuple[str, str, str], AWSOperationSpec] = {}

    @staticmethod
    def _ensure_binary() -> None:
        if not shutil.which("aws"):
            raise RuntimeError("aws CLI is not installed or not on PATH")

    @staticmethod
    def _preferred_region() -> str:
        for value in [
            Config.AWS_CLI_DEFAULT_REGION,
            os.getenv("AWS_REGION", "").strip(),
            os.getenv("AWS_DEFAULT_REGION", "").strip(),
        ]:
            if value:
                return value
        return ""

    @classmethod
    def _candidate_regions(cls) -> list[str]:
        preferred = cls._preferred_region()
        seen: set[str] = set()
        regions: list[str] = []
        if preferred:
            seen.add(preferred)
            regions.append(preferred)
        for item in [value.strip() for value in str(Config.AWS_CLI_FALLBACK_REGIONS or "").split(",") if value.strip()]:
            if item in seen:
                continue
            seen.add(item)
            regions.append(item)
        return regions[: max(1, int(Config.AWS_CLI_AUTO_REGION_FANOUT_MAX))]

    @staticmethod
    def _contains_shell_syntax(tokens: list[str]) -> bool:
        for token in tokens:
            text = str(token or "")
            if text in {"|", "||", "&&", ";", ">", ">>", "<", "<<", "&"}:
                return True
            if re.fullmatch(r"\d*(>>?|<<?)", text):
                return True
            if "$(" in text or text.startswith("<(") or "`" in text:
                return True
        return False

    @staticmethod
    def _normalize_tokens(command: str) -> list[str]:
        tokens = shell_split(command)
        if AWSConnector._contains_shell_syntax(tokens):
            raise ValueError(
                "Shell syntax is not allowed in AWS commands. Provide exactly one AWS CLI command without pipes, redirects, subshells, or process substitution."
            )
        if tokens and tokens[0].lower() == "aws":
            tokens = tokens[1:]
        AWSConnector._command_components(tokens)
        return tokens

    @staticmethod
    def _command_components(tokens: list[str]) -> tuple[int, str, str]:
        index = 0
        while index < len(tokens):
            token = str(tokens[index] or "")
            if token.startswith("-"):
                flag = token.split("=", 1)[0].lower()
                if flag in _GLOBAL_AWS_FLAGS_WITH_VALUE and "=" not in token:
                    index += 2
                else:
                    index += 1
                continue
            service = token
            if index + 1 >= len(tokens):
                raise ValueError("AWS command must include both a service and an operation")
            operation = str(tokens[index + 1] or "")
            if operation.startswith("-"):
                raise ValueError("AWS command must include both a service and an operation before service-specific flags")
            return index, service.lower(), operation.lower()
        raise ValueError("AWS command must include both a service and an operation")

    @staticmethod
    def _service_and_operation(tokens: list[str]) -> tuple[str, str]:
        _, service, operation = AWSConnector._command_components(tokens)
        return service, operation

    @staticmethod
    def _strip_ansi(text: str) -> str:
        return _ANSI_ESCAPE_RE.sub("", str(text or ""))

    def _aws_cli_version(self) -> str:
        if self._cli_version_cache:
            return self._cli_version_cache
        self._ensure_binary()
        result = run_subprocess(["aws", "--version"], timeout_sec=10)
        text = f"{result.stdout}\n{result.stderr}".strip()
        match = re.search(r"aws-cli/([^\s]+)", text)
        version = match.group(1) if match else "unknown"
        self._cli_version_cache = version
        return version

    def _load_operation_spec(self, service: str, operation: str) -> AWSOperationSpec:
        cli_version = self._aws_cli_version()
        cache_key = (cli_version, service, operation)
        cached = self._operation_spec_cache.get(cache_key)
        if cached is not None:
            return cached

        self._ensure_binary()
        result = run_subprocess(["aws", service, operation, "help"], timeout_sec=max(10, Config.AWS_CLI_TIMEOUT_SEC))
        help_text = self._strip_ansi((result.stdout or result.stderr or "").strip())
        lowered = help_text.lower()
        if (not result.ok and not help_text) or "invalid choice" in lowered or "aws: error" in lowered:
            raise ValueError(f"`aws {service} {operation}` is not a valid command for the installed AWS CLI")

        options = {item.lower() for item in _HELP_OPTION_RE.findall(help_text)}
        spec = AWSOperationSpec(
            cli_version=cli_version,
            service=service,
            operation=operation,
            options=options,
            help_text=help_text,
        )
        self._operation_spec_cache[cache_key] = spec
        return spec

    @staticmethod
    def _extract_flag_values(tokens: list[str]) -> dict[str, list[str]]:
        values: dict[str, list[str]] = {}
        index = 0
        while index < len(tokens):
            token = str(tokens[index] or "")
            if not token.startswith("--"):
                index += 1
                continue
            if "=" in token:
                flag, value = token.split("=", 1)
                values.setdefault(flag.lower(), []).append(value)
                index += 1
                continue
            flag = token.lower()
            if index + 1 < len(tokens) and not str(tokens[index + 1] or "").startswith("--"):
                values.setdefault(flag, []).append(str(tokens[index + 1]))
                index += 2
                continue
            values.setdefault(flag, []).append("")
            index += 1
        return values

    @staticmethod
    def _validate_launch_template_value(value: str) -> None:
        text = str(value or "").strip()
        if not text:
            raise ValueError("`--launch-template` requires a value")
        if text.startswith("{"):
            return
        if text.startswith("LaunchTemplate=") or text.startswith("LaunchTemplateSpecification="):
            raise ValueError(
                "`--launch-template` must use the documented shorthand syntax like `LaunchTemplateId=lt-...,Version=4` or a JSON object. Do not wrap it in `LaunchTemplate=` or `LaunchTemplateSpecification=`."
            )
        fields: dict[str, str] = {}
        for segment in [item.strip() for item in text.split(",") if item.strip()]:
            if "=" not in segment:
                raise ValueError(
                    "`--launch-template` must use comma-separated `key=value` pairs, for example `LaunchTemplateId=lt-...,Version=4`."
                )
            key, raw_value = segment.split("=", 1)
            key = key.strip()
            if key not in _LAUNCH_TEMPLATE_ALLOWED_KEYS:
                raise ValueError(
                    "`--launch-template` only accepts `LaunchTemplateId`, `LaunchTemplateName`, and `Version`."
                )
            fields[key] = raw_value.strip()
        if "LaunchTemplateId" not in fields and "LaunchTemplateName" not in fields:
            raise ValueError("`--launch-template` must include either `LaunchTemplateId` or `LaunchTemplateName`.")
        if "Version" not in fields:
            raise ValueError("`--launch-template` must include `Version`.")

    @staticmethod
    def _validate_launch_template_versions_value(value: str) -> None:
        text = str(value or "").strip()
        if text.startswith("$") and text not in {"$Latest", "$Default"}:
            raise ValueError(
                "`--versions` for `ec2 describe-launch-template-versions` accepts `$Latest`, `$Default`, or numeric versions. Use the exact AWS CLI casing from help."
            )

    def _validate_tokens_against_help(self, tokens: list[str]) -> tuple[str, str]:
        start_index, service, operation = self._command_components(tokens)
        spec = self._load_operation_spec(service, operation)
        command_tokens = tokens[start_index + 2 :]
        for token in command_tokens:
            if not str(token or "").startswith("--"):
                continue
            flag = str(token).split("=", 1)[0].lower()
            if flag not in spec.options and flag not in _GLOBAL_AWS_STANDALONE_FLAGS and flag not in _GLOBAL_AWS_FLAGS_WITH_VALUE:
                raise ValueError(
                    f"`{flag}` is not a documented option for `aws {service} {operation}` on the installed AWS CLI."
                )

        values = self._extract_flag_values(command_tokens)
        if service == "autoscaling" and operation == "update-auto-scaling-group":
            for value in values.get("--launch-template", []):
                self._validate_launch_template_value(value)
        if service == "ec2" and operation == "describe-launch-template-versions":
            for value in values.get("--versions", []):
                self._validate_launch_template_versions_value(value)
        return service, operation

    @staticmethod
    def _has_region(tokens: list[str]) -> bool:
        for index, token in enumerate(tokens):
            lowered = str(token).lower()
            if lowered == "--region" and index + 1 < len(tokens):
                return True
            if lowered.startswith("--region="):
                return True
        return False

    @staticmethod
    def _append_region(tokens: list[str], region: str) -> list[str]:
        if not region or AWSConnector._has_region(tokens):
            return list(tokens)
        return [*tokens, "--region", region]

    @staticmethod
    def _is_mutating(operation: str) -> bool:
        lowered = str(operation or "").lower()
        if lowered in _READONLY_OPERATION_EXCEPTIONS:
            return False
        return lowered.startswith(_MUTATING_OPERATION_PREFIXES)

    @staticmethod
    def _base_args(tokens: list[str]) -> list[str]:
        args = ["aws", "--no-cli-pager", *tokens]
        lowered = [str(item).lower() for item in tokens]
        if Config.AWS_CLI_PROFILE and "--profile" not in lowered:
            args.extend(["--profile", Config.AWS_CLI_PROFILE])
        if Config.AWS_CLI_DEFAULT_REGION and "--region" not in lowered:
            args.extend(["--region", Config.AWS_CLI_DEFAULT_REGION])
        return args

    def _run(self, tokens: list[str]) -> tuple[str, Any]:
        self._ensure_binary()
        args = self._base_args(tokens)
        result = run_subprocess(args, timeout_sec=Config.AWS_CLI_TIMEOUT_SEC)
        if not result.ok:
            raise RuntimeError(format_error_summary(result))
        payload = parse_json_output(result.stdout)
        return result.command, payload if payload is not None else result.stdout

    def _run_multi_region(self, tokens: list[str], *, force_all_regions: bool = False) -> tuple[list[str], list[dict[str, Any]]]:
        service, _ = self._service_and_operation(tokens)
        if service in _GLOBAL_SERVICES:
            command, payload = self._run(tokens)
            return [command], [{"region": "global", "payload": payload}]

        commands: list[str] = []
        outputs: list[dict[str, Any]] = []
        regions = self._candidate_regions()
        if self._has_region(tokens):
            command, payload = self._run(tokens)
            return [command], [{"region": "explicit", "payload": payload}]
        if not force_all_regions and Config.AWS_CLI_DEFAULT_REGION:
            command, payload = self._run(self._append_region(tokens, Config.AWS_CLI_DEFAULT_REGION))
            return [command], [{"region": Config.AWS_CLI_DEFAULT_REGION, "payload": payload}]
        for region in regions:
            command, payload = self._run(self._append_region(tokens, region))
            commands.append(command)
            outputs.append({"region": region, "payload": payload})
        return commands, outputs

    def identity(self) -> ToolObservation:
        command, payload = self._run(["sts", "get-caller-identity"])
        summary = f"Authenticated as AWS account `{payload.get('Account', 'unknown')}`." if isinstance(payload, dict) else "Retrieved AWS caller identity."
        return ToolObservation(
            family=self.family,
            action="identity",
            summary=summary,
            structured=payload if isinstance(payload, dict) else {"output": payload},
            commands=[command],
            raw_preview=truncate_text(str(payload), max_chars=1200),
        )

    def regions(self) -> ToolObservation:
        command, payload = self._run(["ec2", "describe-regions", "--all-regions"])
        regions = []
        if isinstance(payload, dict):
            for item in payload.get("Regions") or []:
                if not isinstance(item, dict):
                    continue
                regions.append(
                    {
                        "name": item.get("RegionName"),
                        "opt_in_status": item.get("OptInStatus"),
                    }
                )
        return ToolObservation(
            family=self.family,
            action="regions",
            summary=f"Discovered {len(regions)} AWS regions.",
            structured={"regions": regions},
            commands=[command],
            raw_preview=truncate_text(str(regions[:20]), max_chars=1800),
        )

    def ec2_overview(self, *, states: list[str] | None = None, regions: list[str] | None = None, all_regions: bool = True) -> ToolObservation:
        state_values = states or ["pending", "running", "stopping", "stopped"]
        tokens = [
            "ec2",
            "describe-instances",
            "--filters",
            f"Name=instance-state-name,Values={','.join(state_values)}",
        ]
        commands: list[str] = []
        outputs: list[dict[str, Any]] = []
        if regions:
            for region in regions:
                command, payload = self._run(self._append_region(tokens, region))
                commands.append(command)
                outputs.append({"region": region, "payload": payload})
        else:
            commands, outputs = self._run_multi_region(tokens, force_all_regions=all_regions)

        instances: list[dict[str, Any]] = []
        state_counts: Counter[str] = Counter()
        for item in outputs:
            region = str(item.get("region") or "")
            payload = item.get("payload")
            reservations = payload.get("Reservations") if isinstance(payload, dict) else []
            for reservation in reservations or []:
                if not isinstance(reservation, dict):
                    continue
                for instance in reservation.get("Instances") or []:
                    if not isinstance(instance, dict):
                        continue
                    state = ((instance.get("State") or {}).get("Name")) or "unknown"
                    state_counts[str(state)] += 1
                    instances.append(
                        {
                            "region": region,
                            "instance_id": instance.get("InstanceId"),
                            "state": state,
                            "instance_type": instance.get("InstanceType"),
                            "private_ip": instance.get("PrivateIpAddress"),
                            "name": _tag_name(instance.get("Tags")),
                        }
                    )
        summary = f"Found {len(instances)} EC2 instance(s) across {len({item['region'] for item in outputs}) or 1} region(s)."
        if state_counts:
            summary += " States: " + ", ".join(f"{state}={count}" for state, count in state_counts.most_common())
        return ToolObservation(
            family=self.family,
            action="ec2_overview",
            summary=summary,
            structured={"state_counts": dict(state_counts), "instances": instances[:120], "regions_checked": [item["region"] for item in outputs]},
            commands=commands,
            raw_preview=truncate_text(str(instances[:20]), max_chars=2200),
        )

    def asg_overview(self, *, all_regions: bool = True) -> ToolObservation:
        tokens = ["autoscaling", "describe-auto-scaling-groups"]
        commands, outputs = self._run_multi_region(tokens, force_all_regions=all_regions)
        groups: list[dict[str, Any]] = []
        for item in outputs:
            region = str(item.get("region") or "")
            payload = item.get("payload")
            for group in (payload.get("AutoScalingGroups") if isinstance(payload, dict) else []) or []:
                if not isinstance(group, dict):
                    continue
                groups.append(
                    {
                        "region": region,
                        "name": group.get("AutoScalingGroupName"),
                        "desired": group.get("DesiredCapacity"),
                        "min": group.get("MinSize"),
                        "max": group.get("MaxSize"),
                        "instance_count": len(group.get("Instances") or []),
                    }
                )
        summary = f"Found {len(groups)} Auto Scaling Group(s)."
        return ToolObservation(
            family=self.family,
            action="asg_overview",
            summary=summary,
            structured={"auto_scaling_groups": groups[:120], "regions_checked": [item["region"] for item in outputs]},
            commands=commands,
            raw_preview=truncate_text(str(groups[:20]), max_chars=2200),
        )

    def eks_overview(self, *, all_regions: bool = True) -> ToolObservation:
        commands: list[str] = []
        clusters: list[dict[str, Any]] = []
        for region in self._candidate_regions() if all_regions else [self._preferred_region()]:
            if not region:
                continue
            list_command, list_payload = self._run(self._append_region(["eks", "list-clusters"], region))
            commands.append(list_command)
            names = list_payload.get("clusters") if isinstance(list_payload, dict) else []
            for cluster_name in names or []:
                describe_command, describe_payload = self._run(self._append_region(["eks", "describe-cluster", "--name", str(cluster_name)], region))
                commands.append(describe_command)
                cluster = (describe_payload.get("cluster") if isinstance(describe_payload, dict) else {}) or {}
                cluster_record = {
                    "region": region,
                    "name": cluster.get("name"),
                    "status": cluster.get("status"),
                    "version": cluster.get("version"),
                    "nodegroups": [],
                }
                nodegroup_command, nodegroup_payload = self._run(self._append_region(["eks", "list-nodegroups", "--cluster-name", str(cluster_name)], region))
                commands.append(nodegroup_command)
                nodegroup_names = nodegroup_payload.get("nodegroups") if isinstance(nodegroup_payload, dict) else []
                for nodegroup_name in nodegroup_names or []:
                    ng_command, ng_payload = self._run(
                        self._append_region(
                            ["eks", "describe-nodegroup", "--cluster-name", str(cluster_name), "--nodegroup-name", str(nodegroup_name)],
                            region,
                        )
                    )
                    commands.append(ng_command)
                    nodegroup = (ng_payload.get("nodegroup") if isinstance(ng_payload, dict) else {}) or {}
                    cluster_record["nodegroups"].append(
                        {
                            "name": nodegroup.get("nodegroupName"),
                            "status": nodegroup.get("status"),
                            "desired_size": ((nodegroup.get("scalingConfig") or {}).get("desiredSize")),
                            "instance_types": nodegroup.get("instanceTypes") or [],
                        }
                    )
                clusters.append(cluster_record)
        summary = f"Found {len(clusters)} EKS cluster(s)."
        return ToolObservation(
            family=self.family,
            action="eks_overview",
            summary=summary,
            structured={"clusters": clusters[:60]},
            commands=commands,
            raw_preview=truncate_text(str(clusters[:12]), max_chars=2400),
        )

    def compute_backing_overview(self) -> ToolObservation:
        identity = self.identity()
        ec2 = self.ec2_overview(all_regions=True)
        asg = self.asg_overview(all_regions=True)
        eks = self.eks_overview(all_regions=True)
        structured = {
            "identity": identity.structured,
            "ec2": ec2.structured,
            "asg": asg.structured,
            "eks": eks.structured,
        }
        summary = (
            f"AWS compute overview: {len(ec2.structured.get('instances', []))} EC2 instances, "
            f"{len(asg.structured.get('auto_scaling_groups', []))} ASGs, "
            f"{len(eks.structured.get('clusters', []))} EKS clusters discovered."
        )
        return ToolObservation(
            family=self.family,
            action="compute_backing_overview",
            summary=summary,
            structured=structured,
            commands=[*identity.commands, *ec2.commands, *asg.commands, *eks.commands],
            raw_preview=truncate_text(str(structured), max_chars=3000),
        )

    def raw_read(self, command: str, *, all_regions: bool = True) -> ToolObservation:
        tokens = self._normalize_tokens(command)
        service, operation = self._validate_tokens_against_help(tokens)
        if self._is_mutating(operation):
            raise RuntimeError(f"`{command}` is not a recognized read-only AWS command")
        commands, outputs = self._run_multi_region(tokens, force_all_regions=all_regions)
        structured = {"service": service, "operation": operation, "results": outputs}
        summary = f"Executed read-only AWS command `{command}` across {len(outputs)} target scope(s)."
        return ToolObservation(
            family=self.family,
            action="raw_read",
            summary=summary,
            structured=structured,
            commands=commands,
            raw_preview=truncate_text(str(outputs), max_chars=2600),
        )

    def execute(self, command: str) -> ToolObservation:
        tokens = self._normalize_tokens(command)
        _, operation = self._validate_tokens_against_help(tokens)
        if not self._is_mutating(operation):
            raise RuntimeError("AWS execute was called with a non-mutating command")
        if Config.AWS_CLI_DRY_RUN:
            args = self._base_args(tokens)
            return ToolObservation(
                family=self.family,
                action="raw_write",
                summary="AWS dry-run mode is enabled; no mutating command was executed.",
                structured={"command": " ".join(args), "dry_run": True},
                commands=[" ".join(args)],
                raw_preview="dry run",
            )
        command_text, payload = self._run(tokens)
        return ToolObservation(
            family=self.family,
            action="raw_write",
            summary=f"Executed mutating AWS command `{command}` successfully.",
            structured={"command": command_text, "output": payload},
            commands=[command_text],
            raw_preview=truncate_text(str(payload), max_chars=2400),
        )
