# Product Intent

## Vision

AIOps Agent is an operational control plane for Kubernetes and AWS investigations. It should feel like a disciplined SRE assistant that gathers evidence, reasons over that evidence, proposes safe remediation, and preserves operator control.

## Target User

DevOps/SRE engineers working with EKS/AKS-like clusters, AWS infrastructure, CI/CD systems, observability stacks, and self-hosted DevOps platforms.

## Core Jobs

- Investigate incidents and operational questions.
- Gather live evidence from MCP servers.
- Explain findings with uncertainty and evidence references.
- Propose minimal safe remediations.
- Require approval before mutations.
- Verify after changes.
- Produce reusable traces and handoff packages.

## Experience Principles

- Chat is the entrypoint, not the whole product.
- Operational artifacts must be visible as structured panels.
- The assistant should clarify only when the request is genuinely impossible or dangerous to interpret.
- Deep RCA should be available, but simple questions should stay simple.
- The operator should always see what was checked, what failed, what is uncertain, and what exact action would execute.

## Future Integrations

The architecture should support more MCP integrations later:

- Generic Kubernetes MCP
- Slack
- GitLab
- Jenkins
- Observability/logging/metrics platforms
- Ticketing and incident-management systems
