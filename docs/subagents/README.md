# Subagent Role Cards

These files are reusable prompts for future delegated work. When a future Codex session spawns or simulates a subagent, paste the relevant role card into the task prompt and add the concrete assignment.

## Roles

- `agent-core.md`: run lifecycle, Codex stages, memory, RCA, remediation planning.
- `mcp-integrations.md`: MCP discovery, AWS/EKS/generic connectors, capability metadata.
- `safety-policy.md`: approval gate, policy checks, redaction, auditability.
- `frontend-cockpit.md`: operator UI/UX, panels, settings, responsiveness.
- `testing-qa.md`: unit, integration, fake MCP, e2e, regression coverage.
- `docs-product.md`: docs, runbooks, ADRs, handoff language, product coherence.

## Delegation Rule

Each delegated task should name:

- Owned files or subsystem.
- Exact success criteria.
- Tests to run.
- Files changed in the final report.

Subagents must not revert unrelated work.
