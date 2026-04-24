# ADR 0003: Approval-Gated Mutations

## Decision

Write-capable MCP servers may be configured and discoverable, but write/delete/unknown tool calls cannot execute until an operator approves an exact proposed action.

## Rationale

The planner needs to reason over available remediation tools, but infrastructure mutation must remain under explicit operator control.

## Consequences

- Proposed actions must contain exact tool calls.
- Placeholder arguments are rejected.
- Approval, execution, and verification are durable audit events.
- MCP server-side safeguards are treated as defense in depth, not the primary gate.
