# ADR 0001: Local Cockpit First

## Decision

Build v1 as a localhost-only operator cockpit using local Codex CLI credentials, local AWS/EKS credentials, local MCP subprocesses, and SQLite.

## Rationale

The primary risk is operational safety, not scale. Local-first keeps the trust boundary understandable while the agent, approval gate, evidence chain, and UX mature.

## Consequences

- One operator per instance.
- No centralized auth in v1.
- MCP subprocess health is visible as run trace data.
- A future team deployment will need a new auth/session/isolation ADR.
