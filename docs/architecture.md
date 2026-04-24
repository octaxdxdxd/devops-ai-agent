# Architecture

## Components

- React cockpit: chat, trace, evidence, proposed actions, approval/execution, verification, handoff, and settings panels.
- Fastify API: session/run/action/settings/handoff endpoints and SSE trace streaming.
- SQLite store: sessions, messages, runs, trace events, evidence, tool calls, proposed actions, approvals, verification results, memory, and handoffs.
- Codex provider: local `codex exec` adapter using strict JSON schemas.
- MCP manager: generic MCP client for stdio and streamable HTTP servers.
- Policy layer: safety classification, approval checks, placeholder rejection, evidence enforcement, and redaction.

## Run Lifecycle

The agent creates a durable run record before doing work. Every stage appends trace events so the operator can inspect what happened and export the handoff package later.

Pre-approval stages may execute read-only and sensitive-read tools. Mutating or unknown tools can be included only as proposed actions. Execution requires approval through the API/UI.

## Capability Discovery

Each enabled MCP server is connected to during discovery. The agent lists tools, resources, resource templates, and prompts. Tool schemas and annotations are stored as `CapabilityCard` objects for planner context.

Safety is inferred from MCP annotations where possible:

- `readOnlyHint: true` -> `read`
- `destructiveHint: true` -> `delete`
- missing/ambiguous metadata -> server default, usually `unknown`

Unknown means "approval required."

## Codex Provider

The provider writes a temporary JSON Schema file and calls Codex non-interactively. Every response is parsed with Zod before it enters the run state. `AIOPS_AGENT_MOCK_CODEX=1` enables deterministic responses for local testing.

## Persistence

SQLite is local and file-backed under `.aiops/`. It is intentionally simple for v1. A future team-server deployment should replace it with authenticated multi-user storage and stronger tenant boundaries.
