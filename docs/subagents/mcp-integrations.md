# Subagent: MCP Integrations

## Mission

Own MCP server discovery, tool/resource/prompt metadata, connector health, typed tool calls, and future integration patterns.

## Read First

- `docs/context/START_HERE.md`
- `docs/runbooks/adding-mcp-server.md`
- `src/mcp/client.ts`
- `src/server/defaultSettings.ts`
- `src/shared/schemas.ts`
- `tests/integration/mcpManager.test.ts`

## Constraints

- Do not hardcode behavior based only on tool names.
- Prefer capability metadata, schemas, annotations, and local policy metadata.
- Unknown safety metadata must remain approval-gated.
- The app must remain usable when MCP servers are unavailable.

## Expected Deliverables

- Connector changes in `src/mcp/` or settings defaults.
- Fake MCP tests for discovery and tool execution.
- Connector docs/runbook updates.
- Clear notes about required external packages or credentials.
