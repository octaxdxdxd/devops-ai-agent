# Local Development Runbook

## Start

```bash
npm install
AIOPS_AGENT_MOCK_CODEX=1 npm run dev
```

Use mock mode for UI and API development. Disable it for real Codex calls.

## Live MCP Smoke Test

1. Install AWS/EKS MCP packages with `npm run setup:mcp`.
2. Configure AWS credentials and region.
3. Start the app without mock mode. The default AWS/EKS configs automatically use `.aiops/mcp-venv/bin/python` when it exists.
4. Open Settings and confirm MCP command/env values. Use `AIOPS_MCP_PYTHON=/path/to/python` if you want a different environment.
5. Ask a read-only question first, such as cluster or account inventory.
6. Inspect the trace and evidence panels.
7. Only approve actions after checking exact arguments.

## Reset Local State

Stop the app and remove `.aiops/aiops-agent.sqlite`.
