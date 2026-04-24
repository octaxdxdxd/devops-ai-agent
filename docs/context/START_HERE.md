# Start Here

This is the persistent context entrypoint for future Codex sessions working on the AIOps Agent Local Cockpit.

## Product In One Paragraph

This project is a local, single-operator AIOps cockpit for DevOps/SRE engineers. It uses the local Codex CLI as the temporary LLM backend, discovers AWS/EKS MCP capabilities, gathers live evidence, explains findings, proposes safe remediations, and never mutates infrastructure until an operator approves an exact action.

## Non-Negotiables

- Do not execute write/delete/unknown MCP tools without an approved proposed action.
- Do not use word-based intent routing.
- Do not make root-cause claims without evidence references.
- Do not hardcode AWS/EKS tool names as the only discovery path.
- Do not persist secrets into long-term memory or handoffs.
- Keep v1 localhost-only unless a new architecture decision explicitly changes that.

## Read Order By Task

- Product or planning task: read `PRODUCT_INTENT.md`, `CURRENT_STATE.md`, `NEXT_ACTIONS.md`.
- Agent-core task: read `../subagents/agent-core.md`, `../architecture.md`, `../prompts/structured-stages.md`.
- MCP task: read `../subagents/mcp-integrations.md`, `../runbooks/adding-mcp-server.md`.
- Safety task: read `../subagents/safety-policy.md`, `../policies/safety-policy.md`, ADR 0003.
- UI task: read `../subagents/frontend-cockpit.md` and inspect `src/client/App.tsx`.
- Test task: read `../subagents/testing-qa.md` and inspect `tests/`.
- Docs/task-handoff work: read `../subagents/docs-product.md`.

## Current Runtime

- Node app using npm.
- Backend: Fastify, SQLite, MCP TypeScript SDK.
- Frontend: Vite + React.
- LLM bridge: `codex exec` with output schemas.
- Default model config: `gpt-5.4`, reasoning effort `high`.
- Local database: `.aiops/aiops-agent.sqlite`.

## Useful Commands

```bash
npm run typecheck
npm test
npm run build
npm run dev
AIOPS_AGENT_MOCK_CODEX=1 npm run dev
```
