# Current State

Last updated: 2026-04-24

## Implemented

- Local Fastify API with sessions, runs, trace streaming, proposed actions, approvals, execution, settings, MCP discovery, and handoffs.
- React cockpit with chat, trace, evidence, proposed actions, verification, handoff, and settings panels.
- SQLite persistence for operational artifacts.
- Codex CLI provider using schema-constrained calls.
- Codex CLI provider resolves `codex` from PATH and the VS Code OpenAI extension binary path; `CODEX_COMMAND` can override it.
- LLM provider routing supports `codex`, `copilot`, and `auto` with configurable fallback. Copilot support uses the separate GitHub Copilot CLI, not VS Code extension internals.
- `npm run smoke:llm -- "hi"` tests the configured provider from the same npm/backend environment.
- MCP manager for stdio and streamable HTTP servers.
- Default MCP configs for AWS API MCP and Amazon EKS MCP.
- `npm run setup:mcp` installs AWS API MCP and Amazon EKS MCP into `.aiops/mcp-venv`; default and already-saved Python module configs prefer that interpreter when present.
- MCP stdio startup performs command/module preflight so missing AWS/EKS packages surface as actionable connector errors instead of only `Connection closed`.
- Clarification-only requests from structured classification complete without MCP discovery when live evidence is not required.
- Safety policy for pre-approval blocking, placeholder rejection, claim evidence enforcement, and redaction.
- Unit, integration, and UI tests.
- Docs, ADRs, local runbooks, and safety policy.

## Verified

- `npm run typecheck` passes.
- `npm test` passes.
- `npm run build` passes.
- `npm audit --omit=dev` reports 0 production vulnerabilities.
- Built API health endpoint responds on `http://127.0.0.1:4317/healthz`.
- `http://127.0.0.1:4317/api/mcp/discover` discovers AWS API MCP and Amazon EKS MCP capabilities through `.aiops/mcp-venv` without connector `Connection closed` errors.
- Live non-mock Codex path has been exercised through `POST /api/runs`; the app now reaches Codex and surfaces provider-side errors.
- Full non-mock HTTP run with message `"hi"` now completes through classification/entity extraction and stops with a clarification response when the model says no live evidence is needed.

## Known Environment Notes

- Playwright Chromium was installed, but the host still needs `libasound2t64` for e2e execution.
- Non-interactive sudo cannot install that package because a password is required.
- AWS/EKS MCP discovery requires AWS credentials, region, and kubeconfig/IAM access. If packages are missing, run `npm run setup:mcp`.
- Fastify v5 dependencies warn about Node 20 engines through transitive packages; current Node is v18.19.1. Typecheck/build/tests pass, but upgrading Node to 20+ is recommended.
- If live Codex fails with a missing binary, start the app with `CODEX_COMMAND="$(command -v codex)" npm run dev` or set `codex.command` in Settings.
- If live Codex returns a usage-limit error, the backend is working and the operator must wait for the Codex account limit window to reset or switch to mock mode for UI testing.
- VS Code Copilot Chat is installed, but its `copilotCLIShim.js` only looks for or installs the separate GitHub Copilot CLI. Install `@github/copilot` before setting `llm.provider` or fallback to `copilot`.
- `copilot` is not currently installed on PATH in this environment.

## Important Files

- `src/agent/AiopsAgent.ts`: run lifecycle and approval/execution flow.
- `src/agent/codexProvider.ts`: Codex CLI bridge.
- `src/mcp/client.ts`: MCP discovery and tool call adapter.
- `src/policy/safety.ts`: safety policy and redaction.
- `src/shared/schemas.ts`: core public schemas.
- `src/server/index.ts`: API routes.
- `src/client/App.tsx`: cockpit UI.
- `src/server/defaultSettings.ts`: default Codex and MCP settings.
