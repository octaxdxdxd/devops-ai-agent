# AIOps Agent Local Cockpit

Operator-facing DevOps/SRE assistant for AWS and EKS environments. It runs locally, uses the authenticated Codex CLI as the temporary LLM backend, discovers MCP capabilities at runtime, gathers live evidence, and blocks every write/delete/unknown operation until an operator approves an exact proposed action.

## Quick Start

```bash
npm install
npm run dev
```

Open `http://127.0.0.1:5173`.

The API binds to `127.0.0.1:4317` by default and stores local state in `.aiops/aiops-agent.sqlite`.

## LLM Backend

The v1 backend shells out to:

```bash
codex exec --json --output-schema <schema> -m gpt-5.4 -c model_reasoning_effort=\"high\"
```

Set `AIOPS_AGENT_MOCK_CODEX=1` before starting the server to use deterministic mock responses for UI/testing without model calls.

If the backend cannot find `codex`, set an explicit binary path:

```bash
CODEX_COMMAND="$(command -v codex)" npm run dev
```

Smoke-test the configured LLM provider from the same npm/backend environment:

```bash
npm run smoke:llm -- "hi"
```

To use GitHub Copilot CLI as the LLM provider or fallback, install the separate CLI first:

```bash
npm install -g @github/copilot
```

Then set Settings:

```json
{
  "llm": { "provider": "auto", "fallbackProvider": "copilot" },
  "copilot": {
    "command": "copilot",
    "model": "",
    "timeoutMs": 120000,
    "args": ["-p", "{prompt}", "-s", "--no-ask-user"],
    "mock": false
  }
}
```

`provider: "auto"` tries Codex first and then Copilot. `provider: "copilot"` uses Copilot directly.

## MCP Setup

The default settings include:

- `aws-api`: `.aiops/mcp-venv/bin/python -m awslabs.aws_api_mcp_server.server` when the local venv exists, otherwise `python3 -m awslabs.aws_api_mcp_server.server`
- `eks`: `.aiops/mcp-venv/bin/python -m awslabs.eks_mcp_server.server --allow-write --allow-sensitive-data-access` when the local venv exists, otherwise `python3 -m awslabs.eks_mcp_server.server --allow-write --allow-sensitive-data-access`

Install the AWS/EKS MCP packages into the project-managed Python environment:

```bash
npm run setup:mcp
```

You can override the interpreter with `AIOPS_MCP_PYTHON=/path/to/python`. Then configure AWS/EKS auth using normal AWS profile, region, and kubeconfig/IAM mechanisms. The app discovers tools/resources/prompts at run time, preflights Python module availability, and records connector errors in the trace instead of crashing the cockpit.

## Safety Model

- Read-only and sensitive-read capabilities can be used during evidence gathering.
- Write, delete, and unknown safety classes are available to the planner but blocked before approval.
- Proposed actions must include exact server/tool/arguments, evidence references, risk level, rollback notes, and verification steps.
- Placeholder values such as `<namespace>`, `TODO`, `REPLACE_ME`, and `unknown` are rejected before approval.
- Verification runs after approved execution and is recorded in the run trace.

## Scripts

```bash
npm run dev        # API + Vite UI
npm run build      # Typecheck and build
npm test           # Unit/integration/UI tests
npm run test:e2e   # Playwright layout smoke test
npm run setup:mcp  # Install AWS API and EKS MCP servers into .aiops/mcp-venv
npm run smoke:llm  # Live configured LLM smoke test
```

## Architecture

The run lifecycle is:

1. Classify intent with strict JSON schema.
2. Extract infrastructure entities separately from intent.
3. Discover MCP tools/resources/prompts and local safety metadata.
4. Ask Codex for an evidence plan against discovered capabilities.
5. Execute only pre-approval-safe evidence steps.
6. Generate RCA with evidence references.
7. Generate exact proposed actions when remediation is justified.
8. Require operator approval before mutation.
9. Execute approved action and verify.
10. Export a trace-backed handoff package.

See `docs/architecture.md` and `docs/policies/safety-policy.md` for details.
