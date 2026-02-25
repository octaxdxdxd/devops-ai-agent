# AI Ops Agent Architecture

## Runtime entrypoints
- `app.py`: thin launcher for Streamlit UI.
- `src/ui/app.py`: Streamlit composition root.

## UI layer (`src/ui/`)
- `session.py`: Streamlit session initialization and chat history conversion.
- `sidebar.py`: sidebar controls, model switcher, runtime metadata panel.
- `chat.py`: chat rendering and per-turn execution.

## Agent layer (`src/agents/`)
- `log_analyzer.py`: top-level orchestration (`LogAnalyzerAgent`).
- `tool_loop.py`: iterative model-tool execution loop with budgets and loop protection.
- `approval.py`: pending write-action state, approval helpers, command preview rendering.
- `autonomy_helpers.py`: autonomous scan formatting helpers for UI output.

## Tooling layer (`src/tools/`)
- `k8s_diagnostics.py`: read-only Kubernetes diagnostics tools.
- `k8s_cli.py`: generic kubectl command paths (`kubectl_readonly`, `kubectl_execute`).
- `actions.py`: Kubernetes mutating tools (approval-gated by agent policy).
- `aws_cli.py`: AWS CLI read/write tools with allowlist/blocklist enforcement and auditing.
- `k8s_common.py`: shared kubectl utilities.

## Platform services
- `src/models/`: LLM provider adapters.
- `src/autonomy/`: autonomous incident monitor and notification routing.
- `src/utils/`: tracing and response formatting utilities.
- `src/config.py`: environment-driven configuration.

## Scripts
- `scripts/experiments/`: standalone prototypes and one-off scripts.
