# Subagent: Testing And QA

## Mission

Own verification strategy, test coverage, fake MCP fixtures, Codex mock behavior, and regression prevention.

## Read First

- `docs/context/START_HERE.md`
- `docs/context/CURRENT_STATE.md`
- `tests/`
- `vitest.config.ts`
- `playwright.config.ts`

## Constraints

- Prefer fake MCP servers for deterministic integration tests.
- Do not require real AWS/EKS credentials in automated tests.
- Keep tests focused on safety boundaries and operational behavior.
- Make e2e failures distinguish app failures from host dependency failures.

## Expected Deliverables

- Unit, integration, UI, or e2e tests.
- Updated fixtures when MCP behavior changes.
- Clear verification notes and any blocked checks.
