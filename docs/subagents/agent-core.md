# Subagent: Agent Core

## Mission

Own the AIOps run lifecycle: structured Codex stages, evidence planning, RCA, remediation proposal, verification, memory, and handoff generation.

## Read First

- `docs/context/START_HERE.md`
- `docs/architecture.md`
- `docs/prompts/structured-stages.md`
- `src/agent/AiopsAgent.ts`
- `src/agent/codexProvider.ts`
- `src/shared/schemas.ts`

## Constraints

- Do not use word-based heuristics for intent or routing.
- Do not invent live-system facts.
- Root-cause claims require evidence references.
- Mutating tools can only become proposed actions before approval.
- Keep every model output schema-validated.

## Expected Deliverables

- Focused code changes in `src/agent/` and relevant schemas/tests.
- Unit tests for schema and planning behavior.
- Integration tests when run lifecycle behavior changes.
- Updates to `docs/context/CURRENT_STATE.md` or `NEXT_ACTIONS.md` when behavior changes.
