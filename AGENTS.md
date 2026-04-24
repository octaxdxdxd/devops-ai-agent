# Agent Instructions

This repository implements a focused infrastructure operations agent, not a general chatbot.

## First-Read Context Pack

For any future Codex session, read these files before planning or editing:

1. `docs/context/START_HERE.md`
2. `docs/context/CURRENT_STATE.md`
3. `docs/context/NEXT_ACTIONS.md`
4. `docs/policies/safety-policy.md`
5. The relevant role card in `docs/subagents/`

If context is limited, read `START_HERE.md` first. It points to the minimum files needed for each kind of task.

## Engineering Rules

- Preserve the approval gate. Never add a path that can execute write/delete/unknown MCP tools before an approved proposed action.
- Prefer structured schemas in `src/shared/schemas.ts` over ad hoc objects.
- Do not classify intent with word matching. Add or adjust Codex schema prompts instead.
- MCP tools must be selected from discovered capabilities, not hardcoded names.
- Root-cause claims must cite evidence IDs. Unsupported claims are hypotheses.
- Redact sensitive values before long-term memory or handoff content.
- Keep the local cockpit bound to localhost unless an explicit multi-user/server deployment design is added.

## Useful Commands

```bash
npm run typecheck
npm test
npm run dev
```

## Extension Points

- Add future MCP servers through settings, then map local policy defaults in `src/server/defaultSettings.ts`.
- Add new run stages in `src/agent/AiopsAgent.ts`.
- Add new public schemas in `src/shared/schemas.ts`.
- Add UI panels in `src/client/App.tsx` and keep text compact for repeated SRE use.

## Subagent Role Cards

When delegating future work, use the role cards in `docs/subagents/` as the subagent prompt source. They define ownership, context to read, constraints, and expected deliverables for:

- Agent core orchestration
- MCP integrations
- Safety and policy
- Frontend cockpit UX
- Testing and QA
- Docs and product operations
