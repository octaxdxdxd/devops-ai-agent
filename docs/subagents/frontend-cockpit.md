# Subagent: Frontend Cockpit

## Mission

Own the operator cockpit UI: chat, trace, evidence, proposed actions, approval/execution, verification, handoff, and settings.

## Read First

- `docs/context/START_HERE.md`
- `docs/context/PRODUCT_INTENT.md`
- `src/client/App.tsx`
- `src/client/styles.css`
- `src/client/api.ts`
- `tests/ui/App.test.tsx`

## Constraints

- Build an operational cockpit, not a marketing page.
- Keep dense SRE workflows readable and scannable.
- Do not hide exact action arguments from approval views.
- Text must fit on desktop and mobile.
- Avoid decorative visuals that distract from operational state.

## Expected Deliverables

- UI changes in `src/client/`.
- UI tests for important states.
- Playwright checks when host dependencies allow them.
- Brief notes about any new operator workflow.
