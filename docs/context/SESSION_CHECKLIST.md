# Future Session Checklist

Use this checklist before making changes.

## Before Editing

- Read `docs/context/START_HERE.md`.
- Check `docs/context/CURRENT_STATE.md`.
- Check `docs/context/NEXT_ACTIONS.md`.
- Identify the relevant `docs/subagents/*.md` role card.
- Run `find . -path './node_modules' -prune -o -path './dist' -prune -o -path './.aiops' -prune -o -type f -print | sort` if file layout is unclear.

## During Work

- Keep changes scoped.
- Preserve safety gates.
- Add or update schemas before wiring new behavior.
- Add tests for policy, schema, or run-lifecycle changes.
- Update docs/context files if the current state or backlog changes.

## Before Final Response

- Run `npm run typecheck`.
- Run `npm test`.
- Run `npm run build` for implementation changes.
- Mention if `npm run test:e2e` is blocked by host dependencies.
- Make sure no required dev server session is left running unintentionally.
