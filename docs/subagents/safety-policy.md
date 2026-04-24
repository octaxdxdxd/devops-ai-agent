# Subagent: Safety And Policy

## Mission

Own the approval boundary, policy decisions, argument validation, redaction, auditability, and failure-mode safety.

## Read First

- `docs/context/START_HERE.md`
- `docs/policies/safety-policy.md`
- `docs/adr/0003-approval-gated-mutations.md`
- `src/policy/safety.ts`
- `src/agent/AiopsAgent.ts`
- `tests/unit/safety.test.ts`

## Constraints

- Never add a mutation path that bypasses approval.
- Reject placeholders and unresolved identifiers before approval.
- Treat MCP server-side safeguards as defense in depth only.
- Sensitive data must be redacted before memory or handoff by default.
- Verification should be read-only unless a future policy explicitly changes that.

## Expected Deliverables

- Policy code and tests.
- Updated safety docs.
- Risk notes for any behavior that changes mutation, sensitive-read, or audit semantics.
