# Next Actions

This file is the default backlog for future sessions. Update it when major work lands.

## Highest Priority

1. Install and validate the separate GitHub Copilot CLI if Copilot fallback is desired.
2. Run a live read-only discovery smoke test with real AWS/EKS credentials.
3. Add connector install/update status in Settings for the `.aiops/mcp-venv` AWS/EKS MCP environment.
4. Add explicit connector health cards in the UI instead of only surfacing connector errors in traces/settings.
5. Add a real long-term memory browser/editor with redaction review.
6. Add richer action execution records, including `ExecutionResult` table writes separate from the proposed action status.

## Safety Hardening

1. Add JSON Schema validation of MCP tool arguments against discovered `inputSchema` before any call.
2. Add a stricter local safety override map for known AWS/EKS mutating capabilities.
3. Add policy tests for sensitive reads, unknown tool metadata, and verification-step blocking.
4. Add an approval confirmation modal that displays exact JSON arguments and rollback notes.
5. Add audit export signing or checksum for handoff packages.

## Agent Quality

1. Add fallback evidence-plan repair when Codex references an unavailable capability.
2. Add failed-tool deduplication across runs, not only within a run.
3. Add explicit confidence thresholds and escalation behavior.
4. Add session and long-term memory summarization stages.
5. Add task handoff execution for allowlisted repos after operator approval.

## UI/UX

1. Add connector settings forms instead of raw JSON-only settings.
2. Add trace filters by severity and event type.
3. Add split-pane resizing and persistent panel preferences.
4. Add better empty states for MCP not installed vs no evidence needed.
5. Add mobile-specific verification of panel readability after Playwright deps are installed.

## Testing

1. Install host Playwright dependency `libasound2t64`, then rerun `npm run test:e2e`.
2. Add API route tests with injected fake services.
3. Add fake Codex malformed JSONL and timeout tests.
4. Add fake MCP mutation flow test: blocked -> approved -> executed -> verified.
5. Add visual screenshots for desktop and mobile cockpit states.
