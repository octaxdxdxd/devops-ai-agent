# Structured Codex Stages

The agent uses separate schema-bound calls:

- `IntentClassification`
- `EntityExtraction`
- `EvidencePlan`
- `RcaResponse`
- `RemediationResponse`

Each stage receives only the context needed for that decision. The planner sees discovered capabilities and schemas. RCA sees evidence summaries and evidence IDs. Remediation sees RCA plus evidence references and capability cards.

The implementation lives in `src/agent/AiopsAgent.ts`, `src/agent/prompts.ts`, and `src/agent/codexProvider.ts`.
