# Safety Policy

## Allowed Before Approval

- Intent classification
- Entity extraction
- Capability discovery
- Read-only evidence gathering
- Sensitive reads when the server is configured to allow them
- RCA, explanation, handoff creation

## Blocked Before Approval

- Any `write`, `delete`, or `unknown` safety class tool call
- Any proposed action with placeholder arguments
- Any proposed action without evidence references
- Any proposed action without rollback notes
- Any proposed action without verification steps

## Approval Payload

An approvable action must include:

- MCP server ID
- Tool name
- Exact JSON arguments
- Safety class and risk level
- Evidence references
- Rollback notes
- Verification plan

## Verification

After execution, the agent attempts read-only verification steps and records a `passed`, `failed`, or `inconclusive` result.
