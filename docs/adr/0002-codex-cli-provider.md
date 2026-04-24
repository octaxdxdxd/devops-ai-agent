# ADR 0002: Codex CLI As Temporary LLM Provider

## Decision

Use `codex exec` instead of the OpenAI API for v1 model calls.

## Rationale

The current environment already has an authenticated Codex CLI. The project needs a focused infrastructure agent before committing to a production LLM provider.

## Consequences

- Model calls are local subprocesses.
- Strict JSON schemas are still enforced with `--output-schema` and Zod parsing.
- The provider interface can later be swapped for an API-backed implementation.
