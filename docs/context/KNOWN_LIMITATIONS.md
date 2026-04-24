# Known Limitations

## Local-Only Runtime

The app is designed for one local operator. It has no authentication, authorization, tenancy, or server deployment hardening.

## Codex CLI Provider

The LLM provider is a subprocess adapter around `codex exec`. This is intentionally temporary. It should be replaced by a production provider interface when the product is ready for shared deployment.

## MCP Server Availability

Default AWS/EKS MCP servers require Python packages and local cloud credentials. The cockpit should keep starting even when MCP servers are unavailable.

## Safety Classification

MCP annotations are useful but not sufficient. Unknown tools require approval. Future work should add validated argument schemas and local override policies for high-risk tools.

## Memory

The database schema has memory support, but the UI and review workflow are not yet fully implemented.

## E2E Tests

Playwright tests are configured, but this host needs `libasound2t64` before Chromium can run.
