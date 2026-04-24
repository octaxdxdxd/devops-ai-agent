# Adding An MCP Server

1. Add a server entry in Settings or `src/server/defaultSettings.ts`.
2. Use `transport: "stdio"` for local commands or `transport: "streamableHttp"` for remote endpoints.
3. Set `localPolicy.defaultSafetyClass` to `unknown` unless the server has reliable safety annotations.
4. Run discovery from the Settings/API path.
5. Add fake-server integration tests for critical tool metadata and call behavior.

Future integrations such as Slack, GitLab, Jenkins, observability stacks, and generic Kubernetes MCP servers should follow the same discovery and policy path. Do not add special-case word routing.
