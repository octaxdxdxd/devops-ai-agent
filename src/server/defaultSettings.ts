import { existsSync } from "node:fs";
import { resolve } from "node:path";
import type { Settings } from "../shared/schemas.js";

const LOCAL_MCP_PYTHON = resolve(".aiops/mcp-venv/bin/python");

function defaultMcpPython(): string {
  return process.env.AIOPS_MCP_PYTHON ?? (existsSync(LOCAL_MCP_PYTHON) ? LOCAL_MCP_PYTHON : "python3");
}

export const DEFAULT_SETTINGS: Settings = {
  app: {
    bindHost: "127.0.0.1",
    apiPort: 4317,
    longTermMemoryEnabled: true,
    redactSensitiveMemory: true
  },
  llm: {
    provider: (process.env.AIOPS_LLM_PROVIDER as "codex" | "copilot" | "auto" | undefined) ?? "codex",
    fallbackProvider:
      (process.env.AIOPS_LLM_FALLBACK as "none" | "codex" | "copilot" | undefined) ?? "none"
  },
  codex: {
    command: process.env.CODEX_COMMAND ?? process.env.CODEX_BIN ?? "codex",
    model: "gpt-5.4",
    reasoningEffort: "high",
    sandbox: "read-only",
    timeoutMs: 120000,
    mock: process.env.AIOPS_AGENT_MOCK_CODEX === "1"
  },
  copilot: {
    command: process.env.COPILOT_COMMAND ?? "copilot",
    model: process.env.COPILOT_MODEL ?? "",
    timeoutMs: 120000,
    args: ["-p", "{prompt}", "-s", "--no-ask-user"],
    mock: process.env.AIOPS_AGENT_MOCK_COPILOT === "1"
  },
  mcpServers: [
    {
      id: "aws-api",
      label: "AWS API MCP",
      kind: "aws-api",
      enabled: true,
      transport: "stdio",
      command: defaultMcpPython(),
      args: ["-m", "awslabs.aws_api_mcp_server.server"],
      env: {
        AWS_REGION: process.env.AWS_REGION ?? "us-east-1",
        READ_OPERATIONS_ONLY: "false",
        REQUIRE_MUTATION_CONSENT: "true",
        AWS_API_MCP_ALLOW_UNRESTRICTED_LOCAL_FILE_ACCESS: "workdir"
      },
      timeoutMs: 60000,
      localPolicy: {
        allowSensitiveReads: true,
        allowMutationsAfterApproval: true,
        defaultSafetyClass: "unknown"
      }
    },
    {
      id: "eks",
      label: "Amazon EKS MCP",
      kind: "eks",
      enabled: true,
      transport: "stdio",
      command: defaultMcpPython(),
      args: [
        "-m",
        "awslabs.eks_mcp_server.server",
        "--allow-write",
        "--allow-sensitive-data-access"
      ],
      env: {
        AWS_REGION: process.env.AWS_REGION ?? "us-east-1",
        AWS_PROFILE: process.env.AWS_PROFILE ?? "default",
        FASTMCP_LOG_LEVEL: "ERROR"
      },
      timeoutMs: 60000,
      localPolicy: {
        allowSensitiveReads: true,
        allowMutationsAfterApproval: true,
        defaultSafetyClass: "unknown"
      }
    }
  ],
  policy: {
    requireApprovalFor: ["write", "delete", "unknown"],
    rejectPlaceholderArguments: true,
    requireEvidenceForRootCause: true,
    allowCodexCodingHandoff: false,
    allowedHandoffWorkspaces: []
  }
};
