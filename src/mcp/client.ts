import { execFile } from "node:child_process";
import { existsSync } from "node:fs";
import { delimiter, resolve } from "node:path";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";
import { CallToolResultSchema } from "@modelcontextprotocol/sdk/types.js";
import type { Resource, ResourceTemplate, Tool } from "@modelcontextprotocol/sdk/types.js";
import type {
  CapabilityCard,
  JsonValue,
  McpServerConfig,
  Settings
} from "../shared/schemas.js";
import { classifyCapabilitySafety } from "../policy/safety.js";

export type McpDiscoveryResult = {
  capabilities: CapabilityCard[];
  resources: Array<Resource & { serverId: string }>;
  resourceTemplates: Array<ResourceTemplate & { serverId: string }>;
  prompts: Array<{ serverId: string; name: string; description?: string; arguments?: unknown[] }>;
  errors: Array<{ serverId: string; message: string }>;
};

export class McpManager {
  constructor(private readonly getSettings: () => Settings) {}

  async discover(): Promise<McpDiscoveryResult> {
    const settings = this.getSettings();
    const capabilities: CapabilityCard[] = [];
    const resources: Array<Resource & { serverId: string }> = [];
    const resourceTemplates: Array<ResourceTemplate & { serverId: string }> = [];
    const prompts: Array<{ serverId: string; name: string; description?: string; arguments?: unknown[] }> = [];
    const errors: Array<{ serverId: string; message: string }> = [];

    for (const configuredServer of settings.mcpServers.filter((item) => item.enabled)) {
      const server = normalizeServerConfig(configuredServer);
      try {
        await this.withClient(server, async (client) => {
          const tools = await listAllTools(client);
          for (const tool of tools) {
            capabilities.push(toCapability(server, tool));
          }

          const listedResources = await safeListResources(client);
          resources.push(...listedResources.map((resource) => ({ ...resource, serverId: server.id })));

          const listedTemplates = await safeListResourceTemplates(client);
          resourceTemplates.push(...listedTemplates.map((template) => ({ ...template, serverId: server.id })));

          const listedPrompts = await safeListPrompts(client);
          prompts.push(...listedPrompts.map((prompt) => ({ ...prompt, serverId: server.id })));
        });
      } catch (error) {
        errors.push({ serverId: server.id, message: error instanceof Error ? error.message : String(error) });
      }
    }

    return { capabilities, resources, resourceTemplates, prompts, errors };
  }

  async callTool(serverId: string, toolName: string, args: Record<string, JsonValue>): Promise<unknown> {
    const settings = this.getSettings();
    const configuredServer = settings.mcpServers.find((item) => item.id === serverId && item.enabled);
    if (!configuredServer) throw new Error(`MCP server is not enabled or not configured: ${serverId}`);
    const server = normalizeServerConfig(configuredServer);
    return this.withClient(server, async (client) => {
      return client.callTool(
        { name: toolName, arguments: args },
        CallToolResultSchema,
        { resetTimeoutOnProgress: true, maxTotalTimeout: server.timeoutMs }
      );
    });
  }

  private async withClient<T>(server: McpServerConfig, fn: (client: Client) => Promise<T>): Promise<T> {
    if (server.transport === "stdio") await preflightStdioServer(server);

    const client = new Client(
      { name: "aiops-agent-local-cockpit", version: "0.1.0" },
      { capabilities: {} }
    );
    const transport =
      server.transport === "stdio"
        ? new StdioClientTransport({
            command: server.command ?? "",
            args: server.args,
            env: { ...process.env, ...server.env } as Record<string, string>
          })
        : new StreamableHTTPClientTransport(new URL(server.url ?? ""));

    try {
      await client.connect(transport);
    } catch (error) {
      throw enrichConnectError(server, error);
    }
    try {
      return await fn(client);
    } finally {
      await client.close();
    }
  }
}

function normalizeServerConfig(server: McpServerConfig): McpServerConfig {
  if (server.transport !== "stdio") return server;
  if (!isPythonCommand(server.command) || server.args[0] !== "-m") return server;

  const localPython = process.env.AIOPS_MCP_PYTHON ?? resolve(".aiops/mcp-venv/bin/python");
  if (!existsSync(localPython)) return server;
  return { ...server, command: localPython };
}

async function preflightStdioServer(server: McpServerConfig): Promise<void> {
  const command = server.command ?? "";
  if (!command.trim()) {
    throw new Error(`MCP server ${server.id} has no stdio command configured.`);
  }

  const env = { ...process.env, ...server.env } as Record<string, string>;
  const resolvedCommand = command.includes("/") ? command : findExecutableOnPath(command, env.PATH);
  if (!resolvedCommand) {
    throw new Error(
      `MCP server ${server.id} command was not found: ${command}. Update Settings or install the required server runtime.`
    );
  }
  if (command.includes("/") && !existsSync(command)) {
    throw new Error(`MCP server ${server.id} command does not exist: ${command}. Update Settings for this connector.`);
  }

  const pythonModule = server.args[0] === "-m" ? server.args[1] : undefined;
  if (isPythonCommand(command) && pythonModule) {
    const check = await checkPythonModule(resolvedCommand, pythonModule, env);
    if (!check.ok) {
      throw new Error(
        [
          `MCP server ${server.id} cannot import Python module '${pythonModule}' with ${resolvedCommand}.`,
          "Run `npm run setup:mcp` or set `AIOPS_MCP_PYTHON=/path/to/python` to a Python environment with the AWS/EKS MCP packages installed.",
          check.detail ? `Preflight detail: ${check.detail}` : ""
        ]
          .filter(Boolean)
          .join(" ")
      );
    }
  }
}

function enrichConnectError(server: McpServerConfig, error: unknown): Error {
  const message = error instanceof Error ? error.message : String(error);
  const pythonModule = server.args[0] === "-m" ? server.args[1] : undefined;
  const installHint = pythonModule?.startsWith("awslabs.")
    ? " The AWS/EKS MCP Python packages are expected in `.aiops/mcp-venv`; run `npm run setup:mcp` if this environment was rebuilt."
    : "";
  return new Error(
    `MCP server ${server.id} failed to connect via ${server.transport} (${server.command ?? server.url ?? "unknown"} ${server.args.join(" ")}): ${message}.${installHint}`
  );
}

function isPythonCommand(command: string | undefined): boolean {
  if (!command) return false;
  const normalized = command.split("/").pop() ?? command;
  return normalized === "python" || normalized === "python3" || normalized.startsWith("python3.");
}

function findExecutableOnPath(command: string, pathValue = process.env.PATH): string | undefined {
  for (const directory of (pathValue ?? "").split(delimiter)) {
    if (!directory) continue;
    const candidate = resolve(directory, command);
    if (existsSync(candidate)) return candidate;
  }
  return undefined;
}

function checkPythonModule(
  command: string,
  moduleName: string,
  env: Record<string, string>
): Promise<{ ok: true } | { ok: false; detail: string }> {
  return new Promise((resolveCheck) => {
    const child = execFile(
      command,
      [
        "-c",
        "import importlib.util, sys; module=sys.argv[1]; sys.exit(0 if importlib.util.find_spec(module) else 42)",
        moduleName
      ],
      { env, timeout: 5000 },
      (error, stdout, stderr) => {
        if (!error) {
          resolveCheck({ ok: true });
          return;
        }
        const details = [stderr, stdout, error.message].filter(Boolean).join(" ").trim();
        resolveCheck({ ok: false, detail: details });
      }
    );
    child.on("error", (error) => {
      resolveCheck({ ok: false, detail: error.message });
    });
  });
}

function toCapability(server: McpServerConfig, tool: Tool): CapabilityCard {
  const annotations = (tool.annotations ?? {}) as Record<string, JsonValue>;
  const safetyClass = classifyCapabilitySafety({
    annotations,
    defaultSafetyClass: server.localPolicy.defaultSafetyClass
  });
  return {
    id: `${server.id}:${tool.name}`,
    serverId: server.id,
    serverLabel: server.label,
    serverKind: server.kind,
    name: tool.name,
    title: tool.title,
    description: tool.description,
    inputSchema: (tool.inputSchema ?? undefined) as Record<string, JsonValue> | undefined,
    outputSchema: (tool.outputSchema ?? undefined) as Record<string, JsonValue> | undefined,
    annotations,
    safetyClass,
    available: true,
    transport: server.transport
  };
}

async function listAllTools(client: Client): Promise<Tool[]> {
  const tools: Tool[] = [];
  let cursor: string | undefined;
  do {
    const result = await client.listTools({ cursor });
    tools.push(...result.tools);
    cursor = result.nextCursor;
  } while (cursor);
  return tools;
}

async function safeListResources(client: Client): Promise<Resource[]> {
  try {
    const resources: Resource[] = [];
    let cursor: string | undefined;
    do {
      const result = await client.listResources({ cursor });
      resources.push(...result.resources);
      cursor = result.nextCursor;
    } while (cursor);
    return resources;
  } catch {
    return [];
  }
}

async function safeListResourceTemplates(client: Client): Promise<ResourceTemplate[]> {
  try {
    const templates: ResourceTemplate[] = [];
    let cursor: string | undefined;
    do {
      const result = await client.listResourceTemplates({ cursor });
      templates.push(...result.resourceTemplates);
      cursor = result.nextCursor;
    } while (cursor);
    return templates;
  } catch {
    return [];
  }
}

async function safeListPrompts(
  client: Client
): Promise<Array<{ name: string; description?: string; arguments?: unknown[] }>> {
  try {
    const prompts: Array<{ name: string; description?: string; arguments?: unknown[] }> = [];
    let cursor: string | undefined;
    do {
      const result = await client.listPrompts({ cursor });
      prompts.push(...result.prompts);
      cursor = result.nextCursor;
    } while (cursor);
    return prompts;
  } catch {
    return [];
  }
}
