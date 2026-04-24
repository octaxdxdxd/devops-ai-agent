import { describe, expect, it } from "vitest";
import { resolve } from "node:path";
import { McpManager } from "../../src/mcp/client.js";
import { DEFAULT_SETTINGS } from "../../src/server/defaultSettings.js";

describe("McpManager", () => {
  it("discovers typed capabilities and calls read-only fake MCP tools", async () => {
    const manager = new McpManager(() => ({
      ...DEFAULT_SETTINGS,
      mcpServers: [
        {
          id: "fake-k8s",
          label: "Fake Kubernetes",
          kind: "generic",
          enabled: true,
          transport: "stdio",
          command: "node",
          args: [resolve("tests/fixtures/fake-mcp-server.mjs")],
          env: {},
          timeoutMs: 10000,
          localPolicy: {
            allowSensitiveReads: true,
            allowMutationsAfterApproval: true,
            defaultSafetyClass: "unknown"
          }
        }
      ]
    }));

    const discovery = await manager.discover();
    expect(discovery.errors).toEqual([]);
    expect(discovery.capabilities.map((item) => item.name)).toContain("get_pods");
    expect(discovery.capabilities.find((item) => item.name === "get_pods")?.safetyClass).toBe("read");

    const result = await manager.callTool("fake-k8s", "get_pods", { namespace: "prod" });
    expect(JSON.stringify(result)).toContain("CrashLoopBackOff");
  });

  it("reports actionable Python module preflight errors for stdio MCP servers", async () => {
    const manager = new McpManager(() => ({
      ...DEFAULT_SETTINGS,
      mcpServers: [
        {
          id: "missing-python-mcp",
          label: "Missing Python MCP",
          kind: "generic",
          enabled: true,
          transport: "stdio",
          command: "python3",
          args: ["-m", "aiops_missing_mcp_server_for_test"],
          env: {},
          timeoutMs: 10000,
          localPolicy: {
            allowSensitiveReads: true,
            allowMutationsAfterApproval: true,
            defaultSafetyClass: "unknown"
          }
        }
      ]
    }));

    const discovery = await manager.discover();
    expect(discovery.capabilities).toEqual([]);
    expect(discovery.errors[0]?.message).toContain("cannot import Python module");
    expect(discovery.errors[0]?.message).toContain("npm run setup:mcp");
  });
});
