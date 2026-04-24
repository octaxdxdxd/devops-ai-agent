import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AiopsAgent } from "../../src/agent/AiopsAgent.js";
import type { StructuredLlmProvider } from "../../src/agent/codexProvider.js";
import { RunBus } from "../../src/agent/runBus.js";
import type { McpManager } from "../../src/mcp/client.js";
import { AppDatabase } from "../../src/server/db.js";
import { DEFAULT_SETTINGS } from "../../src/server/defaultSettings.js";

const tempDirs: string[] = [];

afterEach(() => {
  for (const dir of tempDirs.splice(0)) rmSync(dir, { recursive: true, force: true });
});

describe("AiopsAgent", () => {
  it("answers clarification-only requests without discovering MCP capabilities", async () => {
    const dir = mkdtempSync(join(tmpdir(), "aiops-agent-test-"));
    tempDirs.push(dir);
    const db = new AppDatabase(join(dir, "agent.sqlite"));
    const session = db.createSession("Test session");
    const run = db.createRun(session.id, "hi");

    const llm: StructuredLlmProvider = {
      generate: vi.fn(async (request) => {
        if (request.schemaName === "IntentClassification") {
          return {
            intent: "unknown",
            confidence: 0.99,
            urgency: "low",
            needsLiveEvidence: false,
            needsClarification: true,
            clarificationQuestion: "What infrastructure task do you want help with?",
            rationale: "The request is only a greeting."
          };
        }
        throw new Error(`Unexpected LLM request: ${request.schemaName}`);
      })
    };
    const mcp = {
      discover: vi.fn(async () => {
        throw new Error("MCP discovery should not run for clarification-only requests.");
      }),
      callTool: vi.fn()
    } as unknown as McpManager;

    const agent = new AiopsAgent(db, llm, mcp, new RunBus(), () => DEFAULT_SETTINGS);
    await agent.run(run.id);

    expect(mcp.discover).not.toHaveBeenCalled();
    expect(db.getRun(run.id)?.status).toBe("completed");
    expect(db.getRun(run.id)?.finalAnswer).toBe("What infrastructure task do you want help with?");
    expect(db.listMessages(session.id).at(-1)?.content).toBe("What infrastructure task do you want help with?");
    expect(db.listTraceEvents(run.id).map((event) => event.type)).toEqual([
      "run.started",
      "classification.completed",
      "run.completed"
    ]);

    db.close();
  });
});
