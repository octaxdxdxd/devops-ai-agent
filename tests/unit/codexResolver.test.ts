import { chmodSync, mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { describe, expect, it } from "vitest";
import {
  buildCodexOutputSchema,
  resolveCodexCommand,
  resolveCopilotCommand
} from "../../src/agent/codexProvider.js";
import {
  EvidencePlanSchema,
  IntentClassificationSchema,
  RemediationResponseSchema
} from "../../src/shared/schemas.js";
import { LlmEvidencePlanSchema, LlmRemediationResponseSchema } from "../../src/agent/llmSchemas.js";

describe("resolveCodexCommand", () => {
  it("resolves a bare command from PATH", () => {
    const dir = join(tmpdir(), `aiops-codex-path-${Date.now()}`);
    mkdirSync(dir, { recursive: true });
    const executable = join(dir, "codex");
    writeFileSync(executable, "#!/bin/sh\nexit 0\n");
    chmodSync(executable, 0o755);
    const previousPath = process.env.PATH;
    process.env.PATH = `${dir}:${previousPath ?? ""}`;
    try {
      expect(resolveCodexCommand("codex")).toBe(executable);
    } finally {
      process.env.PATH = previousPath;
    }
  });

  it("throws a useful error for missing absolute paths", () => {
    expect(() => resolveCodexCommand("/not/a/real/codex")).toThrow(/does not exist/);
  });

  it("emits a root object output schema for Codex", () => {
    const schema = buildCodexOutputSchema(IntentClassificationSchema, "IntentClassification") as {
      required: string[];
      properties: Record<string, unknown>;
    };
    expect(schema).toMatchObject({
      type: "object",
      properties: expect.any(Object)
    });
    expect(schema.required).toContain("clarificationQuestion");
    expect(schema.properties.clarificationQuestion).toMatchObject({
      type: ["string", "null"]
    });
  });

  it("emits required nested optional properties for strict output schemas", () => {
    const schema = buildCodexOutputSchema(RemediationResponseSchema, "RemediationResponse") as {
      required: string[];
      properties: Record<string, { required?: string[]; items?: { required?: string[] } }>;
    };
    expect(schema.required).toContain("noActionReason");
    expect(schema.properties.actions.items?.required).toContain("createdAt");
  });

  it("does not emit dangling JSON schema refs for arbitrary argument bags", () => {
    const schema = buildCodexOutputSchema(LlmEvidencePlanSchema, "EvidencePlan");
    expect(JSON.stringify(schema)).not.toContain("\"$ref\"");
  });

  it("uses model-facing argument JSON strings for remediation schemas", () => {
    const schema = buildCodexOutputSchema(LlmRemediationResponseSchema, "RemediationResponse") as {
      properties: { actions: { items: { properties: Record<string, unknown> } } };
    };
    expect(schema.properties.actions.items.properties.argumentsJson).toMatchObject({ type: ["string", "null"] });
    expect(schema.properties.actions.items.properties).not.toHaveProperty("arguments");
    expect(JSON.stringify(schema)).not.toContain("\"$ref\"");
  });

  it("makes optional enum defaults nullable with a nullable type", () => {
    const schema = buildCodexOutputSchema(LlmRemediationResponseSchema, "RemediationResponse") as {
      properties: { actions: { items: { properties: Record<string, { type?: unknown; enum?: unknown[] }> } } };
    };
    expect(schema.properties.actions.items.properties.status.type).toEqual(["string", "null"]);
    expect(schema.properties.actions.items.properties.status.enum).toContain(null);
  });

  it("throws a useful Copilot CLI installation error", () => {
    expect(() => resolveCopilotCommand("/not/a/real/copilot")).toThrow(/Copilot command does not exist/);
  });
});
