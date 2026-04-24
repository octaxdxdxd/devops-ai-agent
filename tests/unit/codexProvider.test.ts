import { describe, expect, it } from "vitest";
import { CodexProvider } from "../../src/agent/codexProvider.js";
import { DEFAULT_SETTINGS } from "../../src/server/defaultSettings.js";
import { IntentClassificationSchema } from "../../src/shared/schemas.js";

describe("CodexProvider", () => {
  it("returns schema-valid mock output when mock mode is enabled", async () => {
    const provider = new CodexProvider(() => ({
      ...DEFAULT_SETTINGS,
      codex: { ...DEFAULT_SETTINGS.codex, mock: true }
    }));
    const result = await provider.generate({
      schemaName: "IntentClassification",
      schema: IntentClassificationSchema,
      task: "classify",
      context: { query: "why are pods restarting?" }
    });
    expect(result.intent).toBe("diagnose");
    expect(result.confidence).toBeGreaterThan(0);
  });
});
