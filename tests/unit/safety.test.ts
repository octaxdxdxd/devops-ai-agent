import { describe, expect, it } from "vitest";
import { canExecuteTool, enforceEvidenceForClaims, findPlaceholders, redactJson } from "../../src/policy/safety.js";
import { DEFAULT_SETTINGS } from "../../src/server/defaultSettings.js";

describe("safety policy", () => {
  it("blocks unknown and mutating tools before approval", () => {
    expect(canExecuteTool("unknown", DEFAULT_SETTINGS, false).allowed).toBe(false);
    expect(canExecuteTool("write", DEFAULT_SETTINGS, false).allowed).toBe(false);
    expect(canExecuteTool("delete", DEFAULT_SETTINGS, false).allowed).toBe(false);
    expect(canExecuteTool("write", DEFAULT_SETTINGS, true).allowed).toBe(true);
  });

  it("allows read-only evidence gathering before approval", () => {
    expect(canExecuteTool("read", DEFAULT_SETTINGS, false).allowed).toBe(true);
  });

  it("rejects placeholders in nested arguments", () => {
    expect(findPlaceholders({ namespace: "<namespace>", replicas: 2 })).toEqual(["$.namespace"]);
    expect(findPlaceholders({ patch: { image: "REPLACE_ME" } })).toEqual(["$.patch.image"]);
  });

  it("downgrades unsupported root cause claims", () => {
    const claims = enforceEvidenceForClaims([{ claim: "bad node", evidenceRefs: [], status: "supported" }]);
    expect(claims[0].status).toBe("hypothesis");
  });

  it("redacts sensitive values from memory and evidence payloads", () => {
    expect(redactJson({ token: "abcd1234abcd1234abcd1234", nested: { password: "secret" } })).toEqual({
      token: "[REDACTED]",
      nested: { password: "[REDACTED]" }
    });
  });
});
