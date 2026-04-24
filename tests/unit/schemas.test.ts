import { describe, expect, it } from "vitest";
import { ProposedActionSchema } from "../../src/shared/schemas.js";

describe("shared schemas", () => {
  it("requires evidence, rollback, exact tool, and verification for proposed actions", () => {
    const result = ProposedActionSchema.safeParse({
      title: "Scale deployment",
      description: "Scale deployment after confirming saturation.",
      riskLevel: "medium",
      safetyClass: "write",
      serverId: "eks",
      toolName: "manage_k8s_resource",
      arguments: { namespace: "prod", kind: "deployment", name: "checkout" },
      evidenceRefs: ["evidence-1"],
      rollback: "Scale back to the previous replica count.",
      verificationPlan: [
        {
          id: "verify-pods",
          purpose: "Check pod readiness",
          serverId: "eks",
          toolName: "list_k8s_resources",
          arguments: { namespace: "prod", kind: "pods" },
          expectedOutcome: "Pods are ready",
          safetyClass: "read"
        }
      ]
    });
    expect(result.success).toBe(true);
  });

  it("rejects actions without evidence references", () => {
    const result = ProposedActionSchema.safeParse({
      title: "Patch deployment",
      description: "Patch deployment.",
      riskLevel: "high",
      safetyClass: "write",
      serverId: "eks",
      toolName: "manage_k8s_resource",
      arguments: {},
      evidenceRefs: [],
      rollback: "Revert patch.",
      verificationPlan: []
    });
    expect(result.success).toBe(false);
  });
});
