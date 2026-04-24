import { z } from "zod";
import {
  ActionStatusSchema,
  EvidencePlanSchema,
  ProposedActionSchema,
  RemediationResponseSchema,
  SafetyClassSchema,
  VerificationStepSchema
} from "../shared/schemas.js";

export const LlmEvidencePlanStepSchema = z.object({
  id: z.string().min(1),
  purpose: z.string().min(1),
  serverId: z.string().min(1).optional(),
  toolName: z.string().min(1).optional(),
  argumentsJson: z.string().default("{}"),
  expectedEvidence: z.string().min(1),
  safetyClass: SafetyClassSchema,
  required: z.boolean().default(true)
});

export const LlmEvidencePlanSchema = EvidencePlanSchema.omit({ steps: true }).extend({
  steps: z.array(LlmEvidencePlanStepSchema)
});

export const LlmVerificationStepSchema = VerificationStepSchema.omit({ arguments: true }).extend({
  argumentsJson: z.string().default("{}")
});

export const LlmProposedActionSchema = ProposedActionSchema.omit({
  arguments: true,
  verificationPlan: true
}).extend({
  argumentsJson: z.string().default("{}"),
  verificationPlan: z.array(LlmVerificationStepSchema).min(1),
  status: ActionStatusSchema.default("proposed")
});

export const LlmRemediationResponseSchema = RemediationResponseSchema.omit({ actions: true }).extend({
  actions: z.array(LlmProposedActionSchema)
});

export type LlmEvidencePlan = z.infer<typeof LlmEvidencePlanSchema>;
export type LlmRemediationResponse = z.infer<typeof LlmRemediationResponseSchema>;
