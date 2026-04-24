import { z } from "zod";
import {
  ApprovalDecisionSchema,
  EvidenceItemSchema,
  HandoffPackageSchema,
  ProposedActionSchema,
  RunStatusSchema,
  RunTraceEventSchema,
  SettingsSchema,
  ToolCallRecordSchema,
  VerificationResultSchema
} from "./schemas.js";

export const SessionSchema = z.object({
  id: z.string(),
  title: z.string(),
  createdAt: z.string(),
  updatedAt: z.string()
});

export const MessageSchema = z.object({
  id: z.string(),
  sessionId: z.string(),
  role: z.enum(["user", "assistant", "system"]),
  content: z.string(),
  createdAt: z.string()
});

export const RunSchema = z.object({
  id: z.string(),
  sessionId: z.string(),
  status: RunStatusSchema,
  userQuery: z.string(),
  finalAnswer: z.string().nullable(),
  createdAt: z.string(),
  updatedAt: z.string()
});

export const RunBundleSchema = z.object({
  run: RunSchema,
  session: SessionSchema.optional(),
  messages: z.array(MessageSchema).default([]),
  traceEvents: z.array(RunTraceEventSchema).default([]),
  evidence: z.array(EvidenceItemSchema).default([]),
  toolCalls: z.array(ToolCallRecordSchema).default([]),
  proposedActions: z.array(ProposedActionSchema).default([]),
  approvals: z.array(ApprovalDecisionSchema).default([]),
  verificationResults: z.array(VerificationResultSchema).default([]),
  handoffs: z.array(HandoffPackageSchema).default([])
});

export const CreateRunRequestSchema = z.object({
  sessionId: z.string().optional(),
  message: z.string().min(1)
});

export const CreateRunResponseSchema = z.object({
  runId: z.string(),
  sessionId: z.string()
});

export const ActionDecisionRequestSchema = z.object({
  operator: z.string().default("local-operator"),
  comment: z.string().optional()
});

export const SettingsResponseSchema = SettingsSchema;

export type Session = z.infer<typeof SessionSchema>;
export type Message = z.infer<typeof MessageSchema>;
export type Run = z.infer<typeof RunSchema>;
export type RunBundle = z.infer<typeof RunBundleSchema>;
export type CreateRunRequest = z.infer<typeof CreateRunRequestSchema>;
export type CreateRunResponse = z.infer<typeof CreateRunResponseSchema>;
export type ActionDecisionRequest = z.infer<typeof ActionDecisionRequestSchema>;
