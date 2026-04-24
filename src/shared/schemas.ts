import { z } from "zod";

export const SeveritySchema = z.enum(["debug", "info", "warning", "error"]);
export const RunStatusSchema = z.enum(["queued", "running", "awaiting_approval", "completed", "failed"]);
export const SafetyClassSchema = z.enum(["read", "write", "delete", "sensitive-read", "unknown"]);
export const RiskLevelSchema = z.enum(["low", "medium", "high", "critical"]);
export const ActionStatusSchema = z.enum(["proposed", "approved", "rejected", "executing", "executed", "failed"]);
export const McpTransportSchema = z.enum(["stdio", "streamableHttp"]);
export const LlmProviderSchema = z.enum(["codex", "copilot", "auto"]);

export const JsonValueSchema: z.ZodType<JsonValue> = z.lazy(() =>
  z.union([
    z.string(),
    z.number(),
    z.boolean(),
    z.null(),
    z.array(JsonValueSchema),
    z.record(JsonValueSchema)
  ])
);

export type JsonValue =
  | string
  | number
  | boolean
  | null
  | JsonValue[]
  | { [key: string]: JsonValue };

export const McpServerConfigSchema = z.object({
  id: z.string().min(1),
  label: z.string().min(1),
  kind: z.enum(["aws-api", "eks", "generic"]),
  enabled: z.boolean(),
  transport: McpTransportSchema,
  command: z.string().optional(),
  args: z.array(z.string()).default([]),
  url: z.string().url().optional(),
  env: z.record(z.string()).default({}),
  timeoutMs: z.number().int().positive().default(60000),
  localPolicy: z.object({
    allowSensitiveReads: z.boolean().default(true),
    allowMutationsAfterApproval: z.boolean().default(true),
    defaultSafetyClass: SafetyClassSchema.default("unknown")
  })
});

export const SettingsSchema = z.object({
  app: z.object({
    bindHost: z.string().default("127.0.0.1"),
    apiPort: z.number().int().positive().default(4317),
    longTermMemoryEnabled: z.boolean().default(true),
    redactSensitiveMemory: z.boolean().default(true)
  }),
  llm: z
    .object({
      provider: LlmProviderSchema.default("codex"),
      fallbackProvider: z.enum(["none", "codex", "copilot"]).default("none")
    })
    .default({
      provider: "codex",
      fallbackProvider: "none"
    }),
  codex: z.object({
    command: z.string().default("codex"),
    model: z.string().default("gpt-5.4"),
    reasoningEffort: z.string().default("high"),
    sandbox: z.enum(["read-only", "workspace-write", "danger-full-access"]).default("read-only"),
    timeoutMs: z.number().int().positive().default(120000),
    mock: z.boolean().default(false)
  }),
  copilot: z
    .object({
      command: z.string().default("copilot"),
      model: z.string().default(""),
      timeoutMs: z.number().int().positive().default(120000),
      args: z.array(z.string()).default(["-p", "{prompt}", "-s", "--no-ask-user"]),
      mock: z.boolean().default(false)
    })
    .default({
      command: "copilot",
      model: "",
      timeoutMs: 120000,
      args: ["-p", "{prompt}", "-s", "--no-ask-user"],
      mock: false
    }),
  mcpServers: z.array(McpServerConfigSchema),
  policy: z.object({
    requireApprovalFor: z.array(SafetyClassSchema).default(["write", "delete", "unknown"]),
    rejectPlaceholderArguments: z.boolean().default(true),
    requireEvidenceForRootCause: z.boolean().default(true),
    allowCodexCodingHandoff: z.boolean().default(false),
    allowedHandoffWorkspaces: z.array(z.string()).default([])
  })
});

export const IntentClassificationSchema = z.object({
  intent: z.enum(["answer", "diagnose", "remediate", "explain", "handoff", "unknown"]),
  confidence: z.number().min(0).max(1),
  urgency: z.enum(["low", "normal", "high", "incident"]),
  needsLiveEvidence: z.boolean(),
  needsClarification: z.boolean(),
  clarificationQuestion: z.string().optional(),
  rationale: z.string().min(1)
});

export const EntityExtractionSchema = z.object({
  entities: z.array(
    z.object({
      type: z.enum([
        "cluster",
        "namespace",
        "workload",
        "pod",
        "service",
        "node",
        "aws-account",
        "aws-region",
        "aws-resource",
        "pipeline",
        "repository",
        "time-window",
        "unknown"
      ]),
      value: z.string().min(1),
      confidence: z.number().min(0).max(1),
      sourceText: z.string().optional()
    })
  ),
  unresolvedReferences: z.array(z.string()).default([])
});

export const CapabilityCardSchema = z.object({
  id: z.string().min(1),
  serverId: z.string().min(1),
  serverLabel: z.string().min(1),
  serverKind: z.enum(["aws-api", "eks", "generic"]),
  name: z.string().min(1),
  title: z.string().optional(),
  description: z.string().optional(),
  inputSchema: z.record(JsonValueSchema).optional(),
  outputSchema: z.record(JsonValueSchema).optional(),
  annotations: z.record(JsonValueSchema).default({}),
  safetyClass: SafetyClassSchema,
  available: z.boolean(),
  transport: McpTransportSchema
});

export const EvidencePlanSchema = z.object({
  objective: z.string().min(1),
  confidence: z.number().min(0).max(1),
  steps: z.array(
    z.object({
      id: z.string().min(1),
      purpose: z.string().min(1),
      serverId: z.string().min(1).optional(),
      toolName: z.string().min(1).optional(),
      arguments: z.record(JsonValueSchema).default({}),
      expectedEvidence: z.string().min(1),
      safetyClass: SafetyClassSchema,
      required: z.boolean().default(true)
    })
  ),
  answerWithoutToolsIfUnavailable: z.boolean().default(true)
});

export const EvidenceItemSchema = z.object({
  id: z.string().min(1),
  runId: z.string().min(1),
  source: z.string().min(1),
  summary: z.string().min(1),
  data: JsonValueSchema,
  sensitive: z.boolean(),
  createdAt: z.string()
});

export const ToolCallRecordSchema = z.object({
  id: z.string().min(1),
  runId: z.string().min(1),
  serverId: z.string().min(1),
  toolName: z.string().min(1),
  arguments: z.record(JsonValueSchema),
  status: z.enum(["pending", "running", "succeeded", "failed", "blocked"]),
  safetyClass: SafetyClassSchema,
  result: JsonValueSchema.optional(),
  error: z.string().optional(),
  startedAt: z.string(),
  completedAt: z.string().optional()
});

export const RunTraceEventSchema = z.object({
  id: z.string().min(1),
  runId: z.string().min(1),
  type: z.enum([
    "run.started",
    "classification.completed",
    "entities.extracted",
    "capabilities.discovered",
    "evidence.plan.created",
    "tool.started",
    "tool.completed",
    "tool.blocked",
    "tool.failed",
    "rca.completed",
    "actions.proposed",
    "approval.recorded",
    "execution.started",
    "execution.completed",
    "verification.completed",
    "memory.updated",
    "handoff.created",
    "run.completed",
    "run.failed"
  ]),
  severity: SeveritySchema,
  message: z.string().min(1),
  data: JsonValueSchema.optional(),
  createdAt: z.string()
});

export const RootCauseClaimSchema = z.object({
  claim: z.string().min(1),
  confidence: z.number().min(0).max(1),
  evidenceRefs: z.array(z.string()).default([]),
  status: z.enum(["supported", "hypothesis", "ruled-out"]),
  uncertainty: z.string().optional()
});

export const RcaResponseSchema = z.object({
  answer: z.string().min(1),
  summary: z.string().min(1),
  rootCauseClaims: z.array(RootCauseClaimSchema),
  followUpQuestions: z.array(z.string()).default([]),
  missingEvidence: z.array(z.string()).default([])
});

export const VerificationStepSchema = z.object({
  id: z.string().min(1),
  purpose: z.string().min(1),
  serverId: z.string().min(1).optional(),
  toolName: z.string().min(1).optional(),
  arguments: z.record(JsonValueSchema).default({}),
  expectedOutcome: z.string().min(1),
  safetyClass: SafetyClassSchema.default("read")
});

export const ProposedActionSchema = z.object({
  id: z.string().min(1).optional(),
  runId: z.string().min(1).optional(),
  title: z.string().min(1),
  description: z.string().min(1),
  riskLevel: RiskLevelSchema,
  safetyClass: SafetyClassSchema,
  serverId: z.string().min(1),
  toolName: z.string().min(1),
  arguments: z.record(JsonValueSchema),
  evidenceRefs: z.array(z.string().min(1)).min(1),
  rollback: z.string().min(1),
  verificationPlan: z.array(VerificationStepSchema).min(1),
  status: ActionStatusSchema.default("proposed"),
  createdAt: z.string().optional()
});

export const RemediationResponseSchema = z.object({
  actions: z.array(ProposedActionSchema),
  rationale: z.string().min(1),
  noActionReason: z.string().optional()
});

export const ApprovalDecisionSchema = z.object({
  id: z.string().min(1),
  actionId: z.string().min(1),
  decision: z.enum(["approved", "rejected"]),
  operator: z.string().default("local-operator"),
  comment: z.string().optional(),
  createdAt: z.string()
});

export const ExecutionResultSchema = z.object({
  id: z.string().min(1),
  actionId: z.string().min(1),
  status: z.enum(["succeeded", "failed"]),
  result: JsonValueSchema.optional(),
  error: z.string().optional(),
  createdAt: z.string()
});

export const VerificationResultSchema = z.object({
  id: z.string().min(1),
  actionId: z.string().min(1),
  status: z.enum(["passed", "failed", "inconclusive"]),
  summary: z.string().min(1),
  evidenceRefs: z.array(z.string()).default([]),
  data: JsonValueSchema.optional(),
  createdAt: z.string()
});

export const MemoryRecordSchema = z.object({
  id: z.string().min(1),
  scope: z.enum(["session", "long-term"]),
  key: z.string().min(1),
  value: z.string().min(1),
  sensitive: z.boolean(),
  createdAt: z.string(),
  updatedAt: z.string()
});

export const HandoffPackageSchema = z.object({
  id: z.string().min(1),
  runId: z.string().min(1),
  type: z.enum(["markdown", "json", "codex-task"]),
  title: z.string().min(1),
  content: z.string().min(1),
  createdAt: z.string()
});

export type Severity = z.infer<typeof SeveritySchema>;
export type RunStatus = z.infer<typeof RunStatusSchema>;
export type SafetyClass = z.infer<typeof SafetyClassSchema>;
export type RiskLevel = z.infer<typeof RiskLevelSchema>;
export type ActionStatus = z.infer<typeof ActionStatusSchema>;
export type LlmProvider = z.infer<typeof LlmProviderSchema>;
export type McpServerConfig = z.infer<typeof McpServerConfigSchema>;
export type Settings = z.infer<typeof SettingsSchema>;
export type IntentClassification = z.infer<typeof IntentClassificationSchema>;
export type EntityExtraction = z.infer<typeof EntityExtractionSchema>;
export type CapabilityCard = z.infer<typeof CapabilityCardSchema>;
export type EvidencePlan = z.infer<typeof EvidencePlanSchema>;
export type EvidenceItem = z.infer<typeof EvidenceItemSchema>;
export type ToolCallRecord = z.infer<typeof ToolCallRecordSchema>;
export type RunTraceEvent = z.infer<typeof RunTraceEventSchema>;
export type RootCauseClaim = z.infer<typeof RootCauseClaimSchema>;
export type RcaResponse = z.infer<typeof RcaResponseSchema>;
export type ProposedAction = z.infer<typeof ProposedActionSchema>;
export type RemediationResponse = z.infer<typeof RemediationResponseSchema>;
export type ApprovalDecision = z.infer<typeof ApprovalDecisionSchema>;
export type ExecutionResult = z.infer<typeof ExecutionResultSchema>;
export type VerificationResult = z.infer<typeof VerificationResultSchema>;
export type MemoryRecord = z.infer<typeof MemoryRecordSchema>;
export type HandoffPackage = z.infer<typeof HandoffPackageSchema>;
