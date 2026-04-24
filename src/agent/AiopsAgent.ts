import { randomUUID } from "node:crypto";
import type { AppDatabase } from "../server/db.js";
import { isoNow } from "../server/db.js";
import type { McpManager, McpDiscoveryResult } from "../mcp/client.js";
import type {
  CapabilityCard,
  EvidenceItem,
  EvidencePlan,
  JsonValue,
  ProposedAction,
  RemediationResponse,
  RcaResponse,
  RootCauseClaim,
  RunTraceEvent,
  Settings
} from "../shared/schemas.js";
import {
  EntityExtractionSchema,
  IntentClassificationSchema,
  RcaResponseSchema
} from "../shared/schemas.js";
import {
  LlmEvidencePlanSchema,
  LlmRemediationResponseSchema,
  type LlmEvidencePlan,
  type LlmRemediationResponse
} from "./llmSchemas.js";
import { RunBus } from "./runBus.js";
import type { StructuredLlmProvider } from "./codexProvider.js";
import {
  entityContext,
  evidencePlanContext,
  intentContext,
  rcaContext,
  remediationContext
} from "./prompts.js";
import {
  canExecuteTool,
  enforceEvidenceForClaims,
  redactJson,
  validateProposedAction
} from "../policy/safety.js";

export class AiopsAgent {
  constructor(
    private readonly db: AppDatabase,
    private readonly llm: StructuredLlmProvider,
    private readonly mcp: McpManager,
    private readonly bus: RunBus,
    private readonly getSettings: () => Settings
  ) {}

  async run(runId: string): Promise<void> {
    const run = this.db.getRun(runId);
    if (!run) throw new Error(`Run not found: ${runId}`);

    try {
      this.db.updateRun(runId, { status: "running" });
      this.trace(runId, "run.started", "info", "Run started.", { query: run.userQuery });

      const classification = await this.llm.generate({
        schemaName: "IntentClassification",
        schema: IntentClassificationSchema,
        task: "Classify the operator request for an infrastructure AIOps agent.",
        context: intentContext(run.userQuery)
      });
      this.trace(runId, "classification.completed", "info", "Intent classification completed.", classification);

      if (classification.needsClarification && !classification.needsLiveEvidence) {
        const finalAnswer = formatClarificationAnswer(classification.clarificationQuestion);
        this.db.addMessage(run.sessionId, "assistant", finalAnswer);
        this.db.updateRun(runId, { status: "completed", finalAnswer });
        this.trace(runId, "run.completed", "info", "Run completed with a clarification request.", {
          status: "completed",
          reason: "classification_requested_clarification_without_live_evidence"
        });
        return;
      }

      const rawEntities = await this.llm.generate({
        schemaName: "EntityExtraction",
        schema: EntityExtractionSchema,
        task: "Extract infrastructure entities and unresolved references from the operator request.",
        context: entityContext(run.userQuery, classification)
      });
      const entities = {
        ...rawEntities,
        unresolvedReferences: rawEntities.unresolvedReferences ?? []
      };
      this.trace(runId, "entities.extracted", "info", "Infrastructure entities extracted.", entities);

      const discovery = await this.mcp.discover();
      this.trace(runId, "capabilities.discovered", discovery.errors.length ? "warning" : "info", "MCP capabilities discovered.", {
        capabilityCount: discovery.capabilities.length,
        resourceCount: discovery.resources.length,
        promptCount: discovery.prompts.length,
        errors: discovery.errors
      });

      const rawEvidencePlan = await this.llm.generate({
        schemaName: "EvidencePlan",
        schema: LlmEvidencePlanSchema,
        task:
          "Plan the minimum live evidence needed before answering. Use only discovered capabilities and mark mutating or unknown steps for proposal, not execution. Put tool arguments as a JSON object serialized into argumentsJson.",
        context: evidencePlanContext({
          query: run.userQuery,
          classification,
          entities,
          capabilities: discovery.capabilities,
          discoveryErrors: discovery.errors
        })
      });
      const evidencePlan = toEvidencePlan(LlmEvidencePlanSchema.parse(rawEvidencePlan));
      this.trace(runId, "evidence.plan.created", "info", "Evidence plan created.", evidencePlan);

      const evidence = await this.executeEvidencePlan(runId, evidencePlan, discovery);
      const toolCalls = this.db.listToolCalls(runId);

      const rawRca = await this.llm.generate({
        schemaName: "RcaResponse",
        schema: RcaResponseSchema,
        task: "Answer the operator using only the supplied evidence. Root-cause claims must cite evidence IDs.",
        context: rcaContext({
          query: run.userQuery,
          classification,
          entities,
          evidence,
          toolCalls,
          discoveryErrors: discovery.errors
        })
      });
      const rca: RcaResponse = {
        ...rawRca,
        rootCauseClaims: rawRca.rootCauseClaims.map((claim) => ({
          ...claim,
          evidenceRefs: claim.evidenceRefs ?? []
        })) as RootCauseClaim[],
        followUpQuestions: rawRca.followUpQuestions ?? [],
        missingEvidence: rawRca.missingEvidence ?? []
      };
      const guardedRca: RcaResponse = {
        ...rca,
        rootCauseClaims: enforceEvidenceForClaims(rca.rootCauseClaims)
      };
      this.trace(runId, "rca.completed", "info", "RCA response completed.", guardedRca);

      const rawRemediation = await this.llm.generate({
        schemaName: "RemediationResponse",
        schema: LlmRemediationResponseSchema,
        task:
          "Propose exact, minimal remediation actions only if justified. Every action must use a discovered capability, cite evidence, include rollback, and include read-only verification steps. Put action and verification tool arguments as JSON objects serialized into argumentsJson.",
        context: remediationContext({
          query: run.userQuery,
          rca: guardedRca,
          evidence,
          capabilities: discovery.capabilities
        })
      });
      const remediation = toRemediationResponse(LlmRemediationResponseSchema.parse(rawRemediation));
      const savedActions = this.saveValidActions(runId, remediation, discovery.capabilities);

      this.trace(runId, "actions.proposed", savedActions.length ? "warning" : "info", "Remediation proposal stage completed.", {
        actionCount: savedActions.length,
        rationale: remediation.rationale,
        noActionReason: remediation.noActionReason
      });

      const finalAnswer = formatAnswer(guardedRca, savedActions, discovery.errors);
      this.db.addMessage(run.sessionId, "assistant", finalAnswer);
      this.db.updateRun(runId, {
        status: savedActions.length ? "awaiting_approval" : "completed",
        finalAnswer
      });
      this.trace(
        runId,
        savedActions.length ? "actions.proposed" : "run.completed",
        savedActions.length ? "warning" : "info",
        savedActions.length ? "Run is awaiting operator approval." : "Run completed.",
        { status: savedActions.length ? "awaiting_approval" : "completed" }
      );
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      this.db.updateRun(runId, { status: "failed", finalAnswer: `Run failed: ${message}` });
      this.trace(runId, "run.failed", "error", "Run failed.", { error: message });
    }
  }

  async executeApprovedAction(actionId: string): Promise<void> {
    const action = this.db.getProposedAction(actionId);
    if (!action) throw new Error(`Proposed action not found: ${actionId}`);
    if (action.status !== "approved") {
      throw new Error(`Action must be approved before execution. Current status: ${action.status}`);
    }
    const settings = this.getSettings();
    const policy = canExecuteTool(action.safetyClass, settings, true);
    if (!policy.allowed) throw new Error(policy.reason);

    this.db.updateProposedAction(actionId, { status: "executing" });
    this.trace(action.runId, "execution.started", "warning", `Executing approved action: ${action.title}`, {
      actionId,
      serverId: action.serverId,
      toolName: action.toolName
    });

    try {
      const result = await this.mcp.callTool(action.serverId, action.toolName, action.arguments);
      this.db.updateProposedAction(actionId, {
        status: "executed",
        result: sanitizeMcpResult(result) as JsonValue,
        error: null
      });
      this.trace(action.runId, "execution.completed", "info", `Approved action executed: ${action.title}`, {
        actionId,
        result: sanitizeMcpResult(result)
      });
      await this.verifyAction(action);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      this.db.updateProposedAction(actionId, { status: "failed", error: message });
      this.trace(action.runId, "execution.completed", "error", `Approved action failed: ${action.title}`, {
        actionId,
        error: message
      });
      throw error;
    }
  }

  createHandoff(runId: string, type: "markdown" | "json" | "codex-task" = "markdown") {
    const bundle = this.db.getRunBundle(runId);
    const title = `AIOps handoff for run ${runId.slice(0, 8)}`;
    const content =
      type === "json"
        ? JSON.stringify(bundle, null, 2)
        : [
            `# ${title}`,
            "",
            `Query: ${bundle.run.userQuery}`,
            `Status: ${bundle.run.status}`,
            "",
            "## Answer",
            bundle.run.finalAnswer ?? "No final answer recorded.",
            "",
            "## Evidence",
            ...bundle.evidence.map((item) => `- ${item.id}: ${item.summary} (${item.source})`),
            "",
            "## Proposed Actions",
            ...bundle.proposedActions.map(
              (action) =>
                `- ${action.id}: ${action.title} [${action.status}] via ${action.serverId}/${action.toolName}`
            ),
            "",
            "## Trace",
            ...bundle.traceEvents.map((event) => `- ${event.createdAt} ${event.type}: ${event.message}`)
          ].join("\n");
    const handoff = this.db.addHandoff(runId, type, title, content);
    this.trace(runId, "handoff.created", "info", "Handoff package created.", { handoffId: handoff.id, type });
    return handoff;
  }

  private async executeEvidencePlan(
    runId: string,
    plan: EvidencePlan,
    discovery: McpDiscoveryResult
  ): Promise<EvidenceItem[]> {
    const evidence: EvidenceItem[] = [];
    const attempted = new Set<string>();
    const settings = this.getSettings();

    for (const step of plan.steps) {
      if (!step.serverId || !step.toolName) continue;
      const key = `${step.serverId}/${step.toolName}/${JSON.stringify(step.arguments)}`;
      if (attempted.has(key)) continue;
      attempted.add(key);

      const capability = discovery.capabilities.find(
        (item) => item.serverId === step.serverId && item.name === step.toolName
      );
      if (!capability) {
        this.trace(runId, "tool.blocked", "warning", `Evidence step references an unavailable tool: ${key}`, {
          step
        });
        continue;
      }

      const policy = canExecuteTool(capability.safetyClass, settings, false);
      if (!policy.allowed) {
        this.db.addToolCall({
          runId,
          serverId: step.serverId,
          toolName: step.toolName,
          arguments: step.arguments,
          status: "blocked",
          safetyClass: capability.safetyClass,
          error: policy.reason
        });
        this.trace(runId, "tool.blocked", "warning", policy.reason, {
          step,
          capability: capability.id
        });
        continue;
      }

      const call = this.db.addToolCall({
        runId,
        serverId: step.serverId,
        toolName: step.toolName,
        arguments: step.arguments,
        status: "running",
        safetyClass: capability.safetyClass
      });
      this.trace(runId, "tool.started", "info", `Calling ${step.serverId}/${step.toolName}.`, {
        toolCallId: call.id,
        purpose: step.purpose
      });

      try {
        const result = await this.mcp.callTool(step.serverId, step.toolName, step.arguments);
        const sanitized = sanitizeMcpResult(result);
        this.db.updateToolCall(call.id, {
          status: "succeeded",
          result: sanitized as JsonValue,
          completedAt: isoNow()
        });
        const item = this.db.addEvidence({
          id: randomUUID(),
          runId,
          source: `${step.serverId}/${step.toolName}`,
          summary: step.expectedEvidence,
          data: sanitized as JsonValue,
          sensitive: capability.safetyClass === "sensitive-read"
        });
        evidence.push(item);
        this.trace(runId, "tool.completed", "info", `Collected evidence from ${step.serverId}/${step.toolName}.`, {
          toolCallId: call.id,
          evidenceId: item.id
        });
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        this.db.updateToolCall(call.id, {
          status: "failed",
          error: message,
          completedAt: isoNow()
        });
        this.trace(runId, "tool.failed", "warning", `Tool call failed: ${step.serverId}/${step.toolName}.`, {
          toolCallId: call.id,
          error: message
        });
      }
    }

    if (evidence.length === 0 && discovery.errors.length) {
      const item = this.db.addEvidence({
        runId,
        source: "mcp.discovery",
        summary: "MCP discovery did not provide usable live evidence.",
        data: { errors: discovery.errors },
        sensitive: false
      });
      evidence.push(item);
    }
    return evidence;
  }

  private saveValidActions(
    runId: string,
    remediation: RemediationResponse,
    capabilities: CapabilityCard[]
  ): ProposedAction[] {
    const settings = this.getSettings();
    const saved: ProposedAction[] = [];
    for (const action of remediation.actions) {
      const parsed = {
        ...action,
        id: action.id ?? randomUUID(),
        runId,
        createdAt: isoNow(),
        status: "proposed" as const
      };
      const errors = validateProposedAction(parsed, capabilities, settings);
      if (errors.length) {
        this.trace(runId, "actions.proposed", "warning", `Rejected unsafe proposed action: ${action.title}`, {
          errors,
          action
        });
        continue;
      }
      saved.push(this.db.addProposedAction(parsed));
    }
    return saved;
  }

  private async verifyAction(action: ProposedAction & { runId: string }): Promise<void> {
    const evidenceRefs: string[] = [];
    const data: Record<string, JsonValue> = {};
    let failures = 0;
    let inconclusive = 0;
    for (const step of action.verificationPlan) {
      if (!step.serverId || !step.toolName) {
        inconclusive += 1;
        data[step.id] = { skipped: "No verification tool was specified.", expectedOutcome: step.expectedOutcome };
        continue;
      }
      const policy = canExecuteTool(step.safetyClass, this.getSettings(), false);
      if (!policy.allowed) {
        inconclusive += 1;
        data[step.id] = { blocked: policy.reason };
        continue;
      }
      try {
        const result = await this.mcp.callTool(step.serverId, step.toolName, step.arguments);
        const item = this.db.addEvidence({
          runId: action.runId,
          source: `${step.serverId}/${step.toolName}`,
          summary: `Verification: ${step.purpose}`,
          data: sanitizeMcpResult(result) as JsonValue,
          sensitive: step.safetyClass === "sensitive-read"
        });
        evidenceRefs.push(item.id);
        data[step.id] = { evidenceId: item.id, expectedOutcome: step.expectedOutcome };
      } catch (error) {
        failures += 1;
        data[step.id] = { error: error instanceof Error ? error.message : String(error) };
      }
    }
    const status = failures > 0 ? "failed" : inconclusive > 0 ? "inconclusive" : "passed";
    const result = this.db.addVerificationResult({
      actionId: action.id ?? "",
      status,
      summary:
        status === "passed"
          ? "Verification steps completed."
          : status === "failed"
            ? "One or more verification steps failed."
            : "Verification is inconclusive; manual review is required.",
      evidenceRefs,
      data
    });
    this.trace(action.runId, "verification.completed", status === "failed" ? "error" : "info", result.summary, {
      actionId: action.id,
      verificationResultId: result.id,
      status
    });
  }

  private trace(
    runId: string,
    type: RunTraceEvent["type"],
    severity: RunTraceEvent["severity"],
    message: string,
    data?: unknown
  ): void {
    const event = this.db.addTraceEvent(runId, type, severity, message, toJsonValue(data));
    this.bus.publish(event);
  }
}

function toEvidencePlan(raw: LlmEvidencePlan): EvidencePlan {
  return {
    ...raw,
    steps: raw.steps.map(({ argumentsJson, ...step }) => ({
      ...step,
      arguments: parseJsonObject(argumentsJson),
      required: step.required ?? true
    })),
    answerWithoutToolsIfUnavailable: raw.answerWithoutToolsIfUnavailable ?? true
  };
}

function toRemediationResponse(raw: LlmRemediationResponse): RemediationResponse {
  return {
    ...raw,
    actions: raw.actions.map(({ argumentsJson, verificationPlan, ...action }) => ({
      ...action,
      arguments: parseJsonObject(argumentsJson),
      status: action.status ?? "proposed",
      verificationPlan: verificationPlan.map(({ argumentsJson: verificationArgumentsJson, ...step }) => ({
        ...step,
        arguments: parseJsonObject(verificationArgumentsJson),
        safetyClass: step.safetyClass ?? "read"
      }))
    }))
  };
}

function parseJsonObject(value: string | undefined): Record<string, JsonValue> {
  if (!value?.trim()) return {};
  try {
    const parsed = JSON.parse(value) as unknown;
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      return parsed as Record<string, JsonValue>;
    }
    return {};
  } catch {
    return {};
  }
}

function sanitizeMcpResult(result: unknown): JsonValue {
  return redactJson(JSON.parse(JSON.stringify(result ?? null))) as JsonValue;
}

function toJsonValue(value: unknown): JsonValue {
  return JSON.parse(JSON.stringify(value ?? null)) as JsonValue;
}

function formatClarificationAnswer(question: string | undefined): string {
  return question?.trim() || "What infrastructure task do you want help with?";
}

function formatAnswer(
  rca: RcaResponse,
  actions: ProposedAction[],
  discoveryErrors: Array<{ serverId: string; message: string }>
): string {
  const lines = [rca.answer, ""];
  if (rca.rootCauseClaims.length) {
    lines.push("Root-cause claims:");
    for (const claim of rca.rootCauseClaims) {
      lines.push(`- ${claim.status}: ${claim.claim} (confidence ${Math.round(claim.confidence * 100)}%, evidence ${claim.evidenceRefs.join(", ") || "none"})`);
    }
    lines.push("");
  }
  if (rca.missingEvidence.length) {
    lines.push("Missing evidence:");
    for (const item of rca.missingEvidence) lines.push(`- ${item}`);
    lines.push("");
  }
  if (actions.length) {
    lines.push(`Proposed actions awaiting approval: ${actions.length}`);
    for (const action of actions) lines.push(`- ${action.title} (${action.riskLevel})`);
    lines.push("");
  }
  if (discoveryErrors.length) {
    lines.push("Connector issues:");
    for (const error of discoveryErrors) lines.push(`- ${error.serverId}: ${error.message}`);
  }
  return lines.join("\n").trim();
}
