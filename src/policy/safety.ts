import type {
  CapabilityCard,
  JsonValue,
  ProposedAction,
  SafetyClass,
  Settings,
  ToolCallRecord
} from "../shared/schemas.js";

const PLACEHOLDER_PATTERNS = [
  /^$/,
  /^<.*>$/,
  /^\$\{.*\}$/,
  /\bTBD\b/i,
  /\bTODO\b/i,
  /\bREPLACE_ME\b/i,
  /\bYOUR_[A-Z0-9_]+\b/i,
  /\bexample\b/i,
  /\bplaceholder\b/i,
  /\bunknown\b/i
];

export type PolicyDecision = {
  allowed: boolean;
  reason: string;
  safetyClass: SafetyClass;
};

export function classifyCapabilitySafety(capability: {
  annotations?: Record<string, JsonValue>;
  defaultSafetyClass?: SafetyClass;
}): SafetyClass {
  const annotations = capability.annotations ?? {};
  if (annotations.readOnlyHint === true) return "read";
  if (annotations.destructiveHint === true) return "delete";
  if (annotations.sensitive === true || annotations["aiops.sensitive"] === true) return "sensitive-read";
  if (annotations.idempotentHint === false || annotations.openWorldHint === true) {
    return capability.defaultSafetyClass ?? "unknown";
  }
  return capability.defaultSafetyClass ?? "unknown";
}

export function canExecuteTool(
  safetyClass: SafetyClass,
  settings: Settings,
  approved: boolean
): PolicyDecision {
  const requiresApproval = settings.policy.requireApprovalFor.includes(safetyClass);
  if (requiresApproval && !approved) {
    return {
      allowed: false,
      reason: `Tool safety class '${safetyClass}' requires explicit operator approval.`,
      safetyClass
    };
  }
  if ((safetyClass === "write" || safetyClass === "delete") && !approved) {
    return {
      allowed: false,
      reason: "Mutating operations are blocked until an approved proposed action exists.",
      safetyClass
    };
  }
  return { allowed: true, reason: "Allowed by local policy.", safetyClass };
}

export function validateProposedAction(
  action: ProposedAction,
  capabilities: CapabilityCard[],
  settings: Settings
): string[] {
  const errors: string[] = [];
  const capability = capabilities.find(
    (item) => item.serverId === action.serverId && item.name === action.toolName
  );
  if (!capability) {
    errors.push(`No discovered capability matches ${action.serverId}/${action.toolName}.`);
  }
  if (!action.evidenceRefs.length) {
    errors.push("At least one evidence reference is required.");
  }
  if (!action.rollback.trim()) {
    errors.push("Rollback notes are required.");
  }
  if (!action.verificationPlan.length) {
    errors.push("A verification plan is required.");
  }
  if (settings.policy.rejectPlaceholderArguments) {
    const placeholders = findPlaceholders(action.arguments);
    if (placeholders.length) {
      errors.push(`Placeholder or unresolved argument values found: ${placeholders.join(", ")}.`);
    }
    for (const step of action.verificationPlan) {
      const stepPlaceholders = findPlaceholders(step.arguments);
      if (stepPlaceholders.length) {
        errors.push(
          `Verification step '${step.id}' contains placeholders: ${stepPlaceholders.join(", ")}.`
        );
      }
    }
  }
  return errors;
}

export function enforceEvidenceForClaims<
  T extends { evidenceRefs: string[]; status: "supported" | "hypothesis" | "ruled-out" }
>(
  claims: T[]
): T[] {
  return claims.map((claim) => {
    if (claim.status === "supported" && claim.evidenceRefs.length === 0) {
      return { ...claim, status: "hypothesis" } as T;
    }
    return claim;
  });
}

export function redactSensitiveText(value: string): string {
  return value
    .replace(/AKIA[0-9A-Z]{16}/g, "[REDACTED_AWS_ACCESS_KEY]")
    .replace(/(?<=secretAccessKey["':\s=]{1,20})[A-Za-z0-9/+=]{20,}/gi, "[REDACTED_SECRET]")
    .replace(/(?<=token["':\s=]{1,20})[A-Za-z0-9._-]{20,}/gi, "[REDACTED_TOKEN]")
    .replace(/(?<=password["':\s=]{1,20})[^"',\s]+/gi, "[REDACTED_PASSWORD]");
}

export function redactJson(value: JsonValue): JsonValue {
  if (typeof value === "string") return redactSensitiveText(value);
  if (Array.isArray(value)) return value.map((item) => redactJson(item));
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value).map(([key, nested]) => {
        if (/secret|token|password|credential|key/i.test(key)) {
          return [key, "[REDACTED]"];
        }
        return [key, redactJson(nested)];
      })
    );
  }
  return value;
}

export function summarizeToolCallForAudit(call: ToolCallRecord): string {
  return `${call.status.toUpperCase()} ${call.serverId}/${call.toolName} (${call.safetyClass})`;
}

export function findPlaceholders(value: JsonValue, path = "$"): string[] {
  if (typeof value === "string") {
    const trimmed = value.trim();
    return PLACEHOLDER_PATTERNS.some((pattern) => pattern.test(trimmed)) ? [path] : [];
  }
  if (Array.isArray(value)) {
    return value.flatMap((item, index) => findPlaceholders(item, `${path}[${index}]`));
  }
  if (value && typeof value === "object") {
    return Object.entries(value).flatMap(([key, nested]) => findPlaceholders(nested, `${path}.${key}`));
  }
  return [];
}
