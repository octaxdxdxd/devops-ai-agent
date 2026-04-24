import type {
  CapabilityCard,
  EntityExtraction,
  EvidenceItem,
  IntentClassification,
  ToolCallRecord
} from "../shared/schemas.js";

export function intentContext(query: string): Record<string, unknown> {
  return { query };
}

export function entityContext(query: string, classification: IntentClassification): Record<string, unknown> {
  return { query, classification };
}

export function evidencePlanContext(args: {
  query: string;
  classification: IntentClassification;
  entities: EntityExtraction;
  capabilities: CapabilityCard[];
  discoveryErrors: Array<{ serverId: string; message: string }>;
}): Record<string, unknown> {
  return {
    query: args.query,
    classification: args.classification,
    entities: args.entities,
    capabilities: args.capabilities.map((capability) => ({
      id: capability.id,
      serverId: capability.serverId,
      serverKind: capability.serverKind,
      name: capability.name,
      title: capability.title,
      description: capability.description,
      inputSchema: capability.inputSchema,
      annotations: capability.annotations,
      safetyClass: capability.safetyClass
    })),
    discoveryErrors: args.discoveryErrors
  };
}

export function rcaContext(args: {
  query: string;
  classification: IntentClassification;
  entities: EntityExtraction;
  evidence: EvidenceItem[];
  toolCalls: ToolCallRecord[];
  discoveryErrors: Array<{ serverId: string; message: string }>;
}): Record<string, unknown> {
  return {
    query: args.query,
    classification: args.classification,
    entities: args.entities,
    evidence: args.evidence.map((item) => ({
      id: item.id,
      source: item.source,
      summary: item.summary,
      sensitive: item.sensitive,
      data: item.sensitive ? "[sensitive evidence redacted from model context]" : item.data
    })),
    toolCalls: args.toolCalls.map((call) => ({
      id: call.id,
      serverId: call.serverId,
      toolName: call.toolName,
      status: call.status,
      safetyClass: call.safetyClass,
      error: call.error
    })),
    discoveryErrors: args.discoveryErrors
  };
}

export function remediationContext(args: {
  query: string;
  rca: unknown;
  evidence: EvidenceItem[];
  capabilities: CapabilityCard[];
}): Record<string, unknown> {
  return {
    query: args.query,
    rca: args.rca,
    evidenceRefs: args.evidence.map((item) => ({
      id: item.id,
      source: item.source,
      summary: item.summary
    })),
    capabilities: args.capabilities.map((capability) => ({
      serverId: capability.serverId,
      name: capability.name,
      description: capability.description,
      inputSchema: capability.inputSchema,
      safetyClass: capability.safetyClass,
      annotations: capability.annotations
    }))
  };
}
