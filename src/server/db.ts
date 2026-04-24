import Database from "better-sqlite3";
import { mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { randomUUID } from "node:crypto";
import type {
  ApprovalDecision,
  EvidenceItem,
  HandoffPackage,
  JsonValue,
  ProposedAction,
  RunTraceEvent,
  Settings,
  ToolCallRecord,
  VerificationResult
} from "../shared/schemas.js";
import { SettingsSchema } from "../shared/schemas.js";
import type { Message, Run, RunBundle, Session } from "../shared/contracts.js";
import { DEFAULT_SETTINGS } from "./defaultSettings.js";

type Row = Record<string, unknown>;

const DB_PATH = resolve(process.env.AIOPS_AGENT_DB ?? ".aiops/aiops-agent.sqlite");

export class AppDatabase {
  private readonly db: Database.Database;

  constructor(path = DB_PATH) {
    mkdirSync(dirname(path), { recursive: true });
    this.db = new Database(path);
    this.db.pragma("journal_mode = WAL");
    this.migrate();
    this.ensureSettings();
  }

  close(): void {
    this.db.close();
  }

  createSession(title: string): Session {
    const now = isoNow();
    const session: Session = { id: randomUUID(), title, createdAt: now, updatedAt: now };
    this.db
      .prepare("insert into sessions (id, title, created_at, updated_at) values (?, ?, ?, ?)")
      .run(session.id, session.title, session.createdAt, session.updatedAt);
    return session;
  }

  listSessions(): Session[] {
    return (this.db
      .prepare("select * from sessions order by updated_at desc")
      .all() as Row[])
      .map(mapSession);
  }

  getSession(id: string): Session | undefined {
    const row = this.db.prepare("select * from sessions where id = ?").get(id) as Row | undefined;
    return row ? mapSession(row) : undefined;
  }

  touchSession(id: string): void {
    this.db.prepare("update sessions set updated_at = ? where id = ?").run(isoNow(), id);
  }

  addMessage(sessionId: string, role: Message["role"], content: string): Message {
    const message: Message = { id: randomUUID(), sessionId, role, content, createdAt: isoNow() };
    this.db
      .prepare(
        "insert into messages (id, session_id, role, content, created_at) values (?, ?, ?, ?, ?)"
      )
      .run(message.id, message.sessionId, message.role, message.content, message.createdAt);
    this.touchSession(sessionId);
    return message;
  }

  listMessages(sessionId: string): Message[] {
    return (this.db
      .prepare("select * from messages where session_id = ? order by created_at asc")
      .all(sessionId) as Row[])
      .map(mapMessage);
  }

  createRun(sessionId: string, userQuery: string): Run {
    const now = isoNow();
    const run: Run = {
      id: randomUUID(),
      sessionId,
      status: "queued",
      userQuery,
      finalAnswer: null,
      createdAt: now,
      updatedAt: now
    };
    this.db
      .prepare(
        "insert into runs (id, session_id, status, user_query, final_answer, created_at, updated_at) values (?, ?, ?, ?, ?, ?, ?)"
      )
      .run(run.id, run.sessionId, run.status, run.userQuery, run.finalAnswer, run.createdAt, run.updatedAt);
    return run;
  }

  updateRun(id: string, patch: Partial<Pick<Run, "status" | "finalAnswer">>): Run {
    const current = this.getRun(id);
    if (!current) throw new Error(`Run not found: ${id}`);
    const next = { ...current, ...patch, updatedAt: isoNow() };
    this.db
      .prepare("update runs set status = ?, final_answer = ?, updated_at = ? where id = ?")
      .run(next.status, next.finalAnswer, next.updatedAt, id);
    return next;
  }

  getRun(id: string): Run | undefined {
    const row = this.db.prepare("select * from runs where id = ?").get(id) as Row | undefined;
    return row ? mapRun(row) : undefined;
  }

  listRunsForSession(sessionId: string): Run[] {
    return (this.db
      .prepare("select * from runs where session_id = ? order by created_at asc")
      .all(sessionId) as Row[])
      .map(mapRun);
  }

  addTraceEvent(
    runId: string,
    type: RunTraceEvent["type"],
    severity: RunTraceEvent["severity"],
    message: string,
    data?: JsonValue
  ): RunTraceEvent {
    const event: RunTraceEvent = {
      id: randomUUID(),
      runId,
      type,
      severity,
      message,
      data,
      createdAt: isoNow()
    };
    this.db
      .prepare(
        "insert into trace_events (id, run_id, type, severity, message, data, created_at) values (?, ?, ?, ?, ?, ?, ?)"
      )
      .run(event.id, runId, type, severity, message, encodeJson(data ?? null), event.createdAt);
    return event;
  }

  listTraceEvents(runId: string): RunTraceEvent[] {
    return (this.db
      .prepare("select * from trace_events where run_id = ? order by created_at asc")
      .all(runId) as Row[])
      .map(mapTraceEvent);
  }

  addEvidence(item: Omit<EvidenceItem, "id" | "createdAt"> & { id?: string }): EvidenceItem {
    const evidence: EvidenceItem = { ...item, id: item.id ?? randomUUID(), createdAt: isoNow() };
    this.db
      .prepare(
        "insert into evidence (id, run_id, source, summary, data, sensitive, created_at) values (?, ?, ?, ?, ?, ?, ?)"
      )
      .run(
        evidence.id,
        evidence.runId,
        evidence.source,
        evidence.summary,
        encodeJson(evidence.data),
        evidence.sensitive ? 1 : 0,
        evidence.createdAt
      );
    return evidence;
  }

  listEvidence(runId: string): EvidenceItem[] {
    return (this.db
      .prepare("select * from evidence where run_id = ? order by created_at asc")
      .all(runId) as Row[])
      .map(mapEvidence);
  }

  addToolCall(call: Omit<ToolCallRecord, "id" | "startedAt"> & { id?: string; startedAt?: string }): ToolCallRecord {
    const record: ToolCallRecord = {
      ...call,
      id: call.id ?? randomUUID(),
      startedAt: call.startedAt ?? isoNow()
    };
    this.db
      .prepare(
        "insert into tool_calls (id, run_id, server_id, tool_name, arguments, status, safety_class, result, error, started_at, completed_at) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
      )
      .run(
        record.id,
        record.runId,
        record.serverId,
        record.toolName,
        encodeJson(record.arguments),
        record.status,
        record.safetyClass,
        encodeJson(record.result ?? null),
        record.error ?? null,
        record.startedAt,
        record.completedAt ?? null
      );
    return record;
  }

  updateToolCall(id: string, patch: Partial<ToolCallRecord>): ToolCallRecord {
    const current = this.getToolCall(id);
    if (!current) throw new Error(`Tool call not found: ${id}`);
    const next = { ...current, ...patch };
    this.db
      .prepare(
        "update tool_calls set status = ?, result = ?, error = ?, completed_at = ? where id = ?"
      )
      .run(next.status, encodeJson(next.result ?? null), next.error ?? null, next.completedAt ?? null, id);
    return next;
  }

  getToolCall(id: string): ToolCallRecord | undefined {
    const row = this.db.prepare("select * from tool_calls where id = ?").get(id) as Row | undefined;
    return row ? mapToolCall(row) : undefined;
  }

  listToolCalls(runId: string): ToolCallRecord[] {
    return (this.db
      .prepare("select * from tool_calls where run_id = ? order by started_at asc")
      .all(runId) as Row[])
      .map(mapToolCall);
  }

  addProposedAction(action: ProposedAction & { runId: string }): ProposedAction {
    const id = action.id ?? randomUUID();
    const createdAt = action.createdAt ?? isoNow();
    const saved = { ...action, id, createdAt, status: action.status ?? "proposed" };
    this.db
      .prepare(
        "insert into proposed_actions (id, run_id, title, description, risk_level, safety_class, server_id, tool_name, arguments, evidence_refs, rollback, verification_plan, status, created_at, approved_at, executed_at, result, error) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
      )
      .run(
        saved.id,
        saved.runId,
        saved.title,
        saved.description,
        saved.riskLevel,
        saved.safetyClass,
        saved.serverId,
        saved.toolName,
        encodeJson(saved.arguments),
        encodeJson(saved.evidenceRefs),
        saved.rollback,
        encodeJson(saved.verificationPlan),
        saved.status,
        saved.createdAt,
        null,
        null,
        null,
        null
      );
    return saved;
  }

  updateProposedAction(id: string, patch: Partial<ProposedAction> & { result?: JsonValue; error?: string | null }): ProposedAction {
    const current = this.getProposedAction(id);
    if (!current) throw new Error(`Proposed action not found: ${id}`);
    const next = { ...current, ...patch };
    const now = isoNow();
    this.db
      .prepare(
        "update proposed_actions set status = ?, arguments = ?, evidence_refs = ?, verification_plan = ?, approved_at = coalesce(approved_at, ?), executed_at = ?, result = ?, error = ? where id = ?"
      )
      .run(
        next.status,
        encodeJson(next.arguments),
        encodeJson(next.evidenceRefs),
        encodeJson(next.verificationPlan),
        patch.status === "approved" ? now : null,
        patch.status === "executed" || patch.status === "failed" ? now : null,
        encodeJson(patch.result ?? null),
        patch.error ?? null,
        id
      );
    return next;
  }

  getProposedAction(id: string): (ProposedAction & { runId: string }) | undefined {
    const row = this.db.prepare("select * from proposed_actions where id = ?").get(id) as Row | undefined;
    return row ? mapProposedAction(row) : undefined;
  }

  listProposedActions(runId: string): ProposedAction[] {
    return (this.db
      .prepare("select * from proposed_actions where run_id = ? order by created_at asc")
      .all(runId) as Row[])
      .map(mapProposedAction);
  }

  addApproval(actionId: string, decision: "approved" | "rejected", operator: string, comment?: string): ApprovalDecision {
    const approval: ApprovalDecision = {
      id: randomUUID(),
      actionId,
      decision,
      operator,
      comment,
      createdAt: isoNow()
    };
    this.db
      .prepare(
        "insert into approval_decisions (id, action_id, decision, operator, comment, created_at) values (?, ?, ?, ?, ?, ?)"
      )
      .run(approval.id, actionId, decision, operator, comment ?? null, approval.createdAt);
    return approval;
  }

  listApprovals(runId: string): ApprovalDecision[] {
    return (this.db
      .prepare(
        "select a.* from approval_decisions a join proposed_actions p on p.id = a.action_id where p.run_id = ? order by a.created_at asc"
      )
      .all(runId) as Row[])
      .map(mapApproval);
  }

  addVerificationResult(result: Omit<VerificationResult, "id" | "createdAt">): VerificationResult {
    const saved: VerificationResult = { ...result, id: randomUUID(), createdAt: isoNow() };
    this.db
      .prepare(
        "insert into verification_results (id, action_id, status, summary, evidence_refs, data, created_at) values (?, ?, ?, ?, ?, ?, ?)"
      )
      .run(
        saved.id,
        saved.actionId,
        saved.status,
        saved.summary,
        encodeJson(saved.evidenceRefs),
        encodeJson(saved.data ?? null),
        saved.createdAt
      );
    return saved;
  }

  listVerificationResults(runId: string): VerificationResult[] {
    return (this.db
      .prepare(
        "select v.* from verification_results v join proposed_actions p on p.id = v.action_id where p.run_id = ? order by v.created_at asc"
      )
      .all(runId) as Row[])
      .map(mapVerification);
  }

  addHandoff(runId: string, type: HandoffPackage["type"], title: string, content: string): HandoffPackage {
    const handoff: HandoffPackage = { id: randomUUID(), runId, type, title, content, createdAt: isoNow() };
    this.db
      .prepare(
        "insert into handoffs (id, run_id, type, title, content, created_at) values (?, ?, ?, ?, ?, ?)"
      )
      .run(handoff.id, runId, type, title, content, handoff.createdAt);
    return handoff;
  }

  listHandoffs(runId: string): HandoffPackage[] {
    return (this.db
      .prepare("select * from handoffs where run_id = ? order by created_at asc")
      .all(runId) as Row[])
      .map(mapHandoff);
  }

  getSettings(): Settings {
    const row = this.db.prepare("select data from settings where id = 'default'").get() as Row;
    return SettingsSchema.parse(decodeJson(row.data as string));
  }

  saveSettings(settings: Settings): Settings {
    const parsed = SettingsSchema.parse(settings);
    this.db
      .prepare("insert or replace into settings (id, data, updated_at) values ('default', ?, ?)")
      .run(encodeJson(parsed), isoNow());
    return parsed;
  }

  getRunBundle(runId: string): RunBundle {
    const run = this.getRun(runId);
    if (!run) throw new Error(`Run not found: ${runId}`);
    const session = this.getSession(run.sessionId);
    return {
      run,
      session,
      messages: this.listMessages(run.sessionId),
      traceEvents: this.listTraceEvents(runId),
      evidence: this.listEvidence(runId),
      toolCalls: this.listToolCalls(runId),
      proposedActions: this.listProposedActions(runId),
      approvals: this.listApprovals(runId),
      verificationResults: this.listVerificationResults(runId),
      handoffs: this.listHandoffs(runId)
    };
  }

  private ensureSettings(): void {
    const row = this.db.prepare("select data from settings where id = 'default'").get() as Row | undefined;
    if (!row) this.saveSettings(DEFAULT_SETTINGS);
  }

  private migrate(): void {
    this.db.exec(`
      create table if not exists settings (
        id text primary key,
        data text not null,
        updated_at text not null
      );
      create table if not exists sessions (
        id text primary key,
        title text not null,
        created_at text not null,
        updated_at text not null
      );
      create table if not exists messages (
        id text primary key,
        session_id text not null,
        role text not null,
        content text not null,
        created_at text not null
      );
      create table if not exists runs (
        id text primary key,
        session_id text not null,
        status text not null,
        user_query text not null,
        final_answer text,
        created_at text not null,
        updated_at text not null
      );
      create table if not exists trace_events (
        id text primary key,
        run_id text not null,
        type text not null,
        severity text not null,
        message text not null,
        data text,
        created_at text not null
      );
      create table if not exists evidence (
        id text primary key,
        run_id text not null,
        source text not null,
        summary text not null,
        data text not null,
        sensitive integer not null,
        created_at text not null
      );
      create table if not exists tool_calls (
        id text primary key,
        run_id text not null,
        server_id text not null,
        tool_name text not null,
        arguments text not null,
        status text not null,
        safety_class text not null,
        result text,
        error text,
        started_at text not null,
        completed_at text
      );
      create table if not exists proposed_actions (
        id text primary key,
        run_id text not null,
        title text not null,
        description text not null,
        risk_level text not null,
        safety_class text not null,
        server_id text not null,
        tool_name text not null,
        arguments text not null,
        evidence_refs text not null,
        rollback text not null,
        verification_plan text not null,
        status text not null,
        created_at text not null,
        approved_at text,
        executed_at text,
        result text,
        error text
      );
      create table if not exists approval_decisions (
        id text primary key,
        action_id text not null,
        decision text not null,
        operator text not null,
        comment text,
        created_at text not null
      );
      create table if not exists verification_results (
        id text primary key,
        action_id text not null,
        status text not null,
        summary text not null,
        evidence_refs text not null,
        data text,
        created_at text not null
      );
      create table if not exists memories (
        id text primary key,
        scope text not null,
        key text not null,
        value text not null,
        sensitive integer not null,
        created_at text not null,
        updated_at text not null
      );
      create table if not exists handoffs (
        id text primary key,
        run_id text not null,
        type text not null,
        title text not null,
        content text not null,
        created_at text not null
      );
    `);
  }
}

export function isoNow(): string {
  return new Date().toISOString();
}

function encodeJson(value: unknown): string {
  return JSON.stringify(value ?? null);
}

function decodeJson(value: string | null | undefined): JsonValue {
  if (value == null) return null;
  return JSON.parse(value) as JsonValue;
}

function mapSession(row: Row): Session {
  return {
    id: row.id as string,
    title: row.title as string,
    createdAt: row.created_at as string,
    updatedAt: row.updated_at as string
  };
}

function mapMessage(row: Row): Message {
  return {
    id: row.id as string,
    sessionId: row.session_id as string,
    role: row.role as Message["role"],
    content: row.content as string,
    createdAt: row.created_at as string
  };
}

function mapRun(row: Row): Run {
  return {
    id: row.id as string,
    sessionId: row.session_id as string,
    status: row.status as Run["status"],
    userQuery: row.user_query as string,
    finalAnswer: (row.final_answer as string | null) ?? null,
    createdAt: row.created_at as string,
    updatedAt: row.updated_at as string
  };
}

function mapTraceEvent(row: Row): RunTraceEvent {
  return {
    id: row.id as string,
    runId: row.run_id as string,
    type: row.type as RunTraceEvent["type"],
    severity: row.severity as RunTraceEvent["severity"],
    message: row.message as string,
    data: decodeJson(row.data as string | null),
    createdAt: row.created_at as string
  };
}

function mapEvidence(row: Row): EvidenceItem {
  return {
    id: row.id as string,
    runId: row.run_id as string,
    source: row.source as string,
    summary: row.summary as string,
    data: decodeJson(row.data as string),
    sensitive: Boolean(row.sensitive),
    createdAt: row.created_at as string
  };
}

function mapToolCall(row: Row): ToolCallRecord {
  return {
    id: row.id as string,
    runId: row.run_id as string,
    serverId: row.server_id as string,
    toolName: row.tool_name as string,
    arguments: decodeJson(row.arguments as string) as Record<string, JsonValue>,
    status: row.status as ToolCallRecord["status"],
    safetyClass: row.safety_class as ToolCallRecord["safetyClass"],
    result: decodeJson(row.result as string | null),
    error: (row.error as string | null) ?? undefined,
    startedAt: row.started_at as string,
    completedAt: (row.completed_at as string | null) ?? undefined
  };
}

function mapProposedAction(row: Row): ProposedAction & { runId: string } {
  return {
    id: row.id as string,
    runId: row.run_id as string,
    title: row.title as string,
    description: row.description as string,
    riskLevel: row.risk_level as ProposedAction["riskLevel"],
    safetyClass: row.safety_class as ProposedAction["safetyClass"],
    serverId: row.server_id as string,
    toolName: row.tool_name as string,
    arguments: decodeJson(row.arguments as string) as Record<string, JsonValue>,
    evidenceRefs: decodeJson(row.evidence_refs as string) as string[],
    rollback: row.rollback as string,
    verificationPlan: decodeJson(row.verification_plan as string) as ProposedAction["verificationPlan"],
    status: row.status as ProposedAction["status"],
    createdAt: row.created_at as string
  };
}

function mapApproval(row: Row): ApprovalDecision {
  return {
    id: row.id as string,
    actionId: row.action_id as string,
    decision: row.decision as ApprovalDecision["decision"],
    operator: row.operator as string,
    comment: (row.comment as string | null) ?? undefined,
    createdAt: row.created_at as string
  };
}

function mapVerification(row: Row): VerificationResult {
  return {
    id: row.id as string,
    actionId: row.action_id as string,
    status: row.status as VerificationResult["status"],
    summary: row.summary as string,
    evidenceRefs: decodeJson(row.evidence_refs as string) as string[],
    data: decodeJson(row.data as string | null),
    createdAt: row.created_at as string
  };
}

function mapHandoff(row: Row): HandoffPackage {
  return {
    id: row.id as string,
    runId: row.run_id as string,
    type: row.type as HandoffPackage["type"],
    title: row.title as string,
    content: row.content as string,
    createdAt: row.created_at as string
  };
}
