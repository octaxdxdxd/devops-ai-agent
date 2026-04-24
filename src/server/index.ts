import Fastify from "fastify";
import cors from "@fastify/cors";
import staticPlugin from "@fastify/static";
import { existsSync } from "node:fs";
import { resolve } from "node:path";
import { AppDatabase } from "./db.js";
import { LlmRouter } from "../agent/codexProvider.js";
import { McpManager } from "../mcp/client.js";
import { AiopsAgent } from "../agent/AiopsAgent.js";
import { RunBus } from "../agent/runBus.js";
import {
  ActionDecisionRequestSchema,
  CreateRunRequestSchema
} from "../shared/contracts.js";
import { SettingsSchema } from "../shared/schemas.js";

const db = new AppDatabase();
const bus = new RunBus();
const llm = new LlmRouter(() => db.getSettings());
const mcp = new McpManager(() => db.getSettings());
const agent = new AiopsAgent(db, llm, mcp, bus, () => db.getSettings());

const settings = db.getSettings();
const fastify = Fastify({
  logger: {
    level: process.env.LOG_LEVEL ?? "info"
  }
});

await fastify.register(cors, { origin: true });

fastify.get("/healthz", async () => ({ ok: true, service: "aiops-agent-local-cockpit" }));

fastify.get("/api/sessions", async () => db.listSessions());

fastify.post("/api/sessions", async (request) => {
  const body = (request.body ?? {}) as { title?: string };
  return db.createSession(body.title?.trim() || "New investigation");
});

fastify.get("/api/sessions/:id", async (request, reply) => {
  const { id } = request.params as { id: string };
  const session = db.getSession(id);
  if (!session) return reply.code(404).send({ error: "Session not found" });
  return {
    session,
    messages: db.listMessages(id),
    runs: db.listRunsForSession(id)
  };
});

fastify.post("/api/runs", async (request, reply) => {
  const parsed = CreateRunRequestSchema.safeParse(request.body);
  if (!parsed.success) return reply.code(400).send({ error: parsed.error.format() });
  const message = parsed.data.message.trim();
  const session =
    parsed.data.sessionId && db.getSession(parsed.data.sessionId)
      ? db.getSession(parsed.data.sessionId)
      : db.createSession(titleFromMessage(message));
  if (!session) return reply.code(500).send({ error: "Failed to create session" });
  db.addMessage(session.id, "user", message);
  const run = db.createRun(session.id, message);
  void agent.run(run.id);
  return { runId: run.id, sessionId: session.id };
});

fastify.get("/api/runs/:id", async (request, reply) => {
  const { id } = request.params as { id: string };
  try {
    return db.getRunBundle(id);
  } catch (error) {
    return reply.code(404).send({ error: error instanceof Error ? error.message : String(error) });
  }
});

fastify.get("/api/runs/:id/stream", async (request, reply) => {
  const { id } = request.params as { id: string };
  const run = db.getRun(id);
  if (!run) return reply.code(404).send({ error: "Run not found" });
  reply.raw.writeHead(200, {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache, no-transform",
    Connection: "keep-alive",
    "X-Accel-Buffering": "no"
  });
  for (const event of db.listTraceEvents(id)) {
    reply.raw.write(`event: trace\ndata: ${JSON.stringify(event)}\n\n`);
  }
  const unsubscribe = bus.subscribe(id, (event) => {
    reply.raw.write(`event: trace\ndata: ${JSON.stringify(event)}\n\n`);
  });
  const heartbeat = setInterval(() => {
    reply.raw.write(`event: heartbeat\ndata: {}\n\n`);
  }, 15000);
  request.raw.on("close", () => {
    clearInterval(heartbeat);
    unsubscribe();
    reply.raw.end();
  });
  return reply;
});

fastify.post("/api/proposed-actions/:id/approve", async (request, reply) => {
  const { id } = request.params as { id: string };
  const parsed = ActionDecisionRequestSchema.safeParse(request.body ?? {});
  if (!parsed.success) return reply.code(400).send({ error: parsed.error.format() });
  const action = db.getProposedAction(id);
  if (!action) return reply.code(404).send({ error: "Proposed action not found" });
  if (action.status !== "proposed") {
    return reply.code(409).send({ error: `Action is not approvable in status ${action.status}` });
  }
  const approval = db.addApproval(id, "approved", parsed.data.operator, parsed.data.comment);
  const updated = db.updateProposedAction(id, { status: "approved" });
  const event = db.addTraceEvent(action.runId, "approval.recorded", "warning", `Action approved: ${action.title}`, {
    actionId: id,
    operator: parsed.data.operator
  });
  bus.publish(event);
  return { action: updated, approval };
});

fastify.post("/api/proposed-actions/:id/reject", async (request, reply) => {
  const { id } = request.params as { id: string };
  const parsed = ActionDecisionRequestSchema.safeParse(request.body ?? {});
  if (!parsed.success) return reply.code(400).send({ error: parsed.error.format() });
  const action = db.getProposedAction(id);
  if (!action) return reply.code(404).send({ error: "Proposed action not found" });
  const approval = db.addApproval(id, "rejected", parsed.data.operator, parsed.data.comment);
  const updated = db.updateProposedAction(id, { status: "rejected" });
  const event = db.addTraceEvent(action.runId, "approval.recorded", "info", `Action rejected: ${action.title}`, {
    actionId: id,
    operator: parsed.data.operator
  });
  bus.publish(event);
  return { action: updated, approval };
});

fastify.post("/api/proposed-actions/:id/execute", async (request, reply) => {
  const { id } = request.params as { id: string };
  try {
    await agent.executeApprovedAction(id);
    const action = db.getProposedAction(id);
    return { action };
  } catch (error) {
    return reply.code(400).send({ error: error instanceof Error ? error.message : String(error) });
  }
});

fastify.get("/api/settings", async () => db.getSettings());

fastify.put("/api/settings", async (request, reply) => {
  const parsed = SettingsSchema.safeParse(request.body);
  if (!parsed.success) return reply.code(400).send({ error: parsed.error.format() });
  return db.saveSettings(parsed.data);
});

fastify.get("/api/mcp/discover", async () => mcp.discover());

fastify.post("/api/handoffs", async (request, reply) => {
  const body = (request.body ?? {}) as { runId?: string; type?: "markdown" | "json" | "codex-task" };
  if (!body.runId) return reply.code(400).send({ error: "runId is required" });
  try {
    return agent.createHandoff(body.runId, body.type ?? "markdown");
  } catch (error) {
    return reply.code(400).send({ error: error instanceof Error ? error.message : String(error) });
  }
});

const clientDist = resolve("dist/client");
if (existsSync(clientDist)) {
  await fastify.register(staticPlugin, {
    root: clientDist,
    prefix: "/"
  });
  fastify.setNotFoundHandler((request, reply) => {
    if (request.raw.url?.startsWith("/api")) return reply.code(404).send({ error: "Not found" });
    return reply.sendFile("index.html");
  });
}

const port = Number(process.env.PORT ?? settings.app.apiPort);
const host = process.env.HOST ?? settings.app.bindHost;

try {
  await fastify.listen({ host, port });
} catch (error) {
  fastify.log.error(error);
  process.exit(1);
}

process.on("SIGINT", async () => {
  await fastify.close();
  db.close();
});

function titleFromMessage(message: string): string {
  const squashed = message.replace(/\s+/g, " ").trim();
  return squashed.length > 72 ? `${squashed.slice(0, 69)}...` : squashed || "New investigation";
}
