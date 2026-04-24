import { spawn } from "node:child_process";
import { existsSync, mkdtempSync, readdirSync, rmSync, statSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { tmpdir } from "node:os";
import { delimiter, join } from "node:path";
import type { ZodType } from "zod";
import { zodToJsonSchema } from "zod-to-json-schema";
import type { Settings } from "../shared/schemas.js";

export type LlmRequest<T> = {
  task: string;
  schemaName: string;
  schema: ZodType<T>;
  system?: string;
  context?: unknown;
};

export type StructuredLlmProvider = {
  generate<T>(request: LlmRequest<T>): Promise<T>;
};

export class LlmRouter implements StructuredLlmProvider {
  private codex?: CodexProvider;
  private copilot?: CopilotProvider;

  constructor(private readonly getSettings: () => Settings) {}

  async generate<T>(request: LlmRequest<T>): Promise<T> {
    const settings = this.getSettings();
    const primaryName = settings.llm.provider === "auto" ? "codex" : settings.llm.provider;
    const fallbackName = settings.llm.provider === "auto" ? "copilot" : settings.llm.fallbackProvider;
    const primary = this.getProvider(primaryName);

    try {
      return await primary.generate(request);
    } catch (primaryError) {
      if (fallbackName === "none" || fallbackName === primaryName) throw primaryError;
      const fallback = this.getProvider(fallbackName);
      try {
        return await fallback.generate(request);
      } catch (fallbackError) {
        const primaryMessage = primaryError instanceof Error ? primaryError.message : String(primaryError);
        const fallbackMessage = fallbackError instanceof Error ? fallbackError.message : String(fallbackError);
        throw new Error(
          `${primaryName} provider failed: ${primaryMessage}; ${fallbackName} fallback failed: ${fallbackMessage}`
        );
      }
    }
  }

  private getProvider(name: "codex" | "copilot"): StructuredLlmProvider {
    this.codex ??= new CodexProvider(this.getSettings);
    this.copilot ??= new CopilotProvider(this.getSettings);
    return name === "codex" ? this.codex : this.copilot;
  }
}

export class CodexProvider implements StructuredLlmProvider {
  constructor(private readonly getSettings: () => Settings) {}

  async generate<T>(request: LlmRequest<T>): Promise<T> {
    const settings = this.getSettings();
    if (settings.codex.mock) return request.schema.parse(mockResponse(request.schemaName, request.context));

    const schemaDir = mkdtempSync(join(tmpdir(), "aiops-codex-schema-"));
    const schemaPath = join(schemaDir, `${request.schemaName}.schema.json`);
    writeFileSync(
      schemaPath,
      JSON.stringify(buildCodexOutputSchema(request.schema, request.schemaName), null, 2),
      "utf8"
    );

    const prompt = buildPrompt(request);
    const args = [
      "exec",
      "--skip-git-repo-check",
      "--ephemeral",
      "--json",
      "--ignore-rules",
      "--sandbox",
      settings.codex.sandbox,
      "--output-schema",
      schemaPath,
      "-m",
      settings.codex.model,
      "-c",
      `model_reasoning_effort="${settings.codex.reasoningEffort}"`,
      "-"
    ];

    try {
      const command = resolveCodexCommand(settings.codex.command);
      const stdout = await runCodex(command, args, prompt, settings.codex.timeoutMs);
      const raw = extractFinalPayload(stdout);
      return request.schema.parse(stripNullObjectFields(raw));
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      throw new Error(`Codex ${request.schemaName} request failed: ${message}`);
    } finally {
      rmSync(schemaDir, { recursive: true, force: true });
    }
  }
}

export class CopilotProvider implements StructuredLlmProvider {
  constructor(private readonly getSettings: () => Settings) {}

  async generate<T>(request: LlmRequest<T>): Promise<T> {
    const settings = this.getSettings();
    if (settings.copilot.mock) return request.schema.parse(mockResponse(request.schemaName, request.context));

    const prompt = buildPrompt(request, buildCodexOutputSchema(request.schema, request.schemaName));
    const args = buildCopilotArgs(settings, prompt);

    try {
      const command = resolveCopilotCommand(settings.copilot.command);
      const stdout = await runCommand(command, args, "", settings.copilot.timeoutMs);
      const raw = extractFinalPayload(stdout);
      return request.schema.parse(stripNullObjectFields(raw));
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      throw new Error(`Copilot ${request.schemaName} request failed: ${message}`);
    }
  }
}

export function buildCodexOutputSchema<T>(schema: ZodType<T>, schemaName: string): unknown {
  const raw = zodToJsonSchema(schema, schemaName) as Record<string, unknown>;
  const ref = typeof raw.$ref === "string" ? raw.$ref : undefined;
  if (ref?.startsWith("#/definitions/")) {
    const definitionName = ref.slice("#/definitions/".length);
    const definitions = raw.definitions as Record<string, unknown> | undefined;
    const definition = definitions?.[definitionName];
    if (definition && typeof definition === "object" && !Array.isArray(definition)) {
      const remainingDefinitions = { ...(definitions ?? {}) };
      delete remainingDefinitions[definitionName];
      return toStrictStructuredOutputSchema({
        ...(definition as Record<string, unknown>),
        ...(Object.keys(remainingDefinitions).length ? { definitions: remainingDefinitions } : {})
      });
    }
  }
  return toStrictStructuredOutputSchema(raw);
}

export function resolveCodexCommand(command: string): string {
  if (command.includes("/")) {
    if (!existsSync(command)) {
      throw new Error(
        `Configured Codex command does not exist: ${command}. Update Settings codex.command or set CODEX_COMMAND.`
      );
    }
    return command;
  }

  const fromPath = findExecutableOnPath(command);
  if (fromPath) return fromPath;

  if (command === "codex") {
    const discovered = discoverVsCodeCodexBinary();
    if (discovered) return discovered;
  }

  throw new Error(
    `Could not find '${command}' on the backend PATH. Set Settings codex.command to the full Codex binary path or start the server with CODEX_COMMAND=/path/to/codex.`
  );
}

export function resolveCopilotCommand(command: string): string {
  if (command.includes("/")) {
    if (!existsSync(command)) {
      throw new Error(
        `Configured Copilot command does not exist: ${command}. Install GitHub Copilot CLI or update Settings copilot.command.`
      );
    }
    return command;
  }

  const fromPath = findExecutableOnPath(command);
  if (fromPath) return fromPath;

  const shim = discoverVsCodeCopilotShim();
  const shimHint = shim
    ? ` The VS Code Copilot extension shim exists at ${shim}, but it only launches/installs the separate Copilot CLI.`
    : "";
  throw new Error(
    `Could not find '${command}' on the backend PATH. Install GitHub Copilot CLI with 'npm install -g @github/copilot', or set Settings copilot.command to the full CLI path.${shimHint}`
  );
}

async function runCodex(command: string, args: string[], input: string, timeoutMs: number): Promise<string> {
  return runCommand(command, args, input, timeoutMs);
}

async function runCommand(command: string, args: string[], input: string, timeoutMs: number): Promise<string> {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd: process.cwd(),
      env: process.env,
      stdio: ["pipe", "pipe", "pipe"]
    });
    let stdout = "";
    let stderr = "";
    const timer = setTimeout(() => {
      child.kill("SIGTERM");
      reject(new Error(`Timed out after ${timeoutMs}ms`));
    }, timeoutMs);

    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk: string) => {
      stdout += chunk;
      if (stdout.length > 20 * 1024 * 1024) {
        child.kill("SIGTERM");
        reject(new Error("Codex output exceeded 20 MiB."));
      }
    });
    child.stderr.on("data", (chunk: string) => {
      stderr += chunk;
    });
    child.on("error", (error) => {
      clearTimeout(timer);
      reject(error);
    });
    child.on("close", (code) => {
      clearTimeout(timer);
      if (code === 0) resolve(stdout);
      else {
        const detail = stderr.trim() || tail(stdout.trim(), 4000) || "no stderr/stdout details";
        reject(new Error(`Codex exited with code ${code}: ${detail}`));
      }
    });
    child.stdin.end(input);
  });
}

function tail(value: string, maxLength: number): string {
  if (value.length <= maxLength) return value;
  return value.slice(value.length - maxLength);
}

function findExecutableOnPath(command: string): string | undefined {
  for (const dir of (process.env.PATH ?? "").split(delimiter).filter(Boolean)) {
    const candidate = join(dir, command);
    if (isExecutable(candidate)) return candidate;
  }
  return undefined;
}

function discoverVsCodeCodexBinary(): string | undefined {
  const extensionRoot = join(homedir(), ".vscode-server", "extensions");
  if (!existsSync(extensionRoot)) return undefined;
  const candidates = readdirSync(extensionRoot, { withFileTypes: true })
    .filter((entry) => entry.isDirectory() && entry.name.startsWith("openai.chatgpt-"))
    .map((entry) => [
      join(extensionRoot, entry.name, "bin", "linux-x86_64", "codex"),
      join(extensionRoot, entry.name, "bin", "linux-x64", "codex")
    ])
    .flat()
    .filter(isExecutable)
    .sort()
    .reverse();
  return candidates[0];
}

function discoverVsCodeCopilotShim(): string | undefined {
  const extensionRoot = join(homedir(), ".vscode-server", "extensions");
  if (!existsSync(extensionRoot)) return undefined;
  const candidates = readdirSync(extensionRoot, { withFileTypes: true })
    .filter((entry) => entry.isDirectory() && entry.name.startsWith("github.copilot-chat-"))
    .map((entry) => join(extensionRoot, entry.name, "dist", "copilotCLIShim.js"))
    .filter((path) => existsSync(path))
    .sort()
    .reverse();
  return candidates[0];
}

function isExecutable(path: string): boolean {
  try {
    const stat = statSync(path);
    return stat.isFile() && (stat.mode & 0o111) !== 0;
  } catch {
    return false;
  }
}

function buildPrompt<T>(request: LlmRequest<T>, outputSchema?: unknown): string {
  return [
    request.system ??
      "You are the reasoning engine for a focused AIOps/SRE agent. Return only data that conforms to the provided output schema.",
    "",
    `Task: ${request.task}`,
    "",
    "Operational rules:",
    "- Avoid brittle word heuristics; reason from structured context and available capabilities.",
    "- Do not invent live-system facts. Use evidence references when making root-cause claims.",
    "- Do not propose mutations unless the action contains exact tool arguments, rollback notes, and verification steps.",
    "- Mark uncertainty explicitly.",
    "- Return exactly one JSON object. Do not wrap it in Markdown.",
    outputSchema ? "- The JSON object must conform to the schema included below." : "",
    "",
    outputSchema ? "Output JSON Schema:" : "",
    outputSchema ? JSON.stringify(outputSchema, null, 2) : "",
    outputSchema ? "" : "",
    "Context:",
    JSON.stringify(request.context ?? {}, null, 2)
  ]
    .filter((line) => line !== "")
    .join("\n");
}

function extractFinalPayload(stdout: string): unknown {
  const trimmed = stdout.trim();
  if (!trimmed) throw new Error("Codex returned empty output.");

  const candidates: unknown[] = [];
  for (const line of trimmed.split(/\r?\n/)) {
    const parsed = tryParseJson(line);
    if (!parsed || typeof parsed !== "object") continue;
    candidates.push(parsed);
    const maybeText = findStringPayload(parsed);
    if (maybeText) {
      const maybeJson = tryParseJson(maybeText);
      if (maybeJson) candidates.push(maybeJson);
    }
  }

  const whole = tryParseJson(trimmed);
  if (whole) candidates.push(whole);

  const fenced = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/i);
  if (fenced) {
    const parsed = tryParseJson(fenced[1]);
    if (parsed) candidates.push(parsed);
  }

  const objectMatch = trimmed.match(/\{[\s\S]*\}$/);
  if (objectMatch) {
    const parsed = tryParseJson(objectMatch[0]);
    if (parsed) candidates.push(parsed);
  }

  for (const candidate of candidates.reverse()) {
    if (candidate && typeof candidate === "object") {
      if ("final_output" in candidate) return (candidate as { final_output: unknown }).final_output;
      if ("output" in candidate && typeof (candidate as { output: unknown }).output === "object") {
        return (candidate as { output: unknown }).output;
      }
      if ("message" in candidate && typeof (candidate as { message: unknown }).message === "object") {
        return (candidate as { message: unknown }).message;
      }
      if (!("type" in candidate)) return candidate;
    }
  }
  throw new Error("Could not extract a schema payload from Codex JSONL output.");
}

function findStringPayload(value: unknown): string | undefined {
  if (!value || typeof value !== "object") return undefined;
  const record = value as Record<string, unknown>;
  for (const key of ["final_output", "last_message", "message", "text", "content", "delta"]) {
    if (typeof record[key] === "string") return record[key] as string;
  }
  for (const nested of Object.values(record)) {
    const found = findStringPayload(nested);
    if (found) return found;
  }
  return undefined;
}

function tryParseJson(value: string): unknown | undefined {
  try {
    return JSON.parse(value);
  } catch {
    return undefined;
  }
}

function buildCopilotArgs(settings: Settings, prompt: string): string[] {
  const configured = settings.copilot.args.length
    ? settings.copilot.args
    : ["-p", "{prompt}", "-s", "--no-ask-user"];
  const args: string[] = [];
  let hasPromptPlaceholder = false;
  let hasModelPlaceholder = false;
  for (const arg of configured) {
    if (arg === "{prompt}") {
      args.push(prompt);
      hasPromptPlaceholder = true;
    } else {
      if (arg.includes("{model}")) hasModelPlaceholder = true;
      args.push(arg.replaceAll("{model}", settings.copilot.model));
    }
  }
  if (!hasPromptPlaceholder) args.push("-p", prompt);
  if (settings.copilot.model && !hasModelPlaceholder && !args.includes("--model")) {
    args.push("--model", settings.copilot.model);
  }
  return args;
}

type JsonSchemaObject = Record<string, unknown>;

function toStrictStructuredOutputSchema(schema: unknown, optional = false): unknown {
  if (!schema || typeof schema !== "object" || Array.isArray(schema)) return schema;
  const input = { ...(schema as JsonSchemaObject) };
  if (typeof input.$ref === "string") {
    const replacement = jsonValueSchema(3);
    return optional ? allowNull(replacement) : replacement;
  }
  delete input.$schema;
  delete input.default;
  delete input.definitions;

  const transformed: JsonSchemaObject = {};
  for (const [key, value] of Object.entries(input)) {
    if (key === "properties" && value && typeof value === "object" && !Array.isArray(value)) {
      const originalRequired = new Set(
        Array.isArray(input.required) ? (input.required as unknown[]).filter((item): item is string => typeof item === "string") : []
      );
      const properties = value as Record<string, unknown>;
      transformed.properties = Object.fromEntries(
        Object.entries(properties).map(([propertyName, propertySchema]) => [
          propertyName,
          toStrictStructuredOutputSchema(propertySchema, !originalRequired.has(propertyName))
        ])
      );
      transformed.required = Object.keys(properties);
      continue;
    }
    if (key === "required") continue;
    if (key === "items") {
      transformed.items = toStrictStructuredOutputSchema(value);
      continue;
    }
    if ((key === "anyOf" || key === "oneOf" || key === "allOf") && Array.isArray(value)) {
      transformed[key] = value.map((item) => toStrictStructuredOutputSchema(item));
      continue;
    }
    if (key === "additionalProperties" && value && typeof value === "object" && !Array.isArray(value)) {
      transformed.additionalProperties = toStrictStructuredOutputSchema(value);
      continue;
    }
    transformed[key] = value;
  }

  if (transformed.properties && !("type" in transformed)) transformed.type = "object";
  if (transformed.type === "object" && !("additionalProperties" in transformed)) {
    transformed.additionalProperties = false;
  }

  return optional ? allowNull(transformed) : transformed;
}

function jsonValueSchema(depth: number): JsonSchemaObject {
  if (depth <= 0) {
    return {
      anyOf: [{ type: "string" }, { type: "number" }, { type: "boolean" }, { type: "null" }]
    };
  }
  const nested = jsonValueSchema(depth - 1);
  return {
    anyOf: [
      { type: "string" },
      { type: "number" },
      { type: "boolean" },
      { type: "null" },
      { type: "array", items: nested },
      { type: "object", additionalProperties: nested }
    ]
  };
}

function allowNull(schema: JsonSchemaObject): JsonSchemaObject {
  if (Array.isArray(schema.enum) && !schema.enum.includes(null)) {
    const type = schema.type;
    const nullableType =
      typeof type === "string" ? [type, "null"] : Array.isArray(type) ? [...new Set([...type, "null"])] : type;
    return { ...schema, ...(nullableType ? { type: nullableType } : {}), enum: [...schema.enum, null] };
  }
  const type = schema.type;
  if (typeof type === "string") return { ...schema, type: [type, "null"] };
  if (Array.isArray(type) && !type.includes("null")) return { ...schema, type: [...type, "null"] };
  return { anyOf: [schema, { type: "null" }] };
}

function stripNullObjectFields(value: unknown): unknown {
  if (Array.isArray(value)) return value.map((item) => stripNullObjectFields(item));
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>)
        .filter(([, nested]) => nested !== null)
        .map(([key, nested]) => [key, stripNullObjectFields(nested)])
    );
  }
  return value;
}

function mockResponse(schemaName: string, context: unknown): unknown {
  if (schemaName === "IntentClassification") {
    return {
      intent: "diagnose",
      confidence: 0.74,
      urgency: "normal",
      needsLiveEvidence: true,
      needsClarification: false,
      rationale: "Mock mode treats operator requests as diagnostic unless a later stage proves otherwise."
    };
  }
  if (schemaName === "EntityExtraction") {
    return { entities: [], unresolvedReferences: [] };
  }
  if (schemaName === "EvidencePlan") {
    return {
      objective: "Collect available live evidence from discovered read-only MCP capabilities.",
      confidence: 0.5,
      steps: [],
      answerWithoutToolsIfUnavailable: true
    };
  }
  if (schemaName === "RcaResponse") {
    return {
      answer:
        "Mock Codex mode is enabled. The cockpit recorded the run and connector state, but no live infrastructure claim was made.",
      summary: "Mock investigation completed without live evidence.",
      rootCauseClaims: [],
      followUpQuestions: [],
      missingEvidence: ["Disable mock mode and configure AWS/EKS MCP servers for live diagnosis."]
    };
  }
  if (schemaName === "RemediationResponse") {
    return {
      actions: [],
      rationale: "No remediation is proposed in mock mode without live evidence.",
      noActionReason: "Mock mode avoids infrastructure mutations."
    };
  }
  if (schemaName === "MemorySummary") {
    return { key: "last-run-summary", value: "Mock run completed.", sensitive: false };
  }
  if (schemaName === "HandoffPackage") {
    return context;
  }
  return {};
}
