import { AppDatabase } from "../src/server/db.js";
import { LlmRouter } from "../src/agent/codexProvider.js";
import { intentContext } from "../src/agent/prompts.js";
import { IntentClassificationSchema } from "../src/shared/schemas.js";

const message = process.argv.slice(2).join(" ").trim() || "hi";
const db = new AppDatabase();
const llm = new LlmRouter(() => db.getSettings());

try {
  const settings = db.getSettings();
  console.error(
    `LLM smoke using provider=${settings.llm.provider}, fallback=${settings.llm.fallbackProvider}, codex=${settings.codex.command}, copilot=${settings.copilot.command}`
  );
  const result = await llm.generate({
    schemaName: "IntentClassification",
    schema: IntentClassificationSchema,
    task: "Classify this smoke-test operator request for an infrastructure AIOps agent.",
    context: intentContext(message)
  });
  console.log(JSON.stringify(result, null, 2));
} finally {
  db.close();
}
