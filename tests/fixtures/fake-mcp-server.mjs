import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListResourcesRequestSchema,
  ListResourceTemplatesRequestSchema,
  ListPromptsRequestSchema,
  ListToolsRequestSchema
} from "@modelcontextprotocol/sdk/types.js";

const server = new Server(
  { name: "fake-aiops-mcp", version: "0.1.0" },
  { capabilities: { tools: {}, resources: {}, prompts: {} } }
);

server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [
    {
      name: "get_pods",
      title: "Get Pods",
      description: "List pods in a namespace.",
      inputSchema: {
        type: "object",
        properties: { namespace: { type: "string" } },
        required: ["namespace"]
      },
      annotations: { readOnlyHint: true }
    },
    {
      name: "restart_deployment",
      title: "Restart Deployment",
      description: "Restart a deployment.",
      inputSchema: {
        type: "object",
        properties: { namespace: { type: "string" }, deployment: { type: "string" } },
        required: ["namespace", "deployment"]
      },
      annotations: { destructiveHint: false }
    }
  ]
}));

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  if (request.params.name === "get_pods") {
    return {
      content: [{ type: "text", text: "checkout-abc Ready\ncheckout-def CrashLoopBackOff" }],
      structuredContent: {
        pods: [
          { name: "checkout-abc", phase: "Running", ready: true },
          { name: "checkout-def", phase: "CrashLoopBackOff", ready: false }
        ]
      }
    };
  }
  if (request.params.name === "restart_deployment") {
    return {
      content: [{ type: "text", text: "deployment restarted" }],
      structuredContent: { restarted: true }
    };
  }
  throw new Error(`Unknown tool: ${request.params.name}`);
});

server.setRequestHandler(ListResourcesRequestSchema, async () => ({ resources: [] }));
server.setRequestHandler(ListResourceTemplatesRequestSchema, async () => ({ resourceTemplates: [] }));
server.setRequestHandler(ListPromptsRequestSchema, async () => ({ prompts: [] }));

await server.connect(new StdioServerTransport());
