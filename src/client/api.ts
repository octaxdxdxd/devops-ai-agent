import type { RunBundle, Session } from "../shared/contracts.js";
import type { HandoffPackage, Settings } from "../shared/schemas.js";

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(options?.headers ?? {}) },
    ...options
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.error ? JSON.stringify(body.error) : response.statusText);
  }
  return response.json() as Promise<T>;
}

export const api = {
  listSessions: () => request<Session[]>("/api/sessions"),
  createSession: (title?: string) =>
    request<Session>("/api/sessions", { method: "POST", body: JSON.stringify({ title }) }),
  getSession: (id: string) =>
    request<{ session: Session; messages: RunBundle["messages"]; runs: RunBundle["run"][] }>(
      `/api/sessions/${id}`
    ),
  createRun: (message: string, sessionId?: string) =>
    request<{ runId: string; sessionId: string }>("/api/runs", {
      method: "POST",
      body: JSON.stringify({ message, sessionId })
    }),
  getRun: (id: string) => request<RunBundle>(`/api/runs/${id}`),
  approveAction: (id: string) =>
    request(`/api/proposed-actions/${id}/approve`, {
      method: "POST",
      body: JSON.stringify({ operator: "local-operator" })
    }),
  rejectAction: (id: string) =>
    request(`/api/proposed-actions/${id}/reject`, {
      method: "POST",
      body: JSON.stringify({ operator: "local-operator" })
    }),
  executeAction: (id: string) =>
    request(`/api/proposed-actions/${id}/execute`, {
      method: "POST",
      body: JSON.stringify({})
    }),
  getSettings: () => request<Settings>("/api/settings"),
  saveSettings: (settings: Settings) =>
    request<Settings>("/api/settings", { method: "PUT", body: JSON.stringify(settings) }),
  discoverMcp: () => request("/api/mcp/discover"),
  createHandoff: (runId: string, type: HandoffPackage["type"] = "markdown") =>
    request<HandoffPackage>("/api/handoffs", { method: "POST", body: JSON.stringify({ runId, type }) })
};
