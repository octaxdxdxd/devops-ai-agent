import { useEffect, useMemo, useState, type ReactNode } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  ChevronRight,
  ClipboardList,
  Cloud,
  Code2,
  FileText,
  Gauge,
  History,
  Loader2,
  MessageSquare,
  Play,
  RefreshCcw,
  Save,
  Send,
  Server,
  Settings as SettingsIcon,
  ShieldCheck,
  XCircle
} from "lucide-react";
import { api } from "./api.js";
import type { Message, Run, RunBundle, Session } from "../shared/contracts.js";
import type { HandoffPackage, ProposedAction, RunTraceEvent, Settings } from "../shared/schemas.js";

type Tab = "trace" | "evidence" | "actions" | "verification" | "handoff" | "settings";

export function App() {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [activeSession, setActiveSession] = useState<Session | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [runs, setRuns] = useState<Run[]>([]);
  const [activeRunId, setActiveRunId] = useState<string | null>(null);
  const [bundle, setBundle] = useState<RunBundle | null>(null);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("trace");
  const [settings, setSettings] = useState<Settings | null>(null);
  const [draftSettings, setDraftSettings] = useState("");

  useEffect(() => {
    void bootstrap();
  }, []);

  useEffect(() => {
    if (!activeRunId) return;
    let disposed = false;
    const refresh = async () => {
      try {
        const next = await api.getRun(activeRunId);
        if (!disposed) setBundle(next);
      } catch (err) {
        if (!disposed) setError(err instanceof Error ? err.message : String(err));
      }
    };
    void refresh();
    const source = new EventSource(`/api/runs/${activeRunId}/stream`);
    source.addEventListener("trace", () => void refresh());
    source.onerror = () => void refresh();
    return () => {
      disposed = true;
      source.close();
    };
  }, [activeRunId]);

  const latestRun = bundle?.run ?? runs[runs.length - 1] ?? null;
  const runMessages = useMemo(() => {
    if (!bundle) return messages;
    return bundle.messages;
  }, [bundle, messages]);

  async function bootstrap() {
    try {
      const [sessionList, loadedSettings] = await Promise.all([api.listSessions(), api.getSettings()]);
      setSessions(sessionList);
      setSettings(loadedSettings);
      setDraftSettings(JSON.stringify(loadedSettings, null, 2));
      if (sessionList[0]) await openSession(sessionList[0].id);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function openSession(id: string) {
    const details = await api.getSession(id);
    setActiveSession(details.session);
    setMessages(details.messages);
    setRuns(details.runs);
    const lastRun = details.runs.at(-1);
    setActiveRunId(lastRun?.id ?? null);
    setBundle(lastRun ? await api.getRun(lastRun.id) : null);
  }

  async function submitRun() {
    const message = input.trim();
    if (!message || busy) return;
    setBusy(true);
    setError(null);
    setInput("");
    try {
      const created = await api.createRun(message, activeSession?.id);
      const sessionList = await api.listSessions();
      setSessions(sessionList);
      await openSession(created.sessionId);
      setActiveRunId(created.runId);
      setTab("trace");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  async function newSession() {
    const session = await api.createSession("New investigation");
    setSessions([session, ...sessions]);
    setActiveSession(session);
    setMessages([]);
    setRuns([]);
    setActiveRunId(null);
    setBundle(null);
  }

  async function mutateAction(id: string, operation: "approve" | "reject" | "execute") {
    setError(null);
    try {
      if (operation === "approve") await api.approveAction(id);
      if (operation === "reject") await api.rejectAction(id);
      if (operation === "execute") await api.executeAction(id);
      if (activeRunId) setBundle(await api.getRun(activeRunId));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function saveSettings() {
    try {
      const parsed = JSON.parse(draftSettings) as Settings;
      const saved = await api.saveSettings(parsed);
      setSettings(saved);
      setDraftSettings(JSON.stringify(saved, null, 2));
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function createHandoff(type: HandoffPackage["type"]) {
    if (!activeRunId) return;
    try {
      await api.createHandoff(activeRunId, type);
      setBundle(await api.getRun(activeRunId));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <ShieldCheck size={22} />
          <div>
            <strong>AIOps Cockpit</strong>
            <span>Local control plane</span>
          </div>
        </div>
        <button className="primary full" onClick={newSession}>
          <MessageSquare size={16} /> New session
        </button>
        <div className="session-list">
          {sessions.map((session) => (
            <button
              key={session.id}
              className={`session-item ${activeSession?.id === session.id ? "active" : ""}`}
              onClick={() => void openSession(session.id)}
            >
              <span>{session.title}</span>
              <small>{new Date(session.updatedAt).toLocaleString()}</small>
            </button>
          ))}
        </div>
      </aside>

      <main className="workspace">
        <header className="topbar">
          <div>
            <h1>{activeSession?.title ?? "AIOps investigation"}</h1>
            <p>{latestRun ? `${latestRun.status} · ${latestRun.id.slice(0, 8)}` : "No run selected"}</p>
          </div>
          <div className="status-strip">
            <StatusPill label="Codex" value={settings?.codex.model ?? "loading"} icon={<Code2 size={14} />} />
            <StatusPill label="MCPs" value={`${settings?.mcpServers.filter((item) => item.enabled).length ?? 0} enabled`} icon={<Server size={14} />} />
            <StatusPill label="Policy" value="approval-gated" icon={<ShieldCheck size={14} />} />
          </div>
        </header>

        {error && (
          <div className="error-banner">
            <AlertTriangle size={16} /> {error}
          </div>
        )}

        <section className="content-grid">
          <section className="chat-pane">
            <div className="messages">
              {runMessages.length === 0 && (
                <div className="empty-state">
                  <Gauge size={34} />
                  <h2>Ask for an investigation, explanation, or proposed remediation.</h2>
                  <p>Runs collect traceable evidence first and place any mutation behind approval.</p>
                </div>
              )}
              {runMessages.map((message) => (
                <article key={message.id} className={`message ${message.role}`}>
                  <div className="message-role">{message.role}</div>
                  <pre>{message.content}</pre>
                </article>
              ))}
            </div>
            <form
              className="composer"
              onSubmit={(event) => {
                event.preventDefault();
                void submitRun();
              }}
            >
              <textarea
                value={input}
                onChange={(event) => setInput(event.target.value)}
                placeholder="Investigate why checkout pods are restarting in prod..."
              />
              <button className="primary" disabled={busy || !input.trim()}>
                {busy ? <Loader2 className="spin" size={16} /> : <Send size={16} />} Run
              </button>
            </form>
          </section>

          <aside className="ops-pane">
            <nav className="tabs">
              <TabButton active={tab === "trace"} onClick={() => setTab("trace")} icon={<History size={15} />} label="Trace" />
              <TabButton active={tab === "evidence"} onClick={() => setTab("evidence")} icon={<ClipboardList size={15} />} label="Evidence" />
              <TabButton active={tab === "actions"} onClick={() => setTab("actions")} icon={<ShieldCheck size={15} />} label="Actions" />
              <TabButton active={tab === "verification"} onClick={() => setTab("verification")} icon={<CheckCircle2 size={15} />} label="Verify" />
              <TabButton active={tab === "handoff"} onClick={() => setTab("handoff")} icon={<FileText size={15} />} label="Handoff" />
              <TabButton active={tab === "settings"} onClick={() => setTab("settings")} icon={<SettingsIcon size={15} />} label="Settings" />
            </nav>
            <div className="panel-body">
              {tab === "trace" && <TracePanel events={bundle?.traceEvents ?? []} />}
              {tab === "evidence" && <EvidencePanel bundle={bundle} />}
              {tab === "actions" && <ActionsPanel actions={bundle?.proposedActions ?? []} onAction={mutateAction} />}
              {tab === "verification" && <VerificationPanel bundle={bundle} />}
              {tab === "handoff" && <HandoffPanel bundle={bundle} onCreate={createHandoff} />}
              {tab === "settings" && (
                <SettingsPanel draft={draftSettings} onDraft={setDraftSettings} onSave={saveSettings} />
              )}
            </div>
          </aside>
        </section>
      </main>
    </div>
  );
}

function StatusPill({ label, value, icon }: { label: string; value: string; icon: ReactNode }) {
  return (
    <div className="status-pill">
      {icon}
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function TabButton({
  active,
  onClick,
  icon,
  label
}: {
  active: boolean;
  onClick: () => void;
  icon: ReactNode;
  label: string;
}) {
  return (
    <button className={active ? "active" : ""} onClick={onClick}>
      {icon}
      <span>{label}</span>
    </button>
  );
}

function TracePanel({ events }: { events: RunTraceEvent[] }) {
  return (
    <div className="stack">
      {events.length === 0 && <PanelEmpty text="No trace events yet." />}
      {events.map((event) => (
        <article key={event.id} className={`trace-event ${event.severity}`}>
          <div>
            <TraceIcon event={event} />
            <strong>{event.type}</strong>
            <time>{new Date(event.createdAt).toLocaleTimeString()}</time>
          </div>
          <p>{event.message}</p>
          {event.data != null && <JsonBlock value={event.data} />}
        </article>
      ))}
    </div>
  );
}

function TraceIcon({ event }: { event: RunTraceEvent }) {
  if (event.severity === "error") return <XCircle size={15} />;
  if (event.severity === "warning") return <AlertTriangle size={15} />;
  return <CheckCircle2 size={15} />;
}

function EvidencePanel({ bundle }: { bundle: RunBundle | null }) {
  return (
    <div className="stack">
      {(bundle?.evidence.length ?? 0) === 0 && <PanelEmpty text="Evidence will appear after live tool calls." />}
      {bundle?.evidence.map((item) => (
        <article key={item.id} className="artifact">
          <header>
            <strong>{item.summary}</strong>
            <span>{item.sensitive ? "sensitive" : item.source}</span>
          </header>
          <small>{item.id}</small>
          <JsonBlock value={item.data} />
        </article>
      ))}
      {(bundle?.toolCalls.length ?? 0) > 0 && (
        <>
          <h3>Tool calls</h3>
          {bundle?.toolCalls.map((call) => (
            <article key={call.id} className={`tool-call ${call.status}`}>
              <strong>{call.serverId}/{call.toolName}</strong>
              <span>{call.status} · {call.safetyClass}</span>
              {call.error && <p>{call.error}</p>}
            </article>
          ))}
        </>
      )}
    </div>
  );
}

function ActionsPanel({
  actions,
  onAction
}: {
  actions: ProposedAction[];
  onAction: (id: string, operation: "approve" | "reject" | "execute") => Promise<void>;
}) {
  return (
    <div className="stack">
      {actions.length === 0 && <PanelEmpty text="No proposed actions for this run." />}
      {actions.map((action) => (
        <article key={action.id} className={`action-card ${action.status}`}>
          <header>
            <div>
              <strong>{action.title}</strong>
              <span>{action.riskLevel} risk · {action.safetyClass}</span>
            </div>
            <Badge value={action.status} />
          </header>
          <p>{action.description}</p>
          <div className="call-preview">
            <Cloud size={14} />
            <code>{action.serverId}/{action.toolName}</code>
          </div>
          <JsonBlock value={action.arguments} />
          <details>
            <summary><ChevronRight size={14} /> Rollback and verification</summary>
            <p>{action.rollback}</p>
            <JsonBlock value={action.verificationPlan} />
          </details>
          <div className="button-row">
            {action.status === "proposed" && (
              <>
                <button className="secondary danger" onClick={() => void onAction(action.id ?? "", "reject")}>
                  <XCircle size={15} /> Reject
                </button>
                <button className="primary" onClick={() => void onAction(action.id ?? "", "approve")}>
                  <ShieldCheck size={15} /> Approve
                </button>
              </>
            )}
            {action.status === "approved" && (
              <button className="primary execute" onClick={() => void onAction(action.id ?? "", "execute")}>
                <Play size={15} /> Execute
              </button>
            )}
          </div>
        </article>
      ))}
    </div>
  );
}

function VerificationPanel({ bundle }: { bundle: RunBundle | null }) {
  return (
    <div className="stack">
      {(bundle?.verificationResults.length ?? 0) === 0 && <PanelEmpty text="Verification results appear after approved execution." />}
      {bundle?.verificationResults.map((result) => (
        <article key={result.id} className={`artifact ${result.status}`}>
          <header>
            <strong>{result.status}</strong>
            <span>{new Date(result.createdAt).toLocaleString()}</span>
          </header>
          <p>{result.summary}</p>
          <JsonBlock value={result.data ?? {}} />
        </article>
      ))}
    </div>
  );
}

function HandoffPanel({
  bundle,
  onCreate
}: {
  bundle: RunBundle | null;
  onCreate: (type: HandoffPackage["type"]) => Promise<void>;
}) {
  return (
    <div className="stack">
      <div className="button-row">
        <button className="secondary" disabled={!bundle} onClick={() => void onCreate("markdown")}>
          <FileText size={15} /> Markdown
        </button>
        <button className="secondary" disabled={!bundle} onClick={() => void onCreate("json")}>
          <Code2 size={15} /> JSON
        </button>
      </div>
      {(bundle?.handoffs.length ?? 0) === 0 && <PanelEmpty text="No handoff packages yet." />}
      {bundle?.handoffs.map((handoff) => (
        <article key={handoff.id} className="artifact">
          <header>
            <strong>{handoff.title}</strong>
            <span>{handoff.type}</span>
          </header>
          <pre className="handoff-text">{handoff.content}</pre>
        </article>
      ))}
    </div>
  );
}

function SettingsPanel({
  draft,
  onDraft,
  onSave
}: {
  draft: string;
  onDraft: (value: string) => void;
  onSave: () => Promise<void>;
}) {
  return (
    <div className="settings-editor">
      <div className="button-row">
        <button className="secondary" onClick={() => void onSave()}>
          <Save size={15} /> Save settings
        </button>
        <button className="secondary" onClick={() => window.location.reload()}>
          <RefreshCcw size={15} /> Reload
        </button>
      </div>
      <textarea value={draft} onChange={(event) => onDraft(event.target.value)} spellCheck={false} />
    </div>
  );
}

function JsonBlock({ value }: { value: unknown }) {
  return <pre className="json-block">{JSON.stringify(value, null, 2)}</pre>;
}

function Badge({ value }: { value: string }) {
  return <span className={`badge ${value}`}>{value}</span>;
}

function PanelEmpty({ text }: { text: string }) {
  return <div className="panel-empty">{text}</div>;
}
