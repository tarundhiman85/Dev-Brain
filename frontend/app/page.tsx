"use client";

import { BrainCircuit, ChevronDown, Menu, MessageSquare, Plus, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import ChatWindow, { type Message, type Pending } from "@/components/ChatWindow";
import SettingsPanel from "@/components/SettingsPanel";
import { getSources, streamChat, type Mode, type ToolStep } from "@/services/api";

type Session = { id: string; title: string; messages: Message[] };
type Workspace = { sessions: Session[]; activeId: string; mode: Mode; temperature: number | null };
const initial: Workspace = {
  sessions: [{ id: "first", title: "New conversation", messages: [] }],
  activeId: "first", mode: "auto", temperature: null,
};
const storageKey = "devbrain-workspace-v1";

function withoutObservation({ observation: _observation, ...step }: ToolStep): ToolStep {
  return step;
}

export default function Home() {
  const [workspace, setWorkspace] = useState<Workspace>(initial);
  const [ready, setReady] = useState(false);
  const [storageError, setStorageError] = useState(false);
  const [sidebar, setSidebar] = useState(false);
  const [draft, setDraft] = useState("");
  const [pending, setPending] = useState<Pending>();
  const [error, setError] = useState<string>();
  const controller = useRef<AbortController | null>(null);
  const session = workspace.sessions.find(item => item.id === workspace.activeId) ?? workspace.sessions[0];

  useEffect(() => {
    try {
      const saved = JSON.parse(localStorage.getItem(storageKey) ?? "null") as Workspace | null;
      if (saved?.sessions?.length) setWorkspace(saved);
    } catch { setStorageError(true); }
    setReady(true);
  }, []);

  useEffect(() => {
    if (!ready) return;
    try { localStorage.setItem(storageKey, JSON.stringify(workspace)); }
    catch { setStorageError(true); }
  }, [workspace, ready]);

  function select(id: string) {
    if (controller.current) return;
    setWorkspace(current => ({ ...current, activeId: id }));
    setDraft("");
    setError(undefined);
    setSidebar(false);
  }

  function newChat() {
    if (controller.current) return;
    const id = crypto.randomUUID();
    setWorkspace(current => ({
      ...current, activeId: id,
      sessions: [{ id, title: "New conversation", messages: [] }, ...current.sessions].slice(0, 30),
    }));
    setDraft("");
    setError(undefined);
    setSidebar(false);
  }

  async function send(prompt: string) {
    if (controller.current) return;
    const request = new AbortController();
    controller.current = request;
    setDraft("");
    setError(undefined);
    setPending({ prompt, steps: [] });
    try {
      const reply = await streamChat({
        prompt, mode: workspace.mode, temperature: workspace.temperature,
        history: session.messages.slice(-8).map(message => ({
          prompt: message.prompt, answer: message.reply.answer.slice(0, 16000),
        })),
      }, request.signal, event => setPending(current => {
        if (!current) return current;
        if (event.type === "tool_start") return { ...current, active: event.data.tool };
        return { ...current, active: undefined, steps: [...current.steps, withoutObservation(event.data)] };
      }));
      if (request.signal.aborted) return;
      const message: Message = {
        prompt, sources: getSources(reply.trace), reply: { ...reply, trace: reply.trace.map(withoutObservation) },
      };
      setWorkspace(current => ({
        ...current, sessions: current.sessions.map(item => item.id === session.id ? {
          ...item, title: item.messages.length ? item.title : prompt.replace(/\s+/g, " "),
          messages: [...item.messages, message].slice(-50),
        } : item),
      }));
    } catch (cause) {
      if (!request.signal.aborted) setError(cause instanceof Error ? cause.message : "Chat request failed.");
      setDraft(current => current || prompt);
    } finally {
      controller.current = null;
      setPending(undefined);
    }
  }

  return <main className="workspace">
    {sidebar && <button className="sidebar-backdrop" aria-label="Close sidebar" onClick={() => setSidebar(false)} />}
    <aside id="workspace-sidebar" className={`sidebar ${sidebar ? "sidebar-open" : ""}`}>
      <div className="brand">
        <span className="brand-icon"><BrainCircuit size={22} /></span>
        <span><strong>DevBrain</strong><small>Personal AI workspace</small></span>
        <button className="mobile-close icon-button" aria-label="Close sidebar" onClick={() => setSidebar(false)}><X size={19} /></button>
      </div>
      <button className="new-chat" onClick={newChat} disabled={!!pending}><Plus size={18} /> New chat</button>
      <nav aria-label="Chat history" className="history">
        <div className="section-label history-heading">CONVERSATIONS <ChevronDown size={14} /></div>
        {workspace.sessions.map(item => <button key={item.id} type="button" className={item.id === session.id ? "current" : ""}
          aria-current={item.id === session.id ? "true" : undefined} title={item.title}
          disabled={!!pending} onClick={() => select(item.id)}>
          <MessageSquare size={15} /><span>{item.title}</span>
        </button>)}
      </nav>
      <SettingsPanel mode={workspace.mode} temperature={workspace.temperature} disabled={!!pending}
        setMode={mode => setWorkspace(current => ({ ...current, mode }))}
        setTemperature={temperature => setWorkspace(current => ({ ...current, temperature }))} />
      <div className="sidebar-footer"><span className="status-dot" /> {storageError ? "Chat history could not be saved" : "History stays in this browser"}</div>
    </aside>
    <section className="main-panel">
      <header className="topbar">
        <button className="mobile-menu icon-button" aria-controls="workspace-sidebar" aria-expanded={sidebar}
          aria-label="Open sidebar" onClick={() => setSidebar(true)}><Menu size={21} /></button>
        <div className="topbar-title"><MessageSquare size={16} /><span>{session.title}</span></div>
        <span className="routing-badge"><span className="status-dot" />{
          { auto: "Auto routing", local: "Local only", cloud: "Cloud only" }[workspace.mode]
        }</span>
      </header>
      <ChatWindow key={session.id} messages={session.messages} pending={pending} error={error}
        draft={draft} setDraft={setDraft} onSend={send} onStop={() => controller.current?.abort()} />
    </section>
  </main>;
}
