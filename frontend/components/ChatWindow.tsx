"use client";

import { ArrowUp, ArrowUpRight, BookOpen, BrainCircuit, Check, Code2, Copy, GitPullRequest, Globe2, LoaderCircle, Square } from "lucide-react";
import { useEffect, useRef, useState, type FormEvent } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneLight } from "react-syntax-highlighter/dist/esm/styles/prism";
import type { Reply, Source, ToolStep } from "@/services/api";
import ToolTraceAccordion, { SourceCitations } from "./ToolTraceAccordion";

export type Message = { prompt: string; reply: Reply; sources: Source[] };
export type Pending = { prompt: string; steps: ToolStep[]; active?: string };

const starters = [
  { title: "Search my notes", text: "Find relevant context in your knowledge base.", prompt: "Search my notes for how we handle GitHub rate limits.", icon: BookOpen },
  { title: "Review recent PRs", text: "See what changed in a repository.", prompt: "List recent PRs in ", icon: GitPullRequest },
  { title: "Explain a concept", text: "Get a quick answer or code example.", prompt: "Explain Python tuples with a short example.", icon: Code2 },
  { title: "Research the web", text: "Look up current technical information.", prompt: "Search the web for the current pgvector HNSW index options.", icon: Globe2 },
];

function CopyButton({ value, label = "Copy" }: { value: string; label?: string }) {
  const [status, setStatus] = useState<"idle" | "copied" | "failed">("idle");
  return <button className="copy-button" type="button" onClick={async () => {
    try { await navigator.clipboard.writeText(value); setStatus("copied"); }
    catch { setStatus("failed"); }
  }}>{status === "copied" ? <Check size={14} /> : <Copy size={14} />}
    {status === "copied" ? "Copied" : status === "failed" ? "Copy failed" : label}</button>;
}

function Answer({ reply, sources }: { reply: Reply; sources: Source[] }) {
  return <article className="assistant-message">
    <div className="assistant-heading"><span className="assistant-icon"><BrainCircuit size={17} /></span>
      <strong>DevBrain</strong><span className="reply-meta">{reply.backend === "local" ? "Local" : "Cloud"} · {reply.model}
        {reply.fallback ? " · Fallback" : ""} · {(reply.latency_ms / 1000).toFixed(1)}s</span></div>
    <div className="markdown"><Markdown remarkPlugins={[remarkGfm]} components={{
      a: props => <a {...props} target="_blank" rel="noreferrer" />,
      img: ({ alt }) => <span>{alt}</span>,
      pre: ({ children }) => <>{children}</>,
      code: ({ className, children, node }) => {
        const language = /language-([\w-]+)/.exec(className ?? "")?.[1];
        const text = String(children);
        if (language || text.endsWith("\n") || (node?.position?.start.line !== node?.position?.end.line)) {
          const code = text.replace(/\n$/, "");
          return <div className="code-block"><div className="code-heading"><span>{language ?? "text"}</span>
            <CopyButton value={code} /></div><SyntaxHighlighter language={language ?? "text"} style={oneLight}
              customStyle={{ margin: 0, background: "transparent", fontSize: "0.85rem" }}>{code}</SyntaxHighlighter></div>;
        }
        return <code className={className}>{children}</code>;
      },
    }}>{reply.answer}</Markdown></div>
    <ToolTraceAccordion steps={reply.trace} />
    <SourceCitations sources={sources} />
    <CopyButton value={reply.answer} label="Copy answer" />
  </article>;
}

export default function ChatWindow({ messages, pending, error, draft, setDraft, onSend, onStop }: {
  messages: Message[]; pending?: Pending; error?: string; draft: string; setDraft: (value: string) => void;
  onSend: (prompt: string) => void; onStop: () => void;
}) {
  const feed = useRef<HTMLDivElement>(null);
  const input = useRef<HTMLTextAreaElement>(null);
  useEffect(() => {
    if (feed.current) feed.current.scrollTop = feed.current.scrollHeight;
  }, [messages, pending, error]);

  function submit(event: FormEvent) {
    event.preventDefault();
    if (draft.trim() && !pending) onSend(draft.trim());
  }

  return <div className="chat-window">
    <div className="feed" ref={feed}>
      {!messages.length && !pending && !error ? <section className="welcome">
        <div className="welcome-icon"><BrainCircuit size={33} strokeWidth={1.6} /></div>
        <p className="eyebrow">DEVELOPER WORKSPACE</p>
        <h1>What are we working on?</h1>
        <p className="welcome-description">Ask about your notes, inspect pull requests, or work through a technical question.</p>
        <div className="starter-grid">
          {starters.map(({ title, text, prompt, icon: Icon }) => <button type="button" className="starter" key={title}
            onClick={() => { setDraft(prompt); input.current?.focus(); }}>
            <span className="starter-icon"><Icon size={17} /></span>
            <span><strong>{title}</strong><small>{text}</small></span>
            <ArrowUpRight size={15} className="starter-arrow" />
          </button>)}
        </div>
      </section> : <div className="messages" aria-live="polite">
        {messages.map((message, index) => <div className="turn" key={index}>
          <div className="user-message">{message.prompt}</div>
          <Answer reply={message.reply} sources={message.sources} />
        </div>)}
        {pending && <div className="turn">
          <div className="user-message">{pending.prompt}</div>
          <article className="assistant-message">
            <div className="assistant-heading"><span className="assistant-icon"><BrainCircuit size={17} /></span>
              <strong>DevBrain</strong></div>
            <div className="pending-status"><LoaderCircle className="animate-spin" size={15} />
              {pending.active ? `Running ${pending.active}` : "Preparing an answer…"}</div>
            <ToolTraceAccordion steps={pending.steps} active={pending.active} />
          </article>
        </div>}
        {error && <p role="alert" className="error-message">{error}</p>}
      </div>}
    </div>
    <div className="composer-wrap">
      <form className="composer" onSubmit={submit}>
        <textarea ref={input} aria-label="Message DevBrain" placeholder="Message DevBrain…" rows={2}
          value={draft} maxLength={8000} onChange={event => setDraft(event.target.value)}
          onKeyDown={event => {
            if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
              event.preventDefault();
              event.currentTarget.form?.requestSubmit();
            }
          }} />
        {pending ? <button type="button" className="send" aria-label="Stop request" onClick={onStop}><Square size={16} /></button>
          : <button type="submit" className="send" aria-label="Send message" disabled={!draft.trim()}><ArrowUp size={21} /></button>}
      </form>
      <p className="composer-hint">Enter to send <span>·</span> Shift + Enter for a new line</p>
    </div>
  </div>;
}
