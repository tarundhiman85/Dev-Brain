import { ArrowUpRight, BookOpen, ChevronDown, CircleAlert, CircleCheck, FileText, GitPullRequest, LoaderCircle, Workflow } from "lucide-react";
import type { Source, ToolStep } from "@/services/api";

export default function ToolTraceAccordion({ steps, active }: { steps: ToolStep[]; active?: string }) {
  if (!steps.length && !active) return null;
  return <details className="inspection">
    <summary><Workflow size={16} /><span>Tool activity</span>
      <span className="count">{steps.length + (active ? 1 : 0)}</span><ChevronDown className="chevron" size={15} /></summary>
    <div className="inspection-body">
      {steps.map((step, index) => <details className="tool-step" key={`${step.tool}-${index}`}>
        <summary>{step.status === "ok" ? <CircleCheck size={15} className="good" /> : <CircleAlert size={15} className="bad" />}
          <code>{step.tool}</code><span className="muted tool-time">{step.latency_ms == null ? "" : `${Math.round(step.latency_ms)} ms`}</span>
          <ChevronDown className="chevron" size={14} /></summary>
        {step.error && <p className="bad">{step.error}</p>}
        {step.arguments && <pre>{JSON.stringify(step.arguments, null, 2)}</pre>}
        {step.truncated && <p className="muted">Tool output was truncated.</p>}
      </details>)}
      {active && <div className="running"><LoaderCircle className="animate-spin" size={15} /><code>{active}</code><span>Running</span></div>}
    </div>
  </details>;
}

export function SourceCitations({ sources }: { sources: Source[] }) {
  if (!sources.length) return null;
  return <details className="inspection">
    <summary><BookOpen size={16} /><span>Retrieved sources</span><span className="count">{sources.length}</span>
      <ChevronDown className="chevron" size={15} /></summary>
    <div className="inspection-body">
      {sources.map(source => <details key={source.id} className="source-item">
        <summary>{source.label.includes("PR") ? <GitPullRequest size={16} /> : <FileText size={16} />}
          <span className="source-title">{source.title}</span><ChevronDown className="chevron" size={14} /></summary>
        <div className="source-info">
          <span className="count">{source.label}</span>
          {typeof source.distance === "number" && <span className="count">Cosine similarity {(1 - source.distance).toFixed(3)}</span>}
          {source.url && <a href={source.url} target="_blank" rel="noreferrer">Open source <ArrowUpRight size={13} /></a>}
        </div>
        {source.path && <p className="source-path muted">{source.path}</p>}
        {source.text && <p className="source-text">{source.text}</p>}
      </details>)}
    </div>
  </details>;
}
