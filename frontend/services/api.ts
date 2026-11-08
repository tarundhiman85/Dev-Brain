export type Mode = "auto" | "local" | "cloud";
export type Turn = { prompt: string; answer: string };
export type ToolStep = {
  tool: string;
  status: "ok" | "error";
  arguments?: Record<string, unknown> | null;
  error?: string | null;
  truncated?: boolean;
  latency_ms?: number | null;
  observation?: string;
};
export type Reply = {
  answer: string;
  trace: ToolStep[];
  backend: "local" | "cloud";
  model: string;
  fallback: boolean;
  latency_ms: number;
};
export type Source = {
  id: string;
  title: string;
  label: string;
  url?: string;
  path?: string;
  text?: string;
  distance?: number;
};

type ChatEvent =
  | { type: "tool_start"; data: { tool: string } }
  | { type: "tool_end"; data: ToolStep }
  | { type: "result"; data: Reply }
  | { type: "error"; data: { message: string } };

export async function streamChat(
  request: { prompt: string; mode: Mode; history: Turn[]; temperature: number | null },
  signal: AbortSignal,
  onProgress: (event: Extract<ChatEvent, { type: "tool_start" | "tool_end" }>) => void,
): Promise<Reply> {
  const base = (process.env.NEXT_PUBLIC_DEVBRAIN_API_URL ?? "http://127.0.0.1:8001").replace(/\/$/, "");
  let response: Response;
  try {
    response = await fetch(`${base}/api/v1/agent/chat/stream`, {
      method: "POST", headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify({ ...request, include_evidence: true }), signal,
    });
  } catch (error) {
    if (signal.aborted) throw error;
    throw new Error("Could not reach DevBrain. Check that the API is running and allows this origin.");
  }
  if (!response.ok || !response.body) {
    const payload = await response.json().catch(() => null);
    throw new Error(typeof payload?.detail === "string" ? payload.detail : `Chat request failed (${response.status}).`);
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let boundary: number;
      while ((boundary = buffer.indexOf("\n\n")) >= 0) {
        const frame = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        const lines = frame.split("\n");
        const type = lines.find(line => line.startsWith("event: "))?.slice(7);
        const data = lines.filter(line => line.startsWith("data: ")).map(line => line.slice(6)).join("\n");
        if (!type || !data) continue;
        const event = { type, data: JSON.parse(data) } as ChatEvent;
        if (event.type === "result") return event.data;
        if (event.type === "error") throw new Error(event.data.message);
        onProgress(event);
      }
    }
    throw new Error("The chat stream ended before the answer arrived.");
  } finally {
    await reader.cancel().catch(() => undefined);
    reader.releaseLock();
  }
}

function safeUrl(value?: string) {
  if (!value) return undefined;
  try {
    const url = new URL(value);
    if (["https:", "http:"].includes(url.protocol)) return url.href;
  } catch { /* Invalid URLs remain plain text. */ }
}

type SourceRecord = { title?: string; url?: string; html_url?: string; body?: string; content?: string };
type Evidence = SourceRecord & {
  matches?: { id: string; content: string; distance?: number; metadata?: SourceRecord & { path?: string; source_type?: string } }[];
  pull_requests?: SourceRecord[];
  results?: SourceRecord[];
};

export function getSources(trace: ToolStep[]): Source[] {
  const sources = new Map<string, Source>();
  for (const step of trace) {
    if (!step.observation) continue;
    let data: Evidence;
    try { data = JSON.parse(step.observation); } catch { continue; }
    for (const match of data.matches ?? []) {
      const meta = match.metadata;
      sources.set(match.id, {
        id: match.id, title: meta?.title ?? "Knowledge base chunk",
        label: meta?.source_type === "github_pr" ? "Archived PR" : "Knowledge base",
        url: safeUrl(meta?.url), path: meta?.path, text: match.content?.slice(0, 1600), distance: match.distance,
      });
    }
    const records = data.pull_requests ?? data.results ?? (data.html_url ? [data] : []);
    for (const record of records) {
      const url = safeUrl(record.html_url ?? record.url);
      if (!url) continue;
      const prior = sources.get(url);
      sources.set(url, {
        id: url, url, title: record.title ?? prior?.title ?? url,
        label: step.tool === "web_search" ? "Web" : "GitHub PR",
        text: (record.body ?? record.content ?? prior?.text)?.slice(0, 1600),
      });
    }
  }
  return [...sources.values()];
}
