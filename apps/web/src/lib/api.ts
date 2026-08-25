/**
 * Typed client over the FastAPI surface.
 *
 * Every request and response type is imported from `schema.d.ts`, which is generated
 * from the running API's OpenAPI document by `npm run gen:api`. Nothing here restates a
 * shape by hand, so the client cannot drift from the server: a field renamed in Python
 * fails this file at compile time rather than at runtime in front of a visitor.
 */
import type { components } from "./schema";

export type SourceBlock = components["schemas"]["SourceBlock"];
export type RetrieveResponse = components["schemas"]["RetrieveResponse"];
export type QueryResponse = components["schemas"]["QueryResponse"];
export type CitedSentence = components["schemas"]["CitedSentence"];
export type RetrievalConfig = components["schemas"]["RetrievalConfig"];
export type HealthResponse = components["schemas"]["HealthResponse"];

const BASE = "/api";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    throw new ApiError(await describeFailure(response), response.status);
  }
  return (await response.json()) as T;
}

async function describeFailure(response: Response): Promise<string> {
  // FastAPI validation errors arrive as {detail: [...]}, which stringifies to
  // "[object Object]" if handed straight to a message. Worth unpacking: a 422 the user
  // can read is the difference between fixing their input and refreshing hopefully.
  try {
    const payload = (await response.json()) as { detail?: unknown };
    if (typeof payload.detail === "string") return payload.detail;
    if (payload.detail) return JSON.stringify(payload.detail);
  } catch {
    /* fall through to the status text */
  }
  return response.statusText || `HTTP ${response.status}`;
}

export async function health(): Promise<HealthResponse> {
  const response = await fetch(`${BASE}/health`);
  return (await response.json()) as HealthResponse;
}

/** Retrieval only. Needs no model credentials and costs nothing. */
export function retrieve(question: string, config: RetrievalConfig): Promise<RetrieveResponse> {
  return post<RetrieveResponse>("/retrieve", { question, config });
}

/** Retrieve, generate and bind citations. Needs a model credential. */
export function query(question: string, config: RetrievalConfig): Promise<QueryResponse> {
  return post<QueryResponse>("/query", { question, config });
}

export type StreamEvent =
  | { kind: "token"; text: string }
  | { kind: "sources"; sources: SourceBlock[] }
  | { kind: "stages"; timings: Record<string, number> }
  | { kind: "citations"; cited: number[]; refused: boolean; reason: string | null; cost: number }
  | { kind: "error"; message: string }
  | { kind: "done" };

/**
 * Consume the SSE answer stream.
 *
 * Written against `fetch` and a manual parser rather than `EventSource`, because
 * EventSource cannot send a request body or set headers and would force the whole
 * retrieval configuration into a query string. The parser is small: SSE frames are
 * separated by a blank line and this endpoint only emits `event:` and `data:`.
 */
export async function* streamAnswer(question: string): AsyncGenerator<StreamEvent> {
  const response = await fetch(`${BASE}/query/stream?question=${encodeURIComponent(question)}`);
  if (!response.ok || !response.body) {
    throw new ApiError(await describeFailure(response), response.status);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // A frame may arrive split across reads, so only complete ones are consumed and the
    // remainder stays in the buffer.
    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const frame = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const parsed = parseFrame(frame);
      if (parsed) yield parsed;
      boundary = buffer.indexOf("\n\n");
    }
  }
}

function parseFrame(frame: string): StreamEvent | null {
  let name = "";
  let data = "";
  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) name = line.slice(6).trim();
    else if (line.startsWith("data:")) data += line.slice(5).trim();
  }
  if (!name || !data) return null;

  const payload = JSON.parse(data) as Record<string, unknown>;
  switch (name) {
    case "token":
      return { kind: "token", text: String(payload.text ?? "") };
    case "sources":
      return { kind: "sources", sources: payload as unknown as SourceBlock[] };
    case "stages":
      return { kind: "stages", timings: payload as Record<string, number> };
    case "citations":
      return {
        kind: "citations",
        cited: (payload.cited_chunk_ids as number[]) ?? [],
        refused: Boolean(payload.refused),
        reason: (payload.refusal_reason as string | null) ?? null,
        cost: Number(payload.cost_usd ?? 0),
      };
    case "error":
      return { kind: "error", message: String(payload.message ?? "unknown error") };
    case "done":
      return { kind: "done" };
    default:
      return null;
  }
}

/** The library's own retrieval defaults, so the client never restates them. */
export async function defaultConfig(): Promise<RetrievalConfig> {
  const response = await fetch(`${BASE}/config/default`);
  if (!response.ok) throw new ApiError(await describeFailure(response), response.status);
  return (await response.json()) as RetrievalConfig;
}
