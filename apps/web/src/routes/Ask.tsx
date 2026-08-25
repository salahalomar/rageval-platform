/**
 * The demo.
 *
 * Two buttons on purpose. "Retrieve" runs the retrieval stack alone: it needs no model
 * credential, costs nothing, and is the interesting half -- toggling the cross-encoder and
 * watching the ranking change is the claim this repository is actually making. "Answer"
 * adds generation and is disabled outright when the server reports no credential, because
 * a button that always fails teaches a visitor that the system is broken rather than that
 * it is unconfigured.
 */
import { useEffect, useState } from "react";

import Answer from "../components/Answer";
import ConfigPanel from "../components/ConfigPanel";
import Sources from "../components/Sources";
import Stages from "../components/Stages";
import {
  ApiError,
  defaultConfig,
  health,
  query,
  retrieve,
  type QueryResponse,
  type RetrievalConfig,
  type RetrieveResponse,
  type SourceBlock,
} from "../lib/api";

// Drawn from the golden set so a visitor lands on questions the corpus can actually
// answer, plus one it cannot -- refusal is a feature here and should be easy to see.
const EXAMPLES = [
  "What is the purpose of the warmup phase in learning rate schedules?",
  "How does rotary position embedding encode relative position?",
  "What were Tesla's Q3 2025 delivery numbers?",
];

type Result =
  | { kind: "retrieval"; data: RetrieveResponse }
  | { kind: "answer"; data: QueryResponse };

function sourcesOf(result: Result): SourceBlock[] {
  return result.data.sources;
}

export default function Ask() {
  const [question, setQuestion] = useState("");
  const [config, setConfig] = useState<RetrievalConfig | null>(null);
  const [result, setResult] = useState<Result | null>(null);
  const [asked, setAsked] = useState("");
  const [busy, setBusy] = useState<"retrieve" | "answer" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const [canGenerate, setCanGenerate] = useState<boolean | null>(null);

  useEffect(() => {
    health()
      .then((status) => setCanGenerate(status.generation_configured))
      .catch(() => setCanGenerate(false));
    defaultConfig()
      .then(setConfig)
      .catch(() => setError("Could not reach the API. Is it running on port 8000?"));
  }, []);

  async function run(mode: "retrieve" | "answer", text: string) {
    const trimmed = text.trim();
    if (!trimmed || !config) return;
    setBusy(mode);
    setError(null);
    setSelected(null);
    setAsked(trimmed);
    try {
      const data =
        mode === "retrieve"
          ? ({ kind: "retrieval", data: await retrieve(trimmed, config) } as const)
          : ({ kind: "answer", data: await query(trimmed, config) } as const);
      setResult(data);
    } catch (caught) {
      setResult(null);
      setError(
        caught instanceof ApiError
          ? `${caught.message} (HTTP ${caught.status})`
          : caught instanceof Error
            ? caught.message
            : "The request failed.",
      );
    } finally {
      setBusy(null);
    }
  }

  // Re-running on a config change is what makes the panel feel like an instrument rather
  // than a form: change the fusion mode and the same question reorders underneath you.
  function changeConfig(next: RetrievalConfig) {
    setConfig(next);
    if (asked && result) void rerun(next);
  }


  async function rerun(next: RetrievalConfig) {
    setBusy("retrieve");
    setError(null);
    try {
      setResult({ kind: "retrieval", data: await retrieve(asked, next) });
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "The request failed.");
    } finally {
      setBusy(null);
    }
  }

  const refused = result
    ? result.kind === "answer"
      ? result.data.refused
      : result.data.refused
    : false;

  return (
    <>
      <p className="lede">
        Retrieval over 150 arXiv cs.LG / cs.CL papers. Every number on this page is
        produced by the same library the{" "}
        <a href="/eval">evaluation harness</a> measures.
      </p>

      <form
        className="ask"
        onSubmit={(event) => {
          event.preventDefault();
          void run("retrieve", question);
        }}
      >
        <input
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          placeholder="Ask something about the corpus…"
          aria-label="Question"
          data-testid="question"
        />
        <button
          type="submit"
          className="primary"
          disabled={busy !== null || !question.trim() || !config}
        >
          {busy === "retrieve" ? "Retrieving…" : "Retrieve"}
        </button>
        <button
          type="button"
          disabled={busy !== null || !question.trim() || !config || canGenerate !== true}
          title={
            canGenerate === false
              ? "The server has no model credential, so generation is unavailable. Retrieval works regardless."
              : "Retrieve, then generate an answer with citations"
          }
          onClick={() => void run("answer", question)}
        >
          {busy === "answer" ? "Answering…" : "Answer"}
        </button>
      </form>

      <div className="examples">
        {EXAMPLES.map((example) => (
          <button
            key={example}
            type="button"
            disabled={busy !== null || !config}
            onClick={() => {
              setQuestion(example);
              void run("retrieve", example);
            }}
          >
            {example}
          </button>
        ))}
      </div>

      {config ? (
        <ConfigPanel config={config} onChange={changeConfig} disabled={busy !== null} />
      ) : null}

      {canGenerate === false ? (
        <p className="footnote">
          Generation is off: this deployment has no model credential. Retrieval, reranking,
          refusal and the evaluation table are unaffected — none of them call a paid API.
        </p>
      ) : null}

      <Stages
        timings={result ? result.data.timings_ms : null}
        running={busy ? (busy === "answer" ? "generation_ms" : "dense_ms") : null}
      />

      {error ? (
        <p className="state error" data-testid="error">
          <strong>The request failed.</strong>
          {error}
        </p>
      ) : null}

      {busy && !result ? <p className="state">Searching the corpus…</p> : null}

      {!busy && !result && !error ? (
        <p className="state" data-testid="empty">
          Ask a question, or pick one of the examples above.
        </p>
      ) : null}

      {result && refused ? (
        <p className="state refusal" data-testid="refusal">
          <strong>Refused — no passage cleared the score floor.</strong>
          {result.kind === "answer"
            ? (result.data.refusal_reason ?? "insufficient evidence")
            : (result.data.reason ?? "insufficient evidence")}
          . Refusing is the designed behaviour when retrieval finds nothing good enough; a
          plausible guess would be worse.
        </p>
      ) : null}

      {result && result.kind === "answer" && !refused ? (
        <>
          <Answer text={result.data.answer} selected={selected} onSelect={setSelected} />
          <p className="footnote">
            {result.data.uncited
              ? "Warning: at least one sentence carried no citation. "
              : "Every sentence is bound to a chunk. "}
            {result.data.input_tokens.toLocaleString()} in /{" "}
            {result.data.output_tokens.toLocaleString()} out, ${result.data.cost_usd.toFixed(5)}.
          </p>
        </>
      ) : null}

      {result && sourcesOf(result).length > 0 ? (
        <>
          <Sources
            sources={sourcesOf(result)}
            question={asked}
            selected={selected}
            onSelect={setSelected}
          />
          {result.kind === "retrieval" ? (
            <p className="footnote">
              {result.data.dense_count} dense + {result.data.lexical_count} lexical →{" "}
              {result.data.fused_count} fused
              {result.data.reranked === null || result.data.reranked === undefined
                ? " (reranking off)"
                : ` → ${result.data.reranked} reranked, mean rank movement ${
                    result.data.mean_rank_movement?.toFixed(1) ?? "—"
                  }`}
              . Highlighting marks query terms, not the model's attribution.
            </p>
          ) : null}
        </>
      ) : null}
    </>
  );
}
