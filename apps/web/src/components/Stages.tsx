/**
 * Per-stage timings, shown rather than a spinner.
 *
 * ENGINEERING.md treats latency as a feature, and the server already records a timing for
 * every stage. Rendering those numbers instead of an indeterminate bar means the interface
 * makes the same claim the telemetry does, and a slow reranker is visible rather than
 * hidden behind an animation.
 *
 * The keys are the library's own stage names (`rag/telemetry.py`), not a parallel
 * vocabulary. Anything the server reports that this list has not seen is still rendered,
 * at the end -- a new stage should show up as an unlabelled number rather than vanish.
 */
const ORDER = [
  "embed_query_ms",
  "dense_ms",
  "lexical_ms",
  "fusion_ms",
  "rerank_ms",
  "generation_ms",
] as const;

const LABELS: Record<string, string> = {
  embed_query_ms: "embed",
  dense_ms: "dense",
  lexical_ms: "lexical",
  fusion_ms: "fuse",
  rerank_ms: "rerank",
  generation_ms: "generate",
  encode_ms: "encode",
};

function label(key: string): string {
  return LABELS[key] ?? key.replace(/_ms$/, "");
}

export default function Stages({
  timings,
  running,
}: {
  timings: Record<string, number> | null;
  running: string | null;
}) {
  if (!timings && !running) return null;

  const known = timings ?? {};
  const ordered = ORDER.filter((key) => key in known || key === running);
  const extra = Object.keys(known).filter((key) => !(ORDER as readonly string[]).includes(key));
  const total = Object.values(known).reduce((sum, ms) => sum + ms, 0);

  return (
    <div className="stages" aria-label="Stage timings" data-testid="stages">
      {[...ordered, ...extra].map((key) => {
        const ms = known[key];
        const state = key === running ? "active" : ms === undefined ? "" : "done";
        return (
          <span key={key} className={`stage ${state}`}>
            {label(key)}
            {ms === undefined ? " …" : ` ${ms.toFixed(0)} ms`}
          </span>
        );
      })}
      {timings ? <span className="stage">total {total.toFixed(0)} ms</span> : null}
    </div>
  );
}
