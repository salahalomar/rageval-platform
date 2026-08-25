/**
 * The ablation table -- the actual deliverable.
 *
 * Rows are in matrix order, least machinery to most, never sorted by score. Sorting by the
 * winning column would put the best arm on top regardless of what it cost and would hide
 * the only thing worth reading here: whether each additional piece of machinery earned its
 * place over the row above it.
 */
import { Fragment, useState } from "react";

import {
  COLUMNS,
  ORDERED,
  bestArms,
  refusalAccuracy,
  results,
  type RunResult,
} from "../lib/results";

const WINNER_METRIC = "ndcg@10";

function num(value: number | null | undefined, places = 3): string {
  return value === null || value === undefined ? "—" : value.toFixed(places);
}

function ConfigDetail({ run }: { run: RunResult }) {
  const entries = Object.entries(run.config).sort(([a], [b]) => a.localeCompare(b));
  return (
    <tr>
      <td colSpan={COLUMNS.length + 4} style={{ textAlign: "left" }}>
        <div className="footnote" style={{ margin: 0 }}>
          <code>
            {run.git_sha.slice(0, 8)}
            {run.git_dirty ? "-dirty" : ""}
          </code>{" "}
          · {new Date(run.started_at).toISOString().slice(0, 16).replace("T", " ")} UTC ·{" "}
          {run.duration_s.toFixed(0)}s · python {run.python} ·{" "}
          {run.generation === null ? "retrieval only, no model called" : "judged"}
          <br />
          {entries.map(([key, value]) => `${key}=${String(value)}`).join("  ")}
        </div>
      </td>
    </tr>
  );
}

export default function Evaluation() {
  const runs = results();
  const [open, setOpen] = useState<string | null>(null);
  const winners = bestArms(runs, WINNER_METRIC);

  const golden = runs[0];
  if (golden === undefined) {
    return (
      <p className="state">
        No committed results. Run <code>make ablate</code> to produce them.
      </p>
    );
  }

  return (
    <>
      <p className="lede">
        Every arm below ran against the same golden set on the same commit, through the same
        library the demo uses. Click a row for the configuration that produced it.
      </p>

      <div className="caveat">
        <strong>Read this before the numbers.</strong>
        The golden set is {golden.golden_items} hand-verified hard cases, of which{" "}
        {golden.scored_items} are answerable and {golden.unanswerable_items} are deliberately
        unanswerable. That is small enough that differences under roughly 0.1 are noise, and
        no confidence intervals are computed. Answer-quality columns are absent because the
        LLM judge has not been run — showing zeros would read as “measured and bad” rather
        than “not measured”.
      </div>

      <div style={{ overflowX: "auto" }}>
        <table className="ablation">
          <thead>
            <tr>
              <th>Arm</th>
              {COLUMNS.map((column) => (
                <th key={column.key}>{column.label}</th>
              ))}
              <th>Refusal acc.</th>
              <th>p95 ms</th>
              <th>n</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((run) => (
              <Fragment key={run.arm}>
                <tr
                  className={winners.has(run.arm) ? "winner" : undefined}
                  onClick={() => setOpen(open === run.arm ? null : run.arm)}
                  data-testid="arm-row"
                >
                  <td>{run.arm}</td>
                  {COLUMNS.map((column) => (
                    <td key={column.key} className="num">
                      {num(run.metrics[column.key])}
                    </td>
                  ))}
                  <td className="num">{num(refusalAccuracy(run), 2)}</td>
                  <td className="num">{num(run.latency_ms.p95, 0)}</td>
                  <td className="num">{run.scored_items}</td>
                </tr>
                {open === run.arm ? <ConfigDetail run={run} /> : null}
              </Fragment>
            ))}
          </tbody>
        </table>
      </div>

      <p className="footnote">
        Highlighted {winners.size === 1 ? "row" : `${winners.size} rows`}: best{" "}
        <code>{WINNER_METRIC}</code>. Golden set{" "}
        <code>{golden.golden_file}</code>.{" "}
        {ORDERED
          ? "Rows are in ablation-matrix order, matching the README."
          : "Row order could not be resolved from eval/results/index.json, so these are in filename order."}{" "}
        The <code>rrf-k*</code> arms are identical to <code>hybrid-rerank</code> by
        construction: the cross-encoder rescores the fused top 50 from scratch, so the RRF
        constant cannot survive it. That is a real finding, not a broken run.
      </p>
    </>
  );
}
