/**
 * The committed ablation results, read at build time.
 *
 * These are the same JSON files `eval/ablate.py` writes and CI commits -- imported
 * directly out of `eval/results/`, not copied into the app. A copy would be a second
 * source of truth that goes stale silently, and a stale number in a page whose entire
 * argument is "measure everything" would be the worst possible bug to ship.
 *
 * Row order comes from `index.json`, which `eval/report.py` generates alongside the
 * README table, so this page and the README cannot disagree about which arm goes where.
 */

export type RunResult = {
  arm: string;
  config: Record<string, unknown>;
  metrics: Record<string, number>;
  latency_ms: Record<string, number>;
  git_sha: string;
  git_dirty: boolean;
  started_at: string;
  duration_s: number;
  python: string;
  golden_file: string;
  golden_items: number;
  scored_items: number;
  unanswerable_items: number;
  correct_refusals: number;
  false_refusals: number;
  refusals: number;
  generation: Record<string, unknown> | null;
};

type Manifest = {
  columns: { key: string; label: string }[];
  runs: { arm: string; file: string; git_sha: string; git_dirty: boolean }[];
};

const RESULT_FILES = import.meta.glob<RunResult>("../../../../eval/results/*.json", {
  eager: true,
  import: "default",
});

function basename(path: string): string {
  return path.slice(path.lastIndexOf("/") + 1);
}

const BY_NAME = new Map(
  Object.entries(RESULT_FILES).map(([path, payload]) => [basename(path), payload]),
);

const manifest = BY_NAME.get("index.json") as unknown as Manifest | undefined;

export const COLUMNS: { key: string; label: string }[] = manifest?.columns ?? [
  { key: "recall@1", label: "R@1" },
  { key: "recall@5", label: "R@5" },
  { key: "mrr@10", label: "MRR@10" },
  { key: "ndcg@10", label: "nDCG@10" },
];

/** Every committed run, in the order the README table prints them. */
export function results(): RunResult[] {
  if (manifest) {
    return manifest.runs
      .map((run) => BY_NAME.get(run.file))
      .filter((run): run is RunResult => run !== undefined);
  }
  // No manifest: fall back to filename order rather than rendering nothing. The page
  // says so, because an unordered table is a weaker claim than an ordered one.
  return [...BY_NAME.entries()]
    .filter(([name]) => name !== "index.json")
    .map(([, run]) => run);
}

export const ORDERED = manifest !== undefined;

/** Share of unanswerable items correctly refused; null when none were asked. */
export function refusalAccuracy(run: RunResult): number | null {
  if (run.unanswerable_items === 0) return null;
  return run.correct_refusals / run.unanswerable_items;
}

/**
 * Every arm holding the highest value of `metric` -- a set, not a single winner.
 *
 * Ties are not an edge case here, they are the headline result: once the cross-encoder is
 * on it rescores the fused top 50 from scratch, so the three RRF-k arms come out
 * byte-identical. Picking one of them to crown would invent a difference nothing measured.
 */
export function bestArms(runs: RunResult[], metric: string): Set<string> {
  let best = -Infinity;
  for (const run of runs) {
    const value = run.metrics[metric];
    if (value !== undefined && value > best) best = value;
  }
  if (best === -Infinity) return new Set();
  return new Set(runs.filter((run) => run.metrics[metric] === best).map((run) => run.arm));
}
