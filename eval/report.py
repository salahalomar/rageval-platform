"""Rendering the ablation table, and injecting it into the README.

The table is generated from the committed result JSONs and never hand-edited, so a number
in the README always traces to a file that records the configuration, the commit and the
golden set that produced it.

Rows are printed in matrix order and never sorted by score. Sorting by the winning metric
would make the best arm appear first regardless of what it cost, and would hide the thing
worth showing: whether each additional piece of machinery actually earned its place over
the row above it.
"""

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from eval.runner import DEFAULT_RESULTS_DIR, RunResult

logger = logging.getLogger(__name__)

START_MARKER = "<!-- ABLATION-TABLE:START -->"
END_MARKER = "<!-- ABLATION-TABLE:END -->"

# A machine-readable sibling of the markdown table, written next to the results so the web
# app can render the same rows in the same order without restating ARM_ORDER in TypeScript.
# A third copy of the ordering would drift, and the drift would be invisible: the table
# would still render, just telling a different story than the README.
MANIFEST_NAME = "index.json"

# The order arms appear in, least to most machinery, so the table reads as a story:
# what does lexical alone do, does dense beat it, does fusing help, does reranking help
# on top. Filename order would sort them alphabetically and destroy that reading, and an
# arm that failed to improve on the row above it would no longer be next to it.
#
# Duplicated from `ablate.matrix()` rather than imported, because ablate imports this
# module and the two would form a cycle. `tests/test_eval_harness.py` asserts they agree.
ARM_ORDER: tuple[str, ...] = (
    "lexical-only",
    "dense-small",
    "dense-base",
    "hybrid-rrf",
    "hybrid-rerank",
    "hybrid-rerank-nohdr",
    "rrf-k20",
    "rrf-k120",
    "chunk-256",
    "chunk-1024",
)

# The columns the plan calls for. Answer-quality columns appear only when a judged run
# supplied them; a column of dashes is better than a column of zeros, which would read as
# "measured and bad" rather than "not measured".
COLUMNS: tuple[tuple[str, str], ...] = (
    ("recall@1", "R@1"),
    ("recall@5", "R@5"),
    ("mrr@10", "MRR@10"),
    ("ndcg@10", "nDCG@10"),
)


def _format(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_table(results: Sequence[RunResult]) -> str:
    """Markdown table over a set of runs."""
    if not results:
        return "_No results._"

    header = ["Arm", *[label for _, label in COLUMNS], "Refusal acc.", "p95 ms", "n"]
    rows = []
    for result in results:
        refusal = result.refusal_accuracy
        rows.append(
            [
                result.arm,
                *[_format(result.metrics.get(key)) for key, _ in COLUMNS],
                _format(refusal),
                f"{result.latency_ms.get('p95', 0):.0f}",
                str(result.scored_items),
            ]
        )

    widths = [max(len(row[i]) for row in [header, *rows]) for i in range(len(header))]
    lines = [
        "| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(header)) + " |",
        "|" + "|".join("-" * (widths[i] + 2) for i in range(len(header))) + "|",
    ]
    lines.extend(
        "| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) + " |"
        for row in rows
    )
    return "\n".join(lines)


def load_results(directory: Path) -> list[RunResult]:
    """Read every result JSON in a directory, newest run per arm."""
    by_arm: dict[str, tuple[str, RunResult]] = {}
    for path in sorted(directory.glob("*.json")):
        if path.name == MANIFEST_NAME:
            continue
        payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        result = RunResult(**payload)
        existing = by_arm.get(result.arm)
        if existing is None or existing[0] < result.started_at:
            by_arm[result.arm] = (result.started_at, result)

    def position(name: str) -> int:
        return ARM_ORDER.index(name) if name in ARM_ORDER else len(ARM_ORDER)

    return sorted(
        (result for _, result in by_arm.values()),
        key=lambda result: (position(result.arm), result.arm),
    )


def write_manifest(results: Sequence[RunResult], directory: Path) -> Path:
    """Write the ordering and column choice the markdown table uses, as JSON.

    Exists so that a second renderer -- the `/eval` page in the web app -- shows the same
    arms in the same order for the same reason, rather than sorting alphabetically or by
    score and quietly telling a different story than the README.
    """
    path = directory / MANIFEST_NAME
    payload = {
        "columns": [{"key": key, "label": label} for key, label in COLUMNS],
        "runs": [
            {
                "arm": result.arm,
                "git_sha": result.git_sha,
                "git_dirty": result.git_dirty,
                "started_at": result.started_at,
                "file": f"{result.arm}--{result.git_sha[:8]}.json",
            }
            for result in results
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def inject(readme: Path, table: str) -> bool:
    """Replace the marked block in the README with `table`. Returns True if it changed."""
    text = readme.read_text(encoding="utf-8")
    if START_MARKER not in text or END_MARKER not in text:
        logger.warning("README has no %s / %s markers; not injecting", START_MARKER, END_MARKER)
        return False

    before, _, rest = text.partition(START_MARKER)
    _, _, after = rest.partition(END_MARKER)
    updated = f"{before}{START_MARKER}\n\n{table}\n\n{END_MARKER}{after}"
    if updated == text:
        return False
    readme.write_text(updated, encoding="utf-8")
    return True


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    parser.add_argument("--inject", action="store_true", help="write the table into the README")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s", stream=sys.stderr)

    results = load_results(args.results_dir)
    if not results:
        print(f"no results in {args.results_dir}")
        return 1

    table = render_table(results)
    print(table)
    manifest = write_manifest(results, args.results_dir)
    print(f"\nwrote {manifest}", file=sys.stderr)
    if args.inject and inject(args.readme, table):
        print(f"injected into {args.readme}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
