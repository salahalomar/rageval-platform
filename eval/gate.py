"""The pull-request gate: fail the build when retrieval quality regresses.

Retrieval metrics need no paid API and no network, so this can run on every pull request
without spending anything. That is the whole reason the retrieval and answer halves of
the harness are separate.

The gate compares against a committed baseline and fails when Recall@5 falls more than a
stated tolerance below it. The tolerance exists because a golden set of this size has real
sampling noise: on eighty items a single question changing answer moves Recall@5 by 1.25
points, so a zero-tolerance gate would fail on noise and be switched off within a week —
which is worse than no gate.

A gate that cannot find a corpus reports that and does not fail the build. Failing a pull
request because the evaluation environment was not seeded would train everyone to ignore
it, and a gate people ignore protects nothing.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from eval.ablate import chunking_is_available
from eval.runner import run
from eval.schema import read_items
from rag.config import RetrievalConfig
from rag.db import connect

logger = logging.getLogger(__name__)

DEFAULT_BASELINE = Path("eval/baseline.json")
DEFAULT_GOLDEN = Path("eval/golden/v1.jsonl")
FALLBACK_GOLDEN = Path("eval/golden/hard_cases.jsonl")

# Points of Recall@5 a change may lose before the build fails. Three points, from the plan.
TOLERANCE = 0.03

# Retrieval-only, so the gate stays free and fast. Reranking would add roughly five
# seconds per question on CPU, which is minutes for a smoke subset and would push the
# gate past the point where anyone waits for it.
GATE_CONFIG = RetrievalConfig(rerank_enabled=False)

SMOKE_ITEMS = 20


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns 1 only on a genuine regression."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", type=Path, default=None)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--limit", type=int, default=SMOKE_ITEMS)
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="write the measured metrics to the baseline file instead of comparing",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s", stream=sys.stderr)

    golden = args.golden or (DEFAULT_GOLDEN if DEFAULT_GOLDEN.exists() else FALLBACK_GOLDEN)
    if not golden.exists():
        print(f"gate skipped: no golden set at {golden}")
        return 0

    items = [item for item in read_items(golden) if item.answerable][: args.limit]
    if not items:
        print("gate skipped: no answerable items in the golden set")
        return 0

    with connect() as conn:
        if not chunking_is_available(conn, GATE_CONFIG):
            print("gate skipped: this database has no corpus for the default chunking.")
            print("  Seed it with `rag ingest --ids-file infra/corpus/...` and `rag embed`.")
            return 0

        result = run(items, GATE_CONFIG, conn, arm="gate", golden_file=str(golden))

    measured = result.metrics.get("recall@5", 0.0)
    print(f"gate: recall@5 = {measured:.3f} over {result.scored_items} items")

    if args.update_baseline:
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(
            json.dumps(
                {
                    "recall@5": round(measured, 4),
                    "items": result.scored_items,
                    "golden_file": str(golden),
                    "config": result.config,
                    "git_sha": result.git_sha,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        print(f"wrote baseline to {args.baseline}")
        return 0

    if not args.baseline.exists():
        print(f"gate skipped: no baseline at {args.baseline} (create it with --update-baseline)")
        return 0

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    expected = float(baseline.get("recall@5", 0.0))
    delta = measured - expected

    print(f"baseline: recall@5 = {expected:.3f} over {baseline.get('items', '?')} items")
    print(f"delta:    {delta:+.3f}  (tolerance {-TOLERANCE:+.3f})")

    if baseline.get("items") != result.scored_items:
        print()
        print("  NOTE: the baseline was measured over a different number of items, so this")
        print("  comparison is not like for like. Refresh it with --update-baseline.")

    if delta < -TOLERANCE:
        print()
        print(f"FAIL: recall@5 dropped {abs(delta):.3f} below the baseline")
        return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
