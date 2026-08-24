"""The ablation matrix: run every arm, write one JSON each, emit the table.

Every arm is a `RetrievalConfig`, and nothing else differs between them. That is the
property that makes the table an experiment rather than a collection of anecdotes: an arm
cannot accidentally take a different code path, because there is only one path and the
config selects its behaviour.

Arms that require re-chunking the corpus are declared but not run by default. Changing
`chunk_tokens` produces a chunking that must be ingested and embedded before it can be
retrieved from -- roughly ten minutes of CPU per arm -- so they are opt-in via
`--include-rechunk` after `rag ingest` and `rag embed` have been run for those settings.
The ground-truth remapping in `eval.relevance` is what makes them scoreable at all.
"""

import argparse
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import psycopg

from eval.report import render_table
from eval.runner import DEFAULT_RESULTS_DIR, RunResult, run
from eval.schema import GoldenItem, read_items
from rag.config import RetrievalConfig
from rag.db import connect

logger = logging.getLogger(__name__)

DEFAULT_GOLDEN = Path("eval/golden/v1.jsonl")
FALLBACK_GOLDEN = Path("eval/golden/hard_cases.jsonl")


@dataclass(frozen=True, slots=True)
class Arm:
    """One row of the ablation table."""

    name: str
    label: str
    config: RetrievalConfig
    needs_rechunk: bool = False


def matrix(base: RetrievalConfig | None = None) -> list[Arm]:
    """Every arm of the ablation, in the order the table reports them.

    Ordered from least to most machinery so the table reads as a story: what does lexical
    alone do, what does dense alone do, does fusing them help, does reranking help on top.
    An arm that fails to improve on the row above it is the interesting case and is
    reported in place rather than dropped.
    """
    base = base or RetrievalConfig()
    no_rerank = {"rerank_enabled": False}

    return [
        Arm(
            "lexical-only",
            "Lexical only (ts_rank_cd)",
            base.model_copy(update={"fusion": "lexical_only", **no_rerank}),
        ),
        Arm(
            "dense-small",
            "Dense only (bge-small)",
            base.model_copy(update={"fusion": "dense_only", **no_rerank}),
        ),
        Arm("hybrid-rrf", "Hybrid RRF", base.model_copy(update={"fusion": "rrf", **no_rerank})),
        Arm(
            "hybrid-rerank",
            "Hybrid + rerank",
            base.model_copy(update={"fusion": "rrf", "rerank_enabled": True}),
        ),
        Arm(
            "hybrid-rerank-nohdr",
            "Hybrid + rerank, no contextual headers",
            base.model_copy(
                update={"fusion": "rrf", "rerank_enabled": True, "contextual_headers": False}
            ),
            needs_rechunk=True,
        ),
        Arm(
            "rrf-k20",
            "Hybrid + rerank, RRF k=20",
            base.model_copy(update={"fusion": "rrf", "rerank_enabled": True, "rrf_k": 20}),
        ),
        Arm(
            "rrf-k120",
            "Hybrid + rerank, RRF k=120",
            base.model_copy(update={"fusion": "rrf", "rerank_enabled": True, "rrf_k": 120}),
        ),
        Arm(
            "chunk-256",
            "Hybrid + rerank, 256-token chunks",
            base.model_copy(update={"fusion": "rrf", "rerank_enabled": True, "chunk_tokens": 256}),
            needs_rechunk=True,
        ),
        Arm(
            "chunk-1024",
            "Hybrid + rerank, 1024-token chunks",
            base.model_copy(update={"fusion": "rrf", "rerank_enabled": True, "chunk_tokens": 1024}),
            needs_rechunk=True,
        ),
        Arm(
            "dense-base",
            "Dense only (bge-base, 768-d)",
            base.model_copy(
                update={
                    "fusion": "dense_only",
                    **no_rerank,
                    "embedding_model": "BAAI/bge-base-en-v1.5",
                }
            ),
            needs_rechunk=True,
        ),
    ]


def chunking_is_available(conn: psycopg.Connection, config: RetrievalConfig) -> bool:
    """Whether the corpus has been chunked and embedded for this configuration."""
    row = conn.execute(
        "SELECT count(*) FROM chunks WHERE chunk_config_sha256 = %s",
        (config.chunking_sha256(),),
    ).fetchone()
    return bool(row and row[0])


def run_matrix(
    items: Sequence[GoldenItem],
    conn: psycopg.Connection,
    *,
    golden_file: str,
    arms: Sequence[Arm] | None = None,
    include_rechunk: bool = False,
    results_dir: Path = DEFAULT_RESULTS_DIR,
) -> list[RunResult]:
    """Run every runnable arm and persist one JSON per arm."""
    results: list[RunResult] = []

    for arm in arms or matrix():
        if arm.needs_rechunk and not include_rechunk:
            logger.info("skipping %s: needs a re-chunked corpus (--include-rechunk)", arm.name)
            continue
        if not chunking_is_available(conn, arm.config):
            logger.warning(
                "skipping %s: no chunks for chunking %s — run `rag ingest` and `rag embed` "
                "with those settings first",
                arm.name,
                arm.config.chunking_sha256()[:12],
            )
            continue

        logger.info("running %s", arm.name)
        result = run(items, arm.config, conn, arm=arm.name, golden_file=golden_file)
        path = result.write(results_dir)
        logger.info("wrote %s", path)
        results.append(result)

    return results


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--include-rechunk",
        action="store_true",
        help="also run arms needing a re-chunked corpus (ingest and embed them first)",
    )
    parser.add_argument("--arm", action="append", help="run only these arms, by name")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s", stream=sys.stderr)

    golden = args.golden
    if golden is None:
        golden = DEFAULT_GOLDEN if DEFAULT_GOLDEN.exists() else FALLBACK_GOLDEN
    if not golden.exists():
        print(f"no golden set at {golden}; run the Phase 6 pipeline first")
        return 1

    items = read_items(golden)
    arms = [a for a in matrix() if not args.arm or a.name in args.arm]

    if golden == FALLBACK_GOLDEN:
        print("=" * 78)
        print("  SMOKE RUN — the verified golden set does not exist.")
        print(f"  Scoring against {len(items)} UNVERIFIED hand-written drafts.")
        print("  These numbers demonstrate that the harness runs. They are NOT a")
        print("  quality result and must not be published as one.")
        print("=" * 78)
        print()

    with connect() as conn:
        results = run_matrix(
            items,
            conn,
            golden_file=str(golden),
            arms=arms,
            include_rechunk=args.include_rechunk,
            results_dir=args.results_dir,
        )

    if not results:
        print("no arms ran")
        return 1

    print()
    print(render_table(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
