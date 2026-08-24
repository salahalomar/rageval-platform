"""Running one configuration against the golden set.

This module calls `rag.retrieve.retrieve()` and `rag.generate.answer_question()` and
nothing else. That is the entire argument of the repository: an evaluation with its own
retrieval implementation measures something adjacent to what ships, and the divergence is
invisible until an interviewer asks. There is no retrieval code here to drift.

Retrieval metrics need no model beyond the local embedder and reranker, so a retrieval-only
run costs nothing and can be the CI gate. Answer metrics need a judge and are opt-in.

Every result embeds the full configuration, the git commit, the model identifiers and the
golden-set fingerprint. A number whose provenance cannot be reconstructed is not evidence.
"""

import json
import logging
import platform
import subprocess
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import psycopg

from eval.metrics.retrieval import RetrievalMetrics, average_metrics, score_ranking
from eval.relevance import relevant_ids_for_chunking, resolve_spans
from eval.schema import GoldenItem
from rag.config import GenerationConfig, RetrievalConfig
from rag.retrieve import retrieve
from rag.telemetry import StageTimer

logger = logging.getLogger(__name__)

DEFAULT_RESULTS_DIR = Path("eval/results")


def git_sha() -> str:
    """The commit this result was produced at, or 'unknown' outside a repository."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def git_dirty() -> bool:
    """Whether the working tree had uncommitted changes.

    Recorded because a result produced from a dirty tree cannot be reproduced from its
    commit, and that is worth knowing before anyone quotes the number.
    """
    try:
        return bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            ).stdout.strip()
        )
    except (subprocess.SubprocessError, OSError):
        return True


@dataclass(slots=True)
class ItemResult:
    """What one configuration did on one golden item."""

    item_id: str
    question: str
    answerable: bool
    type: str
    relevant_chunk_ids: list[int]
    retrieved_chunk_ids: list[int]
    refused: bool
    refusal_reason: str | None
    timings_ms: dict[str, float]
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class RunResult:
    """One arm of the ablation: its configuration, its provenance and its numbers."""

    arm: str
    config: dict[str, object]
    generation: dict[str, object] | None
    git_sha: str
    git_dirty: bool
    started_at: str
    duration_s: float
    python: str
    golden_file: str
    golden_items: int
    scored_items: int
    unanswerable_items: int
    refusals: int
    correct_refusals: int
    false_refusals: int
    metrics: dict[str, float]
    latency_ms: dict[str, float]
    items: list[ItemResult]

    @property
    def refusal_accuracy(self) -> float | None:
        """Share of unanswerable items correctly refused.

        None when the set contains no unanswerable items -- reporting 0 or 1 for a
        question that was never asked would be inventing a measurement.
        """
        if self.unanswerable_items == 0:
            return None
        return self.correct_refusals / self.unanswerable_items

    def write(self, directory: Path) -> Path:
        """Persist this result as JSON, named by arm and commit."""
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.arm}--{self.git_sha[:8]}.json"
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True), encoding="utf-8")
        return path


def percentile(values: Sequence[float], fraction: float) -> float:
    """Percentile of an unsorted sequence, 0 when empty."""
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(fraction * (len(ordered) - 1)))]


def run(
    items: Sequence[GoldenItem],
    config: RetrievalConfig,
    conn: psycopg.Connection,
    *,
    arm: str,
    golden_file: str,
    generation: GenerationConfig | None = None,
) -> RunResult:
    """Score one configuration across the golden set.

    Ground truth is resolved to character spans and re-resolved against the chunking this
    configuration actually uses, so an arm that re-chunks the corpus is scored against the
    region of the paper that answers the question rather than against chunk ids belonging
    to a different chunking. Without that, the chunk-size sweep would report zero recall
    for structural reasons.
    """
    started = time.perf_counter()
    started_at = datetime.now(UTC).isoformat()
    chunking = config.chunking_sha256()

    results: list[ItemResult] = []
    scored: list[RetrievalMetrics] = []
    latencies: list[float] = []
    unanswerable = correct_refusals = false_refusals = refusals = 0

    for item in items:
        spans = resolve_spans(conn, item.relevant_chunk_ids)
        relevant = relevant_ids_for_chunking(conn, spans, chunking)

        timer = StageTimer()
        outcome = retrieve(item.question, config, conn, timer=timer)
        latencies.append(timer.total_ms())

        if outcome.refused:
            refusals += 1
        if not item.answerable:
            unanswerable += 1
            if outcome.refused:
                correct_refusals += 1
        elif outcome.refused:
            false_refusals += 1

        metrics = score_ranking(outcome.chunk_ids, relevant)
        if metrics is not None:
            scored.append(metrics)

        results.append(
            ItemResult(
                item_id=item.id,
                question=item.question,
                answerable=item.answerable,
                type=item.type,
                relevant_chunk_ids=relevant,
                retrieved_chunk_ids=outcome.chunk_ids,
                refused=outcome.refused,
                refusal_reason=outcome.reason,
                timings_ms=timer.as_dict(),
                metrics=metrics.as_row() if metrics else {},
            )
        )

    aggregate = average_metrics(scored)
    logger.info("%s: scored %d items, recall@5 %.3f", arm, len(scored), aggregate.recall.get(5, 0))

    return RunResult(
        arm=arm,
        config=config.model_dump(),
        generation=generation.fingerprint() if generation else None,
        git_sha=git_sha(),
        git_dirty=git_dirty(),
        started_at=started_at,
        duration_s=round(time.perf_counter() - started, 2),
        python=platform.python_version(),
        golden_file=golden_file,
        golden_items=len(items),
        scored_items=len(scored),
        unanswerable_items=unanswerable,
        refusals=refusals,
        correct_refusals=correct_refusals,
        false_refusals=false_refusals,
        metrics=aggregate.as_row(),
        latency_ms={
            "p50": round(percentile(latencies, 0.50), 1),
            "p95": round(percentile(latencies, 0.95), 1),
            "mean": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
        },
        items=results,
    )
