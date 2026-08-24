"""Recall@k, Precision@k, MRR and nDCG as pure functions over ranked ids.

No database, no model, no configuration. Every one of these is four lines of arithmetic
that decides what the README claims, so each is unit-tested against numbers computed by
hand. A subtly wrong nDCG produces a plausible table that is wrong in the same direction
for every arm, which is undetectable by inspection.

Two conventions, both stated because both change the numbers:

**Binary relevance.** A chunk either supports the answer or it does not; the golden set
records no graded judgements, so nDCG uses gains of 1 and 0. Claiming graded nDCG over
binary labels would be reporting a more sophisticated metric than the data supports.

**Items with no relevant chunks return `None`, not zero.** Unanswerable questions have no
ground truth to recall, and scoring them zero would drag every average down in proportion
to how many refusal cases the set contains -- punishing the set for testing refusal.
They are excluded from retrieval averages and scored separately by refusal accuracy.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

# The k values reported in the ablation table. Recall is reported at several k because
# the shape of the curve is informative: recall@1 that trails recall@5 badly means the
# right chunk is being found but not ranked first, which is a reranking problem rather
# than a retrieval one.
RECALL_KS = (1, 3, 5, 10)
PRECISION_KS = (1, 5)
MRR_K = 10
NDCG_K = 10


def recall_at_k(ranked: Sequence[int], relevant: Sequence[int], k: int) -> float | None:
    """Share of relevant chunks appearing in the top k.

    Returns None when there is nothing to recall, so the caller can exclude the item
    rather than average in a zero it did not earn.
    """
    truth = set(relevant)
    if not truth:
        return None
    return len(truth & set(ranked[:k])) / len(truth)


def precision_at_k(ranked: Sequence[int], relevant: Sequence[int], k: int) -> float | None:
    """Share of the top k that is relevant.

    Divides by k rather than by the number retrieved, deliberately. Dividing by the
    latter would let a system that returned two results score higher than one that
    returned ten containing the same two, which is the opposite of what precision@k is
    for -- catching recall bought by dumping junk into the context window.
    """
    if not relevant:
        return None
    return len(set(relevant) & set(ranked[:k])) / k


def mrr_at_k(ranked: Sequence[int], relevant: Sequence[int], k: int = MRR_K) -> float | None:
    """Reciprocal of the rank of the first relevant chunk, or 0 if none is in the top k.

    Rewards putting the right chunk *first*, which recall cannot see: a system that
    always ranks the answer tenth scores identically to one that ranks it first on
    recall@10, and very differently here.
    """
    truth = set(relevant)
    if not truth:
        return None
    for position, chunk_id in enumerate(ranked[:k], start=1):
        if chunk_id in truth:
            return 1.0 / position
    return 0.0


def ndcg_at_k(ranked: Sequence[int], relevant: Sequence[int], k: int = NDCG_K) -> float | None:
    """Normalised discounted cumulative gain, binary relevance.

    The metric that handles multi-chunk ground truth properly: recall counts how many
    relevant chunks appeared, MRR looks only at the first, and nDCG accounts for where
    every one of them landed.

    The ideal ranking places all relevant chunks first, so the denominator is capped at
    the number of relevant chunks -- an item with two relevant chunks cannot be penalised
    for failing to produce ten.
    """
    truth = set(relevant)
    if not truth:
        return None
    gain = sum(
        1.0 / math.log2(position + 1)
        for position, chunk_id in enumerate(ranked[:k], start=1)
        if chunk_id in truth
    )
    ideal = sum(1.0 / math.log2(position + 1) for position in range(1, min(len(truth), k) + 1))
    return gain / ideal if ideal else 0.0


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    """Every retrieval metric for one item, or the average across many."""

    recall: dict[int, float]
    precision: dict[int, float]
    mrr: float
    ndcg: float
    scored_items: int = 1

    def as_row(self) -> dict[str, float]:
        """Flattened for a results table or a JSON record."""
        row: dict[str, float] = {}
        for k, value in sorted(self.recall.items()):
            row[f"recall@{k}"] = value
        for k, value in sorted(self.precision.items()):
            row[f"precision@{k}"] = value
        row[f"mrr@{MRR_K}"] = self.mrr
        row[f"ndcg@{NDCG_K}"] = self.ndcg
        return row


def score_ranking(ranked: Sequence[int], relevant: Sequence[int]) -> RetrievalMetrics | None:
    """Every metric for one ranking, or None if the item has no relevant chunks."""
    if not relevant:
        return None
    recall = {k: recall_at_k(ranked, relevant, k) or 0.0 for k in RECALL_KS}
    precision = {k: precision_at_k(ranked, relevant, k) or 0.0 for k in PRECISION_KS}
    return RetrievalMetrics(
        recall=recall,
        precision=precision,
        mrr=mrr_at_k(ranked, relevant) or 0.0,
        ndcg=ndcg_at_k(ranked, relevant) or 0.0,
    )


def average_metrics(items: Sequence[RetrievalMetrics]) -> RetrievalMetrics:
    """Mean of each metric across items, carrying how many were scored.

    The count travels with the average because an arm scored over eleven items and one
    scored over eighty produce numbers that must not be compared as though equal.
    """
    if not items:
        return RetrievalMetrics(recall={}, precision={}, mrr=0.0, ndcg=0.0, scored_items=0)
    count = len(items)
    return RetrievalMetrics(
        recall={k: sum(item.recall.get(k, 0.0) for item in items) / count for k in items[0].recall},
        precision={
            k: sum(item.precision.get(k, 0.0) for item in items) / count for k in items[0].precision
        },
        mrr=sum(item.mrr for item in items) / count,
        ndcg=sum(item.ndcg for item in items) / count,
        scored_items=count,
    )
