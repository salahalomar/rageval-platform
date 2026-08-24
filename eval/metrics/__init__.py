"""Metric implementations.

Retrieval metrics are pure functions over ranked id lists: no database, no model, no
configuration. That is what makes them unit-testable against values computed by hand,
which is the only way anyone can check that the headline number in the README means what
it says.

Answer metrics need a judge and therefore cost money. They are separated for that reason
alone -- the retrieval half of every ablation arm runs free and offline.
"""

from eval.metrics.retrieval import (
    RetrievalMetrics,
    average_metrics,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    score_ranking,
)

__all__ = [
    "RetrievalMetrics",
    "average_metrics",
    "mrr_at_k",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "score_ranking",
]
