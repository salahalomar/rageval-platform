"""Retrieval metrics, checked against values computed by hand.

These four functions decide every number the README will publish. A subtly wrong nDCG
produces a plausible table that is wrong in the same direction for every arm, which no
amount of staring at the output would reveal — so each is pinned to arithmetic a reader
can verify on paper.
"""

import math

import pytest

from eval.metrics.retrieval import (
    RetrievalMetrics,
    average_metrics,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    score_ranking,
)

# A ranking of ten chunks, of which 3 and 7 are relevant (positions 2 and 5).
RANKED = [1, 3, 5, 6, 7, 9, 11, 13, 15, 17]
RELEVANT = [3, 7]


# --- recall -----------------------------------------------------------------


def test_recall_counts_relevant_chunks_in_the_top_k() -> None:
    # top 1 = [1]              -> 0 of 2 = 0.0
    # top 3 = [1,3,5]          -> 1 of 2 = 0.5
    # top 5 = [1,3,5,6,7]      -> 2 of 2 = 1.0
    assert recall_at_k(RANKED, RELEVANT, 1) == 0.0
    assert recall_at_k(RANKED, RELEVANT, 3) == 0.5
    assert recall_at_k(RANKED, RELEVANT, 5) == 1.0


def test_recall_is_one_when_everything_relevant_is_found() -> None:
    assert recall_at_k([9, 8, 7], [7, 8, 9], 3) == 1.0


def test_recall_is_zero_when_nothing_relevant_is_found() -> None:
    assert recall_at_k([1, 2, 3], [99], 3) == 0.0


def test_recall_cannot_exceed_one_with_duplicate_ids() -> None:
    # A ranking containing the same chunk twice must not score 2 of 1.
    assert recall_at_k([7, 7, 7], [7], 3) == 1.0


def test_a_k_beyond_the_ranking_uses_what_exists() -> None:
    assert recall_at_k([3], [3, 7], 10) == 0.5


# --- precision --------------------------------------------------------------


def test_precision_divides_by_k_not_by_results_returned() -> None:
    # Dividing by results-returned would let a system returning two chunks beat one
    # returning ten containing the same two — the opposite of what precision@k is for.
    assert precision_at_k(RANKED, RELEVANT, 5) == pytest.approx(2 / 5)
    assert precision_at_k(RANKED, RELEVANT, 1) == 0.0


def test_precision_of_a_short_ranking_still_divides_by_k() -> None:
    # One relevant chunk out of a possible five, even though only one was returned.
    assert precision_at_k([3], [3], 5) == pytest.approx(1 / 5)


# --- MRR --------------------------------------------------------------------


def test_mrr_is_the_reciprocal_of_the_first_relevant_rank() -> None:
    # First relevant chunk (3) is at position 2 -> 1/2
    assert mrr_at_k(RANKED, RELEVANT) == 0.5


def test_mrr_rewards_ranking_the_answer_first() -> None:
    # The distinction recall@10 cannot see: both rankings contain the answer.
    assert mrr_at_k([7, 1, 2], [7]) == 1.0
    assert mrr_at_k([1, 2, 7], [7]) == pytest.approx(1 / 3)


def test_mrr_is_zero_when_nothing_relevant_is_in_the_top_k() -> None:
    assert mrr_at_k([1, 2, 3, 4, 5], [99], k=5) == 0.0


def test_mrr_respects_its_cutoff() -> None:
    # The relevant chunk sits at position 11, outside MRR@10.
    ranking = [*range(1, 11), 99]
    assert mrr_at_k(ranking, [99], k=10) == 0.0
    assert mrr_at_k(ranking, [99], k=11) == pytest.approx(1 / 11)


# --- nDCG -------------------------------------------------------------------


def test_ndcg_is_one_for_a_perfect_ranking() -> None:
    assert ndcg_at_k([3, 7, 1, 2], [3, 7]) == pytest.approx(1.0)


def test_ndcg_hand_computed_for_a_partial_ranking() -> None:
    # Relevant at positions 2 and 5:
    #   DCG  = 1/log2(3) + 1/log2(6) = 0.6309298 + 0.3868528 = 1.0177826
    #   IDCG = 1/log2(2) + 1/log2(3) = 1.0       + 0.6309298 = 1.6309298
    #   nDCG = 1.0177826 / 1.6309298 = 0.6240...
    dcg = 1 / math.log2(3) + 1 / math.log2(6)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(RANKED, RELEVANT) == pytest.approx(dcg / idcg)
    assert ndcg_at_k(RANKED, RELEVANT) == pytest.approx(0.62405, abs=1e-5)


def test_ndcg_is_zero_when_nothing_relevant_is_retrieved() -> None:
    assert ndcg_at_k([1, 2, 3], [99]) == 0.0


def test_ndcg_ideal_is_capped_at_the_number_of_relevant_chunks() -> None:
    # An item with one relevant chunk must not be penalised for failing to produce ten.
    assert ndcg_at_k([5], [5]) == pytest.approx(1.0)


def test_ndcg_prefers_the_earlier_of_two_equal_recalls() -> None:
    # Both rankings recall the same single chunk; only nDCG and MRR see the difference.
    early = ndcg_at_k([9, 1, 2, 3], [9])
    late = ndcg_at_k([1, 2, 3, 9], [9])
    assert early is not None and late is not None
    assert early > late
    assert recall_at_k([9, 1, 2, 3], [9], 4) == recall_at_k([1, 2, 3, 9], [9], 4)


def test_ndcg_handles_multi_chunk_ground_truth() -> None:
    # The case recall flattens and MRR ignores: three relevant chunks, two found.
    value = ndcg_at_k([1, 2, 3], [1, 3, 99])
    assert value is not None
    assert 0.0 < value < 1.0


# --- the zero-relevant case -------------------------------------------------


def test_every_metric_returns_none_when_there_is_nothing_to_recall() -> None:
    # Unanswerable items have no ground truth. Scoring them zero would drag every
    # average down in proportion to how many refusal cases the set contains, punishing
    # the set for testing refusal at all.
    assert recall_at_k([1, 2], [], 5) is None
    assert precision_at_k([1, 2], [], 5) is None
    assert mrr_at_k([1, 2], []) is None
    assert ndcg_at_k([1, 2], []) is None
    assert score_ranking([1, 2], []) is None


def test_an_empty_ranking_scores_zero_rather_than_none() -> None:
    # Retrieving nothing for an answerable question is a real failure and must be scored,
    # unlike an item that had no ground truth to begin with.
    metrics = score_ranking([], [3])
    assert metrics is not None
    assert metrics.recall[5] == 0.0
    assert metrics.mrr == 0.0


# --- aggregation ------------------------------------------------------------


def test_scoring_produces_every_reported_metric() -> None:
    metrics = score_ranking(RANKED, RELEVANT)
    assert metrics is not None
    row = metrics.as_row()
    assert set(row) >= {"recall@1", "recall@5", "mrr@10", "ndcg@10", "precision@5"}


def test_averaging_takes_the_mean_of_each_metric() -> None:
    perfect = score_ranking([3], [3])
    missed = score_ranking([99], [3])
    assert perfect is not None and missed is not None
    mean = average_metrics([perfect, missed])
    assert mean.recall[1] == 0.5
    assert mean.mrr == 0.5
    assert mean.scored_items == 2


def test_the_scored_count_travels_with_the_average() -> None:
    # An arm scored over eleven items and one scored over eighty produce numbers that
    # must not be compared as though equal, so the count is carried, not discarded.
    single = score_ranking([3], [3])
    assert single is not None
    assert average_metrics([single]).scored_items == 1


def test_averaging_nothing_is_safe() -> None:
    empty = average_metrics([])
    assert empty.scored_items == 0
    assert empty.mrr == 0.0


def test_metrics_are_deterministic() -> None:
    # Two runs of the harness on one commit must produce identical numbers, and that
    # starts with the arithmetic being a pure function of its inputs.
    assert score_ranking(RANKED, RELEVANT) == score_ranking(RANKED, RELEVANT)


def test_relevant_order_does_not_affect_the_result() -> None:
    # Ground truth is a set; the order it happens to be listed in must not change a score.
    assert score_ranking(RANKED, [3, 7]) == score_ranking(RANKED, [7, 3])


def test_metrics_row_is_json_serialisable() -> None:
    metrics = RetrievalMetrics(recall={1: 0.5}, precision={1: 0.5}, mrr=0.5, ndcg=0.5)
    assert all(isinstance(value, float) for value in metrics.as_row().values())
