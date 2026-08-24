"""The harness around the metrics: span remapping, table rendering, Cohen's kappa.

None of this costs anything. The judge is the only paid part of Phase 7 and is covered
here only through its cost projection, which by construction calls nothing.
"""

import json
from pathlib import Path

import pytest

from eval.judge_agreement import Agreement, agreement_by_metric, cohens_kappa
from eval.relevance import EvidenceSpan
from eval.report import END_MARKER, START_MARKER, inject, render_table
from eval.runner import RunResult, percentile

# --- span-based relevance ---------------------------------------------------


def span(start: int, end: int, paper: str = "p1") -> EvidenceSpan:
    return EvidenceSpan(paper_id=paper, char_start=start, char_end=end)


def test_a_chunk_containing_the_evidence_is_relevant() -> None:
    assert span(100, 200).overlaps(span(50, 300))


def test_a_chunk_inside_the_evidence_is_relevant() -> None:
    # A 256-token chunk cannot contain a 512-token span, so demanding full coverage
    # would score the small-chunk arm at zero for structural reasons.
    assert span(0, 1000).overlaps(span(400, 800))


def test_a_disjoint_chunk_is_not_relevant() -> None:
    assert not span(0, 100).overlaps(span(200, 300))


def test_a_chunk_touching_only_at_the_boundary_is_not_relevant() -> None:
    assert not span(0, 100).overlaps(span(100, 200))


def test_a_barely_overlapping_chunk_is_not_relevant() -> None:
    # 10 characters of a 100-character span is 10%, below the 33% threshold.
    assert not span(0, 100).overlaps(span(90, 400))


def test_overlap_is_measured_against_the_shorter_span() -> None:
    # 40 of 100 characters is 40% of the shorter span and only 4% of the longer; scoring
    # against the longer would make every small chunk irrelevant by construction.
    assert span(0, 100).overlaps(span(60, 1100))


def test_evidence_in_a_different_paper_is_never_relevant() -> None:
    # The single most important guard: identical offsets in two papers mean nothing.
    assert not span(0, 500, "p1").overlaps(span(0, 500, "p2"))


def test_the_threshold_is_adjustable() -> None:
    assert span(0, 100).overlaps(span(80, 400), threshold=0.15)
    assert not span(0, 100).overlaps(span(80, 400), threshold=0.5)


def test_a_zero_length_span_does_not_divide_by_zero() -> None:
    assert not span(50, 50).overlaps(span(0, 100))


# --- Cohen's kappa ----------------------------------------------------------


def test_perfect_agreement_on_a_balanced_split_is_one() -> None:
    result = cohens_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"])
    assert result is not None
    assert result.kappa == pytest.approx(1.0)
    assert result.observed == 1.0


def test_kappa_is_zero_at_exactly_chance_agreement() -> None:
    # Human: a,a,b,b. Judge: a,b,a,b. Observed = 2/4 = 0.5.
    # Expected = (2/4 * 2/4) + (2/4 * 2/4) = 0.5. kappa = (0.5-0.5)/(1-0.5) = 0.
    result = cohens_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"])
    assert result is not None
    assert result.observed == 0.5
    assert result.expected == pytest.approx(0.5)
    assert result.kappa == pytest.approx(0.0)


def test_kappa_is_negative_when_agreement_is_worse_than_chance() -> None:
    result = cohens_kappa(["a", "a", "b", "b"], ["b", "b", "a", "a"])
    assert result is not None
    assert result.kappa < 0


def test_a_rubber_stamping_judge_scores_low_despite_high_raw_agreement() -> None:
    # The exact failure kappa exists to catch. The judge says "supported" every time; raw
    # agreement is 90%, which reads as excellent, while the judge exercised no judgement.
    human = ["supported"] * 9 + ["unsupported"]
    judge = ["supported"] * 10
    result = cohens_kappa(human, judge)
    assert result is not None
    assert result.observed == pytest.approx(0.9)
    assert result.kappa == pytest.approx(0.0)


def test_kappa_bands_are_reported() -> None:
    assert Agreement("f", 10, 0.9, 0.5, 0.85).interpretation == "almost perfect"
    assert Agreement("f", 10, 0.5, 0.5, 0.0).interpretation.startswith("poor")


def test_mismatched_or_empty_label_lists_return_nothing() -> None:
    assert cohens_kappa(["a"], ["a", "b"]) is None
    assert cohens_kappa([], []) is None


def test_agreement_is_computed_per_metric() -> None:
    records = [
        {"metric": "faithfulness", "human": "supported", "judge": "supported"},
        {"metric": "faithfulness", "human": "unsupported", "judge": "unsupported"},
        {"metric": "citation", "human": "correct", "judge": "incorrect"},
        {"metric": "citation", "human": "incorrect", "judge": "correct"},
    ]
    results = agreement_by_metric(records)
    assert {r.metric for r in results} == {"faithfulness", "citation"}
    assert next(r for r in results if r.metric == "faithfulness").kappa == pytest.approx(1.0)


# --- reporting --------------------------------------------------------------


def result(arm: str, recall5: float, items: int = 11) -> RunResult:
    return RunResult(
        arm=arm,
        config={},
        generation=None,
        git_sha="abc12345",
        git_dirty=False,
        started_at="2026-01-01T00:00:00+00:00",
        duration_s=1.0,
        python="3.12.0",
        golden_file="eval/golden/v1.jsonl",
        golden_items=items,
        scored_items=items,
        unanswerable_items=0,
        refusals=0,
        correct_refusals=0,
        false_refusals=0,
        metrics={"recall@1": 0.5, "recall@5": recall5, "mrr@10": 0.6, "ndcg@10": 0.6},
        latency_ms={"p50": 10.0, "p95": 20.0},
        items=[],
    )


def test_the_table_renders_one_row_per_arm() -> None:
    table = render_table([result("lexical-only", 0.4), result("hybrid-rerank", 0.8)])
    assert "lexical-only" in table
    assert "hybrid-rerank" in table
    assert table.count("\n") == 3  # header, separator, two rows


def test_rows_keep_matrix_order_rather_than_being_sorted_by_score() -> None:
    # Sorting by the winning metric would put the best arm first regardless of cost, and
    # would hide whether each addition earned its place over the row above it.
    table = render_table([result("worst", 0.1), result("best", 0.9)])
    assert table.index("worst") < table.index("best")


def test_an_unmeasured_metric_renders_as_a_dash_not_a_zero() -> None:
    # A column of zeros reads as "measured and bad"; a dash reads as "not measured".
    assert "—" in render_table([result("arm", 0.5)])  # refusal accuracy, no unanswerables


def test_the_scored_item_count_appears_in_the_table() -> None:
    # Eleven items and eighty items must not be compared as though equal.
    assert "11" in render_table([result("arm", 0.5, items=11)])


def test_rendering_no_results_says_so() -> None:
    assert "No results" in render_table([])


def test_injection_replaces_only_the_marked_block(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(f"before\n{START_MARKER}\nold\n{END_MARKER}\nafter\n", encoding="utf-8")
    assert inject(readme, "new table")
    text = readme.read_text(encoding="utf-8")
    assert "new table" in text
    assert "old" not in text
    assert text.startswith("before")
    assert text.rstrip().endswith("after")


def test_injection_is_idempotent(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(f"{START_MARKER}\n{END_MARKER}\n", encoding="utf-8")
    inject(readme, "table")
    assert not inject(readme, "table")  # second write changes nothing


def test_injection_without_markers_does_nothing(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("no markers here\n", encoding="utf-8")
    assert not inject(readme, "table")
    assert readme.read_text(encoding="utf-8") == "no markers here\n"


# --- run records ------------------------------------------------------------


def test_refusal_accuracy_is_none_when_nothing_was_unanswerable() -> None:
    # Reporting 0 or 1 for a question never asked would be inventing a measurement.
    assert result("arm", 0.5).refusal_accuracy is None


def test_refusal_accuracy_counts_correct_refusals() -> None:
    run = result("arm", 0.5)
    run.unanswerable_items = 4
    run.correct_refusals = 1
    assert run.refusal_accuracy == 0.25


def test_a_result_serialises_to_json(tmp_path: Path) -> None:
    # Every published number must trace to a file recording the config and the commit.
    path = result("arm", 0.5).write(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["arm"] == "arm"
    assert payload["git_sha"] == "abc12345"
    assert "config" in payload


def test_percentiles_handle_short_and_empty_sequences() -> None:
    assert percentile([], 0.95) == 0.0
    assert percentile([5.0], 0.95) == 5.0
    assert percentile([1.0, 2.0, 3.0], 0.5) == 2.0


def test_the_report_arm_order_matches_the_ablation_matrix() -> None:
    # The two lists are duplicated to avoid an import cycle, so a test keeps them honest:
    # an arm added to the matrix and forgotten here would sort to the end of the table.
    from eval.ablate import matrix
    from eval.report import ARM_ORDER

    assert set(ARM_ORDER) == {arm.name for arm in matrix()}


def test_results_are_loaded_in_matrix_order_not_filename_order(tmp_path: Path) -> None:
    # Filenames sort alphabetically, which would put dense-small before lexical-only and
    # destroy the least-to-most-machinery reading the table depends on.
    from eval.report import load_results

    for arm in ("rrf-k20", "lexical-only", "hybrid-rerank", "dense-small"):
        result(arm, 0.5).write(tmp_path)
    assert [r.arm for r in load_results(tmp_path)] == [
        "lexical-only",
        "dense-small",
        "hybrid-rerank",
        "rrf-k20",
    ]
