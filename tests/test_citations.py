"""Citation parsing and binding.

These are the tests behind "no answer without a citation". The rule is only as strong as
the parser: a marker form the parser misses becomes an uncited sentence that was actually
cited, and an out-of-range marker the parser accepts becomes an answer that looks sourced
and is not.
"""

import pytest

from rag.generate.citations import (
    MIN_WORDS_FOR_FACTUAL_SENTENCE,
    bind,
    is_factual,
    parse_markers,
    split_sentences,
    strip_markers,
)
from rag.generate.prompt import INSUFFICIENT_EVIDENCE
from rag.retrieve.types import Candidate


def candidate(chunk_id: int) -> Candidate:
    return Candidate(
        chunk_id=chunk_id,
        score=0.5,
        rank=1,
        content=f"content {chunk_id}",
        section_path="s",
        paper_id=f"p{chunk_id}",
        page_start=1,
        page_end=1,
        paper_title=f"Paper {chunk_id}",
    )


BLOCKS = [candidate(11), candidate(22), candidate(33)]


# --- marker parsing ---------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("A claim [1].", [1]),
        ("A claim [1][2].", [1, 2]),  # adjacent brackets
        ("A claim [1, 2].", [1, 2]),  # comma-separated inside one bracket
        ("A claim [1,2].", [1, 2]),  # no space
        ("A claim [1] and another [3].", [1, 3]),
        ("Repeated [2] and again [2].", [2, 2]),  # not deduplicated, order preserved
        ("Double digits [12].", [12]),
        ("No markers at all.", []),
    ],
)
def test_parses_every_marker_form_models_actually_produce(text: str, expected: list[int]) -> None:
    assert parse_markers(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Brackets with words [see above].",
        "An empty bracket [].",
        "Unclosed [1 and text.",
        "A range [1-3] is not a marker.",
        "Maths like x[i] indexing.",
    ],
)
def test_ignores_things_that_are_not_markers(text: str) -> None:
    assert parse_markers(text) == []


def test_strip_markers_leaves_readable_prose() -> None:
    assert strip_markers("Warmup stabilises training [1][2].") == "Warmup stabilises training."


# --- sentence splitting -----------------------------------------------------


def test_splits_on_sentence_boundaries() -> None:
    assert split_sentences("First one [1]. Second one [2].") == [
        "First one [1].",
        "Second one [2].",
    ]


def test_a_marker_after_the_full_stop_stays_with_its_own_sentence() -> None:
    # Models write both "claim [1]." and "claim. [1]" — the second must not strand the
    # marker onto the following sentence, which would make one sentence look uncited and
    # the next look doubly cited.
    sentences = split_sentences("First claim. [1] Second claim [2].")
    assert parse_markers(sentences[0]) == [1] or parse_markers(sentences[1]) == [1, 2]


def test_empty_answer_yields_no_sentences() -> None:
    assert split_sentences("   ") == []


# --- what counts as a factual sentence --------------------------------------


def test_the_refusal_sentence_never_needs_a_citation() -> None:
    # Refusing is the correct behaviour when evidence is absent. Demanding a citation for
    # it would make correct behaviour register as a violation.
    assert not is_factual(INSUFFICIENT_EVIDENCE)


def test_short_fragments_do_not_require_citations() -> None:
    assert not is_factual("In summary:")
    assert not is_factual("Yes.")


def test_a_real_claim_requires_a_citation() -> None:
    assert is_factual("Learning rate warmup stabilises transformer training substantially.")


def test_the_word_threshold_is_measured_without_markers() -> None:
    # Otherwise "[1][2][3][4][5] short" would pass the length test on its markers alone.
    bare = " ".join(["word"] * MIN_WORDS_FOR_FACTUAL_SENTENCE)
    assert is_factual(bare)
    assert not is_factual(" ".join(["word"] * (MIN_WORDS_FOR_FACTUAL_SENTENCE - 1)))


# --- binding ----------------------------------------------------------------


def test_markers_resolve_positionally_to_chunk_ids() -> None:
    # The model never sees a chunk id, so it cannot cite one. Markers are 1-based
    # positions into the blocks that were sent.
    result = bind("Warmup stabilises transformer training in practice [1][3].", BLOCKS)
    assert result.cited_chunk_ids == (11, 33)
    assert not result.uncited


def test_each_sentence_reports_its_own_citations() -> None:
    answer = "Warmup stabilises transformer training here [1]. Batch size was 512 tokens [2]."
    result = bind(answer, BLOCKS)
    assert [s.chunk_ids for s in result.sentences] == [(11,), (22,)]
    assert all(s.is_cited for s in result.sentences)


def test_an_uncited_factual_sentence_is_reported() -> None:
    answer = "Warmup stabilises transformer training here [1]. Batch size was 512 sequences."
    result = bind(answer, BLOCKS)
    assert result.uncited
    assert result.uncited_sentences == ("Batch size was 512 sequences.",)


def test_an_out_of_range_marker_is_invalid_not_silently_dropped() -> None:
    # An answer citing a block that does not exist is worse than an uncited one: it looks
    # sourced. Three blocks were supplied, so [7] cannot resolve.
    result = bind("Warmup stabilises transformer training substantially [7].", BLOCKS)
    assert result.invalid_markers == (7,)
    assert result.cited_chunk_ids == ()
    assert result.uncited


def test_marker_zero_is_invalid() -> None:
    # Numbering is 1-based; [0] is a model miscounting, not the first block.
    result = bind("Warmup stabilises transformer training substantially [0].", BLOCKS)
    assert result.invalid_markers == (0,)


def test_a_mix_of_valid_and_invalid_markers_keeps_the_valid_ones() -> None:
    result = bind("Warmup stabilises transformer training here [1][9].", BLOCKS)
    assert result.cited_chunk_ids == (11,)
    assert result.invalid_markers == (9,)
    assert result.uncited  # an invalid marker still taints the answer


def test_duplicate_citations_appear_once_in_the_summary() -> None:
    answer = "Warmup stabilises transformer training here [1]. It also helps convergence [1]."
    result = bind(answer, BLOCKS)
    assert result.cited_chunk_ids == (11,)


def test_a_refusal_is_recognised_and_not_marked_uncited() -> None:
    result = bind(INSUFFICIENT_EVIDENCE, BLOCKS)
    assert result.is_refusal
    assert not result.uncited
    assert result.cited_chunk_ids == ()


def test_an_empty_answer_is_not_uncited() -> None:
    # Nothing was claimed, so nothing is unsupported. An empty answer is a different
    # failure and is caught elsewhere.
    result = bind("", BLOCKS)
    assert not result.uncited
    assert result.sentences == ()


def test_binding_against_no_blocks_makes_every_marker_invalid() -> None:
    result = bind("Warmup stabilises transformer training substantially [1].", [])
    assert result.invalid_markers == (1,)
    assert result.cited_chunk_ids == ()
